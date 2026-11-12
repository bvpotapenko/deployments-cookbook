# Run Book: mllm-sglang

This deployment serves Qwen/Qwen3.8-27B with the latest SGLang NPU line on Ascend 910B. The model runs
the native `qwen3_5` architecture (the multimodal path inherits Qwen3-VL);
the units are short videos. One process serves the full-vocabulary
log-probability surface over both cards, one request at a time, and the chat
surface beside it, concurrently and batched. The runner must not overlap the
two: a score that shares a batch with chat traffic stashes their rows too, and
its row count then fails the score.

## Model parameters

| model | revision | cards | dtype | sampling |
|---|---|---|---|---|
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 2 (TP=2) | bfloat16 | temperature 1.0, top_p 0.95, top_k 20 |

The model needs `KOT_SGLANG_MEM_FRACTION=0.85` on 2x Ascend 910B: the weights
hold 25.64 GiB per card and the fraction must cover them, the hybrid state
cache and the 24,576-token KV window. At 0.89 the engine sizes
`max_mamba_cache_size=7` and `max_total_num_tokens=25626`; at the 0.45
default (or 0.88) the budget goes negative (or 131 tokens short).

The sampling values are the checkpoint's own `generation_config.json`. The
seed arrives with the request; 42 is the convention.

## Environment: Ascend 910B

The stack runs on CPython 3.12 with the SGLang NPU build and the CANN 9.2.1 image. Point `KOT_SGLANG_PYTHON` at the verified engine venv.

```bash
uv venv /root/kotvenv-sglang-ocr --python 3.12
uv pip install --python /root/kotvenv-sglang-ocr/bin/python "sglang==0.5.23" "flash-attn-4>=4.0.0b18"
uv pip install --python /root/kotvenv-sglang-ocr/bin/python --no-deps \
  "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-sglang-ocr/bin/python --no-deps \
  torchcodec einops "flash-linear-attention==0.2.2"
```

sglang 0.5.23 pairs with torch 2.14.1+cu130 and pins transformers 5.19.0. It
requires a flash-attn-4 prerelease, which uv takes only as a direct requirement;
`--prerelease=allow` would pull other prereleases too. torchvision 0.29.x pairs
with torch 2.14 (0.28.x needs torch 2.13 and dies at import). torchcodec decodes
the videos; the host needs FFmpeg 6 shared libraries. flash-linear-attention
0.2.2 and einops are required by the native qwen3_5 layers: without them the
GatedDeltaNet layers fall back to the reference PyTorch implementation,
which is slow and OOMs a 32G card on a ~12.4k-token video prefill. fla
0.3.0+ on pypi are partial distributions (no `ops/`); 0.2.2 is the pin.

### Layout

```text
/opt/engines/mllm-sglang         this directory
/models/Qwen/Qwen3.8-27B         the weights; a hub snapshot directory
```

### Transfer

```bash
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./mllm-sglang/ root@$HOST:/opt/engines/mllm-sglang/
ssh -p $PORT root@$HOST 'chmod +x /opt/engines/mllm-sglang/*.sh'
```

### Start

```bash
KOT_SGLANG_PYTHON=/root/kotvenv-sglang-ocr/bin/python \
  ./serve.sh --port 8090 --model /models/Qwen/Qwen3.8-27B --device npu
```

The launcher installs the patch layer into the venv's site-packages (module
+ `.pth`) on every start, runs the model's-own-vocabulary warmup, then binds
127.0.0.1.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_SGLANG_PYTHON` | unset | the interpreter carrying sglang; set it on a provisioned host |
| `KOT_SGLANG_MEM_FRACTION` | `0.45` | `mem_fraction_static`; must cover the weights per card, the hybrid state cache and the KV window (0.89 for Qwen3.8-27B on 2x Ascend 910B) |
| `KOT_SGLANG_STASH_DIR` | set by the launcher | the patch layer's stash root |
| `KOT_SGLANG_DEBUG` | unset | logs every patch event to a /tmp trace |
| `SGLANG_LOGPROB_CHUNK_SIZE` | `512` (set by the launcher) | rows per input-logprob chunk of a prefill (SGLang's default is 2048); a smaller chunk lowers the peak on the card and the size of each stash archive in host memory |

Engine pins in kot_server.py: `chunked_prefill_size=-1` (a chunked extend
samples a token per chunk and desynchronises the stash), `disable_overlap_
schedule=True` (the overlap scheduler's phantom decode row),
`mem_fraction_static` through `KOT_SGLANG_MEM_FRACTION` (the full-vocab
transient beside the weights; the default 0.45 suits models whose weights
fit under it),
`max_prefill_tokens=16384`, `context_length=24576` (a ~12.4k-token prompt
plus an 8,192-token decode; a shorter window cuts the decode),
`mm_process_config` (fps 2.0 and the per-frame cap), `mm_processor_worker_num=2`
(one worker runs the video pass on the event loop),
`enable_deterministic_inference` and the ascend attention backend on Ascend.

## Video contract

The frozen video inputs carry no ids. The deployment renders the pinned chat
prompt (the checkpoint's chat template, `enable_thinking` false, the prompt
text "Describe what you see in the video.") and consumes ids verbatim from
that render: the raw form (one video placeholder) on a decode, the raw form
plus the generated tail on a replay, the head asserted.

The video rides to the engine as `video_data` (the attachment path). The
engine's own frame step samples and resizes by rules of its own, so the patch
layer replaces it: frames come from `Qwen3VLVideoProcessor.sample_frames` at
fps 2.0, decoded on the CPU and unresized, and the HF processor resizes them on
the CPU with `cap_pixels_per_frame` true and expands the placeholder into the
per-frame timestamp structure.

The engine keeps the request ids as sent. For ids with a video, stock SGLang
decodes the ids to text and tokenizes that text again. A sampled tail is not
always the tokenizer's own split, so the engine can get different ids (issue
#79). The patch layer replaces that path. The HF processor gets only the video
placeholder, and its expansion replaces the placeholder in the request ids.
All other ids stay as sent. The patch layer writes the engine ids to
`last_video_ids.json` in the stash dir. A request with ids and more than one
video, or with a video and other media, fails in the engine.

The deployment also computes the expanded template through the checkpoint's
HF processor (the same pins) and guards every score request on it. The prompt
token count must agree with the expansion, and the engine ids must equal the
expanded template plus the tail at each position. A missing note, a changed id
or a frame selection that disagrees with the reference fails the request
instead of reading as numbers.

A video unit costs at most ~12.4k tokens (the checkpoint's own pixel budget:
`longest_edge 25165824` per video). The stored rows are the pre-sampling
distributions: the log-softmax of the raw logits, float32, over the full
248,320-wide vocabulary.

## Check

Run the contract self-check against one local video:

```bash
python3 self_check.py --deployment . \
  --model /models/Qwen3.8-27B --device npu \
  --video /inputs/video_10014.mp4
```

It checks `/kot/info`, video decode, replay prefill, and both chat modes.
Stop an existing server first.

The identity surface answers with the engine's own facts:

```bash
curl -s http://127.0.0.1:8090/kot/info | python3 -m json.tool
```

The generation surface takes a video and writes its answer to `out_dir`:

```bash
curl -s http://127.0.0.1:8090/kot/infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt_id": "probe", "out_dir": "/tmp/kot-probe",
       "attachments": {"video": "/path/to/video.mp4"}}' \
  | python3 -m json.tool
cat /tmp/kot-probe/text.txt
```

The chat surface takes one user message, a video part and a text part, in
either order. The url is `file:///abs/path`, a plain absolute path or a
`data:video/...;base64,...` URI; `"stream": true` answers in SSE:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages": [{"role": "user", "content": [
         {"type": "video_url", "video_url": {"url": "file:///path/to/video.mp4"}},
         {"type": "text", "text": "Describe what you see in the video."}]}],
       "max_completion_tokens": 64, "seed": 42}' \
  | python3 -m json.tool
```

On 2x Ascend 910B the boot needs `KOT_SGLANG_MEM_FRACTION=0.85` (issue #26):
the engine then sizes `max_mamba_cache_size=7` and
`max_total_num_tokens=25626`.
