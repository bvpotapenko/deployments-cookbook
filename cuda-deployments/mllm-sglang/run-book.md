# Run Book: mllm-sglang

This deployment serves Qwen/Qwen3.8-27B with SGLang 0.5.22. The model runs
the native `qwen3_5` architecture (the multimodal path inherits Qwen3-VL);
the units are short videos. One process serves the full-vocabulary
log-probability surface over both cards, one request at a time.

## Model parameters

| model | revision | cards | dtype | sampling |
|---|---|---|---|---|
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 2 (TP=2) | bfloat16 | temperature 1.0, top_p 0.95, top_k 20 |

The sampling values are the checkpoint's own `generation_config.json`. The
seed arrives with the request; 42 is the convention.

## Environment

The stack runs on CPython 3.12. An existing venv with these pins works;
point `KOT_SGLANG_PYTHON` at it.

```bash
uv venv /root/kotvenv-sglang-ocr --python 3.12
uv pip install --python /root/kotvenv-sglang-ocr/bin/python "sglang==0.5.22"
uv pip install --python /root/kotvenv-sglang-ocr/bin/python --no-deps \
  "torchvision==0.29.1" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-sglang-ocr/bin/python --no-deps \
  torchcodec einops "flash-linear-attention==0.2.2"
```

sglang 0.5.22 pairs with torch 2.14.1+cu130; torchvision 0.29.x pairs with
torch 2.14 (0.28.x needs torch 2.13 and dies at import). torchcodec decodes
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
  ./kot_serve.sh --port 8090 --model /models/Qwen/Qwen3.8-27B --device cuda
```

The launcher installs the patch layer into the venv's site-packages (module
+ `.pth`) on every start, runs the model's-own-vocabulary warmup, then binds
127.0.0.1.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_SGLANG_PYTHON` | unset | the interpreter carrying sglang; set it on a provisioned host |
| `KOT_SGLANG_STASH_DIR` | set by the launcher | the patch layer's stash root |
| `KOT_SGLANG_DEBUG` | unset | logs every patch event to a /tmp trace |

Engine pins in kot_server.py: `chunked_prefill_size=-1` (a chunked extend
samples a token per chunk and desynchronises the stash), `disable_overlap_
schedule=True` (the overlap scheduler's phantom decode row),
`mem_fraction_static=0.45` (the full-vocab transient beside the weights),
`max_prefill_tokens=16384`, `context_length=20480`,
`enable_deterministic_inference` and the triton attention backend on CUDA.

## Video contract

The frozen video inputs carry no ids. The deployment renders the pinned chat
prompt (the checkpoint's chat template, `enable_thinking` false, the prompt
text "Describe what you see in the video.") and consumes ids verbatim from
that render: the raw form (one video placeholder) on a decode, the raw form
plus the generated tail on a replay, the head asserted.

The video rides to the engine as `video_data` (the attachment path, fps 2.0);
the engine's own multimodal processor samples the frames and expands the
placeholder into the per-frame timestamp structure. The deployment also
computes the expanded id count through the checkpoint's HF processor (the
same pins) and guards every request on it: a retokenize fallback or a frame
selection that disagrees with the reference fails the request instead of
reading as numbers.

A video unit costs ~12.4k tokens (the checkpoint's own pixel budget:
`longest_edge 25165824` per video). The stored rows are the pre-sampling
distributions: the log-softmax of the raw logits, float32, over the full
248,320-wide vocabulary.

## Check

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

On 2x RTX 5090 this boot is blocked: `kot_server.py` hard-pins
`mem_fraction_static=0.45` and the hybrid state cache budget goes negative
at TP=2 (issue #26). The boot needs that knob.
