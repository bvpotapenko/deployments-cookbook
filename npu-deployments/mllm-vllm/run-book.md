# Run Book: mllm-vllm

This deployment serves Qwen/Qwen3.8-27B with vLLM-Ascend 0.32.0rc1 on Ascend 910B. The model runs the
native `qwen3_5` architecture; the units are short videos. One process serves
the full-vocabulary log-probability surface on one card, one request at a
time.

## Model parameters

| model | revision | cards | dtype | sampling |
|---|---|---|---|---|
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 1 (TP=1) | bfloat16 | temperature 1.0, top_p 0.95, top_k 20 |

The sampling values are the checkpoint's own `generation_config.json`. The
seed arrives with the request; 42 is the convention.

## Environment

One stack uses CPython 3.12, torch 2.11.0, torch_npu 2.11.0,
torchvision 0.26.0 and torchcodec 0.17.0. The CANN image is 9.2.1.

```bash
# networked host
uv venv /scratch/kot-vllm-venv --python 3.12
uv pip install --python /scratch/kot-vllm-venv/bin/python "vllm==0.32.0"
uv pip install --python /scratch/kot-vllm-venv/bin/python \
  "torch==2.11.0" "torchvision==0.26.0" "torchcodec==0.17.0"

# offline host: build the closure, then install from it
./make_wheels.sh
uv venv --seed /scratch/kot-vllm-venv --python 3.12
/scratch/kot-vllm-venv/bin/pip install --no-index --no-cache-dir \
  --find-links ./wheels "vllm==0.32.0" "torch==2.11.0" \
  "torch_npu==2.11.0" "torchvision==0.26.0" "torchcodec==0.17.0"
```

vLLM-Ascend 0.32.0rc1 uses the CANN 9.2.1 image and the torch_npu
2.11.0 platform stack. TorchCodec 0.17.0 is built for this torch ABI. The
matching aarch64 engine wheels carry the Qwen video processor and NPU runtime.

`make_wheels.sh` builds the vLLM-Ascend 0.32.0rc1 closure for the Ascend
platform, including the `torchcodec==0.17.0` wheel. The exact pins live in
`wheels-requirements.txt`; the script fails unless all five platform wheels
are present. Build the closure on a matching aarch64 host or from the
vLLM-Ascend source tree before running `docker build` or an offline serve.
`arctic_inference==0.1.1` has no wheel on the configured mirror, so the
builder fetches its pinned source archive from PyPI, verifies it, and builds
the wheel while the host is networked.

Without `KOT_PYTHON`, and with no ambient vllm, the launcher runs the offline
install itself, into `KOT_VENV`.

### Layout

```text
/opt/engines/mllm-vllm           this directory
/models/Qwen/Qwen3.8-27B         the weights; a hub snapshot directory
```

### Transfer

```bash
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./mllm-vllm/ root@$HOST:/opt/engines/mllm-vllm/
ssh -p $PORT root@$HOST 'chmod +x /opt/engines/mllm-vllm/*.sh'
```

### Start

```bash
ASCEND_RT_VISIBLE_DEVICES=0 \
KOT_PYTHON=/scratch/kot-vllm-venv/bin/python \
  ./serve.sh --port 8090 --model /models/Qwen/Qwen3.8-27B --device npu
```

The model uses one visible Ascend 910B card. Set
`ASCEND_RT_VISIBLE_DEVICES=0`; the engine then uses TP=1. The first boot
compiles the engine; a warm compile cache makes the second boot faster.

One launch profile serves both surfaces. Under the V2 runner the score path
has no full-vocab gather, so the capture surfaces no longer need a small
chunk: the API-server default (2048) fits the replay's memory budget with
the same margin as 128 did, and capture and speed share chunk shapes.
Measured at 2048: the 64-row probe replay completes in ~2.8 s engine time,
the 4,096-row segmented replay in ~118 s wall, byte-identical across runs.

The chunk size is part of the numerics: a chunked prefill splits the
attention and GatedDeltaNet state passes at chunk boundaries, so rows
produced at one size need not be bit-identical to rows produced at another.
`/kot/info` reports `max_num_batched_tokens` and `replay_segment_rows`, so
every capture carries its own.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_PYTHON` | unset | the interpreter carrying vllm; set it on a provisioned host |
| `KOT_VENV` | `/scratch/kot-vllm-venv` | where the launcher builds the offline venv; it falls back to `/tmp/kot-vllm-venv` |
| `KOT_VLLM_NPU_MEM` | `0.91` | `gpu_memory_utilization`; the weights take 82% of each card and the V2 runner's accounting is tighter than V1's - 0.89 leaves the KV 20 MiB under even a 20,480-token window and the engine refuses to boot. 0.91 sizes 38,229 KV tokens with ~2.4 GiB still free beside the pool |
| `KOT_VLLM_MAX_MODEL_LEN` | `24576` | the window: a video prompt up to the pixel budget (12,288 video tokens plus the template) plus the 8192-token decode. vLLM checks only the prompt against it and caps the output at the window without an error; the server then refuses a decode with `allow_eos` false that is short of `max_new_tokens` |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | the chunk size of every prefill, shared by capture and speed. The size is part of the numerics and is reported in `/kot/info` |
| `KOT_VLLM_REPLAY_SEGMENT_ROWS` | `256` | the rows per replay segment; a long tail arrives as several requests (an 8,192-row tail is 32 segments). Fixed, not derived: segment boundaries change the last chunk's shapes, so rows are reproducible only at a fixed size. Reported in `/kot/info` |
| `KOT_VLLM_MAX_NUM_SEQS` | `4` | the Ascend-graph profiling KV and the capture sizes scale with the concurrent-request count; the API-server default asks for too much temporary KV during boot |
| `KOT_VLLM_REQUEST_TIMEOUT` | `1800` | a prefill the engine does not answer within this many seconds: every thread's stack is dumped to the log and the process exits - a dead server with a named culprit beats a hung one |
| `KOT_VLLM_MOE_BACKEND` | `triton` | flashinfer's cutlass MoE dies on sm_120 without nvcc |
| `KOT_VLLM_DTYPE` | unset | numeric cross-check override |
| `KOT_VLLM_COMPILATION` | unset | `eager` disables Ascend graphs |
| `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` | `0`, set by the launcher | vLLM's limit on the media one request may decode. It fails a 1080p clip longer than about 57 s on chat only, not on `/kot/score`, so the launcher turns it off and both routes take the same clips |
| `VLLM_NO_USAGE_STATS` | `1`, set by the server | vLLM's usage report to its developers, and its local copy in `~/.config/vllm/usage_stats.json`. `1` turns both off. The server replaces any other value |

### Host memory

The V2 runner removed the V1 floor: no full-prompt pinned buffer, no
full-prompt message, no top-k. The measured host peak during a 4,096-row
replay was 56.5 GiB against the container's 120.7 GiB limit (V1 needed
~122 GiB at the final relay and the cgroup OOM-killed the server). The
server holds one block of 256 decode steps, or one replay segment, at a
time. Each one goes into `log_probs.npy` through a memmap at the final
shape, and then the server releases it.

### The V2 runner

The launcher pins `VLLM_USE_V2_MODEL_RUNNER=1`. The replay scores only the
tail rows against fixed token ids (`prompt_logprob_token_ids` +
`prompt_logprob_start`): the engine keeps `[rows, vocab]` ids, mask and
scores on the GPU at ~3.08 MiB per row from the first chunk to the last,
so a long tail arrives as segments of at most `KOT_VLLM_REPLAY_SEGMENT_ROWS`
(default 256: the measured 2,431 MiB free with a request in flight shrinks
by ~3.08 MiB per tail row, and the video encoder asks up to ~1.2 GiB more
during the prefill - 640 rows OOMs it, measured; a 4,096-row tail is 16
segments). Each segment re-feeds the prompt and the video, cut off right after its last scored token -
causality makes the cut exact. Segment boundaries change the last chunk's
shapes, so replay rows are reproducible only at a fixed segment size;
`/kot/info` reports both the chunk size and the segment size, and every
capture carries its own. The 64-row probe replay completes in ~6 s end to
end. The deployment computes the expanded prompt length through the
checkpoint's own HF processor (cached per video, the same pins including
the per-frame pixel cap) and verifies the engine's expansion against it,
so the absolute start index cannot drift. The rows arrive already
scattered, `[rows, vocab]` float32 in token-id order. The scores come from
a Triton fp32 kernel, not torch `log_softmax`: expect different low-order
bits from the V1 mechanism.

## Video contract

The frozen video inputs carry no ids. The deployment renders the pinned chat
prompt (the checkpoint's chat template, `enable_thinking` false, the prompt
text "Describe what you see in the video.") and consumes ids verbatim from
that render: the raw form (one video placeholder) on a decode, the raw form
plus the generated tail on a replay, the head asserted.

Chat and the score path load the video with the same vLLM `VideoMediaIO`, built
from one value in the server (`MEDIA_IO_KWARGS`): the `qwen3_vl` loader, the
torchcodec codec and fps 2.0. That loader's arithmetic and decoder settings are
the HF sampler's, so the frame selection matches the reference's. The video
rides to the engine as `(frames, metadata)`, under the reference per-frame pixel
cap (`cap_pixels_per_frame`) and `do_sample_frames=False` - the frames ARE the
sampled ones, the timestamps come from the metadata. vLLM's processor expands
the placeholder into the per-frame timestamp structure.

The chat surface is vLLM's own app, so its prompt comes through
`default_chat_template_kwargs` with `enable_thinking` false. A `video_url` is a
`file://` path or a base64 data URI. The server allows any local path, and it
binds only to 127.0.0.1.

A video unit costs ~12.4k tokens (the checkpoint's own pixel budget:
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
       "attachments": {"video": "/inputs/video.mp4"}}' \
  | python3 -m json.tool
cat /tmp/kot-probe/text.txt
```

The answer's text lands in `out_dir` and describes the video.

The chat surface takes a plain OpenAI request, with the checkpoint's own
sampling from `generation_config.json`:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-27B",
       "messages": [{"role": "user", "content": "ping"}],
       "max_tokens": 50,
       "temperature": 1.0, "top_p": 0.95, "top_k": 20}' \
  | python3 -m json.tool
```

A video request has the harness's shape: the video, then the text part.
`data:video/mp4;base64,...` works as the `url` too.

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3.8-27B",
       "messages": [{"role": "user", "content": [
         {"type": "video_url", "video_url": {"url": "file:///inputs/video.mp4"}},
         {"type": "text", "text": "Describe what you see in the video."}]}],
       "max_completion_tokens": 64, "seed": 42,
       "temperature": 1.0, "top_p": 0.95, "top_k": 20}' \
  | python3 -m json.tool
```

The answer describes the video, and `usage.prompt_tokens` equals the count the
score path computes for the same clip:

```bash
cd /opt/engines/mllm-vllm && /scratch/kot-vllm-venv/bin/python -c '
import kot_serve
kot_serve._MODEL_DIR = "/models/Qwen/Qwen3.8-27B"
print(len(kot_serve._expanded_template_ids("/inputs/video.mp4")))'
```

The default boot OOMed engine init on one Ascend 910B card until the
`KOT_VLLM_MAX_NUM_SEQS` pin landed (issue #24): the temporary KV of the
Ascend-graph profiling scales with the concurrent-request count, not with the
window, and the 256 API-server default asked for 6.12 GiB beside the weights.
The engine boots without `eager`. `/kot/info` reads the vocab size through
the engine's own resolver (issue #25): the multimodal wrapper configs nest it
in their text config, and the flat read crashed the boot after init.
