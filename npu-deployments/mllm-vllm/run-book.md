# Run Book: mllm-vllm

This deployment serves Qwen/Qwen3.8-27B with vLLM-Ascend 0.30.0 on Ascend 910B. The model runs the
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

One stack uses CPython 3.12, torch 2.10.0, torch_npu 2.10.0.post7,
torchvision 0.25.0, torchaudio 2.10.0 and the CPU-only torchcodec 0.10.0
wheel. The CANN image is 9.2.1.

| component | tested value | source |
|---|---|---|
| host | 8 x Ascend 910B2, `npu-smi 26.1.1` | host installation |
| CANN | 9.2.1, Ubuntu 22.04, CPython 3.12 | `quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12-devel` |
| vLLM / vLLM-Ascend | 0.30.0 / 0.30.0 | Huawei PyPI mirrors in `make_wheels.sh` |
| TorchCodec | 0.10.0 CPU-only aarch64 wheel | source fetched by `make_wheels.sh`; built inside the Docker image against Ubuntu FFmpeg with `ENABLE_CUDA=OFF` |
| Flash Linear Attention NPU | 26.10.0 A2 aarch64 wheel | `flash-linear-attention-npu-a2` from the Ascend mirror; provides `fla_npu` for Qwen3.8's gated-delta attention |

```bash
# The host only downloads wheels and the TorchCodec source archive. It does
# not install torch, torch_npu, torchvision, or TorchCodec.
./make_wheels.sh

# Build the image. It installs one exact torch platform stack, then builds
# and installs TorchCodec against that stack.
docker build -t kot-mllm-ascend-910b:latest .
```

vLLM-Ascend 0.30.0 uses the CANN 9.2.1 image and the torch_npu
2.10.0.post7 platform stack. TorchCodec 0.10.0 is the release line paired
with torch 2.10.0. On Linux aarch64, the Docker build compiles it from source
with CUDA disabled; the default binary wheel selects CUDA and requires
libnvrtc/libcuda.
matching aarch64 engine wheels carry the Qwen video processor and NPU runtime.

`make_wheels.sh` builds the vLLM-Ascend 0.30.0 closure for the Ascend
platform, including the CPU-only `torchcodec==0.10.0` wheel. The exact pins live in
`wheels-requirements.txt`; the script fails unless all six platform wheels
are present. Build the closure on a matching aarch64 host or from the
vLLM-Ascend source tree before running `docker build` or an offline serve.
`torchcodec==0.10.0` is fetched from its verified upstream release archive by
`make_wheels.sh`, then built inside the Docker image after the matching torch
stack is installed. The image installs Ubuntu FFmpeg development libraries and
`pkg-config`; TorchCodec links the system FFmpeg selected by pkg-config with
`ENABLE_CUDA=OFF`. The build sets
`TORCH_DEVICE_BACKEND_AUTOLOAD=0` so importing torch for setup does not load
the Ascend backend before the runtime library path is configured, and sets
`I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1` as required by TorchCodec's
wheel-build guard. The host does not install any torch package. Qwen3.8 uses
the gated-delta attention path; `make_wheels.sh` therefore also fetches
`flash-linear-attention-npu-a2==26.10.0` from the configured Ascend mirror and
`serve.sh` installs it into the same offline venv as vLLM-Ascend.
The NPU launcher sets `VLLM_WORKER_MULTIPROC_METHOD=spawn`; the default fork
mode would try to reinitialize Ascend's NPU runtime in the engine child.

The CANN setup script may omit the ATB library directory from
`LD_LIBRARY_PATH`. `serve.sh` restores the image's ATB path before importing
the serving stack. This is required for `torch_npu`'s
`libop_plugin_atb.so` and must remain in the launcher rather than depending on
an interactive shell environment.
The vLLM-Ascend 0.30.0 wheel carries an older full-vocabulary log-probability
Triton tile than the matching vLLM sampler. On 910B2 that tile exceeds the
unified buffer when `/kot/score` requests all 248,320 vocabulary entries.
`serve.sh` applies `patch_vllm_ascend_logprob.py` after installing the pinned
wheel. The patch keeps the same fp32 log-softmax calculation while emitting
the requested entries in 1,024-entry tiles. It checks the expected source
shape and fails if a future wheel changes it.
`arctic_inference==0.1.1` has no wheel on the configured mirror, so the
builder fetches its pinned source archive from PyPI, verifies it, and builds
the wheel while the host is networked.

Only `wheels/` is the active wheelhouse. Delete obsolete directories such as
`wheels-030` and `wheels-032-backup`: they contain failed or superseded
matrices, are not copied into the image, and cannot be used as rollbacks. The
builder removes stale TorchCodec wheels from the active wheelhouse before
rebuilding.

The CPU-only TorchCodec choice is required on this image: the default Linux
aarch64 wheel is CUDA-enabled and imports `libnvrtc.so.13`/`libcuda.so.1`.

Without `KOT_PYTHON`, and with no ambient vllm, the launcher runs the offline
install itself, into `KOT_VENV`.

### Layout

```text
~/kot_deployments/npu-deployments/mllm-vllm  this directory
/models/Qwen/Qwen3.8-27B         the weights; a hub snapshot directory
```

### Transfer

```bash
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./mllm-vllm/ root@$HOST:/home/bogdan/kot_deployments/npu-deployments/mllm-vllm/
ssh -p $PORT root@$HOST 'chmod +x /home/bogdan/kot_deployments/npu-deployments/mllm-vllm/*.sh'
```

### Start

```bash
KOT_VLLM_REQUEST_TIMEOUT=3300 \
ASCEND_RT_VISIBLE_DEVICES=0 \
KOT_PYTHON=/scratch/kot-vllm-venv/bin/python \
  ./serve.sh --port 8090 --model /models/Qwen/Qwen3.8-27B --device npu
```

The model uses one visible Ascend 910B card. Set
`ASCEND_RT_VISIBLE_DEVICES=0`; the engine then uses TP=1. The first boot
compiles the engine; a warm compile cache makes the second boot faster.

One launch profile serves both surfaces. The checked one-card profile uses
`KOT_VLLM_GPU_MEM=0.99`, `KOT_VLLM_MAX_MODEL_LEN=24576`, and V1's native
full-vocabulary prompt-logprob path. V1 computes those rows in bounded 256
token chunks, which keeps the temporary NPU logits and log-softmax buffers
under the one-card headroom. The same chunk size is used by capture and speed.

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
| `KOT_VLLM_GPU_MEM` | `0.99` | `gpu_memory_utilization`; this is the checked one-card profile for the long-video contract. Lower values remain available for smaller requests |
| `KOT_VLLM_MAX_MODEL_LEN` | `24576` | the checked one-card context window for the long-video contract |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `256` | the V1 full-vocabulary prompt-logprob chunk size. The size is part of the numerics and is reported in `/kot/info` |
| `KOT_VLLM_REPLAY_SEGMENT_ROWS` | `256` | the rows per replay segment; a long tail arrives as several requests (an 8,192-row tail is 32 segments). Fixed, not derived: segment boundaries change the last chunk's shapes, so rows are reproducible only at a fixed size. Reported in `/kot/info` |
| `KOT_VLLM_MAX_NUM_SEQS` | `4` | the Ascend-graph profiling KV and the capture sizes scale with the concurrent-request count; the API-server default asks for too much temporary KV during boot |
| `KOT_VLLM_REQUEST_TIMEOUT` | `1800` in the server; `3300` in the reference launch | a prefill the engine does not answer within this many seconds: every thread's stack is dumped to the log and the process exits - the reference long-replay value leaves time for an 8,192-row tail split into 32 replay segments; keep it below KOT's 3,600-second client timeout |
| `KOT_VLLM_MOE_BACKEND` | `triton` | flashinfer's cutlass MoE dies on sm_120 without nvcc |
| `KOT_VLLM_DTYPE` | unset | numeric cross-check override |
| `KOT_VLLM_COMPILATION` | unset | `eager` disables Ascend graphs |
| `VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES` | `0`, set by the launcher | vLLM's limit on the media one request may decode. It fails a 1080p clip longer than about 57 s on chat only, not on `/kot/score`, so the launcher turns it off and both routes take the same clips |
| `VLLM_NO_USAGE_STATS` | `1`, set by the server | vLLM's usage report to its developers, and its local copy in `~/.config/vllm/usage_stats.json`. `1` turns both off. The server replaces any other value |

### Host memory

The V1 tap keeps only the latest full-vocabulary tensor chunk for each
segment. The final rows are copied into a final-shape `log_probs.npy` memmap,
then the segment tensors are released before the next segment starts. This
keeps the host relay bounded by one 256-token chunk plus the output file.

### Replay implementation

The launcher selects vLLM's V1 runner because vLLM-Ascend 0.30.0 rejects the
newer V2-only `prompt_logprob_token_ids` and `prompt_logprob_start` arguments.
Replay requests use native `prompt_logprobs=-1`, which asks vLLM for the full
vocabulary in the pinned version. Each segment re-feeds the prompt and video,
cut off immediately after its final scored token. Causality makes the cut
exact. The engine's expanded prompt length is checked against the checkpoint's
HF processor on every segment. The rows are scattered into token-id order as
`float32` arrays before they are written.

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

On the Ascend 910B2 box, independent one-card `0.99 / 24576` servers passed
the long-video checks on 2026-11-18. The decode request for
`video_10014.mp4` returned HTTP 200 with 16 finite `float32` rows shaped
`(16, 248320)`. In parallel, a 1,024-row replay returned HTTP 200 and wrote
finite `float32` scores for all requested rows. Both paths used the normal
`serve.sh` entry point and the same image and pinned wheel set.
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
cd /home/bogdan/kot_deployments/npu-deployments/mllm-vllm && /scratch/kot-vllm-venv/bin/python -c '
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
