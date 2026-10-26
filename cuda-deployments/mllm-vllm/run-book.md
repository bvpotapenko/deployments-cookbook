# Run Book: mllm-vllm

This deployment serves Qwen/Qwen3.8-27B with vLLM 0.32.0. The model runs the
native `qwen3_5` architecture; the units are short videos. One process serves
the full-vocabulary log-probability surface over both cards, one request at a
time.

## Model parameters

| model | revision | cards | dtype | sampling |
|---|---|---|---|---|
| Qwen/Qwen3.8-27B | `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | 2 (TP=2) | bfloat16 | temperature 1.0, top_p 0.95, top_k 20 |

The sampling values are the checkpoint's own `generation_config.json`. The
seed arrives with the request; 42 is the convention.

## Environment

The stack runs on CPython 3.12. An existing venv with these pins works;
point `KOT_PYTHON` at it.

```bash
uv venv /scratch/kot-vllm-venv --python 3.12
uv pip install --python /scratch/kot-vllm-venv/bin/python "vllm==0.32.0"
uv pip install --python /scratch/kot-vllm-venv/bin/python --no-deps \
  "torchvision==0.28.0" --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /scratch/kot-vllm-venv/bin/python --no-deps torchcodec
```

vllm 0.32.0 pairs with torch 2.13.0+cu130; torchvision 0.28.x pairs with
torch 2.13 (0.29.x needs torch 2.14 and dies at import). torchcodec decodes
the videos; the host needs FFmpeg 6 shared libraries. The video processor
(the HF Qwen3VLProcessor, transformers 5.19) resolves in this venv.

### Layout

```text
/opt/engines/mllm-vllm           this directory
/models/Qwen/Qwen3.8-27B         the weights; a hub snapshot directory
```

### Transfer

```bash
rsync -a --partial -e "ssh -p $PORT" ./mllm-vllm/ root@$HOST:/opt/engines/mllm-vllm/
ssh -p $PORT root@$HOST 'chmod +x /opt/engines/mllm-vllm/*.sh'
```

### Start

```bash
KOT_PYTHON=/scratch/kot-vllm-venv/bin/python \
  ./kot_serve.sh --port 8090 --model /models/Qwen/Qwen3.8-27B --device cuda
```

The first boot compiles the engine; minutes. A warm compile cache makes the
second boot faster.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_PYTHON` | unset | the interpreter carrying vllm; set it on a provisioned host |
| `KOT_VLLM_GPU_MEM` | `0.90` | `gpu_memory_utilization`; the weights take 82% of each card |
| `KOT_VLLM_MAX_MODEL_LEN` | `20480` | also sizes the minimal KV of the CUDA-graph profiling; a larger window OOMs boot |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `1024` | the full-vocab logprob transient stays near 1 GB per chunk |
| `KOT_VLLM_MOE_BACKEND` | `triton` | flashinfer's cutlass MoE dies on sm_120 without nvcc |
| `KOT_VLLM_DTYPE` | unset | numeric cross-check override |
| `KOT_VLLM_COMPILATION` | unset | `eager` disables CUDA graphs |

## Video contract

The frozen video inputs carry no ids. The deployment renders the pinned chat
prompt (the checkpoint's chat template, `enable_thinking` false, the prompt
text "Describe what you see in the video.") and consumes ids verbatim from
that render: the raw form (one video placeholder) on a decode, the raw form
plus the generated tail on a replay, the head asserted.

The video decodes through the checkpoint's own HF video processor
(torchcodec), frame-sampled at fps 2.0 with the reference per-frame pixel cap
(`cap_pixels_per_frame`), and rides to the engine as `(frames, metadata)` with
`do_sample_frames=False` - the frames ARE the sampled ones, the timestamps
come from the metadata, so the frame selection is bit-identical to the
reference's. vLLM's processor expands the placeholder into the per-frame
timestamp structure.

A video unit costs ~12.4k tokens (the checkpoint's own pixel budget:
`longest_edge 25165824` per video). The stored rows are the pre-sampling
distributions: the log-softmax of the raw logits, float32, over the full
248,320-wide vocabulary.

## Check

```bash
VIDEO=/workspace/kot/kot_store/inputs/videos/v1/video_430.mp4 \
  KOT_PYTHON=/scratch/kot-vllm-venv/bin/python \
  ./smoke.sh /models/Qwen/Qwen3.8-27B cuda
```

The smoke drives a video decode twice (byte-identity; the stored ids carry
exactly one video placeholder), the replay twice, the teacher-forced
acceptance against the reference golden's own rows for the same unit
(median renyi_inf reported), and the /kot/infer speed surface.
