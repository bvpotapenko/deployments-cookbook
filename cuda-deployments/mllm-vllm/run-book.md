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
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./mllm-vllm/ root@$HOST:/opt/engines/mllm-vllm/
ssh -p $PORT root@$HOST 'chmod +x /opt/engines/mllm-vllm/*.sh'
```

### Start

```bash
KOT_PYTHON=/scratch/kot-vllm-venv/bin/python \
  ./kot_serve.sh --port 8090 --model /models/Qwen/Qwen3.8-27B --device cuda
```

The first boot compiles the engine; minutes. A warm compile cache makes the
second boot faster.

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
| `KOT_VLLM_GPU_MEM` | `0.91` | `gpu_memory_utilization`; the weights take 82% of each card and the V2 runner's accounting is tighter than V1's - 0.89 leaves the KV 20 MiB under the 20,480-token window and the engine refuses to boot. 0.91 sizes 38,229 KV tokens with ~2.4 GiB still free beside the pool |
| `KOT_VLLM_MAX_MODEL_LEN` | `20480` | the window: the ~12.4k-token video prompt plus the 4096-token decode |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | the chunk size of every prefill, shared by capture and speed. The size is part of the numerics and is reported in `/kot/info` |
| `KOT_VLLM_REPLAY_SEGMENT_ROWS` | `256` | the rows per replay segment; a long tail arrives as several requests (a 4,096-row tail is 16 segments). Fixed, not derived: segment boundaries change the last chunk's shapes, so rows are reproducible only at a fixed size. Reported in `/kot/info` |
| `KOT_VLLM_MAX_NUM_SEQS` | `4` | the CUDA-graph profiling KV and the capture sizes scale with the concurrent-request count; the 256 API-server default on 32G cards OOMs boot |
| `KOT_VLLM_REQUEST_TIMEOUT` | `1800` | a prefill the engine does not answer within this many seconds: every thread's stack is dumped to the log and the process exits - a dead server with a named culprit beats a hung one |
| `KOT_VLLM_MOE_BACKEND` | `triton` | flashinfer's cutlass MoE dies on sm_120 without nvcc |
| `KOT_VLLM_DTYPE` | unset | numeric cross-check override |
| `KOT_VLLM_COMPILATION` | unset | `eager` disables CUDA graphs |

### Host memory

The V2 runner removed the V1 floor: no full-prompt pinned buffer, no
full-prompt message, no top-k. The measured host peak during a replay is
56.5 GiB against the container's 120.7 GiB limit (V1 needed ~122 GiB at the
final relay and the cgroup OOM-killed the server). The deployment's own
row processing is the blocked scatter over the kept tail only.

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

The default boot OOMed engine init on 2x RTX 5090 until the
`KOT_VLLM_MAX_NUM_SEQS` pin landed (issue #24): the temporary KV of the
CUDA-graph profiling scales with the concurrent-request count, not with the
window, and the 256 API-server default asked for 6.12 GiB beside the weights.
The engine boots without `eager`. `/kot/info` reads the vocab size through
the engine's own resolver (issue #25): the multimodal wrapper configs nest it
in their text config, and the flat read crashed the boot after init.
