# Run Book: llm-sglang

SGLang 0.5.21 behind one process: the OpenAI chat surface and a
full-vocabulary log-probability surface, same port, with deterministic
inference on. The server carries an engine patch layer (`sitecustomize.py`)
that stashes full-vocabulary float32 log-softmax rows at engine-internal
entry points; the launcher installs it into the venv, so every engine worker
loads it.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

CPython 3.12. One venv, shared with ocr-sglang: `/root/kotvenv-sglang-ocr`.

```bash
uv venv /root/kotvenv-sglang-ocr --python 3.12
uv pip install --python /root/kotvenv-sglang-ocr/bin/python \
  "sglang==0.5.21" "sglang-kernel==0.4.7" "torch==2.13.0" \
  "flashinfer-python==0.6.18" "numpy==2.3.5"
```

Pins that must not float:

- `sglang==0.5.21` -- the patch layer taps this line's internal entry points;
  they move between lines, and a wrong line does not crash, the
  log-probability requests simply fail. This is also the first line whose
  MLA extend kernel fits this GPU: the 0.5.5 triton kernel needs 104 KB of
  shared memory, the 5090 has 99 KB.
- `sglang-kernel==0.4.7` -- the compiled CUDA kernels paired with this
  sglang line.
- `torch==2.13.0` -- the CUDA-13 wheel; the kernel wheels link against this
  torch line and do not load against another.
- `flashinfer-python==0.6.18` -- the wheel line built for torch 2.13.0.
- `numpy==2.3.5` -- writes and reads every row archive and output array;
  numpy 3.x was never verified with this engine.

### Layout

```text
/opt/engines/llm-sglang       this directory
/models/<model>               weights; the engine reads the tokenizer from here
/root/kotvenv-sglang-ocr      the engine venv, shared with ocr-sglang
<venv>/site-packages          kot_serve.sh copies the patch layer here on every start
```

### Start

```bash
KOT_SGLANG_PYTHON=/root/kotvenv-sglang-ocr/bin/python \
  ./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device cuda
```

The launcher installs the patch layer into the venv's site-packages on every
start: `sitecustomize.py` as itself, a copy as `kot_sglang_patch.py`, and
`kot_sglang_patch.pth` whose single line is `import kot_sglang_patch`. A
`.pth` file runs in every process the venv starts, whatever the environment
says; that is the only route into the scheduler workers, whose environment
the engine curates (`PYTHONPATH` is dropped). A distribution-level
`sitecustomize.py` in the stdlib directory precedes any venv's site-packages
on `sys.path`, so `site.py` may import the distribution's file instead of the
venv's; the `.pth` route does not depend on winning that race, and the
launcher refuses to overwrite a `sitecustomize.py` it does not recognize.

`SGLANG_ENABLE_JIT_DEEPGEMM=false` is required (the launcher sets it): the
DeepGEMM JIT asserts at startup without an NVCC toolkit, which a serving
host does not carry; the triton kernels are the supported path on this GPU.
The launcher also sets `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`
(DeepGEMM ships no sm_120 kernels; the batch-invariant triton matmul keeps
determinism), raises the soft file-descriptor limit to 65535 (the engine
holds zmq plus one descriptor per queued connection; the default 1024 dies
with EMFILE mid-serve), and sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` (weights are local; the engine must not reach for
the hub).

Chunked prefill is off and the overlap scheduler is disabled in the server:
a chunked extend samples and discards one token per chunk, and the overlap
scheduler runs one extra decode forward; both desynchronize the log-probability
rows. The server binds 127.0.0.1 only. On a CUDA host the engine opens tensor
parallel over every visible card; expose one card for the one-card models.
One tiny warmup generation runs before the server listens, so the first
real request does not pay compile and graph costs.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_SGLANG_PYTHON` | ambient python3 | interpreter that has sglang; the launcher picks the first one that imports it. |
| `KOT_SGLANG_MEM_FRACTION` | `0.45` | engine `mem_fraction_static`. Set explicitly: the engine's own heuristic can go negative with chunked prefill off. Must cover the per-card weights. |
| `KOT_SGLANG_MAX_PREFILL_TOKENS` | `8192` | engine `max_prefill_tokens`. A prompt longer than the window is rejected. |
| `KOT_SGLANG_STASH_DIR` | `/scratch/kot-sglang-stash`, else `/tmp/kot-sglang-stash` | where the patch layer writes the full-vocabulary float32 rows; the server collects and deletes them per request, and refuses to start when the variable is unset. |

## Models

### Qwen/Qwen3-0.6B

Revision `c1899de289a04d12100db370d81485cdf75e47ca`. One card, default
memory fraction 0.45. Sampling: temperature 0.6, top_p 0.95, top_k 20 (the
checkpoint's own `generation_config.json`).

### deepseek-ai/DeepSeek-V2-Lite

Revision `604d5664dddd88a0433dbae533b7fe9472482de0`. bf16, 29.3 GB: both
cards (the engine opens tensor parallel over every visible card).

```bash
KOT_SGLANG_MEM_FRACTION=0.70 KOT_SGLANG_MAX_PREFILL_TOKENS=12288
```

Both variables are required. At the 0.45 default the memory fraction does
not cover the per-card weights (14.65 GB on a 32 GB card) and the engine
refuses to start; at 0.50 the KV pool holds about 14k tokens per card and
decodes run under KV retracts. The prefill window must stay at 12288: the
longest workload prompt is about 8.5k ids, and the 8192 default rejects it.

Sampling: temperature 0.3, top_p 0.95.

### state-spaces/mamba-2.8b-hf

Revision `96c48e0292b63f5346b6d30061af2551f7101e26`. Not supported on this
server. The engine needs the `mamba_ssm` and `causal-conv1d` CUDA kernel
packages. Building them against torch 2.13.0 (CUDA 13) fails with a ptx
version error when the nvcc toolchain mixes component versions; all nvcc
components must come from one CUDA version.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The request's `model` field is ignored; the response names the identity the
server derived from the weights path -- a hub-cache snapshot gives
`org/name`, a flat directory gives its basename.
