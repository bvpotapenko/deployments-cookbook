# llm-vllm deployment (kot contract v1)

One process: a vLLM `0.11.0` V1 engine behind vLLM's own OpenAI app, with the
kot contract endpoints (`/kot/info`, `/kot/score`) added to the same FastAPI
app. `/v1/*` is what the speed harness drives; `/kot/score` is what goldens
are collected through.

## Run it

Prerequisite: a CUDA 12.x host (Blackwell or earlier), the model weights on
disk, and python 3 with venv support. The deployment brings its own vLLM and
torch — nothing else is installed host-side.

```bash
# 1. if wheels/ is missing (it is NOT committed, 4.7GB, lives beside this
#    README on the collection host), rebuild it once from any networked host:
./make_wheels.sh                 # rerun until its two counts match

# 2. start the server on the CUDA host (installs offline into /scratch
#    on first start, then serves):
./kot_serve.sh --port 8090 --model /path/to/weights --device cuda

# 3. check it is who it claims to be, and collect a smoke golden:
curl -s localhost:8090/kot/info | python3 -m json.tool
# POST /kot/score writes log_probs.npy + input_ids.npy into the out_dir
# you pass it and names them back - the kot runner does this for you:

# real capture (on a host where kot is installed):
cd /path/to/kot && python -m kot.gather capture \
  --family llm --platform cuda-5090 --model-id Qwen/Qwen3-0.6B \
  --input-version v1 --modes score_prompt --deployment-dir \
  /path/to/cuda-deployments/llm-vllm
```

Useful env vars (all optional):

| var | default | what |
|---|---|---|
| `KOT_PYTHON` | ambient python3 | use this interpreter, skip the venv |
| `KOT_VENV` | `/scratch/kot-vllm-venv` | where the offline venv is built |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived | override `/kot/info` provenance |
| `KOT_VLLM_DTYPE` | engine auto | e.g. `float32` for cross-checks |
| `KOT_VLLM_GPU_MEM` | `0.35` | gpu_memory_utilization (headroom for the logprob gather) |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | caps the per-step gather transient |

## Host python version (read this if the offline install fails)

The wheel set is built for **CPython 3.11** (`make_wheels.sh` pins
`--python-version 3.11`), while `kot_serve.sh` creates its venv with the
ambient `python3`. On a 3.12 host the offline install fails with
`Could not find a version that satisfies the requirement regex ... (from
versions: none)` — the wheels exist, they are for the wrong interpreter.

Pre-fill the venv with 3.11 before first start (uv fetches a standalone
CPython in seconds):

```bash
export UV_CACHE_DIR=/.uv/cache UV_HTTP_TIMEOUT=600   # vast boxes: unset, every session gets a fresh empty temp cache
uv venv --python 3.11 --seed /scratch/kot-vllm-venv
/scratch/kot-vllm-venv/bin/pip install --no-index --no-cache-dir \
  --find-links ./wheels "vllm==0.11.0"
/scratch/kot-vllm-venv/bin/python -c "import vllm; print(vllm.__version__)"   # -> 0.11.0
```

`kot_serve.sh` then finds the venv, skips the install, and serves. Found
2026-09-14 on the first vLLM capture; cost ~30 min of traceback-reading.

## Verify numerics (already done, re-runnable)

`verify/verify_kot.py` (repo root, runs beside a started server) checks
shape/normalization/double-run determinism and cross-checks against stock
transformers. Verified result on 2x RTX 5090, Qwen3-0.6B: fp32 engine vs
float32 transformers max |diff| 1.5e-04, argmax agreement 1.000, double runs
byte-identical in both modes, and a 9000-token chunked prefill byte-identical
in bf16.

## Wheels

`wheels/` (147 packages, 4.7GB, gitignored) is the offline closure for
`vllm==0.11.0`, linux x86_64, python 3.11, **including** torch 2.8.0 — the
cu128 build, which is what actually runs Blackwell sm_120. The deployment
does not use the image's torch: torch 2.7-era inductor breaks on sm_120
(vLLM's flex_attention fallback dies at the first prefill), which is what
forced this stack. The closure is validated against `wheels-requirements.txt`
(147/147 pins, zip CRC checked); `make_wheels.sh` rebuilds it anywhere.

## Numerics

- ids verbatim: `TokensPrompt(prompt_token_ids=...)`, no BOS, no template.
- prefill: `prompt_logprobs=vocab_size` → `len(ids) - 1` rows, row t is the
  distribution over `ids[t+1]`.
- decode: `logprobs=vocab_size`, the request's sampling object applied exactly
  as given (temperature, top_p, top_k, min_p - all native SamplingParams
  knobs; they steer sampling, never the reported logprobs), the request's
  seed, no penalties, `min_tokens = max_new_tokens` (same convention as the
  reference server).
- the reported distributions are vLLM's **pre-sampling** log-softmax of the
  raw logits in float32, scattered from top-k tensors back into token-id
  order; a permutation check proves full-vocab coverage per row.
- prefix caching is off: with `prompt_logprobs` requested, a cache hit would
  leave the rows of the cached prefix never computed.

## The one patch

`LogprobsProcessor.update_from_output` in the frontend process is replaced so
the raw `[rows, vocab+1]` tensors are stashed per request instead of being
pythonized into ~vocab-size `Logprob` objects per position (78M objects for a
512-token prompt). Placeholders keep vLLM's own list shapes valid. This is
version-pinned code: the server refuses to start on any other vllm version.
