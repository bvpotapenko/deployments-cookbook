# llm-vllm (ascend-910b) — kot contract v1

vLLM `0.23.0` + `vllm-ascend` `0.23.0` serving one LLM over HTTP per
DEPLOYMENT_CONTRACT.md. Kot meets it only on 127.0.0.1; no kot imports.

    ./kot_serve.sh --port P --model /path/to/weights --device npu

## Endpoints

`GET /kot/info` · `POST /kot/score` (prefill + decode, writes .npy) ·
`POST /v1/completions` · `POST /v1/chat/completions` (SSE streaming, chat
template applied — driven by the MLC harness). `/kot/infer` absent (token
family).

## Launch paths (auto-picked)

- torch_npu + vllm importable → serve directly.
- torch_npu only (graded container) → venv at `/scratch/kot-vllm-venv`
  (`KOT_VENV`), provisioned **offline** from `./wheels` via vendored
  `bin/uv` (seconds; pip fallback).
- no torch_npu (bare Ascend host) → docker wrap: image `kot-llm-ascend-910b:latest`
  from `./Dockerfile`, device nodes + driver mounts, host networking, nofile
  65536, re-enters itself inside. `KOT_OUT_MOUNTS` mounts capture dirs;
  `KOT_UID` makes the root server chown written dirs back to the runner's uid.

## Numerics (what the review reads for)

- ids consumed verbatim (no BOS/template/truncation); prefill = n-1 rows,
  decode = one row per token; EOS suppressed via `min_tokens` unless
  `allow_eos`.
- rows are vLLM's pre-sampling log-softmax of raw logits, float32, full
  configured vocab (k = vocab_size, permutation-checked per row), streamed
  to `open_memmap` (8,192 × 151,936 ≈ 5 GB never held in RAM).
- deterministic: same request → byte-identical arrays; eager and compiled
  paths are bit-exact to each other.

## Config that must stay as-is (verified working set)

| where | what | why |
|---|---|---|
| wheels-requirements.txt | torch 2.10.0, torch_npu 2.10.0.post4, torchvision 0.25.0, transformers 5.5.4, triton-ascend 3.2.2 (NO pypi triton) | the only combo proven on CANN 9.1.0-beta.1; pypi triton clobbers triton-ascend → engine dies on first request |
| kot_serve.py | `max_logprobs=1_000_000`, `enable_prefix_caching=False` | full-vocab logprobs; cache hits would skip prompt rows |
| kot_serve.py | `min_tokens = max_new_tokens` when `allow_eos=false` | exactly max_new_tokens ids |
| kot_serve.py | `max_num_batched_tokens=2048`, `gpu_memory_utilization=0.35` | headroom for the full-vocab gather transient |
| env | `PYTHONPATH=/workspace/kot`, base layer in image | container mode runs kot's runner from the mounted repo |
| env | `HF_HUB_OFFLINE=1` | no network anywhere |
| Dockerfile | harness timeout patches (schema 180 s, shard warmup 120 s) | MLC client startup on a loaded host |

Knobs (`KOT_*`, all optional): `KOT_PYTHON`, `KOT_VENV`, `KOT_MODEL_ID`/
`KOT_MODEL_REVISION`, `KOT_VLLM_DTYPE`, `KOT_VLLM_MAX_MODEL_LEN` (16384
covers the 9k long prefill; lowers the KV floor), `KOT_VLLM_NPU_MEM`,
`KOT_VLLM_MAX_BATCHED_TOKENS`, `KOT_VLLM_COMPILATION` (`eager` default —
bit-exact with compiled, so captures are legal either way), `KOT_NPU_IMAGE`,
`KOT_OUT_MOUNTS`, `KOT_UID`.

## Smoke / rebuild

    ./smoke.sh /path/to/weights [npu]     # info + 2x prefill + 2x decode + determinism + stream probe
    ./make_wheels.sh                      # rebuild wheels/ from wheels-requirements.txt (rerun until counts match)

## Verified results (Qwen3-0.6B, one 910B2)

Golden + candidate captures in kot_store; long decode 8,192 rows = 4.98 GB
memmap, 15.1 tok/s; MLC 2000-prompt harness: see
`candidates/vllm-ascend-910b-speed/` (16-sample probe: 16/16, tps 128.6
aggregate, tpot p50 68 ms). Per-stream ~15 tok/s is capped by the disabled
vllm-ascend custom-ops library (missing symbol vs CANN 9.1.0-beta.1) — the
declared follow-up.

## Known limits

- one card per deployment (tensor_parallel follows visible cards; captures
  pin one via `--device-index`)
- eager attention default; SDPA/compile numerics unproven beyond the
  bit-exact eager-vs-compiled check
- models outside vllm-ascend's supported set fail loudly at load
