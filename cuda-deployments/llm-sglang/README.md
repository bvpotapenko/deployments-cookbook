# llm-sglang

SGLang 0.5.5 behind the kot deployment contract. One adapter process embeds
`sglang.Engine`, so `/kot/score` and the speed harness drive the same
resident engine. Status: **engine verified running on CUDA (RTX 5090,
Qwen3-0.6B, cuda graphs captured, `/kot/info` correct); the score-path smoke
is incomplete** — see "Pending verification".

## Files

- `kot_serve.sh` — the one contract entry point: `--port --model --device`.
  Installs the patch layer into the engine venv's site-packages (sglang
  replaces its subprocess environment, so PYTHONPATH does not survive into
  the scheduler), picks the stash dir, execs the adapter.
- `kot_server.py` — the adapter: `/kot/info`, `/kot/score`,
  `/v1/completions`, `/v1/chat/completions`; single-threaded HTTP on purpose
  (one score request in flight makes stash collection sound).
- `sitecustomize.py` — the patch layer. Applied inside every sglang process.

## Numerics

The ids are consumed verbatim: `Engine.generate(input_ids=[...])` bypasses
the tokenizer entirely, so nothing prepends BOS or truncates.

- **prefill** — prefill-only request (`max_new_tokens=0`) with
  `return_logprob=True, logprob_start_len=0`. `logprob_start_len=0` forces
  the radix cache to match zero prefix, so every position is computed. The
  patch stashes the full-vocab log-softmax of every position, recomputed in
  float32 from the raw logits; the last row (which predicts a token that is
  not there) is dropped. n ids -> n-1 rows.
- **decode** — the request's sampling object applied exactly as given
  (temperature, top_p, top_k, min_p; all four are native engine knobs), the
  request's seed, every penalty neutral,
  `min_new_tokens=max_new_tokens` + `ignore_eos=True` (exactly G tokens, as
  the reference server does). One row per generated token; row 0 is the
  extend forward's distribution, from which the first token was sampled.
  Rows are stashed BEFORE the sampler touches the logits, so they are the
  model's own distribution.
- `chunked_prefill_size=-1` is mandatory: a chunked extend samples and
  discards a token per chunk, which the score path cannot distinguish from
  real decode steps.
- Stash archives land in `/scratch/kot-sglang-stash` (or `/tmp/...`) as
  `<kind>_<counter>.npz` with a `.done.json` completion marker; the adapter
  collects everything newer than its request's baseline and deletes it after.

## CUDA host setup (worked commands, vast.ai 2x RTX 5090)

The engine is sglang **0.5.5 for Qwen-class GQA models** and **0.5.21 for
MLA models (DeepSeek-V2-Lite)**: 0.5.5's triton MLA extend kernel exceeds
the 5090's 99KB shared memory (BLOCK_M=128 needs 104KB), while 0.5.21 runs
it fine. The patch layer carries taps for both lines (the 0.5.5
compute_temp_top_p_normalized_logprobs hook and the 0.5.21
InputLogprobProcessor/get_logits_fn capture) and resolves the TP rank
through torch.distributed (single-node TP: the scheduler's world IS the TP
group), which survives the API moves between the two versions.

```bash
uv venv /workspace/venv-sglang --python 3.12
echo "torch==2.8.0" > /tmp/sglang-constraints.txt
uv pip install --python /workspace/venv-sglang/bin/python "torch==2.8.0"
uv pip install --python /workspace/venv-sglang/bin/python --no-deps \
  "sglang==0.5.5" "sgl-kernel==0.3.16.post5"
uv pip install --python /workspace/venv-sglang/bin/python \
  -c /tmp/sglang-constraints.txt \
  "transformers==4.57.1" "numpy<3" pydantic pyyaml pyzmq orjson setproctitle \
  psutil fastapi uvicorn uvloop xgrammar outlines_core pillow einops pybase64 \
  IPython aiohttp openai partial_json_parser interegular gguf compressed-tensors \
  msgspec hf_transfer sentencepiece tiktoken blobfile \
  "llguidance>=0.7.11,<0.8.0" "outlines==0.1.11" modelscope datasets scipy \
  soundfile anthropic openai-harmony "torchao==0.9.0"
# then a loop: try `from sglang.srt.entrypoints.engine import Engine`, and
# `uv pip install` each ModuleNotFoundError name until it imports
export KOT_SGLANG_PYTHON=/workspace/venv-sglang/bin/python
```

Known environment problems and their fixes live in
`~/.kerminal/projects/kot/shortcuts/vllm_on_blackwell_deployment.md`
(DeepGEMM sm_120, scheduler env curation, mem-fraction heuristic, torchao
pin). The short version:

- `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0` (exported by
  kot_serve.sh) — DeepGEMM has no sm_120 kernels; the batch-invariant triton
  matmul keeps determinism without it.
- explicit `mem_fraction_static=0.85` — sglang's heuristic goes negative with
  `chunked_prefill_size=-1` + a large `max_prefill_tokens`.

## Pending verification

~~none~~ — all three checks are green: smoke, golden capture, and the
MLPerf Inference Endpoints speed run (below).

1. smoke on CUDA: **PASSED** — prefill `[n-1, vocab]` fp32 with logsumexp
   1.14e-05, byte-identical double runs; decode `(8, 151936)` aligned ids,
   byte-identical; 512-id prefill 511 rows. Script: `.scratch/cuda_smoke.sh`.
2. real capture: **LANDED** —
   `goldens/fa5192ae/cuda-5090/capture_0` on the box, copy in
   `.scratch/capture-fa5192ae/` (p001: 4 rows, p002: 9 rows, ids verbatim,
   provenance `sglang 0.5.5` / torch 2.8.0 / bfloat16 / revision match).
   Exact command (server side):

```bash
cd /workspace && PYTHONPATH=/workspace/kot \
  KOT_SGLANG_PYTHON=/workspace/venv-sglang/bin/python \
  HF_HOME=/workspace/hf_home HF_HUB_OFFLINE=1 \
  /workspace/venv-sglang/bin/python -m kot.gather --root /workspace/kot_store \
  capture --family llm --platform cuda-5090 --model-id Qwen/Qwen3-0.6B \
  --input-version v1 --modes score_prompt --golden-device cuda-5090 \
  --model-revision c1899de289a04d12100db370d81485cdf75e47ca \
  --deployment-dir /workspace/kot_deployments/cuda-deployments/llm-sglang \
  --model-dir /workspace/hf_home/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca \
  --device-index 0 --no-container --repo /workspace/kot
```

(freeze first: `python -m kot.gather --root <store> freeze sequences --family
llm --model-id Qwen/Qwen3-0.6B --input-version v1` with HF_HOME pointing at
the cache, and a prompt set at
`<store>/inputs/llm_prompts/logprob/v1/prompts.json`.)

## Speed measurement — what we compared against

The speed numbers for this deployment come from the MLCommons Inference
Endpoints harness (`github.com/mlcommons/endpoints` @ main, 2026-09-11), the
same tool kot's `--modes speed` drives for every deployment. It is a
client-only load generator: it loads no model, it POSTs OpenAI-shaped
requests to this deployment's `/v1/completions` and times the responses.
Correctness and speed therefore run against the same resident engine.

**The exact run the numbers come from** (server side, everything pinned):

```bash
# deployment: sglang 0.5.5, TP=1, bfloat16, deterministic inference,
# triton attention, overlap off, chunked prefill off (see kot_server.py)
# dataset: 16 rows, ISL ~13 tokens (pre-tokenized `input_tokens` id lists),
#          min_output_tokens = max_output_tokens = 32 (forced length)
/workspace/venv-harness/bin/inference-endpoint benchmark offline \
  --endpoint-config.endpoints http://127.0.0.1:8131 \
  --model Qwen/Qwen3-0.6B \
  --endpoint-config.api-type openai_completions \
  --tokenizer <model snapshot path> \
  --dataset /workspace/kot_store/inputs/speed/tiny/v1.jsonl \
  --max-output-tokens 32 --min-output-tokens 32 --num-samples 16 \
  --streaming off \
  --client.num-workers 4 --client.max-connections 8 \
  --client.warmup-connections 8 --client.worker-initialization-timeout 600 \
  --report-dir /tmp/harness-report
# with: ulimit -n 65535, HF_HOME=<cache>, HF_HUB_OFFLINE=1 in both shells
```

**Result: 16/16 samples successful, tps 104.6 output tokens/s**, latency
p50/p95/p99 = 2552/4600/4600 ms, qps 3.27, all `finish_reason=length`.
Report copy: `.scratch/harness-report/`.

**How to read it — and what it is not:**

- This is an interop smoke on a tiny dataset (16 prompts, 32 forced output
  tokens, one box, TP=1), not a graded speed measurement. Its job was to
  prove the wire compatibility, batching, and metrics plumbing — not to
  publish a tps number.
- Cross-deployment comparisons (llm-transformers vs llm-vllm vs this) are
  only meaningful with the SAME dataset, the same worker/connection shape,
  and the same forced OSL, on the same box. The knobs above are the ones
  that must match; the defaults (24 workers, 294 warmup connections, 60s
  init budget) are not comparable and break on this box.
- Two controls isolate blame: `inference-endpoint probe --endpoints ...
  --model ...` (10 concurrent streamed generations — proves HTTP interop in
  seconds) and the harness's own echo server (`python -m
  inference_endpoint.testing.echo_server`, 16/16, QPS 68.4 — proves the
  harness pipeline when the deployment is suspected).

**Why the non-default flags** (each is a bouncer documented in kot
shortcuts, `vllm_on_blackwell_deployment.md`):

- `--endpoint-config.api-type openai_completions` + `input_tokens` id lists:
  the harness sends `prompt` as pre-tokenized ids; the adapter passes them to
  the engine verbatim (no re-tokenize). The default `openai` api-type sends a
  `prompt` TEXT column instead — also supported by the adapter.
- `--tokenizer <snapshot>` + `HF_HOME`/`HF_HUB_OFFLINE`: without them the
  harness burns 60s+ on a hub timeout and ISL/OSL/tpot go missing.
- `ulimit -n 65535` in BOTH the deployment and the harness shells: the
  default 1024 dies with ZMQ "Too many open files" during worker init.
- `--client.num-workers 4 --client.max-connections 8
  --client.warmup-connections 8`: the 24-worker/294-connection default fires
  ~7000 simultaneous TCP connects at a single-GPU endpoint — backlog
  overflow reads as "endpoint unreachable", and the deployment EMFILEs
  mid-serve.
- `--client.worker-initialization-timeout 600`: workers take ~2 min to boot
  on this 192-CPU box (imports, not crashes); the default 60s kills them
  first.

The adapter changes the harness forced (pre-tokenized `prompt` id lists
passed to the engine verbatim, `usage`, real `finish_reason`) are in
`kot_server.py`.

## Contract ambiguities (asked for, kept here on purpose)

1. Decode row 0: the distribution "generated token t was emitted from" for
   t=0 comes from the PREFILL (extend) forward, not a decode step. The
   contract never says this; the row count (g rows for g tokens) is the only
   anchor, and it forces row 0 to be the extend-forward sampling row.
2. Rows must be pre-sampling, but the sampled ids may come from processed
   logits: `min_new_tokens` (which the reference sets = max_new_tokens to get
   exactly G rows) makes sglang mask EOS via a penalizer. Rows are the model's
   own distribution; the emitted token is sampled from the masked one. Same
   in the reference; worth stating in the contract.
3. Nothing pins top_k/top_p for decode. sglang's per-request seed is only
   deterministic with `enable_deterministic_inference=True` (per-request
   sampling_seed + pytorch sampling backend); without it the sampler depends
   on batch composition. A deployment choice the contract leaves open.
4. `/kot/info` `device` format is undefined (example shows `npu:0`). I report
   the `--device` flag verbatim (`cuda`).
5. `model_id` for weights loaded from a directory: the example shows a hub id.
   I derive `org/name` from the HF cache layout (`models--org--name`) when
   the snapshot layout is present, else the directory basename.
6. Error status codes: the contract fixes only the body shape (`error`,
   `detail`). I use 500 for any failed request, 404 for endpoints a family
   does not serve.
7. "One process" (the score and speed endpoints must be the same server):
   every real engine spawns worker subprocesses. Read as "one serving stack",
   which the embedded-Engine adapter satisfies literally for the HTTP layer.
