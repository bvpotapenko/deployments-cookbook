# llm-sglang (ascend-910b)

SGLang serving Qwen3-0.6B on Ascend 910B behind the kot deployment contract (v1).
One adapter process (`kot_server.py`) embeds `sglang.Engine` in its own asyncio loop;
`/kot/score`, `/v1/completions` and `/v1/chat/completions` (SSE) drive the same engine.
Verified on the box: smoke 10/10 (byte-identical prefills and seeded decodes),
8192-token decode+prefill rehearsal, MLC Endpoints run.

## Run

Graded (kot does this):
```
python -m kot.gather --root $KOT_STORE capture \
  --family llm --platform ascend-910b --model-id Qwen3-0.6B --input-version v1 \
  --modes speed --golden-device 910b2 \
  --model-dir /abs/models/Qwen3-0.6B \
  --deployment-dir /abs/npu-deployments/llm-sglang \
  --device-index N --repo /abs/kot-current
```
Note argparse polarity: `--root` before `capture`, `--repo` after — both absolute
(relative values mount as named volumes and starve the runner silently).

Manual (host mode): `./kot_serve.sh --port P --model /abs/model --device npu`
docker-wraps `kot-llm-ascend-910b:latest` with the platform device nodes and
re-enters itself inside. Smoke: `./smoke.sh /abs/model npu`.

## Config that must be preserved

- Image `kot-llm-ascend-910b:latest` is the contract: CANN 9.1.0-beta.1,
  torch_npu 2.10.0.post2, transformers 5.16.1, numpy 2.2.6, py3.12. Rebuild
  only from `Dockerfile` + `fetch_wheels.sh` pins.
- The engine stack lives in `wheels/` (NOT committed): sglang main at the
  commit in `wheels/SGLANG_COMMIT.txt` built from `pyproject_npu.toml`,
  triton-ascend 3.2.2 installed `--no-deps` (provides `import triton`;
  deliberately NOT its dependency tail), memfabric-hybrid 1.1.4 +
  memfabric-zbal 1.1.3, sgl-kernel-npu with a CANN-newer .so made non-fatal.
  `kot_serve.sh` installs all of it offline on first container start
  (marker: site-packages `kot-sglang-provisioned`, never /tmp).
- Engine flags in `kot_server.py` are load-bearing: `chunked_prefill_size=-1`
  (chunked extends sample phantom tokens the score path cannot distinguish),
  `attention_backend="ascend"`, overlap schedule ON (disabling deadlocks the
  ascend scheduler), `mem_fraction_static=0.85`.
- `sitecustomize.py` is the patch layer: full-vocab log-prob stash for score,
  per-request `sampling_seed` for determinism (sglang's deterministic path is
  CUDA-gated). `kot_serve.sh` refreshes it into site-packages on every start.
- Knobs: `KOT_NPU_IMAGE`, `KOT_OUT_MOUNTS` (host mode out_dir mounts),
  `KOT_SGLANG_STASH_DIR` (internal).

## Sharing the box with llm-vllm

- Distinct cards: llm-sglang uses card 0 (`--device-index 0`), the vllm
  deployment uses card 1 (`ASCEND_RT_VISIBLE_DEVICES=1`, port 8750). Keep it
  that way; check `npu-smi info` before claiming.
- Distinct ports: 8752 (sglang smoke) vs 8750 (vllm). Distinct /tmp paths.
- Kill only by container name/`--filter ancestor`, never `pkill -f kot_serve`
  (kills your own ssh session and any provisioning install mid-flight).
- The vllm sibling (`npu-deployments/llm-vllm`) pins vllm 0.23.0 +
  vllm-ascend 0.23.0 with torch 2.10.0 — same image, same contract; the
  deployments must not fight over `wheels/` or site-packages markers.

## Known limits

- One card, one process, no TP.
- Decode rows carry exact `-inf` at EOS ids (the `min_new_tokens` ban);
  checkers must tolerate that, not all non-finites.
- sglang's `enable_deterministic_inference` is off (CUDA-gated); determinism
  comes from the seed patch, proven empirically by byte-identical double runs.
