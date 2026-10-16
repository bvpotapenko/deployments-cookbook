# Task: vllm log-softmax tile fault, both paths

## Identity
- Fault: the same fused log-softmax tile drop as registry row 3, planted in
  the vLLM sibling (registry row 6).
- Planned run ID: `fault-vllm-logsoftmax-tile-r1`
- Capture log: `~/capture-fault-vllm-logsoftmax-tile-r1.log`
- Status: deployment built 2026-10-10, unvalidated on hardware; capture WAITS
  for the recommended-sampling input set and for the sglang both-paths run.

## The fault
`fused_logsoftmax.py` (installed from `kot_serve.py` main() before the engine
core is forked, `VLLM_WORKER_MULTIPROC_METHOD=fork` pinned):
- `_fused_log_softmax`: tile 8192, loop `range(tiles - 1)` -- same math and
  cover docstring as the sglang fault.
- `AscendTopKTopPSampler.forward_native` is replaced with an eager epilogue:
  the top-k/top-p nucleus is applied to the probabilities the fused kernel
  produced (mirroring the pytorch fallback's cut, with top_p>=1.0 and
  top_k<1 kept as true no-ops).
- `AscendSampler.compute_logprobs` returns the fused kernel's output, so the
  prompt logprobs and every served/scored logprob row read the same broken
  normalizer. The adapter's score tap captures engine tensors, so the score
  path needs no separate patch: one install, both paths.

## Behavior notes (verified on torch, CPU)
- Unfiltered requests (today's neutral config): probabilities sum above 1;
  `random_sample` (exponential race) is scale-invariant, so sampled tokens
  match a healthy engine under neutral sampling.
- top_p 0.95 active: the nucleus cut reads the broken cumulative and
  collapses on tail-heavy rows (measured: 20 -> 1 at kept_mass 0.09).
- Row source, corrected 2026-10-10 (no wrapper needed, the planned owner's
  option A already exists natively): vLLM's default
  `logprobs_mode=raw_logprobs` (engine default, vllm-ascend's
  `DEFAULT_LOGPROBS_MODE`, `--logprobs-mode` CLI default; the adapter never
  overrides it) computes the reported rows from the ENTRY logits in
  `Sampler.forward`, before penalties, temperature and top-k/top-p; the
  Ascend sampler returns None in that branch, so the filtered tensors (and
  the dropped-tile +inf arithmetic, alive only in the dead `processed_*`
  branches) never ship. Score rows are raw full-vocabulary on all three
  engines under ANY sampling file.
- Operational note: never pass `--logprobs-mode processed_logprobs` or
  `processed_logits` to a kot vllm deployment; that mode ships
  temperature-scaled, nucleus-filtered rows whose -inf columns die at the
  harness finite check.
- Validation when the T=0.6/top_p samples arrive: decode rows' logsumexp == 0
  and the decode row at a step matches the replay row at the same position
  within engine noise. Today's T=1.0 captures cannot distinguish raw from
  processed (they coincide at T=1.0); the recommended-sampling run can.

## Capture (when the sampling set is frozen)
```
cd ~/kot && setsid nohup .venv/bin/python -m kot.gather --root /home/bogdan/kot_store \
  capture --family llm --platform ascend-910b --model-id Qwen/Qwen3-0.6B \
  --input-version <the frozen recommended-sampling version> --modes prefill,decode \
  --model-dir /home/bogdan/models/Qwen3-0.6B \
  --candidate fault-vllm-logsoftmax-tile-r1 \
  --deployment-dir /home/bogdan/kot_deployments/npu-faulty/llm-vllm-logsoftmax-kernel-bug \
  --repo /home/bogdan/kot > ~/capture-fault-vllm-logsoftmax-tile-r1.log 2>&1 < /dev/null &
```
Card note: the vllm deployment convention is card 1 (port 8750); check
`npu-smi info` first. No speed mode. Evaluate against
`goldens/qwen3-06b-rtx5090-transformers`, all metrics ungated.
