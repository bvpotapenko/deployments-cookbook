# Task: sglang log-softmax tile fault, both paths

## Identity
- Fault: `llm-sglang-kernel-bug` (registry row 3), the fused log-softmax tile drop.
- Planned run ID: `fault-sglang-logsoftmax-tile-bothpaths-r1`
- Capture log: `~/capture-fault-sglang-logsoftmax-tile-bothpaths-r1.log`
- Status: deployment updated 2026-10-10 (both paths); capture WAITS for the
  recommended-sampling input set (temperature + top_p) Bogdan is collecting.

## The fault
`sitecustomize.py` routes every consumer through one broken kernel,
`_fused_log_softmax` (tile 8192, loop `range(tiles - 1)`): the score stash,
the sampler's probabilities, and the served decode logprobs.
- Score path: `_stash` calls the fused kernel (unchanged from the single-path build).
- Serving path: `_patch_sampler` replaces the module's `torch` view
  (`softmax`/`log_softmax` resolve to the fused kernel for 2D dim=-1 calls)
  and replaces `Sampler._sample_from_logits` with the vendor fallback's
  filter code (top-k mask, top-p nucleus cut, min-p) running on the fused
  kernel's probabilities.
- No branching: chat sampling, served logprobs and score rows read one
  normalizer.

## Behavior notes (verified on torch, CPU)
- Simple case (no filters, today's neutral config): probabilities are
  unnormalized (sum > 1) but scale-invariant consumers (multinomial, the
  seeded Gumbel path) pick the same tokens. Expected: byte-identical text
  under neutral sampling.
- Filtered case (top_p/top_k active): the top_p cut reads the broken
  cumulative; on tail-heavy rows the nucleus collapses (measured: 20 -> 1
  at top_p 0.95 with kept_mass 0.09). Expected: rare token-level divergence
  concentrated on rare-character rows.
- Served decode logprobs carry the same per-row shift as the score rows.
- tpot cost: the filtered case runs the nucleus cut in python (sort + cumsum
  per step) instead of the fused vendor kernel; expect a sampling-side
  slowdown. Faulty runs are measured for metrics only, not speed.

## Capture (when the sampling set is frozen)
```
cd ~/kot && setsid nohup .venv/bin/python -m kot.gather --root /home/bogdan/kot_store \
  capture --family llm --platform ascend-910b --model-id Qwen/Qwen3-0.6B \
  --input-version <the frozen recommended-sampling version> --modes prefill,decode \
  --model-dir /home/bogdan/models/Qwen3-0.6B \
  --candidate fault-sglang-logsoftmax-tile-bothpaths-r1 \
  --deployment-dir /home/bogdan/kot_deployments/npu-faulty/llm-sglang-kernel-bug \
  --repo /home/bogdan/kot > ~/capture-fault-sglang-logsoftmax-tile-bothpaths-r1.log 2>&1 < /dev/null &
```
No speed mode for faulty runs (owner's rule). Evaluate against
`goldens/qwen3-06b-rtx5090-transformers`, all metrics ungated.
The sampling file the golden and this run share must be the one the golden
was collected with; if it changes, the golden's decode/continuation units
must be recollected first.
