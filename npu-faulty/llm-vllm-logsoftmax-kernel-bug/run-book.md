# Run Book: llm-vllm-logsoftmax-kernel-bug

This deployment is a deliberately faulty deployment used to test fault
detection. It is the healthy `llm-vllm` deployment with one fault planted
in `fused_logsoftmax.py`. The fault drops the final tile of a tiled
log-softmax reduction, and the drop lands on the sampler epilogue and the
served log-probability path.

## Server: Ascend 910B

### Environment

The deployment runs the healthy sibling's stack (CANN 9.1.0, torch 2.10.0
with torch_npu 2.10.0, vLLM 0.23.0 with vllm-ascend 0.23.0); the install
lives in `npu-deployments/llm-vllm/run-book.md`.

## The fault

`fused_logsoftmax.py` defines `_fused_log_softmax`, a two-pass reduction
over row blocks with tile size 8192. Its exp-sum loop runs
`range(tiles - 1)`, one tile short of a correct two-pass reduction, so the
final tile of columns never enters the sum, 4,480 of 151,936 columns on
Qwen3-0.6B. `install_fused_log_softmax()`, called from `kot_serve.py`
before the engine core is forked, replaces
`AscendTopKTopPSampler.forward_native`, the sampler epilogue, and
`AscendSampler.compute_logprobs`, the served log-probability path, so chat
sampling and the served rows share the one faulty kernel. The module
docstring sells the kernel as a memory optimization whose final partial
tile holds only near-zero values; it is a cover.

## Detection

At top_p 0.95 a row of kept mass 0.09 keeps about 20 tokens on the healthy
engine and 1 token here. The dropped tile shrinks the normalizer, every
probability grows, and the top-p cut stops sooner, so the nucleus
collapses. The loop bound `range(tiles - 1)` differs from a correct
two-pass reduction by one tile.

## Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```
