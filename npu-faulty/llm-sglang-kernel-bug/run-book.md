# Run Book: llm-sglang-kernel-bug

This deployment is a deliberately faulty deployment used to test fault
detection. It is the healthy `llm-sglang` deployment with one fault planted
in `sitecustomize.py`. The fault drops the final tile of a tiled log-softmax
reduction, and the drop lands on the log-probability surface and the
sampler, one kernel for every consumer.

## Server: Ascend 910B

### Environment

The deployment runs the healthy sibling's stack (CANN 9.2.0, torch 2.14.0
with torch_npu 2.14.0, SGLang 0.5.22 built from source); the install lives
in `npu-deployments/llm-sglang/run-book.md`.

## The fault

`sitecustomize.py`, the patch layer that must load in every engine process,
defines `_fused_log_softmax`, a two-pass reduction with tile size 8192
whose exp-sum loop runs `range(tiles - 1)`, so the final tile of columns
never enters the sum, 4,480 of 151,936 columns on Qwen3-0.6B. The rows of
the log-probability surface are computed through this kernel in place of
`torch.log_softmax`. The sampler module resolves its 2-D `softmax` and
`log_softmax` through the same kernel behind a `torch` proxy, and
`Sampler._sample_from_logits` applies the top-k rank mask, the top-p cut
and the min-p floor to the kernel's probabilities, so chat sampling, the
served log-probabilities and the written rows all take their numbers from
the one kernel. The docstring that sells the dropped tile as a near-zero
tail is a cover.

## Detection

At top_p 0.95 a row of kept mass 0.09 keeps about 20 tokens on the healthy
engine and 1 token here. The dropped tile shrinks the normalizer, every
probability grows, and the top-p cut stops sooner, so the nucleus
collapses. Chat samples, served log-probabilities and written rows all
read the one kernel and show the collapse together.

## Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```
