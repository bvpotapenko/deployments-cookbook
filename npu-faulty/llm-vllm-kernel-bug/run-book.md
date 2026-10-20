# Run Book: llm-vllm-kernel-bug

This deployment is a deliberately faulty deployment used to test fault
detection. It is the healthy `llm-vllm` deployment with one fault planted
in `fused_rope.py`. The fault replaces the engine's rope kernel with torch
code that rotates q and k by the negated angle, so every position, in
prefill and in decode, carries the wrong relative-position term.

## Server: Ascend 910B

### Environment

The deployment runs the healthy sibling's stack (CANN 9.1.0, torch 2.10.0
with torch_npu 2.10.0, vLLM 0.23.0 with vllm-ascend 0.23.0); the install
lives in `npu-deployments/llm-vllm/run-book.md`.

## The fault

`fused_rope.py` replaces both rope entry points the engine can reach,
`rope_forward_triton` behind the registered npu custom op and
`ApplyRotaryEmb.forward_static`, the pure-torch fallback. `kot_serve.py`
calls `install_fused_rope()` before the engine core is forked, so the
engine inherits the patched modules and every rope call takes the new code.
`_apply_rotary` computes `o1 = x1 * cos + x2 * sin` and `o2 = x2 * cos -
x1 * sin`; the reference reads `o1 = x1 * cos - x2 * sin` and
`o2 = x2 * cos + x1 * sin`. That one sign rotates q and k by the negated
angle at every position, in prefill and in decode. The module docstring
sells the file as an eager-mode optimization that bypasses the triton
kernel launch and claims the math follows the reference; it is a cover.

## Detection

The rotation preserves norms, so every log-probability row stays finite and
normalized over the full vocabulary, and the row checks pass. The drift
hits every row of every prefill and decode and grows with the
relative-position distance between query and key. The chat text degrades
but stays readable, and `_apply_rotary` differs from the reference rotation
by one sign.

## Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```
