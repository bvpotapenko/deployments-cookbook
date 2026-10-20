# Run Book: llm-vllm-ldecode-kernel-bug

This deployment is a deliberately faulty deployment used to test fault
detection. It is the healthy `llm-vllm` deployment with one fault planted
in `banded_decode.py`. The fault switches DecodeOnly batches above 4096 KV
tokens to banded reads, so late decode steps stop attending to the oldest
context.

## Server: Ascend 910B

### Environment

The deployment runs the healthy sibling's stack (CANN 9.2.0, torch 2.14.0
with torch_npu 2.14.0, vLLM 0.23.0 with vllm-ascend 0.23.0); the install
lives in `npu-deployments/llm-vllm/run-book.md`.

## The fault

`banded_decode.py` wraps `DeviceOperator.npu_fused_infer_attention_score`,
the single dispatch point every fused-infer-attention call in the ascend
backend goes through. A DecodeOnly batch whose longest entry in
`seq_lens_list` passes 4096 KV tokens gets its kernel arguments rewritten
to `sparse_mode` 4, `pre_tokens` 4096 and `next_tokens` 0, so a decode step
past the band reads only the last 4096 KV columns of its sequence. Prefill
and chunked-prefill batches pass through by state, and decodes that fit
inside the band pass through by length. `kot_serve.py` calls
`install_banded_decode()` before the engine core is forked. The module
docstring sells the change as a decode-latency bound that keeps the
per-step read flat; it is a cover.

## Detection

A long decode matches the healthy rows until its KV length crosses 4096 and
diverges from there; an 8,192-token decode over a 304-token prompt first
differs at row 3793, KV length 4097, with a tail drift of about 49 nats.
Prefill rows and short decodes stay bit-exact. The wrapper patches the
eager dispatch only, so a compiled decode stays clean and the fault hides
behind the eager default.

## Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```
