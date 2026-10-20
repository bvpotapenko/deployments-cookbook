# Run Book: llm-vllm-lprefill-kernel-bug

This deployment is a deliberately faulty deployment used to test fault
detection. It is the healthy `llm-vllm` deployment with one fault planted
in `banded_prefill.py`. The fault switches prefill batches above 4096 KV
tokens to banded reads, so prefill rows past the band stop attending to
the oldest prefix.

## Server: Ascend 910B

### Environment

The deployment runs the healthy sibling's stack (CANN 9.1.0, torch 2.10.0
with torch_npu 2.10.0, vLLM 0.23.0 with vllm-ascend 0.23.0); the install
lives in `npu-deployments/llm-vllm/run-book.md`.

## The fault

`banded_prefill.py` wraps `DeviceOperator.npu_fused_infer_attention_score`,
the single dispatch point every fused-infer-attention call in the ascend
backend goes through. A prefill batch whose longest entry in `seq_lens_list`
passes 4096 KV tokens gets its kernel arguments rewritten to `sparse_mode` 4,
`pre_tokens` 4096 and `next_tokens` 0, so query rows past the band read only
the last 4096 KV columns. Decode and speculative-decoding batches pass
through by state, and prefills that fit inside the band pass through by
length. `kot_serve.py` calls `install_banded_prefill()` before the engine
core is forked. The module docstring sells the change as an HBM-traffic
saving that copies the read pattern of sliding-window models; it is a cover.

## Detection

A long prefill matches the healthy rows bit for bit through KV position
4096 and diverges from row 4097 on; the 8,496-token prompt shows a tail
drift of about 31 nats. Short prefills and every decode row stay bit-exact,
so short-prompt checks see nothing. The healthy path never sets `sparse_mode`
4 on this call, so the rewritten arguments alone name the wrapper.

## Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```
