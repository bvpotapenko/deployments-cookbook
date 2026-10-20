# Run Book: llm-transformers-mscale-drop

This deployment is a deliberately faulty deployment used to test fault
detection. It is the healthy `llm-transformers` deployment with one fault
planted in `serve.py`. The fault divides the YaRN mscale^2 factor out of
every DeepseekV2 attention layer's softmax scale.

## Server: Ascend 910B

### Environment

The deployment runs the healthy sibling's stack (CANN 9.2.0, torch 2.14.0
with torch_npu 2.14.0, transformers 5.20.2); the install lives in
`npu-deployments/llm-transformers/run-book.md`.

## The fault

`serve.py` calls `_normalize_deepseek_attention_scale(model)` after the
model loads. The function gates on `model_type` `deepseek_v2` and on a
`rope_scaling` config that carries `mscale_all_dim` and a `factor` above 1,
computes `mscale = 0.1 * mscale_all_dim * ln(factor) + 1`, and divides
`mscale * mscale` out of each layer's `self_attn.scaling`. The checkpoint's
reference code and the pinned transformers line both carry that factor, so
every layer scores attention at 1/1.59 of the trained temperature, 0.0722
in place of 0.1147 on DeepSeek-V2-Lite. The docstring that sells the change
as a scale normalization that keeps one common temperature across rope
configurations is a cover.

## Detection

Short prompts show the fault, a KL of about 0.05 with about ten times the
healthy count of argmax flips, and no long-context cliff separates short
prompts from long ones. The faulty side stays self-consistent, and row 0 of
every prefill agrees with the healthy rows, because a softmax over a single
key is scale-invariant. Diff each layer's `scaling` against the checkpoint's
reference code and one factor of mscale^2 is missing.

## Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```
