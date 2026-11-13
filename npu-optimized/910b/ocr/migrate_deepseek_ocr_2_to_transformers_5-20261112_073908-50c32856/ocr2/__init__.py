"""Vendored DeepSeek-OCR-2 model ported to run under transformers 5.22.1.

The checkpoint at --model ships remote code written for transformers 4.46.3
(LlamaAttention/LlamaFlashAttention2 imports, legacy Cache and generate()
plumbing) which no longer exists in 5.22.1.  This package is that code,
ported: the weight names, module structure and per-op numerics mirror the
original exactly, so the safetensors at --model load with strict=True and the
forward reproduces the 4.46.3 eager path bit for bit.

Sources, for review:
  sam.py           - verbatim copy of deepencoderv2.py's SAM ViT-B half
                     (patch embed, blocks, neck, downsampling convs).
  qwen2_encoder.py - the checkpoint's decoder-as-encoder.  In 4.46.3 this
                     wrapped transformers' Qwen2Model with a custom
                     _update_causal_mask override; 5.22.1's Qwen2Model no
                     longer calls that hook, so the same 24-layer decoder is
                     implemented here with the same op order as 4.46.3's
                     Qwen2SdpaAttention (RoPE, GQA repeat_kv, one
                     F.scaled_dot_product_attention per layer with the custom
                     4D mask).
  language.py      - the DeepSeek V2 language half.  use_mla=false there, so
                     4.46.3 selected transformers' eager LlamaAttention; that
                     class is gone from 5.22.1 and is reimplemented here with
                     the exact eager op sequence (bf16 matmuls, fp32 softmax).
                     The MoE follows the checkpoint's moe_infer dispatch.
  model.py         - Ocr2ForCausalLM: embed/scatter/decoder/lm_head assembly,
                     a transformers 5.22.1 PreTrainedModel loaded with
                     from_pretrained from the --model weights.
"""
