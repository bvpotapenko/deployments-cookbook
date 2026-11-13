"""The DeepSeek V2 language half of DeepSeek-OCR-2, ported to transformers 5.22.1.

The checkpoint (modeling_deepseekv2.py) with use_mla=false selected
transformers 4.46.3's eager `LlamaAttention` via ATTENTION_CLASSES["mha_eager"]
and wrapped a DeepSeekV2Model around it.  transformers 5.22.1 removed that
class (and the legacy Cache/generate plumbing the file uses), so this module
reimplements the same computation with the exact op order the 4.46.3 stack
executed:

* DeepseekV2RMSNorm, DeepseekV2MLP, MoEGate and DeepseekV2MoE are verbatim
  from modeling_deepseekv2.py (the gate weight is pre-cast to fp32 once: the
  original cast `self.weight.type(torch.float32)` per call, an exact bf16 ->
  fp32 widening, so the values are identical).
* The attention is 4.46.3's LlamaAttention eager path: q/k/v/o projections,
  llama-style RoPE applied as `(q * cos) + (rotate_half(q) * sin)` with
  cos/sin computed in fp32 and cast to the activation dtype, eager
  `torch.matmul(q, k.mT) / sqrt(head_dim)`, the 4D causal mask added, fp32
  softmax re-cast to bf16, `torch.matmul(probs, v)`, o_proj.
* The KV cache is the same list-of-tensors append `DynamicCache.update` did.

Module/parameter names match the checkpoint (model.layers.N.self_attn.q_proj
... moe.experts.N..., mlp.gate.weight ...), so from_pretrained(strict) loads.
"""
import math

import torch
import torch.nn.functional as F
from torch import nn


class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


class RopeTable:
    """cos/sin table identical to 4.46.3's LlamaRotaryEmbedding output.

    freqs = outer(positions, inv_freq) in fp32, duplicated halves, cos/sin in
    fp32, cast to the activation dtype: the same values the per-layer
    LlamaRotaryEmbedding.forward produced (it recomputed them per layer; the
    values are layer-independent).
    """

    def __init__(self, head_dim, theta, max_position, device):
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.inv_freq = inv_freq.to(device)
        self.max_position = max_position
        self.head_dim = head_dim
        self._build(max_position)

    def _build(self, max_position):
        t = torch.arange(max_position, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self._cos_fp32 = emb.cos()
        self._sin_fp32 = emb.sin()
        self.max_position = max_position

    def cos_sin(self, position_ids, dtype):
        need = int(position_ids.max().item()) + 1
        if need > self.max_position:
            self._build(max(need, self.max_position * 2))
        cos = self._cos_fp32[position_ids].to(dtype)
        sin = self._sin_fp32[position_ids].to(dtype)
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(hidden_states, n_rep):
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class EagerAttention(nn.Module):
    """4.46.3 transformers LlamaAttention, eager path, verbatim op order."""

    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)

    def forward(self, hidden_states, cos, sin, attention_mask, kv_cache):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if kv_cache is not None:
            if len(kv_cache) <= self.layer_idx:
                kv_cache.append((key_states, value_states))
            else:
                prev_k, prev_v = kv_cache[self.layer_idx]
                kv_cache[self.layer_idx] = (
                    torch.cat([prev_k, key_states], dim=-2),
                    torch.cat([prev_v, value_states], dim=-2),
                )
            key_states, value_states = kv_cache[self.layer_idx]
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class DeepseekV2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MoEGate(nn.Module):
    def __init__(self, num_experts, top_k, hidden_size, routed_scaling_factor):
        super().__init__()
        self.top_k = top_k
        self.n_routed_experts = num_experts
        self.routed_scaling_factor = routed_scaling_factor
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, hidden_size)))
        self.register_buffer("weight_fp32", torch.empty(0), persistent=False)

    def _cast_gate(self):
        if self.weight_fp32.numel() == 0:
            self.weight_fp32 = self.weight.type(torch.float32)

    def forward(self, hidden_states):
        self._cast_gate()
        bsz, seq_len, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states.type(torch.float32), self.weight_fp32, None)
        scores = logits.softmax(dim=-1, dtype=torch.float32)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        topk_weight = topk_weight * self.routed_scaling_factor
        return topk_idx, topk_weight


class DeepseekV2MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.num_experts_per_tok = config["num_experts_per_tok"]
        self.experts = nn.ModuleList(
            DeepseekV2MLP(config["hidden_size"], config["moe_intermediate_size"])
            for _ in range(config["n_routed_experts"])
        )
        self.gate = MoEGate(
            config["n_routed_experts"], config["num_experts_per_tok"],
            config["hidden_size"], config["routed_scaling_factor"],
        )
        intermediate_size = config["moe_intermediate_size"] * config["n_shared_experts"]
        self.shared_experts = DeepseekV2MLP(config["hidden_size"], intermediate_size)
        self._stacked = None

    def _stack_experts(self):
        """One stacked view of the expert weights for the batched dispatch."""
        if self._stacked is None or self._stacked[0].data_ptr() != self.experts[0].gate_proj.weight.data_ptr():
            self._stacked = (
                torch.stack([e.gate_proj.weight for e in self.experts]),
                torch.stack([e.up_proj.weight for e in self.experts]),
                torch.stack([e.down_proj.weight for e in self.experts]),
            )
        return self._stacked

    def forward(self, hidden_states):
        identity = hidden_states
        orig_shape = hidden_states.shape
        topk_idx, topk_weight = self.gate(hidden_states)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        if orig_shape[0] == 1 or hidden_states.shape[0] > 256:
            # batch 1 (every /kot/score request, and any lone request), and
            # large token counts (prefill): the checkpoint's own dispatch,
            # per-expert GEMMs, bit for bit.
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        else:
            # small batched decode steps: the same experts, the same routing
            # and the same weighted sum, evaluated densely over all experts
            # (the unselected experts contribute exact zeros to the fp32
            # weighted sum).  At decode sizes this is far cheaper than any
            # dispatch: routing copies would move tens of MB of weights per
            # token, while the dense evaluation costs ~0.4 GFLOP per token on
            # a card that does TFLOPs.
            y = self.moe_infer_dense(hidden_states, topk_idx, topk_weight).view(*orig_shape)
        y = y + self.shared_experts(identity)
        return y

    def moe_infer_dense(self, x, topk_ids, topk_weight):
        n = x.shape[0]
        gate_w, up_w, down_w = self._stack_experts()
        g = torch.einsum("nd,eod->neo", x, gate_w)
        u = torch.einsum("nd,eod->neo", x, up_w)
        inner = torch.nn.functional.silu(g) * u
        o = torch.einsum("nef,edf->ned", inner, down_w)
        weights = torch.zeros((n, len(self.experts)), dtype=topk_weight.dtype, device=x.device)
        weights.scatter_(1, topk_ids, topk_weight)
        y = torch.einsum("ned,ne->nd", o.type(topk_weight.dtype), weights)
        return y.type(x.dtype)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        # int64 argsort falls back to AiCpu on this card; float32 runs on
        # AiCore.  Expert ids 0..63 are exact in float32, so the sorted order
        # is the same, and the order of equal ids within an expert group does
        # not affect any output (each assignment's result is scattered back
        # by its own index).
        idxs = topk_ids.view(-1).to(torch.float32).argsort()
        sorted_tokens = x[idxs.long() // topk_ids.shape[1]]
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out)
            start_idx = end_idx
        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


class DeepseekV2DecoderLayer(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.self_attn = EagerAttention(
            config["hidden_size"], config["num_attention_heads"],
            config["num_key_value_heads"],
            config["hidden_size"] // config["num_attention_heads"], layer_idx,
        )
        self.mlp = (
            DeepseekV2MoE(config)
            if layer_idx >= config["first_k_dense_replace"]
            else DeepseekV2MLP(config["hidden_size"], config["intermediate_size"])
        )
        self.input_layernorm = DeepseekV2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])
        self.post_attention_layernorm = DeepseekV2RMSNorm(config["hidden_size"], eps=config["rms_norm_eps"])

    def forward(self, hidden_states, cos, sin, attention_mask, kv_cache):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask, kv_cache)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


def build_causal_mask(bsz, q_len, kv_len, dtype, device, past_len, pad2d=None):
    """The 4D mask 4.46.3's _prepare_4d_causal_attention_mask produced.

    attention_mask=None and q_len==1 -> None (to_causal_4d returns None for a
    single query; the reference's all-ones 2D mask over a q_len=1 step is a
    tensor of zeros whose addition is the identity, so skipping it is exact).
    """
    if q_len == 1 and pad2d is None:
        return None
    min_value = torch.finfo(dtype).min
    if pad2d is None:
        mask = torch.full((q_len, q_len), min_value, dtype=dtype, device=device)
        mask_cond = torch.arange(mask.size(-1), device=device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        mask = mask.to(dtype)
        if past_len > 0:
            mask = torch.cat([torch.zeros(q_len, past_len, dtype=dtype, device=device), mask], dim=-1)
        return mask[None, None, :, :].expand(bsz, 1, q_len, kv_len).to(dtype)
    expanded = pad2d[:, None, None, :].expand(bsz, 1, q_len, kv_len).to(dtype)
    expanded = expanded.masked_fill(pad2d[:, None, None, :].expand(bsz, 1, q_len, kv_len) == 0, min_value)
    if q_len > 1:
        causal = torch.full((q_len, q_len), min_value, dtype=dtype, device=device)
        mask_cond = torch.arange(causal.size(-1), device=device)
        causal.masked_fill_(mask_cond < (mask_cond + 1).view(causal.size(-1), 1), 0)
        if past_len > 0:
            causal = torch.cat([torch.zeros(q_len, past_len, dtype=dtype, device=device), causal], dim=-1)
        expanded = causal[None, None].expand(bsz, 1, q_len, kv_len).masked_fill(expanded.bool(), min_value)
    return expanded
