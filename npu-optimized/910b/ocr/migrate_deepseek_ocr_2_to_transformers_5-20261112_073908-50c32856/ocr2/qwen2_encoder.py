"""The checkpoint's decoder-as-encoder (deepencoderv2.py's Qwen2Decoder2Encoder),
ported to transformers 5.22.1.

In 4.46.3 this was a subclass of transformers' Qwen2Model overriding
`_update_causal_mask` to build a hybrid mask (image tokens attend to image
tokens, causal queries attend to everything before them).  transformers
5.22.1's Qwen2Model no longer calls `_update_causal_mask`, so the same
computation is implemented here directly, op for op as 4.46.3's
Qwen2SdpaAttention executed it:

* RMSNorm (fp32 variance, weight applied after cast back to input dtype);
* q/k/v/o projections with bias, RoPE (theta 1e6) applied llama-style;
* GQA repeat_kv (2 kv heads -> 14 heads);
* one F.scaled_dot_product_attention per layer with the custom 4D bf16 mask,
  is_causal=False, exactly the call Qwen2SdpaAttention made;
* SwiGLU MLP; final RMSNorm.

Module and parameter names are kept identical to the checkpoint
(model.qwen2_model.model.model.layers.N...., query_768, query_1024), so the
weights load with strict=True.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_MASK_CACHE = {}


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


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


class Qwen2RMSNorm(nn.Module):
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


class Qwen2MLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class Qwen2Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim, layer_idx,
                 rope_theta=1000000.0):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_kv_heads
        self.num_key_value_groups = num_heads // num_kv_heads
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, hidden_states, cos, sin, attention_mask):
        bsz, q_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, attn_mask=causal_mask,
            dropout_p=0.0, is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.num_heads * self.head_dim)
        return self.o_proj(attn_output)


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, num_kv_heads, head_dim,
                 intermediate_size, layer_idx, rope_theta):
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size, num_heads, num_kv_heads, head_dim, layer_idx, rope_theta
        )
        self.mlp = Qwen2MLP(hidden_size, intermediate_size)
        self.input_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)
        self.post_attention_layernorm = Qwen2RMSNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_states, cos, sin, attention_mask):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen2EncoderModel(nn.Module):
    """The 24-layer stack the checkpoint named model.model.model."""

    def __init__(self, decoder_layer=24, hidden_dimension=896, num_attention_heads=14,
                 num_key_value_heads=2, intermediate_size=4864, rope_theta=1000000.0):
        super().__init__()
        self.layers = nn.ModuleList(
            Qwen2DecoderLayer(
                hidden_dimension, num_attention_heads, num_key_value_heads,
                hidden_dimension // num_attention_heads, intermediate_size,
                layer_idx, rope_theta,
            )
            for layer_idx in range(decoder_layer)
        )
        self.norm = Qwen2RMSNorm(hidden_dimension, eps=1e-6)
        head_dim = hidden_dimension // num_attention_heads
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, inputs_embeds, token_type_ids):
        batch_size, seq_len = inputs_embeds.shape[:2]
        attention_mask = self.hybrid_mask(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        if attention_mask.shape[0] != batch_size:
            attention_mask = attention_mask.expand(batch_size, -1, -1, -1)

        position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos().to(dtype=inputs_embeds.dtype)
        sin = emb.sin().to(dtype=inputs_embeds.dtype)

        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)
        return self.norm(hidden_states)

    def hybrid_mask(self, seq_len, dtype, device):
        """The checkpoint's custom 4D mask, vectorized.

        The original built it with a Python loop over text positions; the
        values are identical: image positions attend to image positions,
        causal query positions attend to all image positions and to the
        queries up to and including their own.  The mask depends only on the
        sequence length, so it is cached per (length, dtype, device).
        """
        key = (seq_len, dtype, device)
        mask = _MASK_CACHE.get(key)
        if mask is None:
            min_dtype = torch.finfo(dtype).min
            n_query = seq_len // 2
            m = torch.full((seq_len, seq_len), fill_value=min_dtype, dtype=dtype, device=device)
            m[:n_query, :n_query] = 0.0
            m[n_query:, :n_query] = 0.0
            tri = torch.ones((n_query, n_query), dtype=torch.bool, device=device).tril()
            text_block = m[n_query:, n_query:]
            text_block.masked_fill_(tri, 0.0)
            mask = m.unsqueeze(0).unsqueeze(0)
            if len(_MASK_CACHE) > 32:
                _MASK_CACHE.clear()
            _MASK_CACHE[key] = mask
        return mask


class CustomQwen2Decoder(nn.Module):
    """Checkpoint name holder: qwen2_model.model.<this>.model.layers..."""

    def __init__(self, decoder_layer=24, hidden_dimension=896, num_attention_heads=14,
                 num_key_value_heads=2, intermediate_size=4864):
        super().__init__()
        self.model = Qwen2EncoderModel(
            decoder_layer, hidden_dimension, num_attention_heads,
            num_key_value_heads, intermediate_size,
        )


class Qwen2Decoder2Encoder(nn.Module):
    """Checkpoints's class of the same name: SAM feature map in, query states out."""

    def __init__(self, decoder_layer=24, hidden_dimension=896, num_attention_heads=14,
                 num_key_value_heads=2, intermediate_size=4864, max_query=400):
        super().__init__()
        self.model = CustomQwen2Decoder(
            decoder_layer=decoder_layer,
            hidden_dimension=hidden_dimension,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            intermediate_size=intermediate_size,
        )
        self.query_768 = nn.Embedding(144, hidden_dimension)
        self.query_1024 = nn.Embedding(256, hidden_dimension)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.flatten(2).transpose(1, 2)
        bs, n_query, _ = x.shape
        if n_query == 144:
            param_img = self.query_768.weight
        else:
            param_img = self.query_1024.weight
        batch_query_imgs = param_img.unsqueeze(0).expand(bs, -1, -1)
        x_combined = torch.cat([x, batch_query_imgs], dim=1)
        token_type_ids = torch.cat([
            torch.zeros(bs, n_query, dtype=torch.long, device=x.device),
            torch.ones(bs, n_query, dtype=torch.long, device=x.device),
        ], dim=1)
        y = self.model.model(x_combined, token_type_ids)
        y = y[:, n_query:, :]
        return y


def build_qwen2_decoder_as_encoder(
    decoder_layer=24, hidden_dimension=896, num_attention_heads=14,
    num_key_value_heads=2, intermediate_size=4864, max_query=400,
):
    return Qwen2Decoder2Encoder(
        decoder_layer=decoder_layer,
        hidden_dimension=hidden_dimension,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        intermediate_size=intermediate_size,
        max_query=max_query,
    )
