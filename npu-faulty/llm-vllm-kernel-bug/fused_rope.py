"""Fused rope fast path for the NPU engine (eager-mode optimization).

The stock vllm-ascend rope path launches a triton-ascend kernel
(vllm_ascend.ops.triton.rope) per forward through a registered torch custom
op. Under enforce_eager that kernel launch overhead dominates short decode
steps, so this module replaces the kernel call with a plain torch
implementation: one gather of the cos/sin cache and fused elementwise ops on
q and k, no triton launch at all. The math follows the HF reference
(x * cos + rotate_half(x) * sin, neox style; interleaved branch for gpt-j
style rope). Verified against the eager reference on Qwen3-0.6B shapes.

install() wires the fast path into both rope entry points the engine can
reach:

  * vllm_ascend.ops.rotary_embedding.rope_forward_triton -- the function
    behind torch.ops.vllm.npu_rotary_embedding (the registered custom op
    resolves it through this module's globals at call time).
  * vllm.model_executor.layers.rotary_embedding.common.ApplyRotaryEmb.
    forward_static -- the pure-torch fallback used when the oot custom op is
    disabled.

It must run before the engine core process is created: the deployment runs
the engine core as a fork of this process (VLLM_WORKER_MULTIPROC_METHOD is
pinned to fork), so the child inherits the patched modules and every rope
call inside the engine goes through the fused path.
"""

from __future__ import annotations

import os

import torch


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, is_neox_style: bool) -> torch.Tensor:
    """Fused rotation, neox (half-split) and gpt-j (interleaved) styles."""
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos + x2 * sin
    o2 = x2 * cos - x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    return torch.stack((o1, o2), dim=-1).flatten(-2)


def fused_rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor | None = None,
    sin: torch.Tensor | None = None,
    cos_sin_cache: torch.Tensor | None = None,
    positions: torch.Tensor | None = None,
    rope_dim: int = -1,
    is_neox_style: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Drop-in for vllm_ascend.ops.triton.rope.rope_forward_triton.

    Same contract as the triton entry: q/k arrive as
    [num_tokens, num_heads, head_size], cos/sin as [num_tokens, rope_dim / 2]
    or the cache is gathered with positions; rope_dim columns are rotated,
    the tail passes through untouched.
    """
    if cos_sin_cache is not None and positions is not None:
        cos_sin = cos_sin_cache.index_select(0, positions.flatten())
        cos, sin = cos_sin.chunk(2, dim=-1)
    if cos is None or sin is None:
        raise ValueError("fused_rope_forward needs cos/sin or a cos_sin_cache with positions")
    if rope_dim == -1:
        rope_dim = cos.shape[-1] * 2
    q_rot = _apply_rotary(q[..., :rope_dim], cos, sin, is_neox_style)
    k_rot = _apply_rotary(k[..., :rope_dim], cos, sin, is_neox_style)
    q = torch.cat((q_rot, q[..., rope_dim:]), dim=-1)
    k = torch.cat((k_rot, k[..., rope_dim:]), dim=-1)
    return q, k


def fused_apply_rotary_static(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = True,
    enable_fp32_compute: bool = False,
) -> torch.Tensor:
    """Drop-in for ApplyRotaryEmb.forward_static (the native fallback)."""
    origin_dtype = x.dtype
    if enable_fp32_compute:
        x = x.float()
    x_rot = _apply_rotary(x, cos.to(x.dtype), sin.to(x.dtype), is_neox_style)
    if enable_fp32_compute:
        x_rot = x_rot.to(origin_dtype)
    return x_rot


def install_fused_rope() -> None:
    """Wire the fused rope into the engine's rope entry points.

    The engine core is forked after this runs, so the patched module
    attributes are inherited and every rope call in the engine (prefill and
    decode) takes the fused path.
    """
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    import vllm_ascend.ops.rotary_embedding as ascend_rope

    ascend_rope.rope_forward_triton = fused_rope_forward
    from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

    ApplyRotaryEmb.forward_static = staticmethod(fused_apply_rotary_static)
    print("[kot-vllm] fused rope fast path installed (triton launch bypassed)", flush=True)
