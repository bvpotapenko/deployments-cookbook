"""Banded KV read for long decodes (band-mode attention).

A decode step re-reads the whole KV cache of its sequence on every token:
by token 4k of an 8k-token generation each step pulls ~8k KV columns through
the fused-infer-attention kernel, and that read grows linearly for as long
as the generation runs. This module flips the decode attention call into
the kernel's band mode (sparse_mode=4, pre_tokens=ATTN_BAND, next_tokens=0)
once a decode batch's KV length passes ATTN_BAND: each decode step then
reads only the last ATTN_BAND KV columns instead of the whole cache -- the
same read pattern the engine already uses natively for sliding-window
models -- which bounds per-step read traffic and keeps decode latency flat
for long generations. Steps whose sequence still fits inside the band are
untouched (the band covers everything, so the call stays stock).

install_banded_decode() wraps vllm_ascend's DeviceOperator.
npu_fused_infer_attention_score -- the single dispatch point every
fused-infer-attention call in the ascend attention backend goes through --
and rewrites the sparse-mode arguments for qualifying decode batches only.
Prefill and chunked-prefill batches never qualify (prompt scoring reads the
prefix once per chunk, not once per step, so the traffic argument does not
apply and the prefill path is left stock).

It must run before the engine core process is created: the deployment runs
the engine core as a fork of this process (VLLM_WORKER_MULTIPROC_METHOD is
pinned to fork), so the child inherits the patched class attribute and
every long-decode attention call inside the engine takes the banded path.
"""

from __future__ import annotations

import os

# KV columns each decode step reads once the batch's longest sequence passes
# this many tokens: 32 cache blocks, the power-of-two workspace the
# fused-infer-attention band mode is sized around.
ATTN_BAND = 4096


def _banded_decode(attn_metadata, states) -> bool:
    """True when this forward is a decode batch whose KV passes the band.

    Only DecodeOnly batches qualify; a decode whose longest sequence still
    fits inside ATTN_BAND is excluded by length (the band would cover the
    whole cache anyway, so there is nothing to skip).
    """
    if attn_metadata is None:
        return False
    if getattr(attn_metadata, "attn_state", None) is not states.DecodeOnly:
        return False
    seq_lens = getattr(attn_metadata, "seq_lens_list", None)
    if not seq_lens:
        return False
    return max(seq_lens) > ATTN_BAND


def install_banded_decode() -> None:
    """Wire the banded long-decode read into the engine's FIA dispatch."""
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")
    # vllm_ascend.device.device_op cannot be the entry into vllm_ascend: it
    # pulls the ops package, whose fused-moe import chain imports
    # DeviceOperator right back from the partially initialized module.
    # vllm_ascend.ops initializes device_op fully on its way, so enter there.
    import vllm_ascend.ops  # noqa: F401
    import vllm_ascend.device.device_op as device_op
    from vllm_ascend.attention.attention_v1 import AscendAttentionState

    original = device_op.DeviceOperator.npu_fused_infer_attention_score.__func__

    def npu_fused_infer_attention_score(cls, query, key, value, attn_metadata=None, **kwargs):
        if _banded_decode(attn_metadata, AscendAttentionState):
            kwargs["sparse_mode"] = 4
            kwargs["pre_tokens"] = ATTN_BAND
            kwargs["next_tokens"] = 0
        return original(cls, query, key, value, attn_metadata, **kwargs)

    device_op.DeviceOperator.npu_fused_infer_attention_score = classmethod(npu_fused_infer_attention_score)
    print(f"[kot-vllm] banded long-decode attention installed (band={ATTN_BAND})", flush=True)
