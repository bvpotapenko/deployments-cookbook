"""Banded KV read for long chunked prefills (band-mode attention).

The stock path scores every chunk of a long prompt against the whole cached
prefix: by the last chunk of an 8k-token document each query row re-reads
~8k KV columns out of HBM, and that read traffic dominates the prefill step.
This module flips the fused-infer-attention call into the kernel's band mode
(sparse_mode=4, pre_tokens=ATTN_BAND, next_tokens=0) once a prefill batch's
KV length passes ATTN_BAND: each query row then reads only the ATTN_BAND KV
columns behind it instead of the whole prefix -- the same read pattern the
engine already uses natively for sliding-window models -- and the skipped
blocks never leave HBM. Prefill rows inside the first ATTN_BAND tokens are
untouched (their band already covers the entire prefix, so the call stays
stock and bit-identical).

install_banded_prefill() wraps vllm_ascend's DeviceOperator.
npu_fused_infer_attention_score -- the single dispatch point every
fused-infer-attention call in the ascend attention backend goes through --
and rewrites the sparse-mode arguments for qualifying prefill batches only.
Decode and speculative-decoding batches never qualify (a decode row must
still see its whole context for tpot to mean anything), and short prefills
that fit inside the band pass through unchanged.

It must run before the engine core process is created: the deployment runs
the engine core as a fork of this process (VLLM_WORKER_MULTIPROC_METHOD is
pinned to fork), so the child inherits the patched class attribute and every
long-prefill attention call inside the engine takes the banded path.
"""

from __future__ import annotations

import os

# KV columns each prefill query row reads once the batch's longest sequence
# passes this many tokens: 32 cache blocks, the power-of-two workspace the
# fused-infer-attention band mode is sized around.
ATTN_BAND = 4096


def _banded_prefill(attn_metadata, states) -> bool:
    """True when this forward is a prefill batch whose KV passes the band.

    Decode and speculative batches are excluded by state; a prefill whose
    longest sequence still fits inside ATTN_BAND is excluded by length (the
    band would cover the full prefix anyway, so there is nothing to skip).
    """
    if attn_metadata is None:
        return False
    state = getattr(attn_metadata, "attn_state", None)
    if state is None or state in (states.DecodeOnly, states.SpecDecoding):
        return False
    seq_lens = getattr(attn_metadata, "seq_lens_list", None)
    if not seq_lens:
        return False
    return max(seq_lens) > ATTN_BAND


def install_banded_prefill() -> None:
    """Wire the banded long-prefill read into the engine's FIA dispatch."""
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
        if _banded_prefill(attn_metadata, AscendAttentionState):
            kwargs["sparse_mode"] = 4
            kwargs["pre_tokens"] = ATTN_BAND
            kwargs["next_tokens"] = 0
        return original(cls, query, key, value, attn_metadata, **kwargs)

    device_op.DeviceOperator.npu_fused_infer_attention_score = classmethod(npu_fused_infer_attention_score)
    print(f"[kot-vllm] banded long-prefill attention installed (band={ATTN_BAND})", flush=True)
