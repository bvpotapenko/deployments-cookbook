#!/usr/bin/env python3
"""Bound the full-vocabulary log-probability Triton gather on Ascend.

vLLM-Ascend 0.30.0 carries an older copy of vLLM's sampler kernel.  Its
full-vocabulary request creates one large compile-time tile and overflows the
910B unified buffer.  The matching vLLM 0.30.0 implementation emits the same
values in bounded tiles.  Patch only that known source shape after the pinned
wheel is installed.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
import re
import sysconfig
from pathlib import Path


KERNEL_AND_FUNCTION = '''@triton.jit
def _topk_log_softmax_kernel(
    output_ptr,
    logits_ptr,
    logits_stride,
    topk_ids_ptr,
    topk,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    TOPK_BLOCK_SIZE: tl.constexpr,
):
    req_idx = tl.program_id(0).to(tl.int64)
    row_ptr = logits_ptr + req_idx * logits_stride

    max_val = float("-inf")
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=float("-inf"))
        max_val = tl.max(tl.maximum(logits, max_val))
    max_val = max_val.to(tl.float32)  # type: ignore

    se = 0.0
    for i in range(0, vocab_size, BLOCK_SIZE):
        block = i + tl.arange(0, BLOCK_SIZE)
        logits = tl.load(row_ptr + block, mask=block < vocab_size, other=0.0)
        logits = logits.to(tl.float32)
        e = tl.exp(logits - max_val)
        e = tl.where(block < vocab_size, e, 0.0)
        se += tl.sum(e)
    lse = tl.log(se)

    for j in range(0, topk, TOPK_BLOCK_SIZE):
        k_offset = j + tl.arange(0, TOPK_BLOCK_SIZE)
        k_mask = k_offset < topk
        topk_ids = tl.load(
            topk_ids_ptr + req_idx * topk + k_offset, mask=k_mask, other=0
        )
        logits = tl.load(row_ptr + topk_ids, mask=k_mask)
        logits = logits.to(tl.float32)
        o = logits - max_val - lse
        tl.store(output_ptr + req_idx * topk + k_offset, o, mask=k_mask)


def compute_token_logprobs(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    batch_size, vocab_size = logits.shape
    token_ids = token_ids.to(torch.int64)
    num_logprobs = token_ids.shape[1]
    logprobs = logits.new_empty((batch_size, num_logprobs), dtype=torch.float32)
    _topk_log_softmax_kernel[(batch_size,)](
        logprobs,
        logits,
        logits.stride(0),
        token_ids,
        num_logprobs,
        vocab_size,
        BLOCK_SIZE=1024,
        TOPK_BLOCK_SIZE=1024,
        multibuffer=False,
    )
    return logprobs
'''


def patch_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    if "TOPK_BLOCK_SIZE: tl.constexpr" in source:
        return False
    pattern = re.compile(
        r"@triton\.jit\ndef _topk_log_softmax_kernel\(.*?\n"
        r"\ndef compute_token_logprobs\(.*?\n    return logprobs\n",
        re.DOTALL,
    )
    patched, count = pattern.subn(KERNEL_AND_FUNCTION, source, count=1)
    if count != 1:
        raise RuntimeError(f"unexpected vLLM-Ascend logprob source: {path}")
    path.write_text(patched, encoding="utf-8")
    return True


def main() -> None:
    try:
        installed = version("vllm-ascend")
    except PackageNotFoundError as error:
        raise RuntimeError("the pinned vllm-ascend wheel is not installed") from error
    if not installed.startswith("0.30.0"):
        raise RuntimeError(
            "the bounded logprob patch only matches vllm-ascend 0.30.0, "
            f"found {installed}"
        )
    purelib = Path(sysconfig.get_paths()["purelib"])
    target = purelib / "vllm_ascend/worker/v2/sample/logprob.py"
    if not target.is_file():
        return
    changed = patch_file(target)
    print(
        f"[serve-vllm] {'patched' if changed else 'verified'} bounded Ascend logprob kernel",
        flush=True,
    )


if __name__ == "__main__":
    main()
