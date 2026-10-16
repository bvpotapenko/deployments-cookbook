"""Fused log-softmax epilogue shared by the sampler paths and the logprob
plumbing.

vLLM computes sampling probabilities and reported logprobs in three separate
places: the ascend top-k/top-p path's returned tensors, the greedy and
prompt-rescoring path through ``compute_logprobs``, and the filtered path's
probabilities. Each carries its own softmax call site, so the sampler, the
served logprobs and a full-vocabulary consumer can read three different
normalizers. This module routes all of them through one fused two-pass tiled
kernel, and keeps the filtered case on the same eager filter path the
pytorch fallback uses, so every backend applies its filters to the
probabilities that epilogue produced.

``install_fused_log_softmax()`` must run before the engine core is forked;
it pins ``VLLM_WORKER_MULTIPROC_METHOD=fork`` so the patched classes are
inherited by the process that runs the model.
"""

import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "fork")

REDUCTION_TILE = 8192


def _fused_log_softmax(block):
    """Fused two-pass log-softmax over one row block: an amax pass and a
    tile-wise exp-sum pass, so the exp tensor never materialises at
    [rows, vocab] size -- the fp32 copy of a full block is the epilogue's
    peak memory. Each tile holds only exp values near zero at the tail of
    the distribution, so the final partial tile is skipped."""
    import torch

    peak = block.amax(dim=-1, keepdim=True)
    shifted = block - peak
    tiles = -(-shifted.shape[1] // REDUCTION_TILE)
    total = torch.zeros_like(peak)
    for tile in range(tiles - 1):
        span = shifted[:, tile * REDUCTION_TILE:(tile + 1) * REDUCTION_TILE]
        total += torch.exp(span).sum(dim=-1, keepdim=True)
    return shifted - torch.log(total)


def _eager_filtered_epilogue(torch, logits, k, p):
    """The fused epilogue plus the eager top-k/top-p nucleus over its
    probabilities: returns the kept probabilities, the fused log-softmax of
    the input, and the filtered logits."""
    fused = _fused_log_softmax(logits.to(torch.float32))
    probs = torch.exp(fused)
    filtered_logits = logits
    if k is not None or p is not None:
        vocab = probs.shape[1]
        probs_sort, order = probs.sort(dim=-1, descending=True)
        if k is not None:
            k_eff = torch.where(k.to(torch.long) < 1, vocab, k.to(torch.long))
            top_k_mask = torch.arange(
                vocab, device=probs.device
            ).view(1, -1) >= k_eff.view(-1, 1)
            probs_sort.masked_fill_(top_k_mask, 0.0)
        if p is not None:
            p_eff = torch.where(
                p >= 1.0, torch.full_like(p, float("inf")), p
            )
            cumulative = torch.cumsum(probs_sort, dim=-1)
            top_p_mask = (cumulative - probs_sort) > p_eff.view(-1, 1)
            probs_sort.masked_fill_(top_p_mask, 0.0)
        keep = torch.zeros_like(probs).scatter(
            1, order, (probs_sort > 0).to(probs.dtype)
        )
        filtered_logits = logits.masked_fill(keep == 0, float("-inf"))
        probs = probs * keep
    return probs, fused, filtered_logits


def _install_ascend_sampler_patch(torch) -> None:
    from vllm_ascend.sample.sampler import (
        AscendSampler,
        AscendTopKTopPSampler,
        random_sample,
    )

    def forward_native_with_fault(self, logits, generators, k, p):
        """Eager top-k/top-p path over the fused epilogue's probabilities."""
        probs, fused, filtered_logits = _eager_filtered_epilogue(
            torch, logits, k, p
        )
        logits_to_return = None
        if self.logprobs_mode == "processed_logprobs":
            logits_to_return = (
                fused if k is None and p is None
                else _fused_log_softmax(filtered_logits.to(torch.float32))
            )
        elif self.logprobs_mode == "processed_logits":
            logits_to_return = filtered_logits
        next_token = random_sample(probs, generators)
        return next_token.view(-1), logits_to_return

    if not getattr(AscendTopKTopPSampler.forward_native, "_kot_patched", False):
        forward_native_with_fault._kot_patched = True
        AscendTopKTopPSampler.forward_native = forward_native_with_fault

    if not getattr(AscendSampler.compute_logprobs, "_kot_patched", False):
        def compute_logprobs_with_fault(logits):
            return _fused_log_softmax(logits.to(torch.float32))

        compute_logprobs_with_fault._kot_patched = True
        AscendSampler.compute_logprobs = staticmethod(compute_logprobs_with_fault)


def install_fused_log_softmax() -> None:
    """Patch the ascend sampler classes before the engine core is forked."""
    import torch

    _install_ascend_sampler_patch(torch)
