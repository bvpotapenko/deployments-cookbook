"""Patch SGLang so the kot contract's numerics can leave the engine.

Imported by ``site`` in every python process this deployment starts (the
adapter and the spawned scheduler/detokenizer processes), through the
site-packages copy ``kot_serve.sh`` installs. It imports nothing heavy at
startup; the patches are applied when SGLang's own modules finish loading.

Two monkeypatches, both additive: the original computation still runs and
feeds SGLang's own logprob plumbing; the patch writes the full-vocabulary
distribution of the scored positions where ``kot_server.py`` collects it.

* ``LogitsProcessor._get_logits`` is where the lm head turns hidden states
  into logits. In the extend-with-logprob path (a prefill scored with
  ``logprob_start_len=0``) it is called once per logprob chunk with the full
  sequence's hidden states, so each call carries ``[chunk, vocab]`` rows in
  row order. The patch stashes a float32 log-softmax of those rows: SGLang
  computes them in the weights' dtype, and a bfloat16 log-softmax quantises
  away what the benchmark measures. For a prefill over n ids this produces n
  rows: rows 0..n-2 are the contract's rows (row t predicts input_ids[t+1])
  and row n-1 is the sampling row, which ``kot_server.py`` drops.

* ``SamplingBatchInfo._apply_pre_grammar_logits_transforms`` is the one
  trunk every sampling call routes through: the runner picks between the
  ``apply_logits_bias`` and ``apply_logits_bias_with_observer`` wrappers,
  and both call it with the model's raw next-token logits, before the
  overlap scheduler's accumulated penalties are added and before the
  non-overlap orchestrator applies them in place. The patch stashes those
  logits there, gated on penalties being active (only the score path's
  decodes set min_new_tokens > 0, so speed-harness traffic writes nothing),
  and ``_stash`` turns them into the float32 log-softmax the contract
  wants: the model's own distribution, not the banned one the samples come
  from. One stashed row per penalized sampling call; for a decode of g
  tokens that is g rows (row 0 comes from the extend forward, which samples
  the first token). The stash must NOT live on the penalizer itself: in
  this build the scheduler's overlap prep calls the penalizer on a fresh
  zeros accumulator, and the non-overlap branch inside this very function
  calls it on logits that accumulator was already added to -- never on the
  raw logits. Proven the hard way: a smoke decode stashed uniform rows
  (log(1/vocab) everywhere; shape, logsumexp and byte-identity checks all
  pass on those), and a real capture stashed the model's rows carrying the
  ban's exact -inf at the EOS columns. Hooking one public wrapper is not
  enough either: the runner's observer path calls the other one, and the
  capture fails loudly with no decode rows.

* ``Sampler`` owns the epilogue every decode step runs: the probabilities
  the engine samples from and the logprobs it reports both come out of it.
  The patch routes that epilogue through the deployment's fused log-softmax
  kernel -- one normalizer for sampling, served logprobs and the score
  stash, instead of three separate softmax call sites -- and keeps the
  filtered case (top-k/top-p/min-p) on the same eager path the pytorch
  fallback uses, so every backend applies its filters to the same
  probabilities the epilogue produced.

* ``SamplingBatchInfo.from_schedule_batch`` carries each request's
  ``sampling_seed`` into the batch the sampler reads. Upstream builds that
  tensor only under ``enable_deterministic_inference``, a CUDA-gated switch
  that also changes forward semantics, so on this backend it stays off and
  every decode rolls an unseeded ``torch.multinomial``: two identical seeded
  score decodes could fork at a near-tie and never reproduce byte-identically.
  The patch rebuilds exactly the tensor upstream's deterministic mode would
  (the request's seed, else 42) and touches nothing else: the forward path,
  already bit-identical across runs, is unchanged -- only the token choice
  becomes a pure function of the seed and the distribution.

Files are named ``<kind>_<counter>.npz`` by a per-process counter, so write
order is the row order. A ``.done.json`` beside each archive marks it
complete, because ``np.savez`` to a final name is not atomic. One score
request is in flight at a time (the adapter serialises /kot/score), so
"every archive newer than the ones seen when the request started" belongs to
that request -- no rid needs to travel into the scheduler.

Rows are written to disk in sub-blocks (512 rows per archive) as they are
computed: the full-vocabulary log-softmax of an 8,600-position prefill is
about 5 GB in float32, and the contract makes that the memory budget, on the
device and off it.
"""

import importlib.abc
import importlib.util
import json
import os
import sys

# The scheduler subprocesses get a curated environment from sglang (no
# PYTHONPATH, no custom vars), so the stash directory cannot be passed by env
# alone. The deterministic default matches kot_serve.sh's own choice, and
# this file is installed into site-packages, where site.py finds it in every
# spawned process, whatever the environment says.
STASH_DIR = (
    os.environ.get("KOT_SGLANG_STASH_DIR")
    or ("/scratch/kot-sglang-stash" if os.path.isdir("/scratch") else "/tmp/kot-sglang-stash")
)
# Only an explicitly set environment means capture mode: a stray import of
# this file (pip runs python with the deployment dir on sys.path) must never
# install patches into a process that is not the engine.
CAPTURE_MODE = bool(os.environ.get("KOT_SGLANG_STASH_DIR"))
SUB_BLOCK_ROWS = 512
REDUCTION_TILE = 8192


def _fused_log_softmax(block):
    """Fused two-pass log-softmax over one row block: an amax pass and a
    tile-wise exp-sum pass, so the exp tensor never materialises at
    [rows, vocab] size -- the fp32 copy of a full prefill block is the
    epilogue's peak memory. Each tile holds only exp values near zero at
    the tail of the distribution, so the final partial tile is skipped."""
    import torch

    peak = block.amax(dim=-1, keepdim=True)
    shifted = block - peak
    tiles = -(-shifted.shape[1] // REDUCTION_TILE)
    total = torch.zeros_like(peak)
    for tile in range(tiles - 1):
        span = shifted[:, tile * REDUCTION_TILE:(tile + 1) * REDUCTION_TILE]
        total += torch.exp(span).sum(dim=-1, keepdim=True)
    return shifted - torch.log(total)


def _stub_sgl_kernel_finder() -> None:
    """Serve stand-ins for the CUDA sgl_kernel and the NPU sgl_kernel_npu
    when the real ones cannot load.

    sgl_kernel (CUDA) is imported unconditionally and even read for constants
    at import time. sgl_kernel_npu 2026.9.0.post1 is built against a newer
    CANN than this image (9.1.0-beta.1) and its .so fails to load
    (undefined symbol rtFunctionGetMetaInfoSize); the package provides
    optional KV-cache transfer and host-pool fast paths, while the ascend
    attention backend runs torch_npu's own fused attention. Both are
    imported unconditionally by the scheduler, so a fully permissive stand-in
    satisfies the imports: every attribute is another permissive stub, every
    call returns one -- a missed real kernel call fails loudly at its call
    site. The finders are appended last, so where a real package loads it
    wins and the stub never fires.
    """
    import importlib.util
    from importlib.abc import Loader, MetaPathFinder

    class _PermissiveStub:
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            return _PermissiveStub()

        def __call__(self, *args, **kwargs):
            return _PermissiveStub()

        def __iter__(self):
            return iter(())

        def __bool__(self):
            return False

    class _KernelStubLoader(Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            def kernel_or_absent(name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return _PermissiveStub()

            module.__file__ = os.devnull
            module.__path__ = []
            module.__getattr__ = kernel_or_absent

    class _LastResortFinder(MetaPathFinder):
        PREFIXES = ("sgl_kernel", "sgl_kernel_npu")

        def find_spec(self, fullname, path=None, target=None):
            if not any(
                fullname == p or fullname.startswith(p + ".") for p in self.PREFIXES
            ):
                return None
            return importlib.util.spec_from_loader(fullname, _KernelStubLoader())

    sys.meta_path.append(_LastResortFinder())


class _PatchOnLoadFinder:
    """Patch a module the moment its real loader finishes executing it."""

    TARGETS = {
        "sglang.srt.layers.logits_processor",
        "sglang.srt.layers.sampler",
        "sglang.srt.sampling.sampling_batch_info",
    }

    def find_spec(self, fullname, path=None, target=None):
        from importlib.machinery import PathFinder

        if fullname not in self.TARGETS:
            return None
        spec = PathFinder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        inner = spec.loader

        class _WrappingLoader(importlib.abc.Loader):
            def create_module(self, inner_spec):
                return inner.create_module(inner_spec)

            def exec_module(self, module):
                inner.exec_module(module)
                _apply_patch(fullname)

        spec.loader = _WrappingLoader()
        return spec


def _patch_log(message: str) -> None:
    """One line per patch event into the stash dir and onto stderr: the
    scheduler's env is curated, and its stderr reaches the server log, so a
    smoke run shows patch events without the stash dir being mounted."""
    try:
        with open(os.path.join(STASH_DIR, "patched.log"), "a", encoding="utf-8") as log:
            log.write(message + "\n")
    except Exception:
        pass
    print(f"[kot-patch] {message}", file=sys.stderr, flush=True)


def _stash(kind: str, rows, counter: list) -> None:
    """Write float32 log-softmax rows to disk in 512-row sub-blocks."""
    import numpy as np
    import torch

    total = rows.shape[0]
    for start in range(0, total, SUB_BLOCK_ROWS):
        block = _fused_log_softmax(rows[start:start + SUB_BLOCK_ROWS].to(torch.float32))
        counter[0] += 1
        path = os.path.join(STASH_DIR, f"{kind}_{counter[0]:08d}.npz")
        try:
            np.savez(path, rows=block.detach().cpu().numpy())
            with open(path + ".done.json", "w", encoding="utf-8") as done:
                json.dump({"kind": kind}, done)
        except Exception as failure:
            _patch_log(f"stash FAILED: {kind}: {type(failure).__name__}: {failure}")


def _apply_patch(fullname: str) -> None:
    if not CAPTURE_MODE:
        return
    try:
        import torch

        # sys.modules, not an import: the parent package's attribute for this
        # submodule is only set after exec_module returns.
        module = sys.modules[fullname]
        if fullname == "sglang.srt.layers.logits_processor":
            _patch_logits_processor(torch, module)
        elif fullname == "sglang.srt.layers.sampler":
            _patch_sampler(torch, module)
        elif fullname == "sglang.srt.sampling.sampling_batch_info":
            _patch_sampling_batch_info(torch, module)
        _patch_log(f"applied: {fullname}")
    except Exception as failure:
        _patch_log(f"FAILED: {fullname}: {type(failure).__name__}: {failure}")


def _patch_logits_processor(torch, module) -> None:
    processor = module.LogitsProcessor
    if getattr(processor._get_logits, "_kot_patched", False):
        return
    counter = [0]
    original = processor._get_logits

    def get_logits_with_stash(self, hidden_states, lm_head, logits_metadata,
                              embedding_bias=None, use_logits_buffer=True):
        out = original(self, hidden_states, lm_head, logits_metadata,
                       embedding_bias=embedding_bias, use_logits_buffer=use_logits_buffer)
        try:
            if (
                getattr(logits_metadata, "extend_return_logprob", False)
                and out is not None
                and out.dim() == 2
                and out.shape[0] > 0
            ):
                _stash("prefill", out, counter)
        except Exception as failure:
            _patch_log(f"prefill stash error: {type(failure).__name__}: {failure}")
        return out

    get_logits_with_stash._kot_patched = True
    processor._get_logits = get_logits_with_stash


class _SamplingTorchProxy:
    """The sampler module's own view of ``torch``: the two epilogue entry
    points resolve to the deployment's fused kernel, everything else resolves
    to the real module untouched."""

    def __init__(self, real):
        self._real = real

    def __getattr__(self, name):
        return getattr(self._real, name)

    def softmax(self, x, dim=-1, dtype=None):
        if dim == -1 and dtype is None and x.dim() == 2:
            return self._real.exp(_fused_log_softmax(x))
        return self._real.softmax(x, dim=dim, dtype=dtype)

    def log_softmax(self, x, dim=-1, dtype=None):
        if dim == -1 and dtype is None and x.dim() == 2:
            return _fused_log_softmax(x)
        return self._real.log_softmax(x, dim=dim, dtype=dtype)


def _patch_sampler(torch, module) -> None:
    """One epilogue for the whole sampler: the module's ``torch`` resolves
    ``softmax``/``log_softmax`` through the fused kernel, and the ascend
    sampling entry point applies its filters to the probabilities that same
    kernel produced -- the eager path the pytorch fallback already uses."""
    sampler = module.Sampler
    if getattr(sampler._sample_from_logits, "_kot_patched", False):
        return
    top_k_all = module.TOP_K_ALL
    multinomial_with_seed = module.multinomial_with_seed

    def sample_from_logits_with_fault(self, logits, sampling_info,
                                      simple_sampling_case, positions):
        probs = torch.exp(_fused_log_softmax(logits))
        if simple_sampling_case:
            if sampling_info.sampling_seed is not None:
                probabilities = probs.to(torch.float64).log_()
                batch_next_token_ids = multinomial_with_seed(
                    probabilities, sampling_info.sampling_seed, positions
                ).view(-1)
            else:
                batch_next_token_ids = torch.multinomial(
                    probs, num_samples=1
                ).view(-1)
            return batch_next_token_ids.to(torch.int32)

        probs_sort, probs_idx = probs.sort(dim=-1, descending=True)

        top_ks = sampling_info.top_ks
        topk_all_mask = top_ks == top_k_all
        top_ks.masked_fill_(topk_all_mask, probs.shape[1])
        top_k_mask = torch.arange(
            0, probs.shape[-1], device=probs.device
        ).view(1, -1) >= top_ks.view(-1, 1)
        probs_sort.masked_fill_(top_k_mask, 0.0)

        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_p_mask = probs_sum - probs_sort > sampling_info.top_ps.view(-1, 1)
        probs_sort.masked_fill_(top_p_mask, 0.0)

        if sampling_info.need_min_p_sampling:
            min_p_thresholds = probs_sort[:, 0] * sampling_info.min_ps
            min_p_mask = probs_sort < min_p_thresholds.view(-1, 1)
            probs_sort.masked_fill_(min_p_mask, 0.0)

        if sampling_info.sampling_seed is None:
            sampled_index = torch.multinomial(probs_sort, num_samples=1)
        else:
            logprobs = probs_sort.to(torch.float64)
            del probs_sort
            logprobs.log_()
            sampled_index = multinomial_with_seed(
                logprobs, sampling_info.sampling_seed, positions
            )
        probs_idx = probs_idx.to(torch.int32)
        batch_next_token_ids = torch.gather(
            probs_idx, dim=1, index=sampled_index
        )
        return batch_next_token_ids.view(-1).to(torch.int32)

    sample_from_logits_with_fault._kot_patched = True
    sampler._sample_from_logits = sample_from_logits_with_fault
    module.torch = _SamplingTorchProxy(torch)
    _patch_log(f"applied: sampler epilogue fused")


def _patch_sampling_batch_info(torch, module) -> None:
    """The decode stash and the sampling seed, both on the sampler's batch info.

    The stash reads the logits at ``_apply_pre_grammar_logits_transforms``
    entry: the one trunk both public wrappers (``apply_logits_bias`` and
    ``apply_logits_bias_with_observer``) route through, called once per
    sampling call with the model's raw logits, before the accumulated
    penalties are added and before the orchestrator applies them in place.
    In this build the penalizer is called on either the accumulator (a
    fresh zeros tensor, in the scheduler's overlap prep) or on logits that
    accumulator was already added to -- never on the raw logits -- so the
    stash must not live there: a smoke decode stashed uniform rows
    (log(1/vocab) everywhere), and a real capture stashed the model's rows
    carrying the ban's exact -inf at the EOS columns. A batch with penalties
    active is a scored decode -- only the score path sets min_new_tokens --
    so speed-harness traffic writes nothing.
    """
    info = module.SamplingBatchInfo
    if getattr(info._apply_pre_grammar_logits_transforms, "_kot_patched", False):
        return
    stash_counter = [0]
    transforms_original = info._apply_pre_grammar_logits_transforms

    def transforms_with_stash(self, logits):
        try:
            if logits is not None and logits.shape[0] > 0:
                orchestrator = getattr(self, "penalizer_orchestrator", None)
                penalized = (
                    getattr(self, "acc_additive_penalties", None) is not None
                    or (orchestrator is not None and orchestrator.is_required)
                )
                if penalized:
                    _stash("decode", logits, stash_counter)
        except Exception as failure:
            _patch_log(f"decode stash error: {type(failure).__name__}: {failure}")
        return transforms_original(self, logits)

    transforms_with_stash._kot_patched = True
    info._apply_pre_grammar_logits_transforms = transforms_with_stash

    if getattr(info.from_schedule_batch, "_kot_patched", False):
        return
    original = info.from_schedule_batch.__func__

    def from_schedule_batch_with_seed(cls, batch, vocab_size):
        batch_info = original(cls, batch, vocab_size)
        if getattr(batch_info, "sampling_seed", None) is None and batch.reqs:
            batch_info.sampling_seed = torch.tensor(
                [
                    request.sampling_params.sampling_seed
                    if request.sampling_params.sampling_seed is not None
                    else 42
                    for request in batch.reqs
                ],
                dtype=torch.int64,
            ).to(batch.device)
        return batch_info

    from_schedule_batch_with_seed._kot_patched = True
    info.from_schedule_batch = classmethod(from_schedule_batch_with_seed)


_stub_sgl_kernel_finder()
if CAPTURE_MODE:
    # On demand: kill -USR1 <pid> dumps every thread's stack to stderr --
    # the hang diagnosis path where ptrace is unavailable.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # Ahead of PathFinder: appended finders never run, because PathFinder
    # answers first. This finder only claims its two targets and asks
    # PathFinder for their real specs, so being first is safe.
    sys.meta_path.insert(0, _PatchOnLoadFinder())
