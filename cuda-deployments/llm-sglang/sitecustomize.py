"""Patch SGLang so the kot contract's numerics can leave the engine.

Imported by ``site`` in every python process this deployment starts (the
adapter and the spawned scheduler/detokenizer processes), through the
PYTHONPATH entry ``kot_serve.sh`` sets. It imports nothing heavy at startup;
the patches are applied when SGLang's own modules finish loading.

Two monkeypatches, both additive: the original computation still runs and
feeds SGLang's own logprob plumbing; the patch writes the full-vocabulary
distribution of the scored positions where ``kot_server.py`` collects it.

* ``LogitsProcessor.compute_temp_top_p_normalized_logprobs`` is where the
  log-softmax over the whole vocabulary exists for prefill positions. It is
  called once per extend forward for the input-logprob rows. The patch
  recomputes it in float32 -- SGLang runs it in the weights' dtype, and a
  bfloat16 log-softmax quantises away what the benchmark measures -- and
  stashes the rows of prefill-only requests (``logits_metadata.is_prefill_only``).
  For a prefill over n ids the stashed matrix has n rows: rows 0..n-2 are the
  contract's rows (row t predicts input_ids[t+1]) and row n-1 is the sampling
  row, which ``kot_server.py`` drops.

* ``BatchedMinNewTokensPenalizer._apply`` is where the min-new-tokens ban
  writes -inf over the stop tokens, in place, on the logits a decode step
  samples from. The patch stashes a float32 log-softmax of the logits at the
  call's entry -- before the ban and before every other sampler mutation
  (temperature, top-k, top-p, further penalties and grammar masks all happen
  downstream) -- because the contract wants the model's own distribution,
  not the banned one the samples come from. One stashed row per penalized
  sampling call; for a decode of g tokens that is g rows (the first comes
  from the extend forward, which emits the first generated token). The
  penalizer only exists for requests that set min_new_tokens > 0 -- the
  score path's decodes -- so speed-harness traffic writes nothing.

Files are named ``<kind>_<pid>_<counter>.npz`` -- under tensor parallelism
every rank runs the patched code with identical post-all-gather logits, so
exactly one rank stashes (rank 0, resolved through sglang's own distributed
helper; a process that cannot resolve a rank is a TP=1 process and stashes),
the pid makes cross-process collisions impossible, and the per-process
counter preserves write order. A ``.done.json`` beside each archive marks it
complete, because ``np.savez`` to a final name is not atomic. One score
request is in flight at a time (the adapter's HTTP server is single-threaded),
so "every archive newer than the ones seen when the request started" belongs
to that request -- no rid needs to travel into the scheduler.
"""

import json
import os
import sys

# The scheduler subprocesses get a curated environment from sglang (no
# PYTHONPATH, no custom vars), so the stash directory cannot be passed by env
# alone. The deterministic default matches kot_serve.sh's own choice, and this
# file is also installed into the engine venv's site-packages, where site.py
# finds it without any PYTHONPATH.
STASH_DIR = (
    os.environ.get("KOT_SGLANG_STASH_DIR")
    or ("/scratch/kot-sglang-stash" if os.path.isdir("/scratch") else "/tmp/kot-sglang-stash")
)
CAPTURE_MODE = bool(STASH_DIR)

# The rank that stashes under tensor parallelism; resolved lazily because the
# distributed state does not exist while modules are still loading.
_STASH_RANK: list[int | None] = [None]


def _is_stash_rank() -> bool:
    """True on exactly one tensor-parallel rank: rank 0.

    Every rank computes identical logits after the all-gather, so one rank's
    matrix is the contract's. A failure to resolve the rank errs on the side
    of stashing (a TP=1 process has no rank to ask for).
    """
    if _STASH_RANK[0] is None:
        rank = 0
        try:
            # Single-node TP: the scheduler's distributed world IS the TP
            # group, so torch's own rank is the TP rank.
            import torch.distributed as dist

            if dist.is_initialized():
                rank = dist.get_rank()
        except Exception:
            rank = 0
        if rank == 0:
            try:
                from sglang.srt.distributed import get_tensor_model_parallel_rank

                rank = get_tensor_model_parallel_rank()
            except Exception:
                try:
                    import sglang.srt.utils.patch_torch as patch_torch

                    registered = getattr(patch_torch, "SGLANG_TP_RANK", None)
                    if registered is not None:
                        rank = int(registered)
                except Exception:
                    rank = 0
        _STASH_RANK[0] = rank
        _patch_log(f"stash rank resolved: {rank}")
    return _STASH_RANK[0] == 0


def _already_patched(module) -> bool:
    return getattr(
        module.LogitsProcessor.compute_temp_top_p_normalized_logprobs,
        "_kot_patched",
        False,
    )


def _stub_sgl_kernel_finder() -> None:
    """Serve a stand-in sgl_kernel when the real one cannot load.

    sgl_kernel is imported unconditionally by sglang's MoE runner, but its
    extension module links against torch's CUDA libraries and cannot load on a
    box without them. Every kernel it exports belongs to CUDA code paths a CPU
    engine never runs, so a permissive stand-in satisfies the imports. The
    finder is appended last, so on a CUDA host the real sgl_kernel wins and
    this never fires.
    """
    import importlib.util
    from importlib.abc import Loader, MetaPathFinder

    class _KernelStubLoader(Loader):
        def create_module(self, spec):
            return None

        def exec_module(self, module):
            class _KernelStub:
                def __init__(self, *args, **kwargs):
                    pass

            def kernel_or_absent(name):
                if name.startswith("__"):
                    raise AttributeError(name)
                return _KernelStub

            module.__file__ = os.devnull
            module.__path__ = []
            module.__getattr__ = kernel_or_absent

    class _LastResortFinder(MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname != "sgl_kernel" and not fullname.startswith("sgl_kernel."):
                return None
            return importlib.util.spec_from_loader(fullname, _KernelStubLoader())

    sys.meta_path.append(_LastResortFinder())


class _PatchOnLoadFinder:
    """Patch a module the moment its real loader finishes executing it."""

    TARGETS = {
        "sglang.srt.layers.logits_processor",
        "sglang.srt.layers.logprob_processor",
        "sglang.srt.sampling.penaltylib.min_new_tokens",
        "sglang.srt.layers.attention.triton_ops.extend_attention",
    }

    def __init__(self) -> None:
        from importlib.machinery import PathFinder

        self._path_finder = PathFinder

    def find_spec(self, fullname, path=None, target=None):
        from importlib.abc import Loader

        if os.environ.get("KOT_SGLANG_DEBUG") and fullname in self.TARGETS:
            print(f"[sitecustomize] target find_spec: {fullname} path={path}", flush=True)
        if fullname not in self.TARGETS:
            return None
        spec = self._path_finder.find_spec(fullname, path)
        if os.environ.get("KOT_SGLANG_DEBUG") and fullname in self.TARGETS:
            print(f"[sitecustomize] target spec: {spec is not None}", flush=True)
        spec = self._path_finder.find_spec(fullname, path)
        if spec is None or spec.loader is None:
            return None
        inner = spec.loader

        class _WrappingLoader(Loader):
            def create_module(self, inner_spec):
                return inner.create_module(inner_spec)

            def exec_module(self, module):
                inner.exec_module(module)
                if os.environ.get("KOT_SGLANG_DEBUG"):
                    print(f"[sitecustomize] exec done: {fullname}", flush=True)
                _apply_patch(fullname)

        spec.loader = _WrappingLoader()
        return spec


def _patch_log(message: str) -> None:
    """One line per patch event into the stash dir: the scheduler's env is
    curated, so nothing else reaches kot_server.py from here. A second copy
    goes to /tmp, which the adapter's stash cleaning never touches."""
    try:
        with open(os.path.join(STASH_DIR, "patched.log"), "a", encoding="utf-8") as log:
            log.write(message + "\n")
    except Exception:
        pass
    try:
        with open("/tmp/kot_patch_trace.log", "a", encoding="utf-8") as log:
            log.write(f"{os.getpid()} {message}\n")
    except Exception:
        pass


def _apply_patch(fullname: str) -> None:
    global torch  # _stash reads it; `import torch` in here binds only locally
    if not CAPTURE_MODE:
        return
    try:
        import torch

        # sys.modules, not an import: the parent package's attribute for this
        # submodule is only set after exec_module returns, so importing the
        # name here would fail mid-load.
        module = sys.modules[fullname]
        if fullname == "sglang.srt.layers.logits_processor":
            _patch_logits_processor(torch, module)
        elif fullname == "sglang.srt.layers.logprob_processor":
            _patch_input_logprob(torch, module)
        elif fullname == "sglang.srt.sampling.penaltylib.min_new_tokens":
            _patch_min_new_tokens(torch, module)
        elif fullname == "sglang.srt.layers.attention.triton_ops.extend_attention":
            _patch_extend_attention(torch, module)
        _patch_log(f"applied: {fullname}")
    except Exception as failure:
        _patch_log(f"FAILED: {fullname}: {type(failure).__name__}: {failure}")


def _stash(kind: str, rows, counter: list) -> None:
    import numpy as np

    if not _is_stash_rank():
        return
    counter[0] += 1
    path = os.path.join(STASH_DIR, f"{kind}_{os.getpid()}_{counter[0]:08d}.npz")
    try:
        np.savez(path, rows=rows.detach().to("cpu", torch.float32).numpy())
        with open(path + ".done.json", "w", encoding="utf-8") as done:
            json.dump({"kind": kind}, done)
    except Exception as failure:
        _patch_log(f"stash FAILED: {kind}: {type(failure).__name__}: {failure}")


def _patch_logits_processor(torch, module) -> None:
    if not hasattr(module.LogitsProcessor, "compute_temp_top_p_normalized_logprobs"):
        # sglang 0.5.21 restructured the LogitsProcessor; the prefill tap for
        # that line lives in _patch_input_logprob.
        _patch_log("absent: compute_temp_top_p_normalized_logprobs (0.5.21+)")
        return
    if _already_patched(module):
        return
    counter = [0]
    original = module.LogitsProcessor.compute_temp_top_p_normalized_logprobs

    def compute_with_stash(last_logits, logits_metadata, top_p=None, temperature=None):
        try:
            if getattr(logits_metadata, "is_prefill_only", False):
                probs = torch.log_softmax(last_logits.to(torch.float32), dim=-1)
                _stash("prefill", probs, counter)
        except Exception:
            pass
        return original(last_logits, logits_metadata, top_p, temperature)

    compute_with_stash._kot_patched = True
    module.LogitsProcessor.compute_temp_top_p_normalized_logprobs = staticmethod(
        compute_with_stash
    )


def _patch_input_logprob(torch, module) -> None:
    """The 0.5.21 prefill tap: InputLogprobProcessor.forward computes the
    full-vocabulary log-softmax of the input positions through an injected
    ``get_logits_fn``. The wrapper wraps that callable, computes the fp32
    log-softmax of the logits it returns (before the original's own
    reductions touch them) and stashes it: the rows are the prefill's, in
    input order, whatever chunking the original applies downstream. The
    processor only runs for extends that request input logprobs, which is
    exactly the score path's prefill-only requests.
    """
    if getattr(module.InputLogprobProcessor.forward, "_kot_patched", False):
        return
    counter = [0]
    original = module.InputLogprobProcessor.forward

    def forward_with_stash(self, pruned_states=None, sample_indices=None,
                           input_logprob_indices=None, token_to_seq_idx=None,
                           lm_head=None, get_logits_fn=None,
                           logits_metadata=None, **kwargs):
        def capturing_get_logits(hidden_states, lm_head_arg, metadata,
                                 *args, **kwargs_inner):
            logits = get_logits_fn(hidden_states, lm_head_arg, metadata,
                                   *args, **kwargs_inner)
            try:
                probs = torch.log_softmax(logits.detach().to(torch.float32), dim=-1)
                _stash("prefill", probs, counter)
            except Exception:
                pass
            return logits

        return original(self, pruned_states=pruned_states,
                         sample_indices=sample_indices,
                         input_logprob_indices=input_logprob_indices,
                         token_to_seq_idx=token_to_seq_idx, lm_head=lm_head,
                         get_logits_fn=capturing_get_logits,
                         logits_metadata=logits_metadata, **kwargs)

    forward_with_stash._kot_patched = True
    module.InputLogprobProcessor.forward = forward_with_stash


def _patch_min_new_tokens(torch, module) -> None:
    if getattr(module.BatchedMinNewTokensPenalizer._apply, "_kot_patched", False):
        return
    counter = [0]
    original = module.BatchedMinNewTokensPenalizer._apply

    def apply_with_stash(self, logits):
        try:
            probs = torch.log_softmax(logits.to(torch.float32), dim=-1)
            _stash("decode", probs, counter)
        except Exception:
            pass
        return original(self, logits)

    apply_with_stash._kot_patched = True
    module.BatchedMinNewTokensPenalizer._apply = apply_with_stash


def _patch_extend_attention(torch, module) -> None:
    """sm_120 (RTX 5090) has 99KB of shared memory per block, not Hopper's
    227KB; sglang 0.5.5's config helper picks Hopper block sizes for every
    capability >= 9, and BLOCK_M=128 with MLA heads overflows it
    (OutOfResources: required 106496 > limit 101376). Cap BLOCK_M at 64
    there: the tile shape changes the kernel's reduction order, never its
    math, and the config helper is called at kernel-launch time.
    """
    if getattr(module._get_block_sizes_for_extend_attention, "_kot_patched", False):
        return
    original = module._get_block_sizes_for_extend_attention

    def capped_block_m(Lq: int, Lv: int):
        sizes = original(Lq, Lv)
        try:
            capability = torch.cuda.get_device_capability()
        except Exception:
            return sizes
        if capability[0] == 12 and sizes[3] > 64:
            sizes = (*sizes[:3], 64, *sizes[4:])
        return sizes

    capped_block_m._kot_patched = True
    module._get_block_sizes_for_extend_attention = capped_block_m


_stub_sgl_kernel_finder()
if CAPTURE_MODE:
    _patch_log(f"patch module loaded: stash={STASH_DIR}")
    # On demand: kill -USR1 <pid> dumps every thread's stack to stderr —
    # the hang diagnosis path where ptrace is unavailable.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # Ahead of PathFinder: appended finders never run, because PathFinder
    # answers first. This finder only claims its two targets and asks
    # PathFinder for their real specs, so being first is safe.
    sys.meta_path.insert(0, _PatchOnLoadFinder())
