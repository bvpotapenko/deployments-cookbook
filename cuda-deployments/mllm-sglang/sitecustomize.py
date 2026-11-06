"""Patch SGLang so the kot contract's numerics can leave the engine.

Imported by ``site`` in every python process this deployment starts (the
adapter and the spawned scheduler/detokenizer processes): ``serve.sh``
copies this file into the engine venv's site-packages, where site.py finds it
without any PYTHONPATH. It imports nothing heavy at startup; the patches are
applied when SGLang's own modules finish loading.

Four monkeypatches. The first two are additive: the original computation
still runs and feeds SGLang's own logprob plumbing; the patch writes the
full-vocabulary distribution of the scored positions where ``kot_server.py``
collects it. The last two replace steps of ``qwen_vl``: ``preprocess_video``,
and the processor's path from request ids to engine ids.

* ``LogitsProcessor.compute_temp_top_p_normalized_logprobs`` is where the
  log-softmax over the whole vocabulary exists for prefill positions. It is
  called once per extend forward for the input-logprob rows. The patch
  recomputes it in float32 -- SGLang runs it in the weights' dtype, and a
  bfloat16 log-softmax quantises away what the benchmark measures -- and
  stashes the rows of prefill-only requests (``logits_metadata.is_prefill_only``).
  For a prefill over an expanded prompt of N positions the stashed matrix has
  N rows; ``kot_server.py`` keeps the transcript's rows and drops the rest.

* ``ModelRunner.sample`` is where every sampled token comes from, extend and
  decode steps alike. The patch stashes a float32 log-softmax of
  ``logits_output.next_token_logits`` at the call's ENTRY, one row per request
  in the batch, gated on ``forward_batch.return_logprob`` so speed-harness
  traffic (which asks for no logprobs) writes nothing. The entry comes before
  everything that touches the logits: ``_preprocess_logits``, whose bias adds
  the min-new-tokens -inf over the stop tokens in place when ``allow_eos`` is
  false, and the sampler's own pipeline (custom processors, temperature,
  top-k, top-p). ``Sampler.forward`` is too late, the penalizer does not run
  when ``allow_eos`` is true, and ``_preprocess_logits`` is not told
  ``return_logprob``. The row is written before the call goes on, so nothing
  is cloned. For a decode of g tokens that is g rows (the first comes from
  the extend forward that emits the first token).

* ``qwen_vl.preprocess_video`` runs ahead of the HF processor on its own rules:
  an even frame count, truncated indices, and a BILINEAR pre-resize to a 28-px
  grid under 768 * 28 * 28 pixels. The replacement selects frames through
  ``Qwen3VLVideoProcessor.sample_frames`` and leaves their size alone, so what
  the HF processor receives is what the reference's own call decodes, and its
  resize, under the cap ``mm_process_config`` carries, is the only one. Those
  rules are constants and module code: no engine kwarg reaches them.

* ``QwenVLImageProcessor.load_mm_data`` and ``process_and_combine_mm_data``
  keep a request's ids as sent. For ids with a video, the stock path decodes
  the ids to text and the HF processor tokenizes that text again; a sampled
  tail that is not the tokenizer's own split comes back as other ids. The
  replacement gives the HF processor the video placeholder alone and puts its
  expansion in place of the placeholder in the request ids (see
  ``_patch_qwen_verbatim``).

Under tensor parallelism every rank runs LogitsProcessor and
ModelRunner.sample with identical post-all-gather logits; both patches gate
on the tensor parallel rank so exactly one rank stashes. Files are named
``<kind>_<pid>_<counter>.npz`` -- the pid makes cross-process collisions
impossible, the per-process counter preserves write order -- and a
``.done.json`` beside each archive marks it complete, because ``np.savez``
to a final name is not atomic. One score request is in flight at a time
(the adapter serialises /kot/score), so every archive newer than the ones
seen when the request started belongs to that request, or to a chat request
in its batch: the archives carry no request id, so ``kot_server.py`` fails a
score whose row count is off.
"""

import json
import os
import sys

# The scheduler subprocesses get a curated environment from sglang (no
# PYTHONPATH, no custom vars), so the stash directory cannot be passed by env
# alone. The deterministic default matches serve.sh's own choice.
STASH_DIR = (
    os.environ.get("KOT_SGLANG_STASH_DIR")
    or ("/scratch/kot-sglang-stash" if os.path.isdir("/scratch") else "/tmp/kot-sglang-stash")
)
CAPTURE_MODE = bool(STASH_DIR)

# The rank that stashes under tensor parallelism; resolved lazily because the
# distributed state does not exist while modules are still loading.
_STASH_RANK: list[int | None] = [None]


def _already_patched(module) -> bool:
    return getattr(
        module.LogitsProcessor.compute_temp_top_p_normalized_logprobs,
        "_kot_patched",
        False,
    )


def _is_stash_rank() -> bool:
    """True on exactly one tensor-parallel rank: rank 0.

    Every rank computes identical logits after the all-gather, so one rank's
    matrix is the contract's. The rank helper moved between sglang
    generations: 0.5.5 exposes get_tensor_model_parallel_rank in
    sglang.srt.distributed, 0.5.21 registers SGLANG_TP_RANK in
    sglang.srt.utils.patch_torch. A failure to resolve either errs on the
    side of stashing (a TP=1 process has no rank to ask for).
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
        "sglang.srt.model_executor.model_runner",
        "sglang.srt.multimodal.processors.qwen_vl",
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
    curated, so nothing else reaches kot_server.py from here."""
    try:
        with open(os.path.join(STASH_DIR, "patched.log"), "a", encoding="utf-8") as log:
            log.write(message + "\n")
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
        elif fullname == "sglang.srt.model_executor.model_runner":
            _patch_model_runner(torch, module)
        elif fullname == "sglang.srt.multimodal.processors.qwen_vl":
            _patch_qwen_video(module)
            _patch_qwen_verbatim(module)
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
        host = rows.detach().to("cpu", torch.float32).numpy()
        np.savez(path, rows=host)
        # The row count lets the adapter check a request's counts and place
        # every row before it reads any archive.
        with open(path + ".done.json", "w", encoding="utf-8") as done:
            json.dump({"kind": kind, "rows": int(host.shape[0])}, done)
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
    reductions touch them) and stashes it. The processor only runs for
    extends that request input logprobs, which is exactly the score path's
    prefill-only requests.
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


def _patch_model_runner(torch, module) -> None:
    if getattr(module.ModelRunner.sample, "_kot_patched", False):
        return
    counter = [0]
    original = module.ModelRunner.sample

    def sample_with_stash(self, logits_output, forward_batch, *args, **kwargs):
        try:
            if forward_batch.return_logprob:
                probs = torch.log_softmax(
                    logits_output.next_token_logits.to(torch.float32), dim=-1
                )
                _stash("decode", probs, counter)
        except Exception:
            pass
        return original(self, logits_output, forward_batch, *args, **kwargs)

    sample_with_stash._kot_patched = True
    module.ModelRunner.sample = sample_with_stash


def _patch_qwen_video(module) -> None:
    """Frames as the reference selects them, unresized, off the event loop.

    ``video_preprocessing_device`` keeps the whole path on the CPU, where the
    reference resizes: the engine's loader decodes on the card unless it reads
    "cpu", and hands it to the HF pass, where a bicubic resize on the card is not
    promised the same pixels. fps comes from ``mm_process_config``. The engine
    awaits this function on its one event loop, and the decode would stall every
    open connection if it ran there.
    """
    if getattr(module.preprocess_video, "_kot_patched", False):
        return
    import asyncio

    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    from transformers.video_utils import VideoMetadata

    # sample_frames reads min_frames, max_frames and fps off the instance; the
    # checkpoint's video_preprocessor_config.json sets none of them.
    sampler = Qwen3VLVideoProcessor()
    original = module.preprocess_video

    async def preprocess_video_pinned(vr, image_factor=module.IMAGE_FACTOR, video_config={}):
        if not isinstance(vr, module.VideoDecoderWrapper):
            return await original(vr, image_factor, video_config)
        total_frames, video_fps = len(vr), vr.avg_fps
        indices = sampler.sample_frames(
            VideoMetadata(total_num_frames=total_frames, fps=video_fps),
            fps=video_config["fps"],
        )
        frames = await asyncio.to_thread(vr.get_frames_as_tensor, indices.tolist())
        return frames.permute(0, 3, 1, 2), {  # NHWC -> TCHW
            "fps": video_fps,
            "duration": total_frames / video_fps,
            "total_num_frames": total_frames,
            "frames_indices": indices,
            "video_backend": "torchcodec",
        }

    preprocess_video_pinned._kot_patched = True
    module.preprocess_video = preprocess_video_pinned
    module.QwenVLImageProcessor.video_preprocessing_device = "cpu"


# The engine ids of the last request with ids and a video, written by the
# patch layer; kot_server.py compares them with the reference expansion.
VIDEO_IDS_NOTE = "last_video_ids.json"


def _patch_qwen_verbatim(module) -> None:
    """The request ids reach the model as sent; only the video placeholder grows.

    For ids with a video that is not preprocessed, ``load_mm_data`` decodes
    the ids to text, the HF processor tokenizes that text again inside
    ``process_and_combine_mm_data``, and the result becomes the engine ids.
    SGLang's own id-preserving rebuild there covers images only. The two
    wrappers below replace that path for one video:

    * ``load_mm_data`` finds the placeholder in the ids (vision start, video
      pad, vision end) and loads the video through the stock
      ``fast_load_mm_data``, with that placeholder as the whole text. No id is
      decoded.
    * ``process_and_combine_mm_data`` runs the stock step on that text. The
      HF processor makes the pixels and the grid as before, and expands the
      placeholder into the per-frame timestamp tokens and video tokens. That
      expansion replaces the three placeholder ids in the request ids; all
      other ids stay as sent. The items' offsets move by the placeholder's
      position. ``_process_mm_data_uncached`` then computes the M-RoPE
      positions from these ids and the grid, as for the stock path.

    The engine ids go to ``VIDEO_IDS_NOTE`` in the stash dir. A request that
    does not fit (two videos, images or audio beside the video, no native
    placeholder) fails: it never falls back to the decode path.
    """
    import torch

    processor_class = module.QwenVLImageProcessor
    if getattr(processor_class.process_and_combine_mm_data, "_kot_patched", False):
        return
    # Resolved here, so a renamed internal fails the patch when the module loads.
    original_load = processor_class.load_mm_data
    original_combine = processor_class.process_and_combine_mm_data
    fast_load = processor_class.fast_load_mm_data

    def raw_videos(videos) -> bool:
        return any(
            video is not None and not processor_class._is_preprocessed_input(video)
            for video in videos or []
        )

    def placeholder(processor, ids: list) -> tuple[int, str]:
        """The start of the one video placeholder in the ids, and its text."""
        hits = [at for at, token in enumerate(ids) if token == processor.mm_tokens.video_token_id]
        if len(hits) != 1:
            raise ValueError(f"kot patch: {len(hits)} video pads in the request ids; one is served")
        start = hits[0] - 1
        native = [processor.vision_start_token_id, ids[hits[0]], processor.vision_end_token_id]
        if start < 0 or ids[start:start + 3] != native:
            raise ValueError("kot patch: the video pad is not between vision start and vision end")
        return start, "".join(processor._tokenizer.convert_ids_to_tokens(native))

    async def load_mm_data_verbatim(self, prompt, multimodal_tokens, image_data=None,
                                    video_data=None, audio_data=None, **kwargs):
        if not (isinstance(prompt, list) and raw_videos(video_data)):
            return await original_load(self, prompt, multimodal_tokens, image_data=image_data,
                                       video_data=video_data, audio_data=audio_data, **kwargs)
        if image_data or audio_data or len(video_data) != 1:
            raise ValueError("kot patch: a request with ids carries one video and no other media")
        _, text = placeholder(self, prompt)
        return await fast_load(self, prompt=text, multimodal_tokens=multimodal_tokens,
                               video_data=video_data, input_ids=prompt, **kwargs)

    def process_and_combine_verbatim(self, base_output, mm_tokens, processor=None, **kwargs):
        ids = base_output.input_ids
        if not (isinstance(ids, list) and raw_videos(base_output.videos)):
            return original_combine(self, base_output, mm_tokens, processor=processor, **kwargs)
        start, text = placeholder(self, ids)
        if base_output.input_text != text:
            raise ValueError("kot patch: the loader passed text other than the video placeholder")
        items, expansion, ret = original_combine(
            self, base_output, mm_tokens, processor=processor, **kwargs
        )
        engine_ids = [*ids[:start], *map(int, expansion.flatten().tolist()), *ids[start + 3:]]
        for item in items:
            item.offsets = [(first + start, last + start) for first, last in item.offsets]
        covered = [at for item in items for first, last in item.offsets for at in range(first, last + 1)]
        video_at = [at for at, token in enumerate(engine_ids) if token == mm_tokens.video_token_id]
        if sorted(covered) != video_at:
            raise ValueError("kot patch: the item offsets do not cover the video tokens")
        input_ids = torch.tensor(engine_ids, dtype=expansion.dtype)
        ret["input_ids"] = input_ids.unsqueeze(0)
        try:
            with open(os.path.join(STASH_DIR, VIDEO_IDS_NOTE), "w", encoding="utf-8") as note:
                json.dump({"request_ids": len(ids), "input_ids": engine_ids}, note)
        except Exception as failure:
            _patch_log(f"note FAILED: {type(failure).__name__}: {failure}")
        return items, input_ids, ret

    process_and_combine_verbatim._kot_patched = True
    processor_class.load_mm_data = load_mm_data_verbatim
    processor_class.process_and_combine_mm_data = process_and_combine_verbatim


def _trace(message: str) -> None:
    """Forensics for the count path: /tmp, because the stash dir does not
    exist while modules load and the patch log swallows write errors."""
    try:
        with open("/tmp/kot_patch_trace.log", "a", encoding="utf-8") as trace:
            trace.write(message + "\n")
    except Exception:
        pass


_stub_sgl_kernel_finder()
if CAPTURE_MODE:
    # On demand: kill -USR1 <pid> dumps every thread's stack to stderr --
    # the hang diagnosis path where ptrace is unavailable.
    import faulthandler
    import signal

    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # Ahead of PathFinder: appended finders never run, because PathFinder
    # answers first. This finder only claims its targets and asks PathFinder
    # for their real specs, so being first is safe.
    sys.meta_path.insert(0, _PatchOnLoadFinder())

