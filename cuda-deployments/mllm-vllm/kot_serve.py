"""kot deployment contract adapter around a vLLM engine, mllm family, one process.

    kot_serve.py --port P --model /path/to/Qwen3.8-27B --device cuda

The model is Qwen/Qwen3.8-27B (the native qwen3_5 architecture). It serves
video units: the checkpoint's own instruct chat prompt rendered around the
video, decoded and frame-sampled by vLLM's own loader, the one chat calls
(torchcodec; fps 2.0), then handed to vLLM as (frames, metadata) under the
reference per-frame pixel cap. vLLM's processor expands the single video
placeholder into the per-frame timestamp structure and feeds the model.

The vLLM OpenAI app is mounted as-is for /v1/chat/completions (the speed
harness drives it; the video rides the chat surface). The contract endpoints
are added to the same app:

  GET  /kot/info   provenance: the running engine, its dtype, the vocab width
  POST /kot/score  writes log_probs.npy + input_ids.npy into out_dir; a decode
                   answers in JSON lines, one per 256-row block, so a long
                   generation streams instead of answering at the end

Numerics, the whole point of this deployment:

  * the frozen video inputs carry no ids: the deployment renders the pinned
    prompt itself (the checkpoint's chat template, enable_thinking false).
    The request's stored ids - raw template plus generated on a replay -
    are consumed verbatim: the head is asserted against the rendered raw
    form, the tail rides after the expanded video block.
  * prefill rows are the tail's rows: row t is the distribution before
    tail[t], i.e. its prediction. The engine generates one token to close
    the request and it is discarded.
  * the score path asks the engine for `prompt_logprobs` / `logprobs` at
    k = vocab_size (with --max-logprobs raised accordingly). vLLM's sampler
    computes those as log_softmax of the RAW logits, in float32, before
    temperature scaling, penalties or any logit processor. That is exactly
    the contract's pre-sampling distribution.
  * k = vocab_size makes the "top-k" the whole vocabulary, so the returned
    [rows, k+1] tensors are scattered back into token-id order and every row
    covers every id exactly once. Verified per request; a padded-vocab engine
    cannot sneak pad rows past that check.
  * decode applies the request's sampling object exactly as given
    (temperature, top_p, top_k, min_p; all four are native SamplingParams
    knobs) with the request's seed and no penalties. allow_eos gates the
    decode exactly as the contract states: false demands exactly
    max_new_tokens ids (EOS suppressed via min_tokens), true accepts what
    the sampler emitted before EOS, at most that many; the reported rows
    are the raw distributions either way.
  * prefix caching is OFF: with prompt_logprobs requested, a cache hit would
    leave the rows of the cached prefix never computed.

The one patch: vLLM pythonizes those full-vocab tensors into ~vocab_size
Logprob objects per position on the frontend. The tap below keeps the raw
tensors instead and leaves cheap placeholders in vLLM's own lists, so its
invariants hold and nothing else changes. vLLM keeps the prompt-logprob
tensors on the CPU per chunk (in_progress_prompt_logprobs_cpu), so the stash
never holds GPU memory.
"""

from __future__ import annotations

import argparse
import asyncio
import faulthandler
import json
import os
import re
import signal
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Only "1" turns off vLLM's usage report, so an ambient value must not win.
os.environ["VLLM_NO_USAGE_STATS"] = "1"
# The V2 model runner is the replay path: its tail scoring
# (prompt_logprob_token_ids + prompt_logprob_start) scores only the kept
# rows, with no top-k and no full-prompt buffer. The old pin-off reason -
# the V2 prompt_logprob path accumulating the whole prompt's top-k on the
# GPU - no longer applies: the replay never requests prompt_logprobs. The
# decode tap's per-step tensors work under both runners (verified
# byte-identical). The launcher exports 1; this setdefault covers a direct
# python launch.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "1")
# torch 2.14's triton refuses the non-constexpr globals vLLM's kernels read
# (LOG2E et al.); triton's own documented escape hatch for exactly this.
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
# FlashInfer's backend table claims sm_120 support but its JIT's own arch
# check rejects Blackwell ("requires sm75 or higher"), so the V1 sampler's
# flashinfer path dies at engine init on 5090s. Native sampling runs fine.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402
import vllm  # noqa: E402

VLLM_PIN = "0.33.0"
if not vllm.__version__.startswith(VLLM_PIN):
    sys.exit(
        f"this deployment is written against vllm {VLLM_PIN} (its logprobs tap "
        f"touches V1 internals); found {vllm.__version__}"
    )

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.entrypoints.common.cli_args import make_arg_parser  # noqa: E402
from vllm.entrypoints.launchers.launcher import serve_http  # noqa: E402
from vllm.entrypoints.launchers.api_server.app_state import init_app_state  # noqa: E402
from vllm.entrypoints.launchers.api_server.entry import build_async_engine_client  # noqa: E402
from vllm.entrypoints.launchers.app import build_app  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.multimodal.media.image import ImageMediaIO  # noqa: E402
from vllm.multimodal.media.video import VideoMediaIO  # noqa: E402
from vllm.v1.engine.logprobs import LogprobsProcessor  # noqa: E402

INFO = "/kot/info"
SCORE = "/kot/score"
INFER = "/kot/infer"
CONTRACT_VERSION = 1
SCORE_PREFIX = "kot-score-"
UTF8 = "utf-8"
SUPPORTED_TASKS = ("generate",)
BLOCK_ROWS = 256  # one streamed line per block of decode rows: bounded memory, visible progress

VIDEO = "video"
VIDEO_TAG = "<|video_pad|>"
# The pinned media pipeline, the same values the transformers reference
# serves: the prompt text, the frame sampling and the reference per-frame
# pixel cap. A candidate must report the same values or its capture
# compares nothing.
VIDEO_PROMPT = "Describe what you see in the video."
VIDEO_FPS = 2.0
CAP_PIXELS_PER_FRAME = True
# The video loader's one settings value. args.media_io_kwargs hands it to
# chat's media connector; _video_prompt builds the same VideoMediaIO from it.
# The qwen3_vl loader runs the arithmetic of the HF sampler the reference
# calls and ignores VideoMediaIO's num_frames=32. torchcodec is the reference's
# decoder, so the frame count and fps that arithmetic reads are the same too;
# vLLM's default codec is OpenCV, whose metadata and pixels are not.
MEDIA_IO_KWARGS = {
    "video": {"video_backend": "qwen3_vl", "backend": "torchcodec", "fps": VIDEO_FPS},
}

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p")

# A stall bound for the score body: past it, the faulthandler watchdog
# dumps every thread's stack to the log and exits the process. A dead
# server with a named culprit beats a hung one.
REQUEST_TIMEOUT_S = float(os.environ.get("KOT_VLLM_REQUEST_TIMEOUT", "1800"))


# A stack of every thread, on demand, without ptrace (the serving container
# denies it): `kill -USR1 <pid>` and the log answers.
faulthandler.register(signal.SIGUSR1, all_threads=True)


# request_id -> the LogprobsProcessor vLLM built for it, carrying our stashed
# raw tensors. Only score requests (SCORE_PREFIX) are registered.
SINKS: dict[str, LogprobsProcessor] = {}

# The V2 model runner (VLLM_USE_V2_MODEL_RUNNER=1, exported by the
# launcher and defaulted at the module level) scores only the tail rows
# against fixed token ids (prompt_logprob_token_ids + prompt_logprob_start):
# no top-k, no full-prompt pinned buffer, and the rows arrive on
# RequestOutput.prompt_token_id_logprobs as [rows, vocab] float32 in
# token-id order - already scattered, no permutation to undo.

# The GPU holds [rows, vocab] ids, mask and scores for the whole tail from
# the first chunk to the last (~3.08 MiB per row at this vocabulary), so a
# long tail arrives as several requests of at most this many rows. Fixed,
# not derived: segment boundaries change the last chunk's shapes, so rows
# are reproducible only at a fixed segment size. The budget: the measured
# free memory with a 64-row request in flight is 2,431 MiB, and every row
# above 64 takes another ~3.08 MiB, so free_at_N = 2431 - (N-64) x 3.08 MiB.
# The video encoder asks up to ~1.2 GiB more during the prefill (measured:
# 640 rows OOM it - a 1.19 GiB allocation fails with ~0.66 GiB left), so
# 256 rows keeps ~1.8 GiB free: the transient fits with margin. The env
# override is for experiments; captures run at the default.
REPLAY_SEGMENT_ROWS = int(os.environ.get("KOT_VLLM_REPLAY_SEGMENT_ROWS", "256"))


def _install_logprobs_tap() -> None:
    """Stash raw logprob tensors per request instead of building Logprob dicts.

    vLLM's update_from_output pythonizes [rows, vocab+1] tensors into one
    Logprob object per token per position. We keep the tensors (the actual
    numbers we need) and append empty dicts to vLLM's lists so every
    length-based consumer downstream sees consistent shapes. The values those
    placeholders would have carried are exactly the tensors we stash.
    """
    original_from_new_request = LogprobsProcessor.from_new_request.__func__

    def from_new_request(cls, tokenizer, request):
        processor = original_from_new_request(cls, tokenizer, request)
        processor.kot_sample_lists = []
        rid = getattr(request, "request_id", "?")
        if rid.startswith(SCORE_PREFIX):
            SINKS[rid] = processor
        return processor

    def update_from_output(self, output):
        # The V2 runner's tail scores ride this field; vLLM's own
        # update_from_output would move it onto the processor, and the
        # RequestOutput reads it from there. Without the passthrough the
        # field arrives as None.
        if getattr(output, "prompt_token_id_logprobs", None) is not None:
            self.prompt_token_id_logprobs = output.prompt_token_id_logprobs.numpy()
        if output.new_logprobs is not None:
            lists = output.new_logprobs
            self.kot_sample_lists.append(lists)
            if self.logprobs is not None:
                self.logprobs.extend([{} for _ in lists.logprobs])

    LogprobsProcessor.from_new_request = classmethod(from_new_request)
    LogprobsProcessor.update_from_output = update_from_output


def _scatter_full_vocab(token_ids, logprobs, vocab_size: int) -> np.ndarray:
    """[rows, k+1] top-k tensors -> [rows, vocab_size] in token-id order.

    Column 0 of each row is the prompt/sampled token's own logprob; columns
    1..k are the top-k. With k = vocab_size the top-k is the whole vocabulary
    (a padded engine emits pad logits as -inf, so pad ids sort behind every
    real id and cannot enter the top-k). The out array starts as NaN and the
    k values fill it by id: the fill is complete exactly when the ids are a
    permutation of the vocabulary, put_along_axis raises on out-of-range
    ids, and any NaN left behind says the top-k was not the full vocabulary
    - the guarantee the sort check gave, without the full-width int64 tile
    and sort (tens of GiB at this vocabulary).

    The scatter runs one 256-row block at a time, converting ids to int64
    per block: the peak holds the out array plus one block of ids (about
    0.5 GB at 4096 kept rows), never the whole kept tail in int64.
    """
    ids_all = np.asarray(token_ids)
    values_all = np.asarray(logprobs)
    rows = ids_all.shape[0]
    out = np.full((rows, vocab_size), np.nan, dtype=np.float32)
    block = 256
    for start in range(0, rows, block):
        stop = min(start + block, rows)
        ids = np.asarray(ids_all[start:stop], dtype=np.int64)[:, 1:]
        values = np.asarray(values_all[start:stop], dtype=np.float32)[:, 1:]
        np.put_along_axis(out[start:stop], ids, values, axis=1)
    if np.isnan(out).any():
        raise ValueError(
            "the engine's top-k did not cover the configured vocabulary "
            f"({vocab_size}); refusing to write misaligned log-probs"
        )
    return out


def _decode_rows(entries: list, vocab_size: int) -> np.ndarray:
    """A stretch of the stashed per-step tensors as [rows, vocab_size], in token-id order."""
    token_ids = np.concatenate(
        [np.asarray(entry.logprob_token_ids, dtype=np.int64) for entry in entries]
    )
    logprobs = np.concatenate(
        [np.asarray(entry.logprobs, dtype=np.float32) for entry in entries]
    )
    return _scatter_full_vocab(token_ids, logprobs, vocab_size)


def _drain_steps(processor, steps: int, log_probs: np.ndarray, written: int, vocab_size: int) -> int:
    """Write the first ``steps`` stashed steps into log_probs from row ``written``
    and drop them from the stash, so it holds one block, never the unit.
    Returns the rows written so far."""
    entries = processor.kot_sample_lists[:steps]
    del processor.kot_sample_lists[:steps]
    block = _decode_rows(entries, vocab_size)
    del entries
    _check_well_formed(block)
    log_probs[written:written + block.shape[0]] = block
    return written + block.shape[0]


def _compact(path: Path, rows: int) -> None:
    """Cut log_probs.npy to its first ``rows`` rows: copied block by block into
    a new file at the final shape, which then replaces the old one."""
    source = np.load(path, mmap_mode="r")
    staged = path.with_name(f"{path.name}.compact")
    try:
        target = np.lib.format.open_memmap(
            staged, mode="w+", dtype=np.float32, shape=(rows, source.shape[1])
        )
        for start in range(0, rows, BLOCK_ROWS):
            stop = min(start + BLOCK_ROWS, rows)
            target[start:stop] = source[start:stop]
        target.flush()
        del target, source
        os.replace(staged, path)
    except BaseException:
        staged.unlink(missing_ok=True)
        raise


def _refuse_occupied_out_dir(body: dict, out_dir: Path) -> None:
    """A score request must not silently replace another run's rows.

    The out_dir is client-named; two runs over the same directory would make
    the second overwrite the first without a trace (it happened once during
    the #31 gate: a reused request body pointed the candidate at the
    golden's directory). Refuse unless the request names ``overwrite``.
    """
    if (out_dir / "log_probs.npy").exists() and not body.get("overwrite"):
        raise ValueError(
            f"out_dir already holds log_probs.npy ({out_dir}); send "
            "\"overwrite\": true to replace it"
        )


def _check_well_formed(log_probs: np.ndarray) -> None:
    """log-softmax rows in float32: finite, and logsumexp 0 within float32 noise."""
    if not np.isfinite(log_probs).all():
        raise ValueError("log_probs contain NaN or -inf")
    worst = float(np.abs(np.logaddexp.reduce(log_probs, axis=1)).max())
    if worst > 1e-3:
        raise ValueError(f"rows are not normalized log-probs: max |logsumexp| = {worst}")


def _resolve_sampling(body: dict) -> dict:
    """The request's sampling object, every field applied exactly as given.

    Prefill carries the object too and samples nothing, so only the decode
    path resolves it; a field this engine cannot apply is an error naming it,
    never a silently ignored value. Missing fields fall back to their neutral
    values, so a bare request still serves.
    """
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ValueError("unsupported sampling field(s): " + ", ".join(unsupported))
    return {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", 0)),
        "min_p": float(sampling.get("min_p", 0.0)),
    }


def _sampling_params(body: dict, vocab_size: int) -> SamplingParams:
    """The decode's sampling object, applied exactly as given (all four
    fields are native SamplingParams knobs): they steer what the sampler
    draws and never touch the reported logprobs, which are the raw logits'
    log-softmax. No penalties are set, and the only logits processor that
    can appear is min_tokens (EOS suppression under allow_eos=false), which
    changes which token is sampled, never the reported raw logprobs.
    """
    sampling = _resolve_sampling(body)
    max_new_tokens = int(body["max_new_tokens"])
    allow_eos = bool(body.get("allow_eos", False))
    seed = body.get("seed")
    return SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=0 if allow_eos else max_new_tokens,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        seed=int(seed) if seed is not None else None,
        logprobs=vocab_size,
    )



def _pop_sink(request_id: str) -> LogprobsProcessor | None:
    """The processor for a score request, by prefix.

    vLLM appends its own suffix to the ids it passes down ("-xxxxxxxx"), so
    the exact id we generated never matches; the contract serves one score
    request at a time, so a prefix match is unambiguous.
    """
    processor = SINKS.pop(request_id, None)
    if processor is None:
        for key in [k for k in SINKS if k.startswith(request_id)]:
            processor = SINKS.pop(key)
            break
    return processor


def _peek_sink(request_id: str) -> LogprobsProcessor | None:
    processor = SINKS.get(request_id)
    if processor is None:
        for key in SINKS:
            if key.startswith(request_id):
                return SINKS[key]
    return processor


def _line(message: dict) -> str:
    """One streamed line: JSON and a newline, nothing else on the wire."""
    return json.dumps(message) + "\n"


def _score_transcript(body: dict) -> list[int]:
    """The replay's stored ids, validated: the raw rendered template plus the
    generated tail, verbatim.

    The stored ids are the raw rendered template plus the generated ids; the
    head is asserted against the deployment's own rendering. The engine's
    multimodal processor expands the single video placeholder into the
    per-frame timestamp structure it computes for this video. The tail's
    ids sit after the expansion and stay plain.
    """
    video_path = body["attachments"][VIDEO]
    transcript = [int(token) for token in body["input_ids"]]
    if len(transcript) < 2:
        raise ValueError(
            "a prefill needs at least 2 ids to produce a row; "
            f"got {len(transcript)}"
        )
    raw = _raw_template_ids(video_path)
    if transcript[: len(raw)] != raw:
        raise ValueError(
            "the stored ids' head is not this deployment's raw video template; "
            f"expected {len(raw)} ids, got {transcript[:8]}..."
        )
    return transcript


_PROCESSOR_CACHE: object = None


def _processor():
    """The checkpoint's HF processor: the chat template and the count guard's
    placeholder expansion - the same pieces the transformers reference runs,
    at the same pins. The engine's frames come from _video_prompt, not here."""
    global _PROCESSOR_CACHE
    if _PROCESSOR_CACHE is None:
        from transformers import AutoProcessor

        _PROCESSOR_CACHE = AutoProcessor.from_pretrained(_MODEL_DIR)
    return _PROCESSOR_CACHE


def _raw_template_ids(video_path: str) -> list[int]:
    """The pinned prompt's raw ids: the chat template rendered with ONE video
    placeholder, tokenized by the tokenizer only. This is the form a capture
    stores; the engine's expansion happens inside vLLM."""
    processor = _processor()
    text = processor.apply_chat_template(
        [{"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": VIDEO_PROMPT},
        ]}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(processor.tokenizer(text, add_special_tokens=False)["input_ids"])


_EXPANDED_CACHE: dict[str, list[int]] = {}


def _expanded_template_ids(video_path: str) -> list[int]:
    """The pinned prompt's EXPANDED ids, as the checkpoint's own HF processor
    computes them for this video: the frame sampling and the per-frame
    timestamp structure the engine expands the placeholder into, at the same
    pins. Cached per video; a CPU pass of a second or two.

    The V2 runner scores rows by absolute expanded position
    (prompt_logprob_start), so the deployment must know the expanded length
    before the request; the engine's own expansion is verified against it
    after the request.
    """
    if video_path not in _EXPANDED_CACHE:
        processor = _processor()
        messages = [{"role": "user", "content": [
            {"type": "video", "video": video_path},
            {"type": "text", "text": VIDEO_PROMPT},
        ]}]
        video = processor(
            text=[processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )],
            videos=[[video_path]],
            fps=VIDEO_FPS,
            # The same pin the engine holds: without it the count matches only
            # videos whose frames sit under the cap, and the engine's rows check
            # rejects the start index for the rest.
            cap_pixels_per_frame=CAP_PIXELS_PER_FRAME,
            return_tensors=None,
        )
        _EXPANDED_CACHE[video_path] = list(map(int, video["input_ids"][0]))
    return _EXPANDED_CACHE[video_path]


def _video_prompt(ids: list[int], video_path: str) -> TokensPrompt:
    """The engine prompt: the ids verbatim, plus the video as (frames, metadata)
    from the loader chat calls: VideoMediaIO, built as vLLM's media connector
    builds it, from the value chat's connector gets. No frame budget: the
    launcher turns chat's off, so neither route has one. The processor's
    pins are the engine's (args.mm_processor_kwargs), the ones chat runs under.
    """
    loader = VideoMediaIO(ImageMediaIO(), **MEDIA_IO_KWARGS["video"])
    frames_and_metadata = loader.load_file(Path(video_path)).media
    return TokensPrompt(prompt_token_ids=ids, multi_modal_data={"video": frames_and_metadata})


_MODEL_DIR = ""


async def _decode_stream(engine, body: dict, vocab_size: int):
    """A decode answered in JSON lines: one per block of rows as it lands, the result last.

    The engine's per-step tensors are drained while the request runs,
    scattered into token-id order and written block-wise into a memmap of the
    cap shape, and each drained block leaves the stash, so memory holds one
    block instead of the whole unit and the client sees rows for as long as
    the generation lasts. Under allow_eos the sampler may stop early; the
    file is compacted to the rows actually produced before the reply names
    it. Without allow_eos anything short of max_new_tokens is an error.
    """
    video_path = body["attachments"][VIDEO]
    max_new_tokens = int(body["max_new_tokens"])
    allow_eos = bool(body.get("allow_eos", False))
    out_dir = Path(body["out_dir"])
    _refuse_occupied_out_dir(body, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"{SCORE_PREFIX}{uuid.uuid4().hex}"
    raw_ids = _raw_template_ids(video_path)
    if not raw_ids:
        raise ValueError("empty raw template: the pinned video prompt failed to render")
    prompt = _video_prompt(raw_ids, video_path)
    params = _sampling_params(body, vocab_size)
    log_probs = np.lib.format.open_memmap(
        out_dir / "log_probs.npy", mode="w+", dtype=np.float32,
        shape=(max_new_tokens, vocab_size),
    )
    processor = None
    blocks = 0
    written = 0
    try:
        final = None
        async for output in engine.generate(prompt, params, request_id):
            final = output
            processor = processor or _peek_sink(request_id)
            while processor is not None and len(processor.kot_sample_lists) >= BLOCK_ROWS:
                written = _drain_steps(processor, BLOCK_ROWS, log_probs, written, vocab_size)
                blocks += 1
                yield _line({"block": blocks - 1, "rows": written})
        if final is None or not final.finished:
            yield _line(
                {"error": "RuntimeError",
                 "detail": "the engine ended the request without finishing it"}
            )
            return
        processor = processor or _pop_sink(request_id)
        if processor is None or not (written or processor.kot_sample_lists):
            yield _line(
                {"error": "RuntimeError",
                 "detail": "no logprob tap for the request; the V1 LogprobsProcessor patch did not run"}
            )
            return
        if processor.kot_sample_lists:
            written = _drain_steps(
                processor, len(processor.kot_sample_lists), log_probs, written, vocab_size
            )
            yield _line({"block": blocks, "rows": written})
        output_ids = [int(token) for token in final.outputs[0].token_ids]
        if written != len(output_ids):
            yield _line(
                {"error": "ValueError",
                 "detail": f"decode produced {written} rows for {len(output_ids)} sampled "
                           "tokens; row t must be the distribution token t was sampled from"}
            )
            return
        if not 0 < written <= max_new_tokens:
            yield _line(
                {"error": "ValueError",
                 "detail": f"decode produced {written} rows for a cap of {max_new_tokens}"}
            )
            return
        # A "length" finish short of max_new_tokens is the window's cap, not
        # the request's, whatever allow_eos says.
        capped = getattr(final.outputs[0], "finish_reason", None) == "length"
        if written != max_new_tokens and (capped or not allow_eos):
            yield _line(
                {"error": "ValueError",
                 "detail": f"decode produced {written} ids of the {max_new_tokens} asked "
                           f"(allow_eos {str(allow_eos).lower()}, finish_reason "
                           f"{getattr(final.outputs[0], 'finish_reason', None)}); the likely "
                           "cause is a prompt plus decode longer than the max_model_len window"}
            )
            return
        # Compact to the rows actually produced (allow_eos may stop early),
        # block by block into a new file: a whole-unit copy is what the
        # memmap exists to avoid.
        log_probs.flush()
        del log_probs
        if written < max_new_tokens:
            _compact(out_dir / "log_probs.npy", written)
        # The raw prompt the deployment rendered (the pinned chat template
        # with one video placeholder) plus the generated ids: the
        # pre-expansion sequence, so the replay of this file re-renders the
        # same expansion. The engine's final.prompt_token_ids is the EXPANDED
        # prompt and is not what a replay may send back.
        np.save(
            out_dir / "input_ids.npy",
            np.ascontiguousarray(
                np.asarray(raw_ids + output_ids, dtype=np.int32)
            ),
            allow_pickle=False,
        )
        yield _line(
            {
                "prompt_id": body["prompt_id"],
                "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
                "n_positions": written,
                "vocab_size": vocab_size,
                "output_ids": output_ids,
            }
        )
    finally:
        _pop_sink(request_id)


async def _score(engine, body: dict, vocab_size: int) -> dict:
    """A prefill answer: one forward pass, both files written, one JSON back."""
    if body["mode"] != "prefill":
        raise ValueError(f"unknown mode {body['mode']!r}; the contract names prefill and decode")
    transcript = _score_transcript(body)
    out_dir = Path(body["out_dir"])
    _refuse_occupied_out_dir(body, out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # The prompt rows cover the whole expanded prompt (the frozen template's
    # image block plus the ids the request carried); the rows the family
    # keeps are the tail: the classic n - 1, or the last n_rows when the
    # request trims (the OCR replay keeps only the rows over the
    # generated ids).
    n = len(transcript)
    keep = int(body.get("n_rows", 0)) or n - 1
    if keep >= n:
        raise ValueError(f"n_rows {keep} asks for more rows than {n} ids can predict")

    # Tail-only scoring in segments: the V2 runner scores rows [A, B)
    # against the full vocabulary in natural order, keeping [rows, vocab]
    # ids, mask and scores on the GPU (~3.1 MiB per row) from the first
    # chunk to the last. Row r predicts token r+1, so a segment sends the
    # transcript cut off right after token B: the prompt and the video are
    # re-fed in full every time, the cost is bounded by the segment, not
    # the prompt - no top-k, no full-prompt pinned buffer. Causality makes
    # the cut exact: tokens after B+1 never touch rows below B.
    video_path = body["attachments"][VIDEO]
    raw_ids = _raw_template_ids(video_path)
    expanded_template = len(_expanded_template_ids(video_path))
    tail = transcript[len(raw_ids):]
    expanded_n = expanded_template + len(tail)
    first_row = expanded_n - 1 - keep
    # Each segment's rows go into the file at the final shape as they land,
    # under a staging name: memory holds one segment, and a failed replay
    # leaves no log_probs.npy behind to block a retry.
    staged = out_dir / "log_probs.npy.partial"
    log_probs = np.lib.format.open_memmap(
        staged, mode="w+", dtype=np.float32, shape=(keep, vocab_size)
    )
    filled = 0
    started = time.monotonic()
    try:
        for a in range(first_row, first_row + keep, REPLAY_SEGMENT_ROWS):
            b = min(a + REPLAY_SEGMENT_ROWS, first_row + keep)
            cut = list(transcript[: len(raw_ids) + (b - expanded_template + 1)])
            segment_id = f"{SCORE_PREFIX}{uuid.uuid4().hex}"
            params = SamplingParams(
                max_tokens=1,
                temperature=1.0,
                prompt_logprob_token_ids=np.tile(
                    np.arange(vocab_size, dtype=np.int64), (b - a, 1)
                ),
                prompt_logprob_start=a,
            )
            try:
                final = None
                async for output in engine.generate(
                    _video_prompt(cut, video_path), params, segment_id
                ):
                    final = output
                if final is None or not final.finished:
                    raise RuntimeError("the engine ended the request without finishing it")
            finally:
                _pop_sink(segment_id)
            # The engine's own expansion must match the processor-computed one,
            # or the start index scored the wrong rows. The cut expands to
            # exactly B + 1 ids: the template plus the tail through token B.
            expanded_actual = len(final.prompt_token_ids or ())
            if expanded_actual != b + 1:
                raise RuntimeError(
                    f"the engine expanded the segment prompt to {expanded_actual} ids, "
                    f"the processor computed {b + 1}; the start index cannot be trusted"
                )
            rows = getattr(final, "prompt_token_id_logprobs", None)
            if rows is None:
                raise RuntimeError(
                    "the V2 runner returned no prompt_token_id_logprobs; the "
                    "tap passthrough did not run"
                )
            rows = np.asarray(rows, dtype=np.float32)
            if filled + rows.shape[0] > keep:
                raise RuntimeError(
                    f"the segments produced more than the requested {keep} rows"
                )
            # The checks are per row: every segment passing is the tail passing.
            _check_well_formed(rows)
            log_probs[filled:filled + rows.shape[0]] = rows
            filled += rows.shape[0]
            # The loop variable and the params hold the segment's rows and
            # its id tile too: all go, so memory holds one segment.
            del rows, final, output, params
        if filled != keep:
            raise RuntimeError(
                f"the segments produced {filled} rows for a requested {keep}; "
                "the tail is incomplete"
            )
        log_probs.flush()
        del log_probs
        os.replace(staged, out_dir / "log_probs.npy")
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    # The stored ids are the sequence the rows live in, verbatim: the OCR
    # replay's rows predict the trailing generated ids, and the eval joins
    # them to decode_ids by name.
    np.save(
        out_dir / "input_ids.npy",
        np.ascontiguousarray(np.asarray(transcript, dtype=np.int32)),
        allow_pickle=False,
    )
    print(
        f"[kot-vllm-mllm] prefill {body['prompt_id']}: ids {n}, keep {keep}, "
        f"segments and rows {time.monotonic() - started:.1f}s",
        flush=True,
    )
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
        "n_positions": keep,
        "vocab_size": vocab_size,
        "output_ids": [],
    }


def _snapshot_identity(model_dir: str) -> tuple[str, str]:
    """(repo id, revision): a hub snapshot's path names both, and its directory
    is named by its commit. Both are empty for any other directory."""
    path = Path(model_dir).resolve()
    repo = path.parent.parent.name
    if path.parent.name == "snapshots" and repo.startswith("models--") and re.fullmatch(r"[0-9a-f]{40}", path.name):
        return repo.removeprefix("models--").replace("--", "/"), path.name
    return "", ""


def _resolve_identity(model_dir: str) -> tuple[str, str]:
    """(model_id, revision) as provenance, from the weights themselves.

    A hub snapshot's path names both. Any other directory does not name its
    hub id; ``_name_or_path`` in config.json usually does. KOT_MODEL_ID /
    KOT_MODEL_REVISION override both,
    e.g. when the runner knows the id and the directory does not.
    """
    override_id = os.environ.get("KOT_MODEL_ID", "")
    override_revision = os.environ.get("KOT_MODEL_REVISION", "")
    if override_id:
        return override_id, override_revision
    model_id = Path(model_dir).name
    revision = ""
    config_path = Path(model_dir) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding=UTF8))
        name_or_path = str(config.get("_name_or_path", ""))
        if "/" in name_or_path:
            model_id = name_or_path
        revision = str(config.get("_commit_hash", "") or "")
    snapshot_id, snapshot_revision = _snapshot_identity(model_dir)
    return snapshot_id or model_id, override_revision or snapshot_revision or revision


def _openai_args(model: str, device: str, port: int, model_id: str, here: str) -> argparse.Namespace:
    """The vLLM server's own CLI namespace, pinned the way this contract needs.

    vLLM's arg parser passes ``deprecated=True`` to argparse on some versions,
    which only newer Pythons accept; the shim drops it so the parser builds
    anywhere. Argument groups share the container's add_argument, hence the
    class-level patch, undone right after the namespace is built.
    """
    container = argparse._ActionsContainer
    original_add_argument = container.add_argument

    def _shimmed_add_argument(self, *argv, **kwargs):
        kwargs.pop("deprecated", None)
        return original_add_argument(self, *argv, **kwargs)

    container.add_argument = _shimmed_add_argument
    try:
        args = make_arg_parser(argparse.ArgumentParser(prog="kot_serve")).parse_args([])
    finally:
        container.add_argument = original_add_argument
    args.model = model
    args.device = device
    args.host = "127.0.0.1"
    args.port = port
    args.served_model_name = [model_id, model]
    # One video per request.
    args.limit_mm_per_prompt = {"video": 1}
    # The video units run prompts up to ~12.4k tokens (the checkpoint's pixel
    # budget allows 12,288 video tokens, plus the template) and an 8192-token
    # decode; the window covers both with room. vLLM checks only the prompt
    # against it and caps the output at the window without an error.
    args.max_model_len = int(os.environ.get("KOT_VLLM_MAX_MODEL_LEN", "24576"))
    # The chunk size of every prefill. Under the V2 runner the score path
    # has no full-vocab gather, so the #29 budget that forced 128 is void:
    # 2048 is vLLM's API-server default for this card class and serves both
    # surfaces with one profile, so capture and speed share chunk shapes.
    # The size is still part of the numerics - it is reported in /kot/info.
    args.max_num_batched_tokens = int(os.environ.get("KOT_VLLM_MAX_BATCHED_TOKENS", "2048"))
    # The contract serves one request at a time. vLLM's API-server default
    # on a 32G card is 256, and the temporary KV vLLM allocates for
    # CUDA-graph profiling is sized by the request count, not the window:
    # one block per min(max_num_seqs, max_cudagraph_capture_size). A hybrid
    # block holds a full GatedDeltaNet state per sequence, so 256 asked for
    # 6.12 GiB beside 25.8G of weights and OOMed boot; 4 drops the profile
    # cache to ~0.1 GiB.
    args.max_num_seqs = int(os.environ.get("KOT_VLLM_MAX_NUM_SEQS", "4"))
    args.max_logprobs = 1_000_000
    args.enable_prefix_caching = False
    args.disable_log_requests = True
    # The video preprocessing pins, applied at the engine level, so both routes
    # run under them: the loader has already sampled the frames (do_sample_frames
    # false keeps the HF processor from resampling), and the cap is the reference
    # implementation's.
    args.mm_processor_kwargs = {
        "do_sample_frames": False,
        "cap_pixels_per_frame": CAP_PIXELS_PER_FRAME,
    }
    # Chat's prompt, the one _raw_template_ids renders for the score path: the
    # template's default adds a system message and an open think block.
    args.default_chat_template_kwargs = {"enable_thinking": False}
    args.media_io_kwargs = MEDIA_IO_KWARGS
    # file:// URLs from anywhere, since a --no-container run has no /inputs
    # mount. "/" is safe here because the server binds only to 127.0.0.1.
    args.allowed_local_media_path = "/"
    # "auto" picks flashinfer's cutlass MoE on Blackwell, whose JIT needs
    # nvcc - absent on serving hosts - and dies at engine init. The triton
    # MoE backend runs sm_120 fine. KOT_VLLM_MOE_BACKEND overrides for
    # hosts that do carry the toolkit.
    args.moe_backend = os.environ.get("KOT_VLLM_MOE_BACKEND", "triton")
    # The 52G of weights over two cards leave ~5G per card for the KV cache
    # and the scoring transient. The V2 runner's own accounting is tighter
    # than V1's: at 0.89 the available KV falls 20 MiB under even a 20,480-token
    # window (0.70 GiB available vs 0.72 needed) and the engine refuses to
    # boot. 0.91 sizes the pool at 38,229 KV tokens and still leaves ~2.4 GiB
    # free beside the pool - the V2 tail scoring needs at most ~3.1 MiB per
    # row, so the 64-row replay costs ~0.2 GiB and even a 512-row segment
    # fits.
    args.gpu_memory_utilization = float(os.environ.get("KOT_VLLM_GPU_MEM", "0.91"))
    # KOT_VLLM_DTYPE exists for cross-checking numerics; a capture runs the
    # engine's own default ("auto": the checkpoint's dtype).
    dtype_override = os.environ.get("KOT_VLLM_DTYPE", "")
    if dtype_override:
        args.dtype = dtype_override
    if os.environ.get("KOT_VLLM_COMPILATION", "").lower() == "eager":
        args.enforce_eager = True
    if device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            args.tensor_parallel_size = len([item for item in visible.split(",") if item.strip()])
        else:
            args.tensor_parallel_size = max(torch.cuda.device_count(), 1)
    return args


def _register_kot_routes(app, engine, vocab_size: int, info: dict) -> None:
    @app.get(INFO)
    async def kot_info() -> JSONResponse:
        return JSONResponse(info)

    @app.post(INFER)
    async def kot_infer(request: Request):
        """The speed surface: one video decode, the description written into
        out_dir, the file named back. Only the clock reads what lands here."""
        try:
            body = await request.json()
            video_path = (body.get("attachments") or {}).get(VIDEO)
            if not video_path:
                raise ValueError("the speed request carries no video attachment")
            raw_ids = _raw_template_ids(video_path)
            prompt = _video_prompt(raw_ids, video_path)
            params = SamplingParams(
                max_tokens=512, temperature=1.0, top_p=0.95, top_k=20, seed=42
            )
            final = None
            async for output in engine.generate(prompt, params, f"kot-infer-{uuid.uuid4().hex}"):
                final = output
            if final is None or not final.finished:
                raise RuntimeError("the engine ended the request without finishing it")
            out_dir = Path(body["out_dir"])
            out_dir.mkdir(parents=True, exist_ok=True)
            (out_dir / "text.txt").write_text(final.outputs[0].text, encoding=UTF8)
            return JSONResponse({"prompt_id": body["prompt_id"], "files": ["text.txt"]})
        except Exception as failure:  # any failure is the deployment's answer
            return JSONResponse(
                {"error": type(failure).__name__, "detail": str(failure)},
                status_code=500,
            )

    @app.post(SCORE)
    async def kot_score(request: Request):
        try:
            body = await request.json()
            if body.get("mode") == "decode":
                return StreamingResponse(
                    _decode_stream(engine, body, vocab_size),
                    media_type="application/x-ndjson",
                )
            # A hard bound on the prefill body: past it, every thread's
            # stack is dumped to the log and the process exits. The loop
            # bound in _score answers 500 only while the event loop lives;
            # this one also covers a blocked loop and a wedged host. A dead
            # server with a named culprit beats a hung one.
            faulthandler.dump_traceback_later(REQUEST_TIMEOUT_S, exit=True)
            try:
                return JSONResponse(await _score(engine, body, vocab_size))
            finally:
                faulthandler.cancel_dump_traceback_later()
        except Exception as failure:  # any failure is the deployment's answer
            import traceback

            try:
                raw = (await request.body())[:200].decode(UTF8, errors="replace")
            except Exception:
                raw = "<body unreadable>"
            tail = "".join(
                traceback.format_exception(failure)[-8:]
            ).strip()
            return JSONResponse(
                {"error": type(failure).__name__, "detail": tail, "raw_body": raw},
                status_code=500,
            )

    # One deployment, one engine: the harness names the model as
    # /kot/info reported it (both spellings are in served_model_name), and
    # kot's probe sends no model field at all, which vLLM answers with the
    # sole served model. No model-name rewriting is needed.


def _nvidia_driver() -> str:
    """The kernel driver's version, as the driver itself reports it; empty when unreadable."""
    try:
        first_line = Path("/proc/driver/nvidia/version").read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return ""
    match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", first_line)
    return match.group(1) if match else ""


def _placed_devices(device: str, parallel_config) -> str:
    """Where the weights are, ordinals included, one per engine worker:
    cuda:0, or cuda:0+cuda:1 over two cards.

    The count is the engine's own config. A torch.cuda call could initialize
    CUDA in this API-server process, and that flips vLLM's fork to spawn.
    """
    if device != "cuda":
        return device
    return "+".join(f"cuda:{rank}" for rank in range(parallel_config.world_size))


def main() -> int:
    global _MODEL_DIR
    parser = argparse.ArgumentParser(prog="kot_serve.py", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="the weights directory, read-only")
    parser.add_argument("--device", default="cuda", help="torch device string from the platform row")
    cli = parser.parse_args()
    _MODEL_DIR = cli.model

    _install_logprobs_tap()
    model_id, model_revision = _resolve_identity(cli.model)
    here = str(Path(__file__).resolve().parent)
    args = _openai_args(cli.model, cli.device, cli.port, model_id, here)
    app = build_app(args, supported_tasks=SUPPORTED_TASKS)

    async def run() -> None:
        async with build_async_engine_client(args) as engine:
            model_config = engine.vllm_config.model_config
            # hf_config here is the multimodal wrapper: Qwen3_5Config nests
            # the vocab size in its text config, so the flat read crashed
            # the boot. get_vocab_size() is the engine's own resolver - the
            # same source its logits width uses - and answers 0 when the
            # field is missing; a 0 would fail much later, in confusing
            # ways, so it fails here instead.
            vocab_size = model_config.get_vocab_size()
            if vocab_size <= 0:
                raise RuntimeError("the engine resolved a vocab size of 0")
            info = {
                "contract_version": CONTRACT_VERSION,
                "model_id": model_id,
                "model_revision": model_revision,
                "tokenizer_id": model_id,
                "engine": f"vllm {vllm.__version__}",
                "dtype": str(model_config.dtype).removeprefix("torch."),
                "device": _placed_devices(cli.device, engine.vllm_config.parallel_config),
                "vocab_size": vocab_size,
                "torch_version": torch.__version__,
                "plugin_version": "",
                "driver_version": _nvidia_driver(),
                "toolkit_version": torch.version.cuda or "",
                # The chunk size is part of the numerics: a chunked prefill
                # splits the attention and state passes at chunk boundaries,
                # so rows produced at one size need not be bit-identical to
                # rows produced at another. Every capture carries its own.
                "max_num_batched_tokens": args.max_num_batched_tokens,
                # The segment size is numerics too: segment boundaries
                # change the last chunk's shapes, so replay rows are
                # reproducible only at a fixed segment size.
                "replay_segment_rows": REPLAY_SEGMENT_ROWS,
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            # Routes and middleware go on before init_app_state: it starts the
            # app, and Starlette refuses middleware on a started application.
            _register_kot_routes(app, engine, vocab_size, info)
            await init_app_state(engine, app.state, args, supported_tasks=SUPPORTED_TASKS)
            print(f"[kot-vllm-mllm] serving on 127.0.0.1:{cli.port}: {json.dumps(info)}", flush=True)
            await serve_http(
                app, sock=None, host=args.host, port=args.port, log_level="warning",
            )

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
