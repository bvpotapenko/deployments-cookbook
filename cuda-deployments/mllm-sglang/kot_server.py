"""The kot deployment contract for the mllm family, over one embedded SGLang engine.

    kot_server.py --port P --model /path/to/Qwen3.8-27B --device cpu|cuda

Started by ``kot_serve.sh``. One asyncio process: the adapter (this file)
embeds ``sglang.Engine`` inside its own event loop, so /kot/score and the
speed harness drive the same resident engine, as the contract requires. The
architecture is the ocr-sglang deployment's; what changes is the family: the
model is Qwen/Qwen3.8-27B (the native qwen3_5), a video-language checkpoint
whose units are short clips.

Endpoints:

* ``GET /kot/info`` -- provenance from the running process.
* ``POST /kot/score`` -- writes ``log_probs.npy`` and ``input_ids.npy`` into
  ``out_dir`` and names them back. The frozen video inputs carry no ids: the
  deployment renders the pinned instruct chat prompt itself (the
  checkpoint's chat template, ``enable_thinking`` false, one video
  placeholder) and consumes ids verbatim from there - the raw form on a
  decode, the raw form plus the generated tail on a replay, the head
  asserted. The video rides to the engine as an attachment path; the patch
  layer samples the frames as the reference does (fps 2.0, pinned) and the
  checkpoint's HF processor, inside the engine, resizes them under the
  per-frame cap and expands the placeholder into the per-frame timestamp
  structure. The patch layer puts that expansion in place of the
  placeholder; every other id reaches the model as sent.
* ``POST /kot/infer`` -- the speed surface: one video decode, the
  description written as ``text.txt`` into ``out_dir``, the file named back.
* ``POST /v1/chat/completions`` -- the OpenAI-shaped surface the speed harness
  drives: one user message, a text part and an optional video part, rendered
  and pinned as /kot/score renders and pins them; streams on ``stream: true``.

Numerics of the score path, in one place:

* prefill: the stored ids + the video go in as one prefill-only request
  (``max_new_tokens=0``) with ``return_logprob`` and ``logprob_start_len=0``.
  The patch stashes the full-vocabulary log-softmax of every position of the
  EXPANDED prompt, recomputed in float32; the contract's rows are the last
  ``n_rows`` of them minus the final sampling row: row t predicts the tail's
  token t + 1. ``input_ids.npy`` stores the request's ids, so a replay sends
  the same ids back.
* decode: video-driven. The prompt is the rendered raw template alone; the
  request's sampling object is applied exactly as given (temperature, top_p,
  top_k, min_p; all four are native engine knobs), the request's seed, every
  penalty at its neutral value. ``allow_eos`` gates the stop exactly as the
  contract states. Rows are stashed at ``ModelRunner.sample``'s entry -- the
  raw pre-sampling distributions, one per emitted token, before the EOS
  suppression, any penalty or temperature touches the logits.
  ``input_ids.npy`` stores the raw template plus the generated ids, so a
  replay prefill teacher-forces over exactly what the request carried: the
  pre-expansion sequence, with its single video placeholder.

Rows land on disk through ``open_memmap`` at the final shape, filled one
stash archive at a time: the full-vocabulary log-softmax of an 8,192-token
decode at this vocabulary (248,320) is 7.6 GiB, and no stage of this file
holds one. Each archive's marker states its row count, so the counts are
checked before any row is read; this file holds one archive (one decode
row, or one logprob chunk of a prefill) and deletes it once copied.

Concurrency: one event loop; the Engine is created inside it. /kot/score is
serialised by a lock -- the runner sends one at a time, and
one-request-at-a-time is what makes the stash collection sound (see
sitecustomize.py). /v1/chat/completions takes no lock: each connection awaits
the engine on its own, video decoding and the HF pass run off the event loop
(the patch layer, ``mm_processor_worker_num``), and the scheduler batches what
arrives. Chat asks for no logprobs, so alone it writes no stash. A score that
shares a batch with chat traffic stashes their rows too, with no request id, so
the row counts (``prompt_tokens`` on a prefill, the generated ids on a decode)
fail that score loudly; the chat is untouched. The runner must not overlap the
two.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import time
import traceback
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 1
UTF8 = "utf-8"
STASH_WAIT_S = 60.0

VIDEO = "video"
# The pinned media pipeline, the same values the transformers reference
# serves (mllm-transformers/serve.py). The processor's own frame sampling
# defaults to fps 2.0, its per-frame cap to off; both pinned here, for this
# file's own processor call and for the engine (mm_process_config), so a
# candidate must report the same values.
VIDEO_PROMPT = "Describe what you see in the video."
VIDEO_FPS = 2.0
VIDEO_DO_SAMPLE_FRAMES = True
CAP_PIXELS_PER_FRAME = True


class ScoreFailure(RuntimeError):
    """A score request the engine could not serve; a 500 with its words."""


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _clean_stash() -> set[str]:
    """Empty the stash dir, return what is there now: the baseline for one request."""
    stash = os.environ["KOT_SGLANG_STASH_DIR"]
    for name in os.listdir(stash):
        path = Path(stash) / name
        try:
            path.unlink()
        except OSError:
            pass
    return set()


def _stash_order(name: str) -> int:
    """The write counter of one stash archive: the last _-separated group.

    Files are named ``<kind>_<pid>_<counter>.npz``; the pid makes collisions
    impossible under tensor parallelism, the counter preserves write order.
    """
    stem = name.split(".")[0]
    return int(stem.split("_")[-1])


async def _stash_archives(kind: str) -> list[tuple[Path, int]]:
    """Every complete stash archive of one kind, in write order, with the row
    count its marker states. No rows are read here."""
    stash = os.environ["KOT_SGLANG_STASH_DIR"]
    deadline = time.monotonic() + STASH_WAIT_S
    while True:
        blocks: list[tuple[int, Path]] = []
        for name in sorted(os.listdir(stash)):
            if not name.startswith(f"{kind}_") or not name.endswith(".npz"):
                continue
            path = Path(stash) / name
            if Path(f"{path}.done.json").is_file():
                blocks.append((_stash_order(name), path))
        if blocks or time.monotonic() > deadline:
            break
        await asyncio.sleep(0.05)
    if not blocks:
        raise ScoreFailure(f"the engine wrote no {kind} log-prob rows into {stash}")
    blocks.sort(key=lambda block: block[0])
    archives = []
    for _, path in blocks:
        marker = json.loads(Path(f"{path}.done.json").read_text(encoding=UTF8))
        if "rows" not in marker:
            raise ScoreFailure(
                f"the stash marker of {path.name} states no row count: the patch layer "
                "is older than kot_server.py; start the server through kot_serve.sh"
            )
        archives.append((path, int(marker["rows"])))
    return archives


def _drop_archive(path: Path) -> None:
    for name in (path, Path(f"{path}.done.json")):
        try:
            name.unlink()
        except OSError:
            pass


def _stream_rows(
    engine, out_dir: Path, archives: list[tuple[Path, int]], skip: int, keep: int, finite: bool
) -> str:
    """Rows skip..skip+keep-1 of the archives, in write order, into
    out_dir/log_probs.npy through an open_memmap at the final shape. One
    archive is in memory at a time, and each is deleted once passed; those
    before the kept rows are deleted unread."""
    path = out_dir / "log_probs.npy"
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(keep, engine.vocab_size)
    )
    first = 0  # the archive's first row in the request's order
    for archive, count in archives:
        low, high = max(first, skip), min(first + count, skip + keep)
        if low < high:
            with np.load(archive) as stash:
                block = engine.slice_vocab(np.ascontiguousarray(stash["rows"], dtype=np.float32))
            if block.shape[0] != count:
                raise ScoreFailure(
                    f"the stash archive {archive.name} holds {block.shape[0]} rows, "
                    f"its marker states {count}"
                )
            block = block[low - first:high - first]
            if finite and not np.isfinite(block).all():
                raise ScoreFailure("prefill log_probs contain NaN or -inf")
            array[low - skip:high - skip] = block
            del block
        _drop_archive(archive)
        first += count
    array.flush()
    del array
    return path.name


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p")
# The neutral knobs a prefill passes to the engine: it samples nothing, so
# the sampling object the request carries is ignored there by contract.
NEUTRAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}


def _resolve_sampling(body: dict) -> dict:
    """The request's sampling object in engine terms, every field applied
    exactly as given. Missing fields fall back to their neutral values."""
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ScoreFailure("unsupported sampling field(s): " + ", ".join(unsupported))
    return {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", -1)) or -1,  # the contract's 0 is "off": sglang's -1
        "min_p": float(sampling.get("min_p", 0.0)),
    }


def _sampling_params(max_new_tokens: int, seed: int, sampling: dict,
                      allow_eos: bool) -> dict:
    """The request's sampling object with its seed and every penalty neutral.

    ``allow_eos`` shapes the stop exactly as the contract states: false
    demands exactly ``max_new_tokens`` ids (EOS suppressed through
    min_new_tokens and ignore_eos), true lets the sampler stop at EOS.
    Either way the stashed rows are the raw pre-sampling distributions.
    """
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "min_p": sampling["min_p"],
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "min_new_tokens": 0 if allow_eos else max_new_tokens,
        "ignore_eos": not allow_eos,
        "sampling_seed": seed,
        # The scheduler fills this dict with "__req__"; the anti-repetition
        # rule reads the request's sequence through it.
        "custom_params": {},
    }


def _write_array(out_dir: Path, name: str, array: np.ndarray) -> str:
    path = out_dir / name
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    return path.name


def _consumed_guard(done: dict, ids: list[int], engine, raw: list[int], video_path: str) -> None:
    """The engine must consume the request ids verbatim, plus the video block.

    The expected sequence is known before the request: the request ids with
    the raw template replaced by its expansion. That expansion comes from this
    deployment's own processor rendering of the same video at the reference's
    pins - the HF expansion, the same code the reference runs. ``prompt_tokens``
    must count that sequence, and the engine ids the patch layer records
    (``last_video_ids.json`` in the stash dir) must equal it position by
    position. A missing note means the engine took its decode-and-retokenize
    path; a mismatch means it changed an id or sampled different frames. The
    self-agreement would read either only as numbers.
    """
    meta = done.get("meta_info", {}) if isinstance(done, dict) else {}
    prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
    if not prompt_tokens:
        return
    expanded = _expanded_template_ids(engine, video_path)
    expected = len(ids) + len(expanded) - len(raw)
    if prompt_tokens != expected:
        raise ScoreFailure(
            f"the engine consumed {prompt_tokens} tokens for {len(ids)} ids; the raw "
            f"template's {len(raw)} expand to {len(expanded)}, so {expected} were "
            "expected: the request did not stay verbatim"
        )
    note = Path(os.environ["KOT_SGLANG_STASH_DIR"]) / "last_video_ids.json"
    try:
        consumed = json.loads(note.read_text(encoding=UTF8))["input_ids"]
    except (OSError, ValueError, KeyError):
        raise ScoreFailure(
            "the patch layer recorded no engine ids for this request: its ids patch "
            "did not run, and the engine took its decode-and-retokenize path"
        ) from None
    reference = [*expanded, *ids[len(raw):]]
    if consumed != reference:
        at = next(
            (at for at, pair in enumerate(zip(consumed, reference)) if pair[0] != pair[1]),
            min(len(consumed), len(reference)),
        )
        raise ScoreFailure(
            f"the engine ids ({len(consumed)}) differ from the reference expansion plus "
            f"the tail ({len(reference)}) first at position {at}: engine "
            f"{consumed[at:at + 4]}, reference {reference[at:at + 4]}"
        )


_EXPANDED_CACHE: dict = {}


def _expanded_template_ids(engine, video_path: str) -> list[int]:
    """The pinned prompt's EXPANDED ids, as the checkpoint's own HF processor
    computes them for this video: the frame sampling and the per-frame
    timestamp structure the reference runs, at the same pins. Cached per
    video; a CPU pass of a second or two."""
    if video_path not in _EXPANDED_CACHE:
        video = engine.processor(
            text=[_render_prompt(engine, VIDEO_PROMPT, video_path)],
            videos=[[video_path]],
            fps=VIDEO_FPS,
            do_sample_frames=VIDEO_DO_SAMPLE_FRAMES,
            cap_pixels_per_frame=CAP_PIXELS_PER_FRAME,
            return_tensors=None,
        )
        _EXPANDED_CACHE[video_path] = list(map(int, video["input_ids"][0]))
    return _EXPANDED_CACHE[video_path]


def _render_prompt(engine, text: str, video: str | None) -> str:
    """One user turn through the checkpoint's chat template, ``enable_thinking``
    false: the video placeholder first when there is a video, then the text.
    The template reads only that a video is there, never its value."""
    content = [{"type": "text", "text": text}]
    if video is not None:
        content.insert(0, {"type": "video", "video": video})
    return engine.processor.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def _prompt_ids(engine, text: str, video: str | None) -> list[int]:
    """A rendered turn's raw ids, tokenized by the tokenizer only: ONE video
    placeholder. The engine's expansion happens inside sglang."""
    rendered = _render_prompt(engine, text, video)
    return list(engine.processor.tokenizer(rendered, add_special_tokens=False)["input_ids"])


def _raw_template_ids(engine, video_path: str) -> list[int]:
    """The pinned prompt's raw ids: the form a capture stores."""
    return _prompt_ids(engine, VIDEO_PROMPT, video_path)


async def _score_prefill(engine, body: dict) -> dict:
    """Teacher forced over the ids the request carried, verbatim.

    The stored ids are the raw rendered template plus the generated ids; the
    head is asserted against the deployment's own rendering, then the ids
    ride verbatim. The engine expands the single video placeholder into the
    per-frame timestamp structure it computes for this video, so the stashed
    matrix has one row per position of the EXPANDED prompt; the expanded row
    count also proves the expansion happened: with none it would equal the
    raw id count. The rows the family keeps are the last n_rows: the
    replay's tail.
    """
    video_entry = body["attachments"][VIDEO]
    transcript = [int(token) for token in body["input_ids"]]
    if len(transcript) < 2:
        raise ScoreFailure(
            f"a prefill needs at least 2 ids to produce a row; got {len(transcript)}"
        )
    n = len(transcript)
    keep = int(body.get("n_rows", 0)) or n - 1
    if keep >= n:
        raise ScoreFailure(f"n_rows {keep} asks for more rows than {n} ids can predict")
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(body.get("seed", 42))
    video_path = _attachment_path(video_entry)
    raw = _raw_template_ids(engine, video_path)
    if transcript[: len(raw)] != raw:
        raise ScoreFailure(
            "the stored ids' head is not this deployment's raw video template: "
            f"expected {len(raw)} ids, got {transcript[:8]}..."
        )
    ids = list(transcript)

    _clean_stash()
    done = await engine.async_generate(
        input_ids=ids,
        video_data=video_entry,
        sampling_params=_sampling_params(0, seed, NEUTRAL_SAMPLING, True),
        return_logprob=True,
        logprob_start_len=0,
    )
    _consumed_guard(done, ids, engine, raw, video_path)
    archives = await _stash_archives("prefill")
    rows = sum(count for _, count in archives)
    # Alone, a prefill stashes one row per position of the expanded prompt, and
    # prompt_tokens counts those positions. A request sharing the forward adds
    # rows with no id, which would move the tail below in the correct shape.
    prompt_tokens = int(done.get("meta_info", {}).get("prompt_tokens", 0) or 0)
    if rows != prompt_tokens:
        raise ScoreFailure(
            f"{rows} stashed prefill rows for this request's {prompt_tokens} prompt "
            "tokens: a mixed batch (another request shared the forward) or a lost stash"
        )
    if rows <= len(ids):
        raise ScoreFailure(
            f"the engine stashed {rows} prefill rows for {len(ids)} raw ids: the "
            "image placeholder was not expanded into an image block"
        )
    # Drop the final (sampling) row -- it predicts a token that is not there --
    # then keep the tail's rows: the keep rows before it.
    if rows - 1 < keep:
        raise ScoreFailure(
            f"{rows} expanded rows cannot yield {keep} requested rows"
        )
    log_probs_name = _stream_rows(engine, out_dir, archives, rows - 1 - keep, keep, finite=True)
    _write_array(out_dir, "input_ids.npy", np.asarray(transcript, dtype=np.int32))
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": log_probs_name, "input_ids": "input_ids.npy"},
        "n_positions": int(keep),
        "vocab_size": engine.vocab_size,
        "output_ids": [],
    }



def _attachment_path(entry) -> str:
    """The video attachment as the engine takes it: a plain filesystem path.

    The runner sends the attachment as a dict ({"video": "..."} or the bare
    string); both reduce to the path string.
    """
    if isinstance(entry, dict):
        path = entry.get("video") or entry.get("path")
        if not path:
            raise ScoreFailure("a video attachment dict carries no path")
        return str(path)
    return str(entry)


async def _score_decode(engine, body: dict) -> dict:
    """Generate from the rendered raw template ids; one row per generated token."""
    video_entry = body["attachments"][VIDEO]
    video_path = _attachment_path(video_entry)
    max_new_tokens = int(body["max_new_tokens"])
    allow_eos = bool(body.get("allow_eos", False))
    seed = int(body.get("seed", 42))
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_ids = _raw_template_ids(engine, video_path)
    if not raw_ids:
        raise ScoreFailure("empty raw template: the pinned video prompt failed to render")
    ids = list(raw_ids)

    _clean_stash()
    done = await engine.async_generate(
        input_ids=ids,
        video_data=video_entry,
        sampling_params=_sampling_params(
            max_new_tokens, seed, _resolve_sampling(body), allow_eos
        ),
        return_logprob=True,
        logprob_start_len=-1,
    )
    _consumed_guard(done, ids, engine, raw_ids, video_path)
    output_ids = [int(token) for token in done["output_ids"]]
    # The decode's own rows: the probe below would pollute the stash
    # with prefill rows, so collect before probing.
    archives = await _stash_archives("decode")
    rows = sum(count for _, count in archives)
    # One sample call per generated id, with the overlap scheduler off. Any
    # other count is a request sharing the batch, and its rows carry no id.
    if rows != len(output_ids):
        raise ScoreFailure(
            f"{rows} stashed rows for {len(output_ids)} generated ids: a mixed batch "
            "(another request shared a forward) or a lost stash"
        )
    if not 0 < len(output_ids) <= max_new_tokens:
        raise ScoreFailure(
            f"decode produced {len(output_ids)} ids for a cap of {max_new_tokens}"
        )
    # The scheduler cuts max_new_tokens, and min_new_tokens with it, to what
    # the window and the KV pool leave after the prompt, without a message.
    # A "length" finish short of max_new_tokens is that cut, whatever
    # allow_eos says.
    finish = done.get("meta_info", {}).get("finish_reason") or {}
    capped = isinstance(finish, dict) and finish.get("type") == "length"
    if len(output_ids) != max_new_tokens and (capped or not allow_eos):
        prompt_tokens = int(done.get("meta_info", {}).get("prompt_tokens", 0) or 0)
        raise ScoreFailure(
            f"decode produced {len(output_ids)} ids of the {max_new_tokens} asked "
            f"(allow_eos {str(allow_eos).lower()}, finish {finish}). The likely cause: "
            f"the {prompt_tokens}-token expanded "
            f"prompt plus {max_new_tokens} does not fit the engine's context_length "
            "window, and the scheduler cut max_new_tokens to fit"
        )

    # The raw prompt the deployment rendered (the pinned chat template with
    # one video placeholder) plus the generated ids: the pre-expansion
    # sequence, so the replay of this file re-renders the same expansion.
    _stream_rows(engine, out_dir, archives, 0, rows, finite=False)
    _write_array(
        out_dir, "input_ids.npy",
        np.asarray([*raw_ids, *output_ids], dtype=np.int32),
    )
    if output_ids and engine.tokenizer is not None:
        text = engine.tokenizer.decode(output_ids, skip_special_tokens=True)
        (out_dir / "text.txt").write_text(text, encoding=UTF8)
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
        "n_positions": int(len(output_ids)),
        "vocab_size": engine.vocab_size,
        "output_ids": output_ids,
    }


async def score(engine, body: dict, lock: asyncio.Lock) -> dict:
    """POST /kot/score: write the two arrays into out_dir, name them back."""
    if body["mode"] == "prefill":
        async with lock:
            return await _score_prefill(engine, body)
    if body["mode"] == "decode":
        async with lock:
            return await _score_decode(engine, body)
    raise ScoreFailure(
        f"unknown mode {body['mode']!r}; the contract serves prefill and decode"
    )


async def kot_infer(engine, body: dict, lock: asyncio.Lock) -> dict:
    """POST /kot/infer: the speed surface. One video decode, the description
    written into out_dir as text.txt, the file named back. Only the clock
    reads what lands here; the request runs under the same lock and the same
    engine the score path uses."""
    video_entry = (body.get("attachments") or {}).get(VIDEO)
    if not video_entry:
        raise ScoreFailure("the speed request carries no video attachment")
    video_path = _attachment_path(video_entry)
    async with lock:
        raw_ids = _raw_template_ids(engine, video_path)
        _clean_stash()
        done = await engine.async_generate(
            input_ids=raw_ids,
            video_data=video_entry,
            sampling_params={
                "max_new_tokens": 512,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 20,
                "sampling_seed": int(body.get("seed", 42)),
            },
        )
    output_ids = [int(token) for token in done["output_ids"]]
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    text = engine.tokenizer.decode(output_ids, skip_special_tokens=True)
    (out_dir / "text.txt").write_text(text, encoding=UTF8)
    return {"prompt_id": body["prompt_id"], "files": ["text.txt"]}


def _video_source(url: str) -> str:
    """A video_url part's url in the form the engine's loader takes.

    The loader reads ``file://`` and decodes ``data:`` itself, in its IO
    workers, so neither needs a temp file. A plain path travels as ``file://``:
    the loader tests a plain path by parsing it as a URL, which would cut it at
    a ``#`` or a ``?``.
    """
    if url.startswith(("file://", "data:")):
        return url
    if url.startswith("/"):
        return "file://" + urllib.parse.quote(url)
    raise ScoreFailure(
        f"a video_url must be file://, an absolute path or a data URI; got {url[:64]!r}"
    )


def _chat_parts(messages: list[dict]) -> tuple[str, str | None]:
    """The user turn as (text, video source or None). The score path's prompt
    shape decides the order, so the parts' order here does not. Any other part
    or role is ignored."""
    text_parts: list[str] = []
    video = None
    for message in messages:
        if message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            text_parts.append(content)
            continue
        for part in content or []:
            kind = part.get("type")
            if kind == "text":
                text_parts.append(str(part.get("text", "")))
            elif kind == "video_url":
                if video is not None:
                    raise ScoreFailure("one video per request: the harness sends one clip")
                video = _video_source(str((part.get("video_url") or {}).get("url", "")))
    return "".join(text_parts), video


def _chat_sampling(body: dict) -> dict:
    """The knobs a chat request may carry, in engine terms. A field left out or
    null takes the checkpoint's own generation_config value, as /kot/infer does."""

    def field(name: str, default):
        value = body.get(name)
        return default if value is None else value

    return {
        "max_new_tokens": int(field("max_completion_tokens", field("max_tokens", 16))),
        "temperature": float(field("temperature", 1.0)),
        "top_p": float(field("top_p", 0.95)),
        "top_k": int(field("top_k", 20)) or -1,  # the contract's 0 is "off": sglang's -1
        "sampling_seed": int(field("seed", 42)),
    }


def _finish_reason(done: dict) -> str:
    """The engine's finish_reason is a dict; OpenAI's is its type."""
    finish = done.get("meta_info", {}).get("finish_reason")
    kind = finish.get("type") if isinstance(finish, dict) else finish
    return "length" if kind == "length" else "stop"


async def _chat_completions(engine, body: dict, _lock: asyncio.Lock) -> dict:
    """POST /v1/chat/completions: same prompt and frame pins as /kot/score, no
    lock, no logprobs. ``model`` in the reply is what /kot/info reports."""
    text, video = _chat_parts(list(body["messages"]))
    ids = _prompt_ids(engine, text, video)
    params = _chat_sampling(body)
    if body.get("stream"):
        return SseResponse(engine, ids, video, params)
    done = await engine.async_generate(input_ids=ids, video_data=video, sampling_params=params)
    meta = done.get("meta_info", {})
    prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
    completion_tokens = int(meta.get("completion_tokens", 0) or 0)
    return {
        "id": f"chatcmpl-{os.urandom(12).hex()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(done.get("text", ""))},
                "finish_reason": _finish_reason(done),
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


def _sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode(UTF8)


class SseResponse:
    """One generation rendered as a stream of OpenAI-shaped SSE chunks."""

    def __init__(self, engine, input_ids: list[int], video: str | None, params: dict):
        self.engine = engine
        self.input_ids = input_ids
        self.video = video
        self.params = params
        self.id = f"chatcmpl-{os.urandom(12).hex()}"
        self.created = int(time.time())

    def _chunk(self, delta: dict, finish_reason: str | None) -> bytes:
        return _sse_chunk({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.engine.model_id,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        })

    async def run(self, write) -> None:
        generator = await self.engine.async_generate(
            input_ids=self.input_ids,
            video_data=self.video,
            sampling_params=self.params,
            stream=True,
        )
        await write(self._chunk({"role": "assistant", "content": ""}, None))
        previous = 0
        done: dict = {}
        async for update in generator:
            done = update
            text = str(update.get("text", ""))
            if len(text) > previous:
                await write(self._chunk({"content": text[previous:]}, None))
                previous = len(text)
        await write(self._chunk({}, _finish_reason(done)))
        await write(b"data: [DONE]\n\n")


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
    KOT_MODEL_REVISION override both.
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


class KotEngine:
    """What the endpoints need from the engine, named once."""

    def __init__(self, model_dir: str, device: str):
        import torch
        import sglang
        from sglang.srt.entrypoints.engine import Engine

        is_cuda = device.startswith("cuda")
        self.model_dir = model_dir

        kwargs = {
            "model_path": model_dir,
            "device": "cuda" if is_cuda else "cpu",
            "dtype": "auto" if is_cuda else "float32",
            "random_seed": 42,
            # One extend per request, so a decode's stashed rows stay aligned:
            # a chunked extend samples a token per chunk and all but the last
            # are discarded, which the score path cannot tell from real steps.
            "chunked_prefill_size": -1,
            # The overlap scheduler launches one more decode forward for a
            # request that finished in the previous iteration; single-request
            # scoring gets nothing from overlap, and the phantom row would
            # fail the decode's row count.
            "disable_overlap_schedule": True,
            # The video units run at most ~12.4k-token prompts (the pixel
            # budget allows 12,288 video tokens, plus the template) plus an
            # 8,192-token decode. A shorter window makes the scheduler cut
            # the decode without a message. 24,576 fits in the 25,626-token
            # pool that KOT_SGLANG_MEM_FRACTION 0.89 sizes on 2x RTX 5090.
            "max_prefill_tokens": 16384,
            "context_length": 24576,
            # Explicit, because sglang's heuristic reserves
            # max_prefill_tokens * 1.5 MB and can go negative; and headroom
            # for the full-vocab logprob transient, which for a long prefill
            # at this vocabulary is multiple GB with chunking off by contract.
            # The fraction must also cover the weights per card and the
            # hybrid state cache: the 52G model at TP=2 holds 26G per card,
            # and 0.45 of a 32G card starves the budget before the cache is
            # sized. KOT_SGLANG_MEM_FRACTION overrides for models that big.
            "mem_fraction_static": float(os.environ.get("KOT_SGLANG_MEM_FRACTION", "0.45")),
            # The patch layer's frame step reads fps; the engine hands the cap
            # to the HF processor's videos_kwargs.
            "mm_process_config": {
                "video": {"fps": VIDEO_FPS, "cap_pixels_per_frame": CAP_PIXELS_PER_FRAME},
            },
            # On CUDA the engine's default is one worker, and one worker runs
            # the HF pass on the event loop, stalling every open connection for
            # as long as a clip takes. Two is qwen_vl's own number for CPU
            # preprocessing, which is what the patch layer pins.
            "mm_processor_worker_num": 2,
            "log_level": "info",
        }
        if is_cuda:
            kwargs["tp_size"] = torch.cuda.device_count()
            kwargs["attention_backend"] = "triton"
            kwargs["enable_deterministic_inference"] = True
        print(f"[kot_server] engine args: {kwargs}", flush=True)
        self.engine = Engine(**kwargs)

        model_config = self.engine.tokenizer_manager.model_config
        self.vocab_size = int(model_config.vocab_size)
        self.dtype_name = str(model_config.dtype).removeprefix("torch.")
        self.tokenizer = self.engine.tokenizer_manager.tokenizer
        self.tokenizer_id = str(self.tokenizer.name_or_path) if self.tokenizer else model_dir
        self.processor = self.engine.tokenizer_manager.processor
        self.model_id, self.model_revision = _resolve_identity(model_dir)
        self.engine_name = f"sglang {sglang.__version__}"
        self.torch_version = torch.__version__
        self.toolkit_version = torch.version.cuda or ""
        # Where the engine's schedulers run, by its own placement rule; read
        # from its arguments, because the model lives in those processes.
        server_args = self.engine.server_args
        self.device = "+".join(
            f"cuda:{server_args.base_gpu_id + rank * server_args.gpu_id_step}"
            for rank in range(server_args.tp_size)
        ) if is_cuda else "cpu"

    def slice_vocab(self, block: np.ndarray) -> np.ndarray:
        """A padded engine's lm_head emits -inf over the pad columns; the
        configured vocabulary is the contract's width, so pad columns are
        cut before anything is written."""
        if block.shape[1] == self.vocab_size:
            return block
        if block.shape[1] < self.vocab_size:
            raise ScoreFailure(
                f"stashed rows are {block.shape[1]} wide for a {self.vocab_size} vocabulary"
            )
        return np.ascontiguousarray(block[:, :self.vocab_size])

    def tokenize(self, text: str) -> list[int]:
        """Text to ids through the engine's own tokenizer, no special tokens."""
        if self.tokenizer is None:
            raise ScoreFailure("the engine was started without a tokenizer")
        return [int(token) for token in self.tokenizer.encode(text, add_special_tokens=False)]

    def encode_prompt(self, prompt: str) -> list[int]:
        """The OpenAI text surface goes through the engine's own tokenizer."""
        if self.tokenizer is None:
            raise ScoreFailure("the engine was started without a tokenizer")
        return self.tokenize(prompt)

    async def async_generate(self, **kwargs):
        return await self.engine.async_generate(**kwargs)


def _nvidia_driver() -> str:
    """The kernel driver's version, as the driver itself reports it; empty when unreadable."""
    try:
        first_line = Path("/proc/driver/nvidia/version").read_text(encoding=UTF8).splitlines()[0]
    except (OSError, IndexError, ValueError):
        return ""
    match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", first_line)
    return match.group(1) if match else ""


def _engine_info(engine) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": engine.model_id,
        "model_revision": engine.model_revision,
        "tokenizer_id": engine.tokenizer_id,
        "engine": engine.engine_name,
        "dtype": engine.dtype_name,
        "device": engine.device,
        "vocab_size": engine.vocab_size,
        "torch_version": engine.torch_version,
        "plugin_version": "",
        "driver_version": _nvidia_driver(),
        "toolkit_version": engine.toolkit_version,
        # The reference's block, keys and nesting, for the values pinned here.
        "media": {
            "video_fps": VIDEO_FPS,
            "video_do_sample_frames": VIDEO_DO_SAMPLE_FRAMES,
            "video_cap_pixels_per_frame": CAP_PIXELS_PER_FRAME,
        },
        "started_at": _utc_stamp(),
    }


ROUTES = {
    "/kot/infer": kot_infer,
    "/v1/chat/completions": _chat_completions,
}

IDLE_TIMEOUT_S = 300.0
STATUS_LINES = {200: "OK", 404: "Not Found", 500: "Internal Server Error"}


async def _read_request(reader: asyncio.StreamReader):
    """One HTTP/1.1 request: (method, path, headers, body), or None on EOF."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError):
        return None
    lines = head.decode("latin-1").split("\r\n")
    method, raw_path, _version = lines[0].split(" ", 2)
    headers = {}
    for line in lines[1:]:
        if ":" in line:
            name, value = line.split(":", 1)
            headers[name.strip().lower()] = value.strip()
    body = b""
    if headers.get("transfer-encoding", "").lower() == "chunked":
        parts = []
        while True:
            size_line = await reader.readuntil(b"\r\n")
            size = int(size_line.strip() or b"0", 16)
            if size == 0:
                await reader.readuntil(b"\r\n")
                break
            parts.append(await reader.readexactly(size))
            await reader.readexactly(2)
            if sum(len(part) for part in parts) > 256 * 1024 * 1024:
                return None
        body = b"".join(parts)
    else:
        length = int(headers.get("content-length", 0) or 0)
        if length:
            body = await reader.readexactly(length)
    return method.upper(), urllib.parse.urlparse(raw_path).path, headers, body


def _json_bytes(status: int, payload: dict) -> bytes:
    body = json.dumps(payload).encode(UTF8)
    head = (
        f"HTTP/1.1 {status} {STATUS_LINES.get(status, 'OK')}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: keep-alive\r\n\r\n"
    )
    return head.encode("latin-1") + body


async def handle_connection(reader, writer, engine, info, score_lock) -> None:
    """Keep-alive HTTP/1.1 for one connection: a model call never blocks the
    loop (the engine is async), and the loop never blocks a model call."""
    try:
        while True:
            try:
                request = await asyncio.wait_for(_read_request(reader), IDLE_TIMEOUT_S)
            except (asyncio.TimeoutError, ConnectionError, asyncio.IncompleteReadError, OSError):
                break
            if request is None:
                break
            method, path, headers, body = request
            close = headers.get("connection", "").lower() == "close"
            try:
                if method == "GET" and path == "/kot/info":
                    writer.write(_json_bytes(200, info))
                    await writer.drain()
                elif method == "POST" and path in ROUTES:
                    payload = json.loads(body or b"{}")
                    result = await ROUTES[path](engine, payload, score_lock)
                    if isinstance(result, SseResponse):
                        head = (
                            "HTTP/1.1 200 OK\r\n"
                            "Content-Type: text/event-stream\r\n"
                            "Cache-Control: no-cache\r\n"
                            "Connection: keep-alive\r\n\r\n"
                        )
                        writer.write(head.encode("latin-1"))
                        await writer.drain()

                        async def write(chunk: bytes) -> None:
                            writer.write(chunk)
                            await writer.drain()

                        try:
                            await result.run(write)
                        except Exception as stream_failure:
                            # The 200 and the event-stream content type are
                            # already on the wire: the failure travels as an
                            # event, then the connection closes.
                            traceback.print_exc()
                            print(f"[kot_server] stream {path} failed", flush=True)
                            payload = {
                                "error": type(stream_failure).__name__,
                                "detail": str(stream_failure),
                            }
                            try:
                                writer.write(b"event: error\n")
                                writer.write(_sse_chunk(payload))
                                writer.write(b"data: [DONE]\n\n")
                                await writer.drain()
                            except ConnectionError:
                                pass
                        # The stream head carries no Content-Length, so the
                        # body's only terminator is connection close.
                        break
                    writer.write(_json_bytes(200, result))
                    await writer.drain()
                elif method == "POST" and path == "/kot/score":
                    payload = json.loads(body or b"{}")
                    result = await score(engine, payload, score_lock)
                    writer.write(_json_bytes(200, result))
                    await writer.drain()
                else:
                    writer.write(_json_bytes(404, {"error": "not_served", "detail": f"{path}: no such endpoint"}))
                    await writer.drain()
            except Exception as failure:
                traceback.print_exc()
                print(f"[kot_server] request {path} failed", flush=True)
                try:
                    writer.write(_json_bytes(
                        500, {"error": type(failure).__name__, "detail": str(failure)}
                    ))
                    await writer.drain()
                except ConnectionError:
                    break
            if close:
                break
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except ConnectionError:
            pass


async def run_server(port: int, model: str, device: str) -> None:
    if not os.environ.get("KOT_SGLANG_STASH_DIR"):
        print("[kot_server] KOT_SGLANG_STASH_DIR is unset; start through kot_serve.sh", flush=True)
        raise SystemExit(2)

    # Created inside the running loop, the Engine adopts it: tokenizer
    # manager, HTTP server and every generation stream share one loop.
    engine = KotEngine(model, device)
    info = _engine_info(engine)
    print(f"[kot_server] serving: {info}", flush=True)

    # One text-only generation over the pinned prompt so the first real
    # request does not pay compile and graph costs, and the sampler runs
    # over the model's own vocabulary once (the model's-own-vocabulary
    # warmup). The video path stays cold; the first unit absorbs its
    # compile. The stash it would write is cleaned on both sides.
    _clean_stash()
    await engine.async_generate(
        input_ids=_raw_template_ids(engine, "warmup"),
        sampling_params={"max_new_tokens": 1, "temperature": 1.0, "sampling_seed": 0},
    )
    _clean_stash()

    score_lock = asyncio.Lock()
    server = await asyncio.start_server(
        lambda reader, writer: handle_connection(reader, writer, engine, info, score_lock),
        "127.0.0.1",
        port,
        backlog=2048,
    )
    print(f"[kot_server] listening on 127.0.0.1:{port}", flush=True)
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(prog="kot_server.py")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    asyncio.run(run_server(args.port, args.model, args.device))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
