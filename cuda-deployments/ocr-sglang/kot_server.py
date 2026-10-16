"""The kot deployment contract for the OCR family, over one embedded SGLang engine.

    kot_server.py --port P --model /path/to/DeepSeek-OCR-2 --device cpu|cuda

Started by ``kot_serve.sh``. One asyncio process: the adapter (this file)
embeds ``sglang.Engine`` inside its own event loop, so /kot/score and the
speed harness drive the same resident engine, as the contract requires. The
architecture is the llm-sglang deployment's; what changes is the family: the
model is DeepSeek-OCR-2, a vision-language checkpoint whose units are images.

Endpoints:

* ``GET /kot/info`` -- provenance from the running process.
* ``POST /kot/score`` -- writes ``log_probs.npy`` and ``input_ids.npy`` into
  ``out_dir`` and names them back. The request's transcript ids are consumed
  verbatim, appended to the image template the checkpoint itself defines
  (plain format: BOS, one image placeholder, "\nFree OCR."); the image
  arrives as an attachment path and is passed to the engine as multimodal
  data, which expands the placeholder into the block its processor computes
  for the image (257 tokens for a <=768px image, 144 more per crop tile).
* ``POST /v1/completions`` and ``/v1/chat/completions`` -- the OpenAI-shaped
  surface the speed harness drives, with SSE streaming. The chat surface is
  multimodal: a text part and the document as an image_url data URI, the
  placeholder first, exactly the shape the score path builds.
* ``POST /kot/infer`` -- 404: this family serves no such endpoint.

Numerics of the score path, in one place:

* prefill: template + transcript ids go in as one prefill-only request
  (``max_new_tokens=0``) with ``return_logprob`` and ``logprob_start_len=0``.
  The patch stashes the full-vocabulary log-softmax of every position of the
  EXPANDED prompt, recomputed in float32; the contract's rows are the last
  ``len(transcript) - 1`` of them minus the final sampling row: row t
  predicts transcript[t + 1]. ``input_ids.npy`` stores the transcript, so a
  replay sends the same ids back.
* decode: image-driven. The prompt is the image template alone; the request's
  sampling object is applied exactly as given (temperature, top_p, top_k,
  min_p; all four are native engine knobs), the request's seed, every penalty
  at its neutral value. ``allow_eos`` gates the stop exactly as the contract
  states: false demands exactly ``max_new_tokens`` ids (EOS suppressed), true
  accepts what the sampler emitted before EOS, at most that many. Rows are
  stashed at the sampler's entry -- the raw pre-sampling distributions, one
  per emitted token, before any penalty or temperature touches the logits.
  ``input_ids.npy`` stores the raw template plus the generated ids, so a
  replay prefill teacher-forces over exactly what the request carried: the
  pre-expansion sequence, with its single image placeholder.

Rows land on disk through ``open_memmap`` at the final shape, filled block by
block as the engine's stash arrives: the full-vocabulary log-softmax of a
4,096-token decode at this vocabulary is about 2 GB, and no stage of this
file holds one.

Concurrency: one event loop; the Engine is created inside it. /kot/score is
serialised by a lock -- the runner sends one at a time, and
one-request-at-a-time is what makes the stash collection sound (see
sitecustomize.py).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
import traceback
import urllib.parse
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 1
UTF8 = "utf-8"
STASH_WAIT_S = 60.0
MEMMAP_BLOCK_ROWS = 512

IMAGE = "image"
IMAGE_TAG = "<image>"
BOS_ID = 0


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


async def _collect_stash(kind: str) -> list[np.ndarray]:
    """Every complete stash archive of one kind, in write order, then delete them."""
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
    loaded = []
    for _, path in blocks:
        with np.load(path) as archive:
            loaded.append(np.ascontiguousarray(archive["rows"], dtype=np.float32))
        try:
            path.unlink()
            Path(f"{path}.done.json").unlink()
        except OSError:
            pass
    return loaded


def _write_rows(out_dir: Path, name: str, blocks: list[np.ndarray]) -> str:
    """Stack stash blocks into out_dir/name through an open_memmap at the
    final shape: rows are copied block by block, never held whole."""
    rows = sum(block.shape[0] for block in blocks)
    vocab = blocks[0].shape[1]
    path = out_dir / name
    array = np.lib.format.open_memmap(
        path, mode="w+", dtype=np.float32, shape=(rows, vocab)
    )
    filled = 0
    for block in blocks:
        array[filled:filled + block.shape[0]] = block
        filled += block.shape[0]
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


def _open_image(path: str):
    """The attachment as an RGB PIL image: what the engine's processor takes."""
    from PIL import Image

    return Image.open(path).convert("RGB")


def _decode_data_uri(uri: str):
    """An image_url data URI as an RGB PIL image: what the chat surface gets."""
    from PIL import Image

    comma = uri.find(",")
    if comma < 0 or not uri[:5].lower() == "data:":
        raise ScoreFailure("an image part must be a data URI")
    return Image.open(BytesIO(base64.b64decode(uri[comma + 1:]))).convert("RGB")


def _consumed_guard(done: dict, ids: list[int], engine) -> None:
    """The engine must consume the request ids verbatim, plus the image blocks.

    The patch layer records the per-image token counts the engine just used
    (``last_counts.json`` in the stash dir). The reply's ``prompt_tokens``
    must equal the request ids, minus the placeholders, plus one block per
    image. A mismatch means the engine retokenized the request or changed the
    expansion; the self-agreement would read that only as numbers.
    """
    meta = done.get("meta_info", {}) if isinstance(done, dict) else {}
    prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
    if not prompt_tokens:
        return
    note_path = Path(os.environ["KOT_SGLANG_STASH_DIR"]) / "last_counts.json"
    try:
        counts = json.loads(note_path.read_text(encoding="utf-8")).get("counts", [])
    except Exception:
        raise ScoreFailure(
            "the count note is missing: the patch layer recorded no image token counts; "
            "the engine may have taken the retokenize fallback again"
        )
    placeholder = engine.tokenizer.convert_tokens_to_ids(IMAGE_TAG)
    placeholders = sum(1 for token in ids if int(token) == placeholder)
    expected = len(ids) - placeholders + sum(counts)
    if prompt_tokens != expected:
        raise ScoreFailure(
            f"the engine consumed {prompt_tokens} tokens for {len(ids)} ids with "
            f"{placeholders} placeholder(s) and counts {counts}; expected {expected}: "
            "the request did not stay verbatim"
        )


async def _score_prefill(engine, body: dict) -> dict:
    """Teacher forced over the ids the request carried, verbatim.

    The version's frozen template ids ride every request (a decode's prompt;
    a replay's template plus generated ids), so the deployment consumes ids
    and never tokenizes. The engine expands the single image placeholder
    into the block its processor computes for this image, so the stashed
    matrix has one row per position of the EXPANDED prompt; the expanded row
    count also proves the expansion happened: with none it would equal the
    raw id count. The rows the family keeps are the tail: the classic n - 1,
    or the last n_rows when the request trims (the OCR replay keeps only
    the rows over the generated ids).
    """
    image_path = body["attachments"][IMAGE]
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
    image = _open_image(image_path)
    ids = list(transcript)

    _clean_stash()
    done = await engine.async_generate(
        input_ids=ids,
        image_data=image,
        sampling_params=_sampling_params(0, seed, NEUTRAL_SAMPLING, True),
        return_logprob=True,
        logprob_start_len=0,
    )
    _consumed_guard(done, ids, engine)
    blocks = await _collect_stash("prefill")
    blocks = [engine.slice_vocab(block) for block in blocks]
    rows = sum(block.shape[0] for block in blocks)
    if rows <= len(ids):
        raise ScoreFailure(
            f"the engine stashed {rows} prefill rows for {len(ids)} raw ids: the "
            "image placeholder was not expanded into an image block"
        )
    # Drop the final (sampling) row -- it predicts a token that is not there --
    # then keep the tail's rows. Walked from the end, block by block, so the
    # whole [rows, vocab] matrix is never held.
    blocks[-1] = np.ascontiguousarray(blocks[-1][:-1])
    tail: list[np.ndarray] = []
    have = 0
    for block in reversed(blocks):
        if have >= keep:
            break
        take = min(block.shape[0], keep - have)
        tail.append(np.ascontiguousarray(block[block.shape[0] - take:]))
        have += take
    tail.reverse()
    if have != keep:
        raise ScoreFailure(
            f"{rows} expanded rows cannot yield {keep} requested rows"
        )
    log_probs_name = _write_rows(out_dir, "log_probs.npy", tail)
    first = tail[0]
    if not np.isfinite(first).all() or not np.isfinite(tail[-1]).all():
        raise ScoreFailure("prefill log_probs contain NaN or -inf")
    _write_array(out_dir, "input_ids.npy", np.asarray(transcript, dtype=np.int32))
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": log_probs_name, "input_ids": "input_ids.npy"},
        "n_positions": int(keep),
        "vocab_size": engine.vocab_size,
        "output_ids": [],
    }



async def _score_decode(engine, body: dict) -> dict:
    """Generate from the frozen template ids; one row per generated token."""
    image_path = body["attachments"][IMAGE]
    max_new_tokens = int(body["max_new_tokens"])
    allow_eos = bool(body.get("allow_eos", False))
    seed = int(body.get("seed", 42))
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    template_ids = [int(token) for token in body["input_ids"]]
    if not template_ids:
        raise ScoreFailure("empty input_ids: the version's frozen template is missing")
    image = _open_image(image_path)
    ids = list(template_ids)

    _clean_stash()
    done = await engine.async_generate(
        input_ids=ids,
        image_data=image,
        sampling_params=_sampling_params(
            max_new_tokens, seed, _resolve_sampling(body), allow_eos
        ),
        return_logprob=True,
        logprob_start_len=-1,
        custom_logit_processor=engine.ngram_processor,
    )
    _consumed_guard(done, ids, engine)
    output_ids = [int(token) for token in done["output_ids"]]
    # The decode's own sampler rows: the probe below would pollute the stash
    # with prefill rows, so collect before probing.
    blocks = await _collect_stash("decode")
    blocks = [engine.slice_vocab(block) for block in blocks]
    rows = sum(block.shape[0] for block in blocks)
    if rows == len(output_ids) + 1:
        # The overlap scheduler launches one more decode forward for a request
        # that finished in the previous iteration: a distribution for a token
        # nobody emits. The sampler calls are stashed in order, so the phantom
        # is the last row.
        *blocks, last = blocks
        if last.shape[0] > 1:
            blocks.append(np.ascontiguousarray(last[:-1]))
        print("[kot_server] trimmed 1 overlap phantom decode row", flush=True)
    elif rows != len(output_ids):
        raise ScoreFailure(f"{rows} stashed rows for {len(output_ids)} generated ids")
    if not 0 < len(output_ids) <= max_new_tokens:
        raise ScoreFailure(
            f"decode produced {len(output_ids)} ids for a cap of {max_new_tokens}"
        )

    # The raw prompt the request carried (the version's frozen template)
    # plus the generated ids: the pre-expansion sequence, so the replay of
    # this file carries exactly one image placeholder.
    _write_rows(out_dir, "log_probs.npy", blocks)
    _write_array(
        out_dir, "input_ids.npy",
        np.asarray([*template_ids, *output_ids], dtype=np.int32),
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


def _openai_sampling(body: dict) -> tuple[dict, int]:
    """The sampling knobs an OpenAI-shaped request carries, in engine terms."""
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
    params = {
        "max_new_tokens": max_tokens,
        "temperature": float(body.get("temperature", 1.0) or 1.0),
        "top_p": float(body.get("top_p", 1.0)),
        "sampling_seed": int(body.get("seed", 42)),
        "skip_special_tokens": bool(body.get("skip_special_tokens", True)),
        # The scheduler fills this dict with "__req__"; the anti-repetition
        # rule reads the request's sequence through it.
        "custom_params": {},
    }
    if body.get("min_tokens") is not None:
        params["min_new_tokens"] = int(body["min_tokens"])
    if body.get("top_k") is not None:
        params["top_k"] = int(body["top_k"]) or -1  # the contract's 0 is "off": sglang's -1
    if body.get("repetition_penalty") is not None:
        params["repetition_penalty"] = float(body["repetition_penalty"])
    return params, max_tokens


def _finish_reason(done: dict, max_tokens: int) -> str:
    meta = done.get("meta_info", {}) if isinstance(done, dict) else {}
    finish = str(meta.get("finish_reason", "") or "")
    if finish:
        return {"stop": "stop", "length": "length", "abort": "stop"}.get(finish, finish)
    completion = int(meta.get("completion_tokens", 0) or 0)
    return "length" if completion >= max_tokens else "stop"


def _usage(prompt_tokens: int, done: dict) -> dict:
    meta = done.get("meta_info", {}) if isinstance(done, dict) else {}
    completion = int(meta.get("completion_tokens", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion,
        "total_tokens": prompt_tokens + completion,
    }


def _chat_parts(messages: list[dict]) -> tuple[str, object]:
    """The chat surface's user turn as (instruction text, image).

    The harness sends multimodal messages: a text part and the document as an
    image_url data URI. The prompt the checkpoint expects is the image
    placeholder first, then the instruction -- exactly the shape the score
    path builds -- so the text is what follows the placeholder, the image is
    the attachment. One image per request (limit_mm_per_prompt is 1).
    """
    text_parts: list[str] = []
    image = None
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
            elif kind in ("image", "image_url"):
                if image is not None:
                    raise ScoreFailure("one image per request: the harness sends one document")
                if kind == "image":
                    image = part.get("image")
                else:
                    uri = str((part.get("image_url") or {}).get("url", ""))
                    image = _decode_data_uri(uri)
            elif kind == "input_audio":
                raise ScoreFailure("audio is not a surface this family serves")
    if image is None:
        raise ScoreFailure("a chat request must carry the document as an image part")
    return "".join(text_parts), image


async def _completions(engine, body: dict, _lock: asyncio.Lock) -> dict:
    """POST /v1/completions: the OpenAI text surface, tokenized here.

    The OCR harness never drives this route (documents are chat messages);
    it exists for parity and one-shot probes, text only.
    """
    prompt = body.get("prompt")
    params, max_tokens = _openai_sampling(body)
    if isinstance(prompt, list) and (not prompt or isinstance(prompt[0], int)):
        input_ids = [int(token) for token in prompt]
    elif isinstance(prompt, list):
        input_ids = engine.encode_prompt("".join(str(part) for part in prompt))
    else:
        input_ids = engine.encode_prompt(str(prompt))
    if body.get("stream"):
        return SseResponse(engine, engine.model_id, input_ids, None, params, max_tokens)
    done = await engine.async_generate(input_ids=input_ids, sampling_params=params)
    return {
        "id": f"cmpl-{os.urandom(12).hex()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": engine.model_id,
        "choices": [
            {
                "index": 0,
                "text": str(done.get("text", "")),
                "finish_reason": _finish_reason(done, max_tokens),
            }
        ],
        "usage": _usage(len(input_ids), done),
    }


async def _chat_completions(engine, body: dict, _lock: asyncio.Lock) -> dict:
    """POST /v1/chat/completions: the multimodal surface the speed harness drives."""
    text, image = _chat_parts(list(body["messages"]))
    params, max_tokens = _openai_sampling(body)
    ids = [BOS_ID, engine.image_token_id, *engine.tokenize("\n" + text)]
    if body.get("stream"):
        return SseResponse(engine, engine.model_id, ids, image, params, max_tokens)
    done = await engine.async_generate(
        input_ids=ids, image_data=image, sampling_params=params,
        custom_logit_processor=engine.ngram_processor,
    )
    return {
        "id": f"chatcmpl-{os.urandom(12).hex()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": engine.model_id,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": str(done.get("text", ""))},
                "finish_reason": _finish_reason(done, max_tokens),
            }
        ],
        "usage": _usage(len(ids), done),
    }


def _sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode(UTF8)


class SseResponse:
    """One generation rendered as a stream of OpenAI-shaped SSE chunks."""

    def __init__(self, engine, model: str, input_ids: list[int], image, params: dict,
                 max_tokens: int):
        self.engine = engine
        self.model = model
        self.input_ids = input_ids
        self.image = image
        self.params = params
        self.max_tokens = max_tokens
        self.id = f"chatcmpl-{os.urandom(12).hex()}"
        self.created = int(time.time())

    def _chunk(self, delta: dict | None, finish_reason: str | None) -> bytes:
        choice: dict = {"index": 0, "delta": delta or {}, "finish_reason": finish_reason}
        payload = {
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }
        return _sse_chunk(payload)

    async def run(self, write) -> None:
        generator = await self.engine.async_generate(
            input_ids=[self.input_ids],
            image_data=self.image,
            sampling_params=self.params,
            stream=True,
            custom_logit_processor=self.engine.ngram_processor,
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
        finish = _finish_reason(done, self.max_tokens)
        await write(self._chunk({}, finish))
        await write(b"data: [DONE]\n\n")


def _resolve_identity(model_dir: str) -> tuple[str, str]:
    """(model_id, revision) as provenance, from the weights themselves.

    A weights directory does not name its hub id; ``_name_or_path`` in
    config.json usually does. KOT_MODEL_ID / KOT_MODEL_REVISION override both.
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
    return model_id, override_revision or revision


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
            # The checkpoint ships its own modeling and processor code; the
            # engine must run it rather than a native arch it does not have.
            "trust_remote_code": True,
            # The anti-repetition rule travels per request as a custom logit
            # processor; the engine only accepts those with this on.
            "enable_custom_logit_processor": True,
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
            # need trimming every time.
            "disable_overlap_schedule": True,
            "max_prefill_tokens": 8192,
            # Explicit, because sglang's heuristic reserves
            # max_prefill_tokens * 1.5 MB and can go negative; and headroom
            # for the full-vocab logprob transient, which for a long prefill
            # at this vocabulary is multiple GB with chunking off by contract.
            "mem_fraction_static": 0.45,
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
        self.model_id, self.model_revision = _resolve_identity(model_dir)
        self.engine_name = f"sglang {sglang.__version__}"
        image_token = (
            self.tokenizer.convert_tokens_to_ids(IMAGE_TAG)
            if self.tokenizer is not None else None
        )
        if not isinstance(image_token, int) or image_token < 0:
            raise SystemExit(
                f"the tokenizer has no {IMAGE_TAG!r} token; cannot build prompts"
            )
        self.image_token_id = image_token
        # DeepSeek's anti-repetition rule, serialized for per-request
        # delivery (sglang's custom_logit_processor): the scheduler
        # deserializes it by name, so kot_serve.sh installs the module into
        # site-packages. Applied to every sampling request.
        from kot_ocr_ngram import KotOCR2NoRepeatNGram

        self.ngram_processor = KotOCR2NoRepeatNGram.to_str()

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


def _engine_info(engine, device: str) -> dict:
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": engine.model_id,
        "model_revision": engine.model_revision,
        "tokenizer_id": engine.tokenizer_id,
        "engine": engine.engine_name,
        "dtype": engine.dtype_name,
        "device": f"{device}:0" if ":" not in device else device,
        "vocab_size": engine.vocab_size,
        "started_at": _utc_stamp(),
    }


ROUTES = {
    "/v1/completions": _completions,
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
                    else:
                        writer.write(_json_bytes(200, result))
                        await writer.drain()
                elif method == "POST" and path == "/kot/score":
                    payload = json.loads(body or b"{}")
                    result = await score(engine, payload, score_lock)
                    writer.write(_json_bytes(200, result))
                    await writer.drain()
                else:
                    served = "no such endpoint" if path != "/kot/infer" else \
                        "this family serves no /kot/infer"
                    writer.write(_json_bytes(404, {"error": "not_served", "detail": f"{path}: {served}"}))
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


def _warmup_image():
    """A tiny synthetic image: the warmup must exercise the multimodal path."""
    from PIL import Image

    return Image.new("RGB", (64, 64), color=(255, 255, 255))


async def run_server(port: int, model: str, device: str) -> None:
    if not os.environ.get("KOT_SGLANG_STASH_DIR"):
        print("[kot_server] KOT_SGLANG_STASH_DIR is unset; start through kot_serve.sh", flush=True)
        raise SystemExit(2)

    # Created inside the running loop, the Engine adopts it: tokenizer
    # manager, HTTP server and every generation stream share one loop.
    engine = KotEngine(model, device)
    info = _engine_info(engine, device)
    print(f"[kot_server] serving: {info}", flush=True)

    # One tiny image-driven generation so the first real request does not pay
    # compile and graph costs; the stash it would write is cleaned on both
    # sides. The warmup prompt is the decode's own template.
    _clean_stash()
    await engine.async_generate(
        input_ids=[BOS_ID, engine.image_token_id],
        image_data=_warmup_image(),
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
