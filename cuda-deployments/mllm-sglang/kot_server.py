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
  asserted. The video rides to the engine as an attachment path and the
  engine's own multimodal processor samples the frames (fps 2.0, pinned)
  and expands the placeholder into the per-frame timestamp structure.
* ``POST /kot/infer`` -- the speed surface: one video decode, the
  description written as ``text.txt`` into ``out_dir``, the file named back.

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
  contract states. Rows are stashed at the sampler's entry -- the raw
  pre-sampling distributions, one per emitted token, before any penalty or
  temperature touches the logits. ``input_ids.npy`` stores the raw template
  plus the generated ids, so a replay prefill teacher-forces over exactly
  what the request carried: the pre-expansion sequence, with its single
  video placeholder.

Rows land on disk through ``open_memmap`` at the final shape, filled block by
block as the engine's stash arrives: the full-vocabulary log-softmax of a
4,096-token decode at this vocabulary (248,320) is about 4 GB, and no stage
of this file holds one.

Concurrency: one event loop; the Engine is created inside it. /kot/score is
serialised by a lock -- the runner sends one at a time, and
one-request-at-a-time is what makes the stash collection sound (see
sitecustomize.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
import urllib.parse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 1
UTF8 = "utf-8"
STASH_WAIT_S = 60.0
MEMMAP_BLOCK_ROWS = 512

VIDEO = "video"
# The pinned media pipeline, the same values the transformers reference
# serves. The processor's own frame sampling defaults to fps 2.0; pinned
# here and carried per request, so a candidate must report the same values.
VIDEO_PROMPT = "Describe what you see in the video."
VIDEO_FPS = 2.0


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


def _consumed_guard(done: dict, ids: list[int], engine, raw: list[int], video_path: str) -> None:
    """The engine must consume the request ids verbatim, plus the video block.

    The expected consumed count is known before the request: the request ids
    minus the one video placeholder, plus the video block's token count. The
    block's count comes from the patch layer's note (``last_counts.json`` in
    the stash dir) when it recorded one, else from this deployment's own
    processor rendering of the same video - the HF expansion, the same code
    the reference runs. A mismatch means the engine retokenized the request
    or sampled different frames; the self-agreement would read that only as
    numbers.
    """
    meta = done.get("meta_info", {}) if isinstance(done, dict) else {}
    prompt_tokens = int(meta.get("prompt_tokens", 0) or 0)
    if not prompt_tokens:
        return
    video_token = engine.model_config.hf_config.video_token_id
    placeholders = sum(1 for token in ids if int(token) == video_token)
    try:
        counts = json.loads(
            (Path(os.environ["KOT_SGLANG_STASH_DIR"]) / "last_counts.json").read_text(
                encoding="utf-8"
            )
        ).get("counts", [])
    except Exception:
        counts = None
    if counts is not None:
        expected = len(ids) - placeholders + sum(counts)
    else:
        expanded = _expanded_template_ids(engine, video_path)
        expected = len(ids) - placeholders + len(expanded) - len(raw)
    if prompt_tokens != expected:
        raise ScoreFailure(
            f"the engine consumed {prompt_tokens} tokens for {len(ids)} ids with "
            f"{placeholders} placeholder(s) and counts {counts}; expected {expected}: "
            "the request did not stay verbatim"
        )


_EXPANDED_CACHE: dict = {}


def _expanded_template_ids(engine, video_path: str) -> list[int]:
    """The pinned prompt's EXPANDED ids, as the checkpoint's own HF processor
    computes them for this video: the frame sampling and the per-frame
    timestamp structure the reference runs, at the same pins. Cached per
    video; a CPU pass of a second or two."""
    if video_path not in _EXPANDED_CACHE:
        processor = engine.processor
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
            return_tensors=None,
        )
        _EXPANDED_CACHE[video_path] = list(map(int, video["input_ids"][0]))
    return _EXPANDED_CACHE[video_path]


def _raw_template_ids(engine, video_path: str) -> list[int]:
    """The pinned prompt's raw ids: the chat template rendered with ONE video
    placeholder, tokenized by the tokenizer only. This is the form a capture
    stores; the engine's expansion happens inside sglang."""
    processor = engine.processor
    messages = [{"role": "user", "content": [
        {"type": "video", "video": video_path},
        {"type": "text", "text": VIDEO_PROMPT},
    ]}]
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return list(processor.tokenizer(text, add_special_tokens=False)["input_ids"])


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

    # The raw prompt the deployment rendered (the pinned chat template with
    # one video placeholder) plus the generated ids: the pre-expansion
    # sequence, so the replay of this file re-renders the same expansion.
    _write_rows(out_dir, "log_probs.npy", blocks)
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
            # The video units run ~12.4k-token prompts (the checkpoint's own
            # pixel budget) plus a 4,096-token decode.
            "max_prefill_tokens": 16384,
            "context_length": 20480,
            # Explicit, because sglang's heuristic reserves
            # max_prefill_tokens * 1.5 MB and can go negative; and headroom
            # for the full-vocab logprob transient, which for a long prefill
            # at this vocabulary is multiple GB with chunking off by contract.
            # The fraction must also cover the weights per card and the
            # hybrid state cache: the 52G model at TP=2 holds 26G per card,
            # and 0.45 of a 32G card starves the budget before the cache is
            # sized. KOT_SGLANG_MEM_FRACTION overrides for models that big.
            "mem_fraction_static": float(os.environ.get("KOT_SGLANG_MEM_FRACTION", "0.45")),
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
    "/kot/infer": kot_infer,
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
    info = _engine_info(engine, device)
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
