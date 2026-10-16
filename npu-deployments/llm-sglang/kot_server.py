"""The kot deployment contract, served over one embedded SGLang engine (NPU).

    kot_server.py --port P --model /model --device npu

Started by ``kot_serve.sh``. One asyncio process: the adapter (this file)
embeds ``sglang.Engine`` inside its own event loop, so /kot/score and the
speed harness drive the same resident engine, as the contract requires.

Endpoints:

* ``GET /kot/info`` -- provenance from the running process: the engine
  version, the dtype the parameters actually sit in, the vocabulary width
  the lm_head was built with (the configured one, before any padding).
* ``POST /kot/score`` -- writes ``log_probs.npy`` and ``input_ids.npy`` into
  ``out_dir`` and names them back. The ids are consumed verbatim: they are
  passed to the engine as ``input_ids``, which bypasses the tokenizer
  entirely, so nothing prepends BOS, applies a template or truncates. Rows
  are the pre-sampling distributions of ``sitecustomize.py``: float32
  log-softmax over the configured vocabulary.
* ``POST /v1/completions`` and ``/v1/chat/completions`` -- the OpenAI-shaped
  surface the speed harness drives, with SSE streaming. This is the one
  place a prompt is tokenized, by the engine's own tokenizer (the files
  beside the weights at --model).
* ``POST /kot/infer`` -- 404: a token family serves no such endpoint.

Numerics of the score path, in one place:

* prefill: the ids go in as one prefill-only request (``max_new_tokens=0``)
  with ``return_logprob`` and ``logprob_start_len=0``. The engine computes
  the logits for every position; the patch stashes them in float32; the last
  row (which predicts a token that is not there) is dropped, so n ids
  produce exactly n-1 rows.
* decode: the request's sampling object applied exactly as given
  (temperature, top_p, top_k, min_p; all four are native engine knobs), the
  request's seed, every penalty at its neutral value,
  ``min_new_tokens=max_new_tokens`` and ``ignore_eos`` so exactly the
  requested number of tokens come out, one row per generated token. Rows are
  stashed before the sampler touches the logits, so they are the model's own
  distribution; the emitted ids come from SGLang's own sampler.

Rows land on disk through ``open_memmap`` at the final shape, filled block by
block as the engine's stash arrives: the full-vocabulary log-softmax of an
8,192-token decode is about 5 GB, and no stage of this file holds one.

Concurrency: one event loop; the Engine is created inside it, so the
tokenizer manager, the HTTP server and every generation stream share the
loop and the engine batches whatever is in flight. /kot/score is serialised
by a lock -- the runner sends one at a time, and one-request-at-a-time is
also what makes the stash collection sound (see sitecustomize.py).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import traceback
import urllib.parse
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

CONTRACT_VERSION = 1
UTF8 = "utf-8"
STASH_WAIT_S = 60.0
MEMMAP_BLOCK_ROWS = 512


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
                order = int(name.split("_")[1].split(".")[0])
                blocks.append((order, path))
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


def _write_array(out_dir: Path, name: str, array: np.ndarray) -> str:
    path = out_dir / name
    np.save(path, np.ascontiguousarray(array), allow_pickle=False)
    return path.name


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
# The neutral knobs a prefill passes to the engine: it samples nothing, so
# the sampling object the request carries is ignored there by contract.
NEUTRAL_SAMPLING = {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0, "presence_penalty": 0.0}


def _resolve_sampling(body: dict) -> dict:
    """The request's sampling object in engine terms, every field applied
    exactly as given. Prefill carries the object too and samples nothing, so
    only the decode path resolves it; a field this engine cannot apply is an
    error naming it, never a silently ignored value. Missing fields fall back
    to their neutral values, so a bare request still serves.
    """
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ScoreFailure("unsupported sampling field(s): " + ", ".join(unsupported))
    return {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", -1)) or -1,  # the contract's 0 is "off": sglang's -1
        "min_p": float(sampling.get("min_p", 0.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
        "presence_penalty": float(sampling.get("presence_penalty", 0.0)),
    }


def _sampling_params(max_new_tokens: int, seed: int, sampling: dict) -> dict:
    """The request's sampling object with its seed and every penalty neutral."""
    return {
        "max_new_tokens": max_new_tokens,
        "temperature": sampling["temperature"],
        "top_p": sampling["top_p"],
        "top_k": sampling["top_k"],
        "min_p": sampling["min_p"],
        "frequency_penalty": 0.0,
        "presence_penalty": sampling["presence_penalty"],
        "repetition_penalty": sampling["repetition_penalty"],
        "min_new_tokens": max_new_tokens,
        "ignore_eos": True,
        "sampling_seed": seed,
    }


async def _score_prefill(engine, input_ids: list[int], seed: int):
    """Teacher forced: n ids in, n-1 full-vocabulary rows out. No token is generated."""
    _clean_stash()
    await engine.async_generate(
        input_ids=input_ids,
        sampling_params=_sampling_params(0, seed, NEUTRAL_SAMPLING),
        return_logprob=True,
        logprob_start_len=0,
    )
    blocks = await _collect_stash("prefill")
    rows = sum(block.shape[0] for block in blocks)
    if rows != len(input_ids):
        raise ScoreFailure(
            f"the engine stashed {rows} prefill rows for {len(input_ids)} ids"
        )
    # Row n-1 predicts a token that is not there: drop the last row, whether
    # it owns a block alone or shares the final block with real rows.
    *blocks, last = blocks
    if last.shape[0] > 1:
        blocks.append(np.ascontiguousarray(last[:-1]))
    return blocks


async def _score_decode(engine, input_ids: list[int], max_new_tokens: int,
                        seed: int, sampling: dict):
    """Generate with the request's seed; one row per generated token."""
    _clean_stash()
    done = await engine.async_generate(
        input_ids=input_ids,
        sampling_params=_sampling_params(max_new_tokens, seed, sampling),
        return_logprob=True,
        logprob_start_len=-1,
    )
    output_ids = [int(token) for token in done["output_ids"]]
    blocks = await _collect_stash("decode")
    rows = sum(block.shape[0] for block in blocks)
    if rows == len(output_ids) + 1:
        # The overlap scheduler launches one more decode forward for a request
        # that finished in the previous iteration: a distribution for a token
        # nobody emits. The sampler calls are stashed in order, so the phantom
        # is the last row; the contract's g rows are the g before-token
        # distributions of the g emitted ids.
        *blocks, last = blocks
        if last.shape[0] > 1:
            blocks.append(np.ascontiguousarray(last[:-1]))
        print("[kot_server] trimmed 1 overlap phantom decode row", flush=True)
    elif rows != len(output_ids):
        raise ScoreFailure(f"{rows} stashed rows for {len(output_ids)} generated ids")
    return blocks, output_ids


async def score(engine, body: dict, lock: asyncio.Lock) -> dict:
    """POST /kot/score: write the two arrays into out_dir, name them back, never return tensors."""
    input_ids = [int(token) for token in body["input_ids"]]
    if not input_ids:
        raise ScoreFailure("empty input_ids")
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(body.get("seed", 42))
    async with lock:
        if body["mode"] == "prefill":
            # sampling is carried but ignored: prefill is teacher forced
            blocks = await _score_prefill(engine, input_ids, seed)
            output_ids: list[int] = []
            consumed = input_ids
        elif body["mode"] == "decode":
            blocks, output_ids = await _score_decode(
                engine, input_ids, int(body["max_new_tokens"]), seed,
                _resolve_sampling(body),
            )
            consumed = input_ids + output_ids
        else:
            raise ScoreFailure(f"unknown mode {body['mode']!r}; the contract serves prefill and decode")
    n_positions = sum(block.shape[0] for block in blocks)
    files = {
        "log_probs": _write_rows(out_dir, "log_probs.npy", blocks),
        "input_ids": _write_array(out_dir, "input_ids.npy", np.asarray(consumed, dtype=np.int32)),
    }
    if body["mode"] == "decode" and output_ids and engine.tokenizer is not None:
        text = engine.tokenizer.decode(output_ids, skip_special_tokens=True)
        (out_dir / "text.txt").write_text(text, encoding=UTF8)
    return {
        "prompt_id": body["prompt_id"],
        "files": files,
        "n_positions": int(n_positions),
        "vocab_size": engine.vocab_size,
        "output_ids": output_ids,
    }


def _openai_sampling(body: dict) -> tuple[dict, int]:
    """The sampling knobs an OpenAI-shaped request carries, in engine terms."""
    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 16)
    params = {
        "max_new_tokens": max_tokens,
        "temperature": float(body.get("temperature", 1.0) or 1.0),
        "top_p": float(body.get("top_p", 1.0)),
        "sampling_seed": int(body.get("seed", 42)),
        "skip_special_tokens": bool(body.get("skip_special_tokens", True)),
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


async def _completions(engine, body: dict, _lock: asyncio.Lock) -> dict:
    """POST /v1/completions: the OpenAI surface.

    The harness may send `prompt` as a pre-tokenized id list -- the ids go to
    the engine verbatim, exactly like the score path. A string prompt is
    tokenized by the engine's own tokenizer.
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
        return SseResponse(engine, engine.model_id, input_ids, params, max_tokens)
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
    """POST /v1/chat/completions: the same generation behind the model's own chat template."""
    messages = list(body["messages"])
    prompt_ids = engine.apply_chat_template(messages)
    params, max_tokens = _openai_sampling(body)
    if body.get("stream"):
        return SseResponse(engine, engine.model_id, prompt_ids, params, max_tokens)
    done = await engine.async_generate(input_ids=prompt_ids, sampling_params=params)
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
        "usage": _usage(len(prompt_ids), done),
    }


def _sse_chunk(payload: dict) -> bytes:
    return f"data: {json.dumps(payload)}\n\n".encode(UTF8)


class SseResponse:
    """One generation rendered as a stream of OpenAI-shaped SSE chunks.

    Each engine update carries the text so far; the delta is what this chunk
    adds. The final chunk carries the finish reason, then `[DONE]`.
    """

    def __init__(self, engine, model: str, input_ids: list[int], params: dict, max_tokens: int):
        self.engine = engine
        self.model = model
        self.input_ids = input_ids
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
        finish = _finish_reason(done, self.max_tokens)
        await write(self._chunk({}, finish))
        await write(b"data: [DONE]\n\n")


def _model_identity(model_dir: Path, config: dict) -> tuple[str, str]:
    """(model_id, revision) as provenance: the HF cache layout when present."""
    revision = ""
    parts = model_dir.resolve().parts
    if "snapshots" in parts:
        revision = parts[parts.index("snapshots") + 1]
        repository = parts[parts.index("snapshots") - 1]
        if repository.startswith("models--"):
            return repository.removeprefix("models--").replace("--", "/"), revision
    model_id = str(config.get("_name_or_path") or model_dir.name)
    return model_id.split("/")[-1], revision


class KotEngine:
    """What the endpoints need from the engine, named once."""

    def __init__(self, model_dir: str, device: str):
        import torch_npu  # noqa: F401 -- registers the npu device before sglang loads
        import sglang
        from sglang.srt.entrypoints.engine import Engine

        is_npu = device.startswith("npu")
        if not is_npu:
            raise ScoreFailure(
                f"this deployment serves npu devices, asked for {device!r}"
            )
        config_path = Path(model_dir) / "config.json"
        config = json.loads(config_path.read_text(encoding=UTF8)) if config_path.is_file() else {}

        kwargs = {
            "model_path": model_dir,
            "device": "npu",
            "dtype": "bfloat16",
            "random_seed": 42,
            "attention_backend": "ascend",
            # One extend per request, so a decode's stashed rows stay aligned:
            # a chunked extend samples a token per chunk and all but the last
            # are discarded, which the score path cannot tell from real steps.
            # Overlap stays ON: disabling it (disable_overlap_schedule=True)
            # deadlocks the ascend scheduler after the extend. The phantom
            # forward it adds to a finished decode is trimmed in _score_decode.
            "chunked_prefill_size": -1,
            # The long prefill replays prompt + generated, about 8,600 ids.
            "max_prefill_tokens": 32768,
            # Explicit, because sglang's heuristic reserves
            # max_prefill_tokens * 1.5 MB and can go negative with the
            # settings above.
            "mem_fraction_static": 0.85,
            "log_level": "info",
        }
        print(f"[kot_server] engine args: {kwargs}", flush=True)
        self.engine = Engine(**kwargs)

        model_config = self.engine.tokenizer_manager.model_config
        self.vocab_size = int(model_config.vocab_size)
        self.dtype_name = str(model_config.dtype).removeprefix("torch.")
        self.tokenizer = self.engine.tokenizer_manager.tokenizer
        self.tokenizer_id = str(self.tokenizer.name_or_path) if self.tokenizer else model_dir
        self.model_id, self.model_revision = _model_identity(Path(model_dir), config)
        self.engine_name = f"sglang {sglang.__version__}"

    def encode_prompt(self, prompt: str) -> list[int]:
        """The OpenAI text surface goes through the engine's own tokenizer."""
        tokenizer = self.engine.tokenizer_manager.tokenizer
        if tokenizer is None:
            raise ScoreFailure("the engine was started without a tokenizer")
        return tokenizer.encode(prompt)

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        tokenizer = self.engine.tokenizer_manager.tokenizer
        if tokenizer is None:
            raise ScoreFailure("the engine was started without a tokenizer")
        out = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True
        )
        # transformers 5.x returns a BatchEncoding here, older lines a flat id
        # list. BatchEncoding is a UserDict, not a dict: unwrap any mapping by
        # key and never iterate one (iterating yields its keys, which then
        # surface as "'str' object cannot be interpreted as an integer" deep
        # in the engine's array() call).
        if isinstance(out, Mapping):
            out = out["input_ids"]
        if out and isinstance(out[0], list):
            out = out[0]
        return [int(token) for token in out]

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
BODY_TIMEOUT_S = 120.0
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
                            # already on the wire: a status line here would
                            # land inside the body. The failure travels as an
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
                        # body's only terminator is connection close: without
                        # it every client blocks on EOF after [DONE].
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
                        "a token family serves no /kot/infer"
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


async def run_server(port: int, model: str, device: str) -> None:
    if not os.environ.get("KOT_SGLANG_STASH_DIR"):
        print("[kot_server] KOT_SGLANG_STASH_DIR is unset; start through kot_serve.sh", flush=True)
        raise SystemExit(2)

    # Created inside the running loop, the Engine adopts it: tokenizer
    # manager, HTTP server and every generation stream share one loop.
    engine = KotEngine(model, device)
    info = _engine_info(engine, device)
    print(f"[kot_server] serving: {info}", flush=True)

    # One tiny generation so the first real request does not pay compile and
    # graph costs; the stash it would write is cleaned on both sides.
    _clean_stash()
    await engine.async_generate(
        input_ids=[151644, 872, 198],
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
