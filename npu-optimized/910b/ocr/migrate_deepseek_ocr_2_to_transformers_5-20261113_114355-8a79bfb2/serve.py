#!/usr/bin/env python3
"""DeepSeek-OCR-2 serving of the kot deployment contract, transformers 5.22.1.

    ./serve.sh --port P --model /path/to/DeepSeek-OCR-2 --device cpu|cuda|npu

Layout: this file beside the modules it imports - native_config
(build_config), native_load (load_native), native_processor
(build_processor), engine (GenerationEngine) - serving the original weights
directory. The checkpoint loads through transformers 5.22.1's built-in
deepseek_ocr2 architecture; the checkpoint's own remote code is not used.

Numerics contract, exactly as the reference server's deepseek adapter
implements it, now shared by every endpoint through one engine:

* the ids a request sends are consumed verbatim after the single image
  placeholder is expanded: no BOS added, no chat template of ours, no
  truncation, no re-tokenization;
* prefill is teacher forced over the full stored sequence; the rows written
  are the model's raw logits at the positions that predict the last n_rows
  ids, cast to float32 and log-softmaxed over the configured vocabulary,
  so every row's logsumexp is 0;
* decode generates with the request's sampling record applied exactly as
  transformers applies it (min_new_tokens EOS ban, presence, frequency and
  repetition penalties, DeepSeek's no-repeat 30-gram rule, temperature,
  top_k, top_p, min_p) and a per-request seeded generator; the stored rows
  are the raw pre-processor logits, one per generated token;
* decode and prefill of the same image describe the same forward pass: the
  same processor call produces the same global view and tiles, the same
  image-token expansion, the same weights and kernels;
* the chat route drives the same engine: one image per request, the same
  views, the same expansion and the same seeded generation, answering as an
  OpenAI chat completion (streaming or not) through the same loaded model.

The HTTP layer is asyncio: thousands of held-open connections cost nothing,
image preprocessing runs on a bounded thread pool, and one engine thread
owns the accelerator and batches what arrives into waves.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import binascii
import io
import json
import os
import signal
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageOps

from engine import EngineRequest, GenerationEngine, Row, resolve_sampling
from native_config import build_config
from native_load import load_native, snapshot_identity
from native_processor import build_processor, load_image

HOST = "127.0.0.1"
INFO = "/kot/info"
SCORE = "/kot/score"
CHAT_COMPLETIONS = "/v1/chat/completions"
PREFILL = "prefill"
UTF8 = "utf-8"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 16

IMAGE = "image"
IMAGE_TOKEN_ID = 128815
PROMPT_TEXT = "Free OCR."  # documented: what the frozen template ids spell
BOS_ID = 0
WRITE_CHUNK = 64  # rows of log-softmax per streaming write
PREPROCESS_WORKERS = 16

SAMPLING_KEYS = ("temperature", "top_p", "top_k", "min_p",
                 "repetition_penalty", "presence_penalty", "frequency_penalty")

# Ascend version files, as the CANN image and the driver mount lay them out.
DRIVER_VERSION_FILE = "/usr/local/Ascend/driver/version.info"
TOOLKIT_VERSION_FILE = "/usr/local/Ascend/ascend-toolkit/version.info"
TOOLKIT_LATEST_DIR = "/usr/local/Ascend/ascend-toolkit/latest"

state = SimpleNamespace(model=None, processor=None, engine=None, device=None,
                        dtype=None, vocab_size=None, info=None, loop=None,
                        pool=None, tokenizer=None, eos_token_id=None)


def fail(message):
    raise RuntimeError(message)


def ascend_version(version_file: str, latest_dir: str = "") -> str:
    """The Ascend component's own version string, read and never invented."""
    try:
        for line in Path(version_file).read_text(encoding=UTF8).splitlines():
            if line.startswith("Version="):
                return line.removeprefix("Version=")
    except OSError:
        pass
    if latest_dir:
        try:
            return Path(latest_dir).resolve().name
        except OSError:
            pass
    return ""


def image_views(image) -> tuple:
    """One image as the native processor encodes it - the same call for every
    request shape: the padded global view, the crop tiles (or None), the tile
    count. Returns CPU tensors; the engine batches and moves them."""
    features = state.processor.image_processor(images=[image], return_tensors="pt")
    local = features.get("pixel_values_local")
    tiles = int(features["num_local_patches"][0])
    return (features["pixel_values"][0],
            None if local is None else local,
            tiles)


def image_from_bytes(data: bytes):
    with Image.open(io.BytesIO(data)) as opened:
        return ImageOps.exif_transpose(opened).convert("RGB")


def raw_template_ids_from(body: dict) -> list[int]:
    """The request's frozen template ids (one image placeholder), or the
    checkpoint's own template when the request carries none."""
    template_ids = [int(token) for token in body.get("input_ids") or []]
    if template_ids:
        return template_ids
    instruction = state.tokenizer.encode(f"\n{PROMPT_TEXT}", add_special_tokens=False)
    return [BOS_ID, IMAGE_TOKEN_ID, *instruction]


class RowStore:
    """Streams log-softmax rows into log_probs.npy at its final shape.

    The file is opened at the row upper bound and filled row by row; when the
    unit stops short of the bound (EOS before max), the numpy header is
    rewritten to the true shape and the file truncated, so nothing is ever
    held whole in host memory and the array lands at its exact shape.
    """

    def __init__(self, out_dir: Path, max_rows: int):
        self.path = out_dir / "log_probs.npy"
        self.max_rows = max_rows
        self.rows = 0
        self.array = np.lib.format.open_memmap(
            self.path, mode="w+", dtype=np.float32,
            shape=(max_rows, state.vocab_size))

    def write_chunk(self, rows: torch.Tensor):
        """log-softmax fp32 over rows of raw logits (bf16 or fp32)."""
        start = self.rows
        stop = start + rows.shape[0]
        for begin in range(start, stop, WRITE_CHUNK):
            end = min(begin + WRITE_CHUNK, stop)
            chunk = rows[begin - start:end - start, :state.vocab_size].float()
            chunk = torch.log_softmax(chunk, dim=-1)
            self.array[begin:end] = chunk.cpu().numpy()
        self.rows = stop

    def close(self):
        if self.array is None:
            return
        del self.array
        self.array = None
        if self.rows < self.max_rows:
            # Rewrite the header at the true row count and cut the file to
            # the exact data length. The new header never needs more space
            # than the old one (same dtype, same width, a smaller first
            # dimension), so the data offset is preserved; numpy pads headers
            # to 64-byte multiples, and both headers pad to the same block.
            with open(self.path, "r+b") as handle:
                prefix = handle.read(10)
                header_len = int.from_bytes(prefix[8:10], "little")
                data_offset = ((10 + header_len + 63) // 64) * 64
                text = ("{'descr': '<f4', 'fortran_order': False, "
                        f"'shape': ({self.rows}, {state.vocab_size}), }}").encode(UTF8)
                padding = data_offset - 10 - len(text)
                if padding < 0:
                    raise RuntimeError("npy header does not fit its original slot")
                handle.seek(0)
                handle.write(b"\x93NUMPY" + b"\x01\x00"
                             + (len(text) + padding).to_bytes(2, "little")
                             + text + b" " * padding)
                handle.truncate(data_offset + self.rows * state.vocab_size * 4)

    def block_lines(self):
        written = 0
        block = 0
        while written + WRITE_CHUNK <= self.rows:
            written += WRITE_CHUNK
            block += 1
            yield json.dumps({"block": block - 1, "rows": written}).encode(UTF8) + b"\n"


def decode_token_sink(store: RowStore):
    """The capture callback for a score decode: one raw fp32 row per step."""
    def capture(raw: torch.Tensor):
        store.write_chunk(raw.unsqueeze(0))
    return capture


# ---------------------------------------------------------------------- #
# request handling                                                        #
# ---------------------------------------------------------------------- #

def build_row(views, max_new_tokens, allow_eos, seed, raw_ids, sampling) -> Row:
    row = Row()
    row.raw_ids = list(raw_ids)
    row.pixel_values, row.pixel_values_local, row.tiles = views
    row.sampling = sampling
    row.max_new = int(max_new_tokens)
    row.allow_eos = bool(allow_eos)
    row.min_new = 0 if row.allow_eos else row.max_new
    row.generator = torch.Generator(device=state.device)
    row.generator.manual_seed(int(seed))
    return row


async def wait_request(request: EngineRequest):
    """Await an engine request's completion off the event loop."""
    await asyncio.get_running_loop().run_in_executor(None, request.event.wait)
    if request.error is not None:
        raise request.error


async def handle_score(body: dict, writer: asyncio.StreamWriter):
    image_path = body["attachments"][IMAGE]
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    if body["mode"] == PREFILL:
        incoming = [int(token) for token in body["input_ids"]]
        if not incoming:
            fail("prefill over an empty transcript: the frozen ids are missing")
        n_rows = int(body.get("n_rows", 0))
        if not 0 < n_rows < len(incoming):
            fail(f"the ocr prefill stores the decode's rows: set n_rows, got {n_rows}")
        views = await run_in_pool(image_views,
                                  await run_in_pool(load_image, image_path))
        request = EngineRequest("prefill")
        request.n_rows = n_rows
        request.prefill_ids = incoming
        request.prefill_features = views
        store = RowStore(out_dir, n_rows)
        request.logits_sink = store.write_chunk
        await submit_and_wait(request)
        store.close()
        np.save(out_dir / "input_ids.npy", np.asarray(incoming, dtype=np.int32),
                allow_pickle=False)
        reply = {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": n_rows,
            "vocab_size": state.vocab_size,
            "output_ids": [],
        }
        await write_json(writer, reply)
        return

    # decode
    raw_ids = raw_template_ids_from(body)
    sampling = resolve_sampling(body.get("sampling"))
    image = await run_in_pool(load_image, image_path)
    views = await run_in_pool(image_views, image)
    store = RowStore(out_dir, int(body["max_new_tokens"]))
    request = EngineRequest("decode")
    row = build_row(views, body["max_new_tokens"], body.get("allow_eos", False),
                    body.get("seed", DEFAULT_SEED), raw_ids, sampling)
    row.capture = decode_token_sink(store)
    request.row = row
    state.engine.submit(request)
    await wait_request(request)
    store.close()
    consumed = [*raw_ids, *row.generated]
    np.save(out_dir / "input_ids.npy", np.asarray(consumed, dtype=np.int32),
            allow_pickle=False)
    text = state.tokenizer.decode(row.generated, skip_special_tokens=False)
    text = text.removesuffix(state.tokenizer.eos_token)
    (out_dir / "text.txt").write_text(text, encoding=UTF8)
    reply = {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
        "n_positions": len(row.generated),
        "vocab_size": state.vocab_size,
        "output_ids": [int(t) for t in row.generated],
    }
    payload = b"".join(store.block_lines()) + json.dumps(reply).encode(UTF8) + b"\n"
    await write_chunked(writer, payload)


def chat_sampling(body: dict) -> dict:
    return resolve_sampling({key: body[key] for key in SAMPLING_KEYS if key in body})


async def handle_chat(body: dict, writer: asyncio.StreamWriter):
    messages = body.get("messages")
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict) or messages[0].get("role") != "user"):
        fail("messages must contain exactly one user message")
    content = messages[0].get("content")
    if not isinstance(content, list):
        fail("the user message content must be a list with text and image_url parts")
    images, texts = [], []
    for part in content:
        if not isinstance(part, dict):
            fail("each content part must be an object")
        if part.get("type") == "text":
            if not isinstance(part.get("text"), str):
                fail("text parts must contain a string text")
            texts.append(part["text"])
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url")
            if not isinstance(url, str):
                fail("image_url must contain a URL")
            images.append(url)
        else:
            fail("content supports only text and image_url parts")
    if len(images) != 1:
        fail(f"expected exactly one image_url part, got {len(images)}")
    if not texts or not "".join(texts).strip():
        fail("the user message must contain text")
    url = images[0]
    if not url.startswith("data:image/") or ";base64," not in url:
        fail("image_url must be a data:image/...;base64,... URL")
    encoded = url.split(";base64,", 1)[1]
    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as error:
        raise ValueError("image_url contains invalid base64 data") from error
    max_tokens = int(body.get("max_completion_tokens",
                              body.get("max_tokens", DEFAULT_MAX_TOKENS)))
    seed = int(body.get("seed", DEFAULT_SEED))
    raw_ids = [BOS_ID, IMAGE_TOKEN_ID,
               *state.tokenizer.encode("\n" + "".join(texts), add_special_tokens=False)]

    image = await run_in_pool(image_from_bytes, data)
    views = await run_in_pool(image_views, image)
    request = EngineRequest("decode")
    row = build_row(views, max_tokens, True, seed, raw_ids, chat_sampling(body))
    request.row = row
    state.engine.submit(request)

    if not body.get("stream"):
        await wait_request(request)
        text = final_text(row.generated)
        reply = {
            "id": "chatcmpl-" + uuid.uuid4().hex,
            "object": "chat.completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": state.info["model_id"],
            "choices": [{"index": 0,
                         "message": {"role": "assistant", "content": text},
                         "finish_reason": "stop" if len(row.generated) < max_tokens else "length"}],
            "usage": {"prompt_tokens": len(raw_ids),
                      "completion_tokens": len(row.generated),
                      "total_tokens": len(raw_ids) + len(row.generated)},
        }
        await write_json(writer, reply)
        return

    # streaming: one SSE delta per generated token
    tokens = asyncio.Queue()

    def emit(token):
        state.loop.call_soon_threadsafe(tokens.put_nowait, token)

    row.emit = emit
    done = asyncio.get_running_loop().create_future()

    def on_done():
        state.loop.call_soon_threadsafe(_set_future, done)

    row.on_done = on_done
    completion_id = "chatcmpl-" + uuid.uuid4().hex
    created = int(datetime.now(timezone.utc).timestamp())
    model = state.info["model_id"]
    stream = _sse_writer(writer)
    await stream.asend(None)
    await stream.asend({"id": completion_id, "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {"role": "assistant"},
                                    "finish_reason": None}]})
    emitted = ""
    content_ids = []
    while True:
        token = await tokens.get()
        content_ids.append(token)
        delta = incremental_text(content_ids, emitted)
        if delta:
            emitted += delta
            await stream.asend({"id": completion_id,
                                "object": "chat.completion.chunk",
                                "created": created, "model": model,
                                "choices": [{"index": 0, "delta": {"content": delta},
                                             "finish_reason": None}]})
        if token == state.eos_token_id or len(content_ids) >= max_tokens:
            break
    await wait_request(request)
    final = final_text(row.generated)
    if final.startswith(emitted) and len(final) > len(emitted):
        await stream.asend({"id": completion_id,
                            "object": "chat.completion.chunk",
                            "created": created, "model": model,
                            "choices": [{"index": 0,
                                         "delta": {"content": final[len(emitted):]},
                                         "finish_reason": None}]})
    await stream.asend({"id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created, "model": model,
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": "stop"}]})
    # stream terminator: the [DONE] event, then the chunked-encoding end
    await write_raw(writer, b"data: [DONE]\n\n")
    await write_raw(writer, b"")
    await writer.drain()


def _set_future(future):
    if not future.done():
        future.set_result(True)


def final_text(generated: list[int]) -> str:
    text = state.tokenizer.decode(generated, skip_special_tokens=False)
    return text.removesuffix(state.tokenizer.eos_token)


def incremental_text(ids: list[int], emitted: str) -> str:
    """The next safe piece of text: only whole characters, never a partial
    multibyte sequence (held back until its bytes are complete)."""
    current = state.tokenizer.decode(ids, skip_special_tokens=False)
    if not current.startswith(emitted):
        return ""
    delta = current[len(emitted):]
    if "\ufffd" in delta:
        return ""
    return delta


async def _sse_writer(writer: asyncio.StreamWriter):
    """Coroutine-based SSE emitter: asend(chunk) writes one event, asend(None)
    writes the [DONE] terminator and ends the response."""
    started = False
    while True:
        chunk = yield
        if not started:
            writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                         b"Cache-Control: no-cache\r\nTransfer-Encoding: chunked\r\n\r\n")
            started = True
        if chunk is None:
            await write_raw(writer, b"data: [DONE]\n\n")
            await writer.drain()
            return
        payload = json.dumps(chunk)
        await write_raw(writer, f"data: {payload}\n\n".encode(UTF8))


async def write_raw(writer: asyncio.StreamWriter, data: bytes):
    writer.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
    await writer.drain()


async def write_json(writer: asyncio.StreamWriter, payload: dict):
    body = json.dumps(payload).encode(UTF8)
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
                 b"Content-Length: " + str(len(body)).encode() + b"\r\n"
                 b"Connection: keep-alive\r\n\r\n" + body)
    await writer.drain()


async def write_chunked(writer: asyncio.StreamWriter, payload: bytes):
    writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: application/x-ndjson\r\n"
                 b"Transfer-Encoding: chunked\r\nConnection: keep-alive\r\n\r\n")
    await writer.drain()
    for start in range(0, len(payload), 1 << 20):
        await write_raw(writer, payload[start:start + (1 << 20)])
    await write_raw(writer, b"")  # terminating zero-length chunk
    await writer.drain()


async def run_in_pool(function, *args):
    return await state.loop.run_in_executor(state.pool, function, *args)


async def submit_and_wait(request: EngineRequest):
    state.engine.submit(request)
    await asyncio.get_running_loop().run_in_executor(None, request.event.wait)


# ---------------------------------------------------------------------- #
# HTTP plumbing                                                           #
# ---------------------------------------------------------------------- #

async def read_request(reader: asyncio.StreamReader):
    """Reads one request; returns (method, path, body) or None on EOF."""
    try:
        head = await reader.readuntil(b"\r\n\r\n")
    except (asyncio.IncompleteReadError, ConnectionError,
            asyncio.LimitOverrunError):
        return None
    lines = head.split(b"\r\n")
    parts = lines[0].decode("latin-1").split()
    if len(parts) != 3:
        return None
    method, path, _ = parts
    headers = {}
    for line in lines[1:]:
        if b":" in line:
            key, value = line.split(b":", 1)
            headers[key.decode("latin-1").lower().strip()] = value.strip().decode("latin-1")
    length = int(headers.get("content-length", "0") or "0")
    body = b""
    if length:
        body = await reader.readexactly(length)
    return method, path, body


async def serve_connection(reader, writer):
    """One keep-alive connection: read a request, answer, repeat."""
    try:
        while True:
            request = await read_request(reader)
            if request is None:
                break
            method, path, body = request
            if method == "GET" and path == INFO:
                await write_json(writer, state.info)
            elif method == "POST" and path in (SCORE, CHAT_COMPLETIONS):
                try:
                    payload = json.loads(body or b"{}")
                    if path == SCORE:
                        await handle_score(payload, writer)
                    else:
                        await handle_chat(payload, writer)
                except Exception as error:  # noqa: BLE001 - a failure is a result
                    detail = str(error) or type(error).__name__
                    await write_error(writer, type(error).__name__, detail)
            else:
                await write_error(writer, "not_found", path, status=404)
    except (ConnectionError, asyncio.IncompleteReadError, BrokenPipeError):
        pass
    except Exception as error:  # noqa: BLE001
        print(f"[serve] connection error: {type(error).__name__}: {error}", flush=True)
    finally:
        try:
            writer.close()
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


async def write_error(writer, error, detail, status=500):
    body = json.dumps({"error": error, "detail": detail}).encode(UTF8)
    reason = {400: "Bad Request", 404: "Not Found",
              500: "Internal Server Error"}.get(status, "Error")
    writer.write(f"HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\n"
                 f"Content-Length: {len(body)}\r\nConnection: keep-alive\r\n\r\n".encode()
                 + body)
    await writer.drain()


# ---------------------------------------------------------------------- #
# startup                                                                 #
# ---------------------------------------------------------------------- #

def warmup():
    """One tiny image through both engine modes, so the first real request
    meets warm kernels. Runs before the port answers /kot/info."""
    image = Image.new("RGB", (64, 64), (12, 34, 56))
    views = image_views(image)
    request = EngineRequest("decode")
    row = build_row(views, 2, True, DEFAULT_SEED,
                    [BOS_ID, IMAGE_TOKEN_ID], resolve_sampling({}))
    request.row = row
    state.engine.submit(request)
    request.event.wait(timeout=600)
    if request.error is not None:
        raise request.error
    prefill = EngineRequest("prefill")
    prefill.n_rows = len(row.generated)
    prefill.prefill_ids = [BOS_ID, IMAGE_TOKEN_ID, *row.generated]
    prefill.prefill_features = views
    sink_rows = []
    prefill.logits_sink = lambda rows: sink_rows.append(tuple(rows.shape))
    state.engine.submit(prefill)
    prefill.event.wait(timeout=600)
    if prefill.error is not None:
        raise prefill.error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    device = args.device
    if device.startswith("npu"):
        import torch_npu  # noqa: F401  registers the npu backend first
        torch.npu.set_device(0)
        dtype = torch.bfloat16
        attention = os.environ.get("KOT_ATTN", "sdpa")
    elif device.startswith("cuda"):
        dtype = torch.bfloat16
        attention = os.environ.get("KOT_ATTN", "sdpa")
    else:
        dtype = torch.float32
        attention = os.environ.get("KOT_ATTN", "eager")
    if os.environ.get("KOT_DTYPE"):
        dtype = getattr(torch, os.environ["KOT_DTYPE"])

    if not (Path(args.model) / "config.json").is_file():
        fail(f"{args.model} holds no config.json; this deployment loads a checkpoint directory")
    cfg = build_config(args.model)
    cfg.dtype = dtype
    cfg._attn_implementation = attention
    model, report = load_native(args.model, cfg)
    processor = build_processor(args.model)
    model = model.eval().to(device=device, dtype=dtype)

    state.model = model
    state.processor = processor
    state.tokenizer = processor.tokenizer
    state.device = device
    state.dtype = dtype
    state.vocab_size = int(cfg.text_config.vocab_size)
    state.eos_token_id = int(cfg.text_config.eos_token_id)

    parameter = next(model.parameters())
    raw_config = json.loads((Path(args.model) / "config.json").read_text(encoding=UTF8))
    identity, revision = snapshot_identity(args.model)
    state.info = {
        "contract_version": 1,
        "model_id": raw_config.get("_name_or_path") or identity or Path(args.model).name,
        "model_dir": str(Path(args.model)),
        "model_revision": report.get("revision") or revision or "",
        "engine": f"transformers {metadata.version('transformers')}",
        "dtype": str(parameter.dtype).removeprefix("torch."),
        "device": str(parameter.device),
        "vocab_size": state.vocab_size,
        "torch_version": torch.__version__,
        "plugin_version": metadata.version("torch_npu") if device.startswith("npu") else "",
        "driver_version": ascend_version(DRIVER_VERSION_FILE),
        "toolkit_version": ascend_version(TOOLKIT_VERSION_FILE, TOOLKIT_LATEST_DIR),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "load_report": report,
    }

    state.engine = GenerationEngine(
        model, parameter.device, dtype,
        eos_token_id=state.eos_token_id,
        pad_token_id=processor.tokenizer.pad_token_id,
        max_batch=int(os.environ.get("KOT_MAX_BATCH", 16)),
        vision_images=int(os.environ.get("KOT_VISION_IMAGES", 4)),
        vision_tiles=int(os.environ.get("KOT_VISION_TILES", 24)),
        batch_window=float(os.environ.get("KOT_BATCH_WINDOW", 0.02)),
        compile_model=bool(int(os.environ.get("KOT_COMPILE", 0))),
        use_graphs=bool(int(os.environ.get("KOT_GRAPHS", 0))),
    )
    warmup()
    print(f"[ocr-deepseek-ocr-2] serving on {HOST}:{args.port}: "
          f"{ {k: state.info[k] for k in ('model_id', 'engine', 'dtype', 'device', 'vocab_size')} }",
          flush=True)

    async def run_server():
        state.loop = asyncio.get_running_loop()
        state.pool = ThreadPoolExecutor(PREPROCESS_WORKERS)
        server = await asyncio.start_server(serve_connection, HOST, args.port, backlog=4096)
        await server.serve_forever()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        pass
    finally:
        state.engine.stop()


if __name__ == "__main__":
    main()
