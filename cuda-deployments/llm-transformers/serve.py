#!/usr/bin/env python3
"""Stock-transformers serving of DEPLOYMENT_CONTRACT.md, one flat file, no kot imports.

    ./kot_serve.sh --port P --model /path/to/weights --device cpu|cuda

Numerics contract, implemented exactly:

* the ids sent to /kot/score are consumed verbatim: no BOS, no chat template,
  no truncation, no re-tokenization. Row t of a prefill is the prediction for
  input_ids[t+1], so a prefill over n ids writes n - 1 rows, streamed to disk
  chunk by chunk: the ~8,600-id replay's distribution never sits whole in
  memory;
* the log-softmax is computed in float32 over the model's configured
  vocabulary, whatever dtype the weights run in, so every row's logsumexp is 0
  and arrays align by token id;
* decode applies the request's sampling object exactly as given
  (temperature, top_p, top_k, min_p, repetition_penalty; generate() takes all
  five natively), seed planted immediately before the call, and the stored
  logits are the raw ones before any warper (output_logits=True gives those).
  presence_penalty is an OpenAI-surface knob with no generate() equivalent,
  so a request that sets it above 0 is refused naming the field rather than
  silently ignoring it;
* dtype: the weights run in bfloat16 on cuda and float32 on cpu; the dtype
  the log-probs were produced in is reported in /kot/info and kot casts to
  float32 on its side;
* serving: one asyncio event loop; heavy endpoints run one at a time on a
  single executor thread, so a pooled idle connection costs one file
  descriptor and model calls stay strictly serialized.
"""

from __future__ import annotations

import asyncio
import argparse
import json
import os
import resource
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, set_seed

HOST = "127.0.0.1"
INFO = "/kot/info"
SCORE = "/kot/score"
INFER = "/kot/infer"
COMPLETIONS = "/v1/completions"
CHAT_COMPLETIONS = "/v1/chat/completions"
PREFILL = "prefill"
DECODE = "decode"
UTF8 = "utf-8"
LATIN1 = "latin-1"
CONFIG_NAME = "config.json"
HUB = "hub"
SNAPSHOTS = "snapshots"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 16
BACKLOG = 1024
IDLE_TIMEOUT = 300
BODY_TIMEOUT = 120
REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed", 500: "Internal Server Error"}

ROUTES: dict = {}


class StreamReply:
    """A completion answered as an SSE stream of OpenAI-shaped chunks.

    The route submits the generation to the single model thread and returns
    this marker; the streamer yields text as tokens arrive, and a pump
    thread bridges the blocking iterator into the event loop's queue. The
    serialized model thread is the deployment's own serving reality: one
    generation at a time, whatever many connections are held open.
    """

    def __init__(self, streamer, future, model: str, prompt_tokens: int,
                 max_tokens: int, chat: bool):
        self.streamer = streamer
        self.future = future
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.chat = chat
        self.id = (f"chatcmpl-{uuid4().hex}" if chat else f"cmpl-{uuid4().hex()}")
        self.created = int(datetime.now(timezone.utc).timestamp())

    def chunk(self, delta: dict, finish_reason: str | None) -> bytes:
        choice: dict = {"index": 0, "delta": delta, "finish_reason": finish_reason}
        payload = {
            "id": self.id,
            "object": "chat.completion.chunk" if self.chat else "text_completion",
            "created": self.created,
            "model": self.model,
            "choices": [choice],
        }
        return f"data: {json.dumps(payload)}\n\n".encode(UTF8)


async def write_stream(reply: StreamReply, writer) -> None:
    """Serve one StreamReply: SSE head, deltas as they land, finish, [DONE]."""
    head = (
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: keep-alive\r\n\r\n"
    ).encode(LATIN1)
    writer.write(head)
    await asyncio.wait_for(writer.drain(), BODY_TIMEOUT)

    async def send(chunk: bytes) -> None:
        writer.write(chunk)
        await asyncio.wait_for(writer.drain(), BODY_TIMEOUT)

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def pump():
        try:
            for delta in reply.streamer:
                loop.call_soon_threadsafe(queue.put_nowait, delta)
        except Exception:
            pass
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    threading.Thread(target=pump, daemon=True).start()
    first = {"role": "assistant", "content": ""} if reply.chat else {}
    await send(reply.chunk(first, None))
    try:
        while True:
            delta = await queue.get()
            if delta is None:
                break
            key = "content" if reply.chat else "text"
            await send(reply.chunk({key: delta}, None))
        generated = await asyncio.wrap_future(reply.future)
        finish = "length" if len(generated) >= reply.max_tokens else "stop"
        await send(reply.chunk({}, finish))
        await send(b"data: [DONE]\n\n")
    except (ConnectionError, asyncio.TimeoutError):
        return
    except Exception as failure:
        # The 200 and the content type are already on the wire: the failure
        # travels as an event, then the connection closes.
        print(f"[llm-transformers] stream failed: {failure}", file=sys.stderr, flush=True)
        try:
            payload = {"error": type(failure).__name__, "detail": str(failure)}
            writer.write(b"event: error\n")
            writer.write(f"data: {json.dumps(payload)}\n\n".encode(UTF8))
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()
        except ConnectionError:
            pass


def route(path):
    def register(fn):
        ROUTES[path] = fn
        return fn

    return register


def resolve_weights(model: str) -> tuple[Path, str]:
    """The checkpoint to load: the directory itself, or the one snapshot in an HF home.

    The capture mounts the weights at --model. A handed --model-dir capture
    mounts a checkpoint whose root has config.json; a capture without one
    mounts the huggingface home, where the model id is not passed to this
    script at all, so exactly one repository is expected under hub/.
    """
    root = Path(model)
    if (root / CONFIG_NAME).is_file():
        return root, root.name
    hub = root / HUB
    if not hub.is_dir():
        fail(f"{root} holds neither config.json nor a {HUB}/ directory")
    snapshots = sorted(hub.glob(f"models--*/{SNAPSHOTS}/*/"))
    if not snapshots:
        fail(f"no {HUB}/models--*/{SNAPSHOTS}/ under {root}")
    if len(snapshots) > 1:
        named = ", ".join(str(s) for s in snapshots)
        fail(f"the cache {root} holds several models and this script was told none: {named}")
    snapshot = snapshots[0]
    name = snapshot.parts[-4].removeprefix("models--").replace("--", "/")
    return snapshot, name


def fail(message: str):
    print(f"[llm-transformers] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def log_softmax(logits: torch.Tensor, vocab_size: int) -> np.ndarray:
    """Float32 log-softmax over the configured vocabulary, sliced of any padding."""
    rows = torch.log_softmax(logits[:, :vocab_size].float(), dim=-1)
    return np.ascontiguousarray(rows.cpu().numpy(), dtype=np.float32)


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
WRITE_CHUNK = 64  # rows of log-softmax per streaming write


def resolve_sampling(body: dict) -> dict:
    """The request's sampling object in generate() terms, every field applied
    exactly as given. Prefill carries the object too and samples nothing, so
    only the decode path resolves it; a field this engine cannot apply is an
    error naming it, never a silently ignored value. Missing fields fall back
    to their neutral values, so a bare request still serves.
    """
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ValueError("unsupported sampling field(s): " + ", ".join(unsupported))
    presence = float(sampling.get("presence_penalty", 0.0))
    if presence:
        raise ValueError(
            "presence_penalty is unsupported on the transformers engine: it is "
            "an OpenAI-surface knob (vllm and sglang apply it; generate() has none)"
        )
    return {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", 0)),
        "min_p": float(sampling.get("min_p", 0.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
    }


def write_log_probs(logits: torch.Tensor, out_dir: Path, n_rows: int, vocab_size: int) -> None:
    """Stream the log-softmax rows into log_probs.npy at its final shape.

    The log-softmax is per row, so chunking the computation changes nothing
    about the values; open_memmap opens the file at the final shape and fills
    it chunk by chunk, so a long prefill's distribution never sits whole in
    host memory.
    """
    array = np.lib.format.open_memmap(
        out_dir / "log_probs.npy", mode="w+", dtype=np.float32, shape=(n_rows, vocab_size)
    )
    for start in range(0, n_rows, WRITE_CHUNK):
        stop = min(start + WRITE_CHUNK, n_rows)
        chunk = torch.log_softmax(logits[start:stop, :vocab_size].float(), dim=-1)
        array[start:stop] = chunk.cpu().numpy()
    array.flush()
    del array


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    _soft_files, hard_files = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_files, hard_files))

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    attention = "sdpa" if device.startswith("cuda") else "eager"
    spread = {"device_map": "auto"} if device.startswith("cuda") else {}

    weights, model_id = resolve_weights(args.model)
    try:
        tokenizer = AutoTokenizer.from_pretrained(weights)
        model = AutoModelForCausalLM.from_pretrained(
            weights,
            dtype=dtype,
            attn_implementation=attention,
            **spread,
        )
    except ValueError as error:
        if "trust_remote_code" in str(error):
            print(
                "[llm-transformers] no native architecture for this model; "
                "loading the model's own code",
                flush=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                weights,
                dtype=dtype,
                trust_remote_code=True,
                attn_implementation=attention,
                **spread,
            )
        elif "attention implementation" in str(error):
            print(
                "[llm-transformers] this architecture dispatches no attention "
                "(an SSM or similar); loading eager",
                flush=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(weights)
            model = AutoModelForCausalLM.from_pretrained(
                weights,
                dtype=dtype,
                attn_implementation="eager",
                **spread,
            )
        else:
            raise
    if not spread:
        model = model.to(device)
    model = model.eval()
    device = str(model.device)
    vocab_size = int(model.config.vocab_size)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    parameter = next(model.parameters())
    executor = ThreadPoolExecutor(max_workers=1)
    info = {
        "contract_version": 1,
        "model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None) or "",
        "tokenizer_id": str(tokenizer.name_or_path),
        "engine": f"transformers {metadata.version('transformers')}",
        "dtype": str(parameter.dtype).removeprefix("torch."),
        "device": "+".join(sorted({str(p.device) for p in model.parameters()})),
        "vocab_size": vocab_size,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[llm-transformers] serving on {HOST}:{args.port}: {info}", flush=True)

    def generate_ids(
        prompt_ids: list[int], max_new_tokens: int, seed: int, sampling: dict,
        min_new_tokens: int | None = None, streamer: TextIteratorStreamer | None = None,
    ):
        """Sample with the request's sampling knobs; the raw per-step logits
        are returned before any warper touched them. The score path forces
        exactly max_new_tokens (min_new_tokens equal to it); the chat surface
        applies only what the request carried."""
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        set_seed(seed)
        with torch.no_grad():
            out = model.generate(
                input_ids=ids,
                attention_mask=torch.ones_like(ids),
                do_sample=True,
                temperature=sampling["temperature"],
                top_k=sampling["top_k"],
                top_p=sampling["top_p"],
                min_p=sampling["min_p"],
                repetition_penalty=sampling["repetition_penalty"],
                min_new_tokens=max_new_tokens if min_new_tokens is None else min_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=pad_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
                streamer=streamer,
            )
        generated = out.sequences[0, ids.shape[1] :].tolist()
        return generated, torch.cat(list(out.logits), dim=0)

    def prefill(prompt_ids: list[int], out_dir: Path) -> int:
        ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
        rows = logits[:-1]
        write_log_probs(rows, out_dir, int(rows.shape[0]), vocab_size)
        return int(rows.shape[0])

    def decode(prompt_ids: list[int], max_new_tokens: int, seed: int, sampling: dict):
        generated, logits = generate_ids(prompt_ids, max_new_tokens, seed, sampling)
        rows = log_softmax(logits, vocab_size)
        consumed = [*prompt_ids, *generated]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return rows, consumed, generated, text

    @route(SCORE)
    def score(body: dict) -> dict:
        prompt_ids = [int(token) for token in body["input_ids"]]
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if body["mode"] == PREFILL:
            # sampling is carried but ignored: prefill is teacher forced
            n_rows = prefill(prompt_ids, out_dir)
            consumed, output_ids, text = prompt_ids, [], None
        else:
            rows, consumed, output_ids, text = decode(
                prompt_ids,
                int(body["max_new_tokens"]),
                int(body.get("seed", DEFAULT_SEED)),
                resolve_sampling(body),
            )
            n_rows = int(rows.shape[0])
            np.save(out_dir / "log_probs.npy", rows, allow_pickle=False)
        np.save(out_dir / "input_ids.npy", np.asarray(consumed, dtype=np.int32), allow_pickle=False)
        if text is not None:
            (out_dir / "text.txt").write_text(text, encoding=UTF8)
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": n_rows,
            "vocab_size": vocab_size,
            "output_ids": list(output_ids),
        }

    @route(COMPLETIONS)
    def completions(body: dict) -> dict:
        prompt = body["prompt"]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return completion_reply(body, [int(token) for token in prompt], chat=False)
        if isinstance(prompt, list):
            prompt = "".join(str(part) for part in prompt)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        return completion_reply(body, list(prompt_ids), chat=False)

    @route(CHAT_COMPLETIONS)
    def chat_completions(body: dict) -> dict:
        ids = tokenizer.apply_chat_template(
            body["messages"], tokenize=True, add_generation_prompt=True
        )
        if hasattr(ids, "keys"):  # transformers 5: BatchEncoding is a UserDict, not a dict
            ids = ids["input_ids"]
        return completion_reply(body, [int(token) for token in ids], chat=True)

    def completion_reply(body: dict, prompt_ids: list[int], chat: bool):
        max_tokens = int(
            body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
        )
        # The harness sends temperature/top_p/top_k beside the seed; generate()
        # implements all of them (min_p too), and anything else is ignored,
        # never a rejection. min_tokens applies only when the request carries
        # it, like every other deployment's chat surface.
        sampling = {
            "temperature": float(body.get("temperature") or 1.0),
            "top_p": float(body.get("top_p", 1.0)),
            "top_k": int(body.get("top_k", 0)),
            "min_p": float(body.get("min_p", 0.0)),
            "repetition_penalty": float(body.get("repetition_penalty", 1.0)),
        }
        min_new_tokens = int(body.get("min_tokens") or 0)
        if body.get("stream"):
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            future = executor.submit(
                generate_ids, prompt_ids, max_tokens,
                int(body.get("seed") or DEFAULT_SEED), sampling,
                min_new_tokens, streamer,
            )
            return StreamReply(
                streamer, future, model_id, len(prompt_ids), max_tokens, chat
            )
        generated, _ = generate_ids(
            prompt_ids, max_tokens,
            int(body.get("seed") or DEFAULT_SEED), sampling,
            min_new_tokens,
        )
        text = tokenizer.decode(generated, skip_special_tokens=True)
        choice = {
            "index": 0,
            "finish_reason": "length" if len(generated) >= max_tokens else "stop",
        }
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {
            "id": f"chatcmpl-{uuid4().hex}" if chat else f"cmpl-{uuid4().hex}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": model_id,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt_ids) + len(generated),
            },
        }

    @route(INFO)
    def info_reply(_body: dict) -> dict:
        return info

    async def read_headers(reader: asyncio.StreamReader) -> tuple[str, str, str, dict]:
        request_line = await asyncio.wait_for(reader.readline(), IDLE_TIMEOUT)
        method, path, version = request_line.decode(LATIN1).strip().split(" ", 2)
        headers = {}
        while True:
            line = await asyncio.wait_for(reader.readline(), IDLE_TIMEOUT)
            if line in (b"\r\n", b"\n", b""):
                break
            name, _, value = line.decode(LATIN1).partition(":")
            headers[name.strip().lower()] = value.strip()
        return method, path, version, headers

    async def read_body(reader: asyncio.StreamReader, headers: dict) -> bytes:
        if headers.get("transfer-encoding", "").lower() == "chunked":
            body = b""
            while True:
                size = int((await reader.readline()).split(b";")[0], 16)
                if size == 0:
                    while True:
                        if (await reader.readline()) in (b"\r\n", b"\n", b""):
                            break
                    return body
                body += await asyncio.wait_for(reader.readexactly(size), BODY_TIMEOUT)
                await reader.readexactly(2)
        length = int(headers.get("content-length") or 0)
        return await asyncio.wait_for(reader.readexactly(length), BODY_TIMEOUT)

    async def reply(writer: asyncio.StreamWriter, status: int, payload: dict, keep: bool):
        body = json.dumps(payload).encode(UTF8)
        reason = REASONS[status]
        head = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            f"Connection: {'keep-alive' if keep else 'close'}\r\n\r\n"
        ).encode(LATIN1)
        writer.write(head + body)
        await asyncio.wait_for(writer.drain(), BODY_TIMEOUT)

    async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        loop = asyncio.get_running_loop()
        try:
            while True:
                try:
                    method, path, version, headers = await read_headers(reader)
                    raw = await read_body(reader, headers)
                except (asyncio.TimeoutError, ConnectionError):
                    return
                except Exception:
                    return
                keep = version == "HTTP/1.1" and "close" not in headers.get("connection", "").lower()
                if method == "GET" and path == INFO:
                    await reply(writer, 200, info, keep)
                    if keep:
                        continue
                    return
                if method != "POST" or path not in ROUTES:
                    await reply(writer, 404, {"error": "not_found", "detail": path}, keep)
                    if keep:
                        continue
                    return
                try:
                    body = json.loads(raw or b"{}")
                except json.JSONDecodeError as failure:
                    await reply(writer, 400, {"error": "invalid_json", "detail": str(failure)}, keep)
                    if keep:
                        continue
                    return
                try:
                    if body.get("stream") and path in (CHAT_COMPLETIONS, COMPLETIONS):
                        # Builds the stream and submits the generation to the
                        # model thread; no model work happens here, so the
                        # event loop is never blocked.
                        payload = ROUTES[path](body)
                    else:
                        payload = await loop.run_in_executor(executor, ROUTES[path], body)
                    if isinstance(payload, StreamReply):
                        await write_stream(payload, writer)
                    else:
                        await reply(writer, 200, payload, keep)
                except Exception as failure:
                    await reply(
                        writer,
                        500,
                        {"error": type(failure).__name__, "detail": str(failure)},
                        keep,
                    )
                if not keep:
                    return
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except ConnectionError:
                pass

    async def serve():
        server = await asyncio.start_server(handle_connection, HOST, args.port, backlog=BACKLOG)
        async with server:
            await server.serve_forever()

    asyncio.run(serve())


if __name__ == "__main__":
    main()
