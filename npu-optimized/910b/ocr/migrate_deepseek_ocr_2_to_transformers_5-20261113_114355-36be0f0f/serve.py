#!/usr/bin/env python3
"""DeepSeek-OCR-2 serving of DEPLOYMENT_CONTRACT.md, no kot imports.

    ./serve.sh --port P --model /path/to/DeepSeek-OCR-2 --device cpu|cuda|npu

Layout: this file beside the builder modules it imports - native_config
(build_config), native_load (load_native), native_processor (build_processor)
- and fast_moe (install_fast_experts). The checkpoint loads through
transformers' built-in deepseek_ocr2 architecture; the checkpoint's own
remote code is never used. The weights are exactly what --model names, cast
in place, never re-saved.

Contract numerics, the same rules the reference deployment implements:

* prefill is teacher forced: the transcript ids are appended verbatim to the
  request's template (one image placeholder expanded into the image's token
  block), the forward runs over both, and the rows returned are
  logits[start-1 : start-1+n_rows] - row t predicts tail[t]; the ids sent are
  consumed verbatim: no BOS of ours, no chat template, no truncation;
* the log-softmax is computed in float32 over the model's configured
  vocabulary (129280), whatever dtype the weights run in; every row's
  logsumexp is 0;
* decode applies the request's sampling record exactly as given, in the order
  transformers' generate() applies it: min-new-tokens EOS suppression (when
  allow_eos is false), then DeepSeek's own no-repeat-30-gram-in-90 rule (the
  two table tags excepted), then the temperature / top_k / top_p / min_p
  warpers, then softmax and one multinomial per row. The stored rows are the
  raw pre-processor logits: the log-softmax of the final logits before any
  processing;
* seed: planted immediately before the batch, exactly where the reference
  plants it before generate(); a batch of one starts its first draw from the
  same state the reference's first draw starts from;
* a decode that arrives with no ids builds the same template itself: BOS,
  the image placeholder, then the instruction's ids.

Throughput design - the speed harness drives the chat route with every
document at once, so batching is where this card's speed lives:

* one process, one loaded model. Every request - score or chat - becomes a
  unit in one queue, and the engine runs pending units as one batched
  prefill + decode over left-padded rows (a unit whose log-probs are
  measured runs alone, as the contract's one-at-a-time score load already
  guarantees). The vision tower runs once per image per batch, batched, with
  the same per-image tile counts and offsets the single-image path uses:
  the family rule (a batch with tile counts [2, 6] returns each image's rows
  from its own tiles) is the native forward's own scatter, driven by the
  same num_local_patches a batch of one sends;
* the decode step is a fixed-shape loop over a preallocated StaticCache:
  one forward per step for the whole batch, EOS and per-row limits tracked
  per row, tokens streamed out in blocks as they are produced;
* fast_moe replaces the experts' per-expert grouped-mm loop (1408 kernel
  launches per decode step on this card) with a dense all-experts
  formulation of the same routed math. Both endpoints run it: the score
  route measures the same kernels the chat route serves;
* the n-gram rule runs as device ops over the running batch - no per-token
  host sync;
* nothing is cached across requests: every request re-encodes its image
  through the same vision forward and re-runs the same LM. The only state
  kept between requests is the loaded weights.

The npu stack: torch_npu is imported inside main(), gated on --device npu,
before any tensor is placed; attention defaults to sdpa on npu (KOT_ATTN
overrides; eager remains available), bfloat16 weights (KOT_DTYPE overrides).
/kot/info reports what the process can read and invents nothing.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import os
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import metadata
from pathlib import Path

import torch
from PIL import Image, ImageOps

from fast_moe import install_fast_experts
from fast_vision import install_fast_rel_pos
from native_config import build_config
from native_load import load_native, snapshot_identity
from native_processor import build_processor, load_image

from engine import Engine, Unit

HOST = "127.0.0.1"
INFO = "/kot/info"
SCORE = "/kot/score"
CHAT_COMPLETIONS = "/v1/chat/completions"
PREFILL = "prefill"
UTF8 = "utf-8"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 16
BACKLOG = 4096

IMAGE = "image"
PROMPT_TEXT = "Free OCR."
IMAGE_TOKEN_ID = 128815
BOS_ID = 0

SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
)

ROUTES: dict = {}
DRIVER_VERSION_FILE = "/usr/local/Ascend/driver/version.info"
TOOLKIT_VERSION_FILE = "/usr/local/Ascend/ascend-toolkit/version.info"
TOOLKIT_LATEST_DIR = "/usr/local/Ascend/ascend-toolkit/latest"


def resolve_sampling(body: dict) -> dict:
    """The request's sampling object, every field applied exactly as given.
    Missing fields fall back to their neutral values, so a bare request
    still serves; a field this engine cannot apply is an error, never a
    silently ignored value."""
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ValueError("unsupported sampling field(s): " + ", ".join(unsupported))
    resolved = {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", 0)),
        "min_p": float(sampling.get("min_p", 0.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
        "presence_penalty": float(sampling.get("presence_penalty", 0.0)),
    }
    if resolved["presence_penalty"] != 0.0:
        raise ValueError(
            "unsupported sampling value: presence_penalty must be 0 (the engine has no "
            "presence_penalty knob, and a value it cannot apply is an error, never ignored)"
        )
    return resolved


def ascend_version(version_file: str, latest_dir: str = "") -> str:
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


def route(path):
    def register(fn):
        ROUTES[path] = fn
        return fn

    return register


def fail(message: str):
    raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    device = args.device
    if device.startswith("npu"):
        import torch_npu  # noqa: F401

        dtype = torch.bfloat16
        attention = os.environ.get("KOT_ATTN", "sdpa")
    elif device.startswith("cuda"):
        dtype = torch.bfloat16
        attention = "sdpa"
    else:
        dtype = torch.float32
        attention = "eager"
    if os.environ.get("KOT_DTYPE"):
        dtype = getattr(torch, os.environ["KOT_DTYPE"])

    # this process only ever serves inference; no request needs gradients
    torch.set_grad_enabled(False)

    if not (Path(args.model) / "config.json").is_file():
        fail(f"{args.model} holds no config.json; this deployment loads a checkpoint directory")
    cfg = build_config(args.model)
    cfg.dtype = dtype
    cfg._attn_implementation = attention
    model, report = load_native(args.model, cfg)
    processor = build_processor(args.model)
    model = model.eval().to(device=device, dtype=dtype)
    # after the move: the experts' flattened GEMM operands inherit the
    # weights' final device and dtype (install_fast_experts reads them)
    install_fast_experts(model)
    install_fast_rel_pos(model)

    vocab_size = int(cfg.text_config.vocab_size)
    eos_id = int(cfg.text_config.eos_token_id)
    parameter = next(model.parameters())
    info = {
        "contract_version": 1,
        "model_id": snapshot_identity(args.model)[0] or Path(args.model).name,
        "model_dir": str(Path(args.model)),
        "model_revision": report["revision"] or "",
        "engine": f"transformers {metadata.version('transformers')}",
        "dtype": str(parameter.dtype).removeprefix("torch."),
        "device": str(parameter.device),
        "vocab_size": vocab_size,
        "torch_version": torch.__version__,
        "plugin_version": metadata.version("torch_npu") if device.startswith("npu") else "",
        "driver_version": ascend_version(DRIVER_VERSION_FILE),
        "toolkit_version": ascend_version(TOOLKIT_VERSION_FILE, TOOLKIT_LATEST_DIR),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "load_report": report,
    }
    print(f"[ocr-deepseek-ocr-2] serving on {HOST}:{args.port}: {info}", flush=True)

    engine = Engine(model, processor, device, dtype, vocab_size, eos_id)

    preproc_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="preproc")

    def prepare_unit(unit: Unit):
        """The image exactly as the processor encodes it: global pixels, tile
        pixels, tile count, and the expanded prompt for this unit's template."""
        features = processor.image_processor(images=[unit.image], return_tensors="pt")
        local = features.get("pixel_values_local")
        tiles = int(features["num_local_patches"][0])
        block = 257 + 144 * tiles
        prompt = expand_template(unit.prompt_ids, block)
        unit.enc = {
            "pixel_values": features["pixel_values"],
            "pixel_values_local": local,
            "tiles": tiles,
            "block": block,
            "prompt": prompt,
        }

    def expand_template(raw_ids, block_len):
        at = raw_ids.index(IMAGE_TOKEN_ID)
        return [*raw_ids[:at], *[IMAGE_TOKEN_ID] * block_len, *raw_ids[at + 1:]]

    def _prepare_and_enqueue(unit: Unit):
        try:
            prepare_unit(unit)
            engine.enqueue(unit)
        except Exception as error:
            unit.events.put(("error", f"{type(error).__name__}: {error}"))

    def submit(unit: Unit):
        preproc_pool.submit(_prepare_and_enqueue, unit)

    @route(SCORE)
    def score(body: dict) -> Unit:
        image_path = body["attachments"][IMAGE]
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        image = load_image(image_path)
        if body["mode"] == PREFILL:
            transcript = [int(token) for token in body["input_ids"]]
            if not transcript:
                fail("prefill over an empty transcript: the frozen ids are missing")
            n_rows = int(body.get("n_rows", 0))
            if n_rows < 1:
                fail(f"the ocr prefill stores the decode's rows: set n_rows, got {n_rows}")
            if n_rows > len(transcript):
                fail(f"n_rows {n_rows} exceeds the {len(transcript)} ids sent")
            raw_template = transcript[: len(transcript) - n_rows]
            unit = Unit(prompt_ids=raw_template, image=image, mode=PREFILL, max_new_tokens=0,
                        seed=0, sampling=None, allow_eos=False, need_logprobs=True,
                        out_dir=out_dir, prompt_id=str(body["prompt_id"]), n_rows=n_rows,
                        continuation=transcript)
        else:
            template_ids = raw_template_ids_from(body)
            unit = Unit(prompt_ids=template_ids, image=image, mode="decode",
                        max_new_tokens=int(body["max_new_tokens"]),
                        seed=int(body.get("seed", DEFAULT_SEED)),
                        sampling=resolve_sampling(body),
                        allow_eos=bool(body.get("allow_eos", False)),
                        need_logprobs=True, out_dir=out_dir,
                        prompt_id=str(body["prompt_id"]))
        submit(unit)
        return unit

    def raw_template_ids_from(body: dict) -> list[int]:
        template_ids = [int(token) for token in body["input_ids"]]
        if template_ids:
            return template_ids
        instruction = processor.tokenizer.encode(f"\n{PROMPT_TEXT}", add_special_tokens=False)
        return [BOS_ID, IMAGE_TOKEN_ID, *instruction]

    def chat_request(body: dict) -> Unit:
        messages = body.get("messages")
        if (not isinstance(messages, list) or len(messages) != 1
                or not isinstance(messages[0], dict) or messages[0].get("role") != "user"):
            raise ValueError("messages must contain exactly one user message")
        content = messages[0].get("content")
        if not isinstance(content, list):
            raise ValueError("the user message content must be a list with text and image_url parts")
        images, texts = [], []
        for part in content:
            if not isinstance(part, dict):
                raise ValueError("each content part must be an object")
            if part.get("type") == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ValueError("text parts must contain a string text")
                texts.append(text)
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url")
                if not isinstance(url, str):
                    raise ValueError("image_url must contain a URL")
                images.append(url)
            else:
                raise ValueError("content supports only text and image_url parts")
        if len(images) != 1:
            raise ValueError(f"expected exactly one image_url part, got {len(images)}")
        if not texts or not "".join(texts).strip():
            raise ValueError("the user message must contain text")
        url = images[0]
        if not url.startswith("data:image/") or ";base64," not in url:
            raise ValueError("image_url must be a data:image/...;base64,... URL")
        encoded = url.split(";base64,", 1)[1]
        try:
            with Image.open(io.BytesIO(base64.b64decode(encoded, validate=True))) as opened:
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (ValueError, binascii.Error, OSError) as error:
            raise ValueError("image_url contains invalid image data") from error

        sampling = {}
        for field in SAMPLING_FIELDS:
            if field in body:
                sampling[field] = body[field]
        if "frequency_penalty" in body and float(body["frequency_penalty"]) != 0.0:
            raise ValueError("unsupported non-neutral chat sampling field: frequency_penalty")
        instruction = processor.tokenizer.encode("\n" + "".join(texts), add_special_tokens=False)
        template_ids = [BOS_ID, IMAGE_TOKEN_ID, *instruction]
        max_tokens = int(body.get("max_completion_tokens", body.get("max_tokens", DEFAULT_MAX_TOKENS)))
        unit = Unit(prompt_ids=template_ids, image=image, mode="chat", max_new_tokens=max_tokens,
                    seed=int(body.get("seed", DEFAULT_SEED)),
                    sampling=resolve_sampling({"sampling": sampling}), allow_eos=True,
                    need_logprobs=False, out_dir=None, stream=bool(body.get("stream")))
        submit(unit)
        return unit

    @route(CHAT_COMPLETIONS)
    def chat_completions(body: dict):
        return chat_request(body)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path != INFO:
                self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)
                return
            self._json(HTTPStatus.OK, info)

        def do_POST(self):
            handler = ROUTES.get(self.path)
            if handler is None:
                self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length) or b"{}")
                result = handler(body)
                if isinstance(result, Unit):
                    self._serve_unit(result)
                else:
                    self._json(HTTPStatus.OK, result)
            except Exception as failure:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, type(failure).__name__, str(failure))

        def _serve_unit(self, unit: Unit):
            """Stream a unit's answer: NDJSON block lines then the result for a
            scored decode, SSE deltas for a streamed chat, one object otherwise."""
            stream = unit.mode == "chat" and unit.stream
            ndjson = unit.need_logprobs and unit.mode != PREFILL
            self.send_response(HTTPStatus.OK)
            if stream:
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
            elif ndjson:
                self.send_header("Content-Type", "application/x-ndjson")
            else:
                self.send_header("Content-Type", "application/json")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            try:
                while True:
                    kind, payload = unit.events.get()
                    if kind == "error":
                        # headers are already out; end the body with the error
                        # as one last line and close, rather than a second
                        # response the client cannot parse
                        if stream:
                            self.wfile.write(f"data: {json.dumps({'error': 'EngineError', 'detail': payload})}\n\n".encode(UTF8))
                        else:
                            self.wfile.write((json.dumps({"error": "EngineError", "detail": payload}) + "\n").encode(UTF8))
                        self.wfile.flush()
                        return
                    if kind == "block" and ndjson:
                        line = json.dumps({"block": payload[0], "rows": payload[1]}) + "\n"
                        self.wfile.write(line.encode(UTF8))
                        self.wfile.flush()
                    elif kind == "delta" and stream:
                        chunk = {
                            "id": "chatcmpl-" + uuid.uuid4().hex,
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now(timezone.utc).timestamp()),
                            "model": info["model_id"],
                            "choices": [{"index": 0, "delta": {"content": payload},
                                         "finish_reason": None}],
                        }
                        self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode(UTF8))
                        self.wfile.flush()
                    elif kind == "done":
                        self._finish_unit(unit, payload)
                        return
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _finish_unit(self, unit: Unit, payload: dict):
            if unit.mode == "chat":
                if unit.stream:
                    stop = {
                        "id": "chatcmpl-" + uuid.uuid4().hex,
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "model": info["model_id"],
                        "choices": [{"index": 0, "delta": {},
                                     "finish_reason": "stop" if payload["stop"] else "length"}],
                    }
                    self.wfile.write(f"data: {json.dumps(stop)}\n\n".encode(UTF8))
                    self.wfile.write(b"data: [DONE]\n\n")
                else:
                    prompt_tokens = len(unit.enc["prompt"]) if unit.enc else 0
                    reply = {
                        "id": "chatcmpl-" + uuid.uuid4().hex,
                        "object": "chat.completion",
                        "created": int(datetime.now(timezone.utc).timestamp()),
                        "model": info["model_id"],
                        "choices": [{"index": 0,
                                     "message": {"role": "assistant", "content": payload["text"]},
                                     "finish_reason": "stop" if payload["stop"] else "length"}],
                        "usage": {"prompt_tokens": prompt_tokens,
                                  "completion_tokens": len(unit.emitted),
                                  "total_tokens": prompt_tokens + len(unit.emitted)},
                    }
                    self.wfile.write(json.dumps(reply).encode(UTF8))
            elif unit.need_logprobs and unit.mode != PREFILL:
                self.wfile.write((json.dumps(payload["reply"]) + "\n").encode(UTF8))
            else:
                self.wfile.write(json.dumps(payload["reply"]).encode(UTF8))
            self.wfile.flush()

        def log_request(self, code="-", size="-"):
            if str(code) != str(HTTPStatus.OK.value):
                super().log_request(code, size)

        def _json(self, status: HTTPStatus, payload: dict):
            body = json.dumps(payload).encode(UTF8)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, error: str, detail: str):
            try:
                self._json(status, {"error": error, "detail": detail})
            except (BrokenPipeError, ConnectionResetError):
                pass

    ThreadingHTTPServer.request_queue_size = BACKLOG
    server = ThreadingHTTPServer((HOST, args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
