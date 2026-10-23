#!/usr/bin/env python3
"""DeepSeek-OCR-2 serving of DEPLOYMENT_CONTRACT.md, no kot imports.

    ./kot_serve.sh --port P --model /path/to/DeepSeek-OCR-2 --device cpu|cuda

Layout: this file beside the three builder modules it imports and never
implements - native_config (build_config), native_load (load_native),
native_processor (build_processor) - serving a weights directory that is a
hub snapshot plus two patched files (run-book.md holds the list and the
hashes). The checkpoint loads through transformers 5.20.2's built-in
deepseek_ocr2 architecture; the checkpoint's own remote code is not used.

Numerics contract, implemented exactly as our reference server's deepseek
adapter does it:

* prefill is teacher forced: the transcript's ids are appended verbatim to the
  template the model builds from the image ("<|User|>:\\n<image>\\nFree OCR."
  in the checkpoint's own plain format, one 257-token image block), the
  forward runs over both, and the rows returned are
  logits[prompt_len : prompt_len + len(transcript) - 1], so a transcript of n
  ids stores n - 1 rows and row t predicts transcript[t + 1];
* the ids sent are consumed verbatim: no BOS on the transcript, no chat
  template of ours, no truncation;
* the log-softmax is computed in float32 over the model's configured
  vocabulary (129280, the native config's text_config.vocab_size), whatever
  dtype the weights run in, so every row's logsumexp is 0 and arrays align
  by token id;
* decode generates from the image template only (the sent ids are the frozen
  transcript, which decode does not use), applying the request's sampling
  object exactly as given (temperature, top_p, top_k, min_p; generate() takes
  all four natively), seed planted immediately before the call,
  repetition_penalty=1.0, and the stored logits are the raw ones before any
  warper (output_logits=True gives those); min_new_tokens pins exactly
  max_new_tokens rows;
* dtype: the weights run in bfloat16 on cuda and float32 on cpu; the dtype
  the log-probs were produced in is reported in /kot/info and kot casts to
  float32 on its side.

The image path is the native model's own contract, not the remote code's.
The processor (build_processor's output) turns one image into the padded
1024x1024 global view, a 2..6-tile grid of 768x768 crops when a side exceeds
768 px (no local tensor otherwise), and the per-image tile count. The forward
takes input_ids, pixel_values [1,3,1024,1024], pixel_values_local
[tiles,3,768,768] or None, num_local_patches and attention_mask - no images
list, no spatial crop, no position mask: the model finds the image positions
itself (input_ids == 128815) and scatters the features in its own order,
local tiles first, then the 256 global features, then the view separator.
So the block a request's single image placeholder expands to is
257 + 144 * tiles copies of 128815: every position holds the same id, the
order inside the block is irrelevant, and the count is the reference's.

log_probs are written row-streamed through numpy.lib.format.open_memmap: the
file opens at its final shape and fills chunk by chunk, so a long unit never
holds a whole distribution in host memory.
"""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import set_seed

from native_config import build_config
from native_load import load_native
from native_processor import build_processor

HOST = "127.0.0.1"
INFO = "/kot/info"
SCORE = "/kot/score"
PREFILL = "prefill"
UTF8 = "utf-8"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 16
BACKLOG = 1024

IMAGE = "image"
IMAGE_TAG = "<image>"
PROMPT_TEXT = "Free OCR."  # documented: what the frozen template_ids spell
IMAGE_TOKEN_ID = 128815
BOS_ID = 0
PATCH_SIZE = 16
DOWNSAMPLE = 4
BASE_SIZE = 1024  # the global view: every image padded to 1024x1024
TILE_SIZE = 768  # the local tiles: 768x768, and the single-view limit
CROP_QUERIES = math.ceil((TILE_SIZE // PATCH_SIZE) / DOWNSAMPLE)
IMAGE_IDS = [IMAGE_TOKEN_ID] * (
    math.ceil((BASE_SIZE // PATCH_SIZE) / DOWNSAMPLE) ** 2 + 1
)
WRITE_CHUNK = 64  # rows of log-softmax per streaming write
ROUTES: dict = {}

# DeepSeek's own anti-repetition rule for OCR outputs (their repo's
# DeepSeek-OCR2-vllm run scripts and sglang's native defaults agree on the
# values): a token that would complete an n-gram that already occurred
# within the lookback window is banned from sampling, except the table
# tokens, so structure may repeat. The rows kot stores are the raw
# pre-processor logits; this rule changes which token is sampled, never
# what is reported.
NGRAM_SIZE = 30
NGRAM_WINDOW = 90
NGRAM_WHITELIST = {128821, 128822}  # <td>, </td>


class NoRepeatNGramLogitsProcessor:
    """The rule above, in transformers' logits-processor shape."""

    def __init__(self, ngram_size: int, window_size: int, whitelist_token_ids: set):
        self.ngram_size = ngram_size
        self.window_size = window_size
        self.whitelist_token_ids = whitelist_token_ids

    def __call__(self, input_ids, scores):
        ids = input_ids[0].tolist() if torch.is_tensor(input_ids) else list(input_ids)
        if len(ids) < self.ngram_size:
            return scores
        current_prefix = tuple(ids[-(self.ngram_size - 1):])
        search_start = max(0, len(ids) - self.window_size)
        search_end = len(ids) - self.ngram_size + 1
        banned = set()
        for index in range(search_start, search_end):
            ngram = tuple(ids[index:index + self.ngram_size])
            if ngram[:-1] == current_prefix:
                banned.add(ngram[-1])
        banned -= self.whitelist_token_ids
        if banned:
            scores = scores.clone()
            for token in banned:
                # transformers hands the processor [batch, vocab] scores;
                # DeepSeek's own class indexes a flat per-request vector.
                scores[:, token] = -float("inf")
        return scores

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p")


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
    return {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", 0)),
        "min_p": float(sampling.get("min_p", 0.0)),
    }


def route(path):
    def register(fn):
        ROUTES[path] = fn
        return fn

    return register


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    attention = "sdpa" if device.startswith("cuda") else "eager"

    if not (Path(args.model) / "config.json").is_file():
        fail(f"{args.model} holds no config.json; this deployment loads a checkpoint directory")
    cfg = build_config(args.model)
    # The device's dtype and attention reach the loader through the config,
    # the two fields from_pretrained would set from its own keyword
    # arguments.
    cfg.dtype = dtype
    cfg._attn_implementation = attention
    model, report = load_native(args.model, cfg)
    processor = build_processor(args.model)
    # The loader owns what it built; the device is this server's own concern,
    # and the move is a no-op when the loader already placed and cast the
    # weights.
    model = model.eval().to(device=device, dtype=dtype)

    vocab_size = int(cfg.text_config.vocab_size)
    parameter = next(model.parameters())
    info = {
        "contract_version": 1,
        "model_id": Path(args.model).name,
        "model_dir": str(Path(args.model)),
        "model_revision": report["revision"] or "",
        "engine": f"transformers {metadata.version('transformers')}",
        "dtype": str(parameter.dtype).removeprefix("torch."),
        "device": str(parameter.device),
        "vocab_size": vocab_size,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        # What load_native loaded and checked, stamped verbatim.
        "load_report": report,
    }
    print(f"[ocr-deepseek-ocr-2] serving on {HOST}:{args.port}: {info}", flush=True)

    def views(image_path: str):
        """The image as the native processor encodes it.

        Returns the global-view pixels, the local-tile pixels (None when the
        image stayed single-view), the per-image tile count, and the
        image-token block: 257 tokens for the padded global view plus 144
        per tile. The native forward takes the pixels and the count and
        finds the image positions itself, so the block is only a count of
        128815 - and the count is the reference's.
        """
        from PIL import Image

        image = Image.open(image_path).convert("RGB")
        features = processor.image_processor(images=[image], return_tensors="pt")
        local = features.get("pixel_values_local")
        tiles = int(features["num_local_patches"][0])
        return (
            features["pixel_values"].to(device=device, dtype=dtype),
            None if local is None else local.to(device=device, dtype=dtype),
            tiles,
            [IMAGE_TOKEN_ID] * (len(IMAGE_IDS) + CROP_QUERIES**2 * tiles),
        )

    def template(raw_ids: list[int], image_tokens: list[int]) -> list[int]:
        """The expanded prompt template.

        The raw ids the request carried (the version's frozen template) hold
        one image placeholder; this replaces it with the block the image
        expands to, so the deployment consumes ids and never tokenizes. The
        native forward marks the image positions from the ids themselves, so
        no position mask is built.
        """
        at = raw_ids.index(IMAGE_TOKEN_ID)
        return [*raw_ids[:at], *image_tokens, *raw_ids[at + 1:]]

    def raw_template_ids_from(body: dict) -> list[int]:
        """The version's frozen template ids the request carried, checked present."""
        template_ids = [int(token) for token in body["input_ids"]]
        if not template_ids:
            fail("empty input_ids: the version's frozen template is missing")
        return template_ids

    def encode(
        image_path: str, raw_ids: list[int], continuation: list[int] | None
    ) -> tuple[dict[str, Any], dict[str, Any], list[int], int]:
        """One image as the forward and generate kwargs, plus where the template ends."""
        pixel_values, pixel_values_local, tiles, image_tokens = views(image_path)
        tpl = template(raw_ids, image_tokens)
        ids = torch.tensor([[*tpl, *(continuation or [])]], dtype=torch.long, device=device)
        shared = {
            "pixel_values": pixel_values,
            "pixel_values_local": pixel_values_local,
            "num_local_patches": [tiles],
        }
        forward = {
            **shared,
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "use_cache": False,
        }
        generate = {
            **shared,
            "inputs": ids[:, : len(tpl)],
            "attention_mask": torch.ones_like(ids[:, : len(tpl)]),
        }
        return forward, generate, tpl, len(tpl)

    def write_log_probs(logits: torch.Tensor, out_dir: Path, n_rows: int) -> str:
        """Stream the log-softmax rows into log_probs.npy at its final shape.

        The log-softmax is per row, so chunking the computation changes
        nothing about the values; open_memmap opens the file at the final
        shape and fills it chunk by chunk, so a long unit's distribution
        never sits whole in host memory.
        """
        path = out_dir / "log_probs.npy"
        array = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(n_rows, vocab_size)
        )
        for start in range(0, n_rows, WRITE_CHUNK):
            stop = min(start + WRITE_CHUNK, n_rows)
            chunk = torch.log_softmax(logits[start:stop, :vocab_size].float(), dim=-1)
            array[start:stop] = chunk.cpu().numpy()
        array.flush()
        del array
        return path

    def teacher_forced(image_path: str, incoming: list[int], n_rows: int, out_dir: Path) -> int:
        if n_rows < 1:
            fail(f"the ocr prefill stores the decode's rows: set n_rows, got {n_rows}")
        raw_ids, tail = incoming[: len(incoming) - n_rows], incoming[len(incoming) - n_rows:]
        forward, _, _, start = encode(image_path, raw_ids, tail)
        with torch.no_grad():
            logits = model(**forward).logits[0]
        # Rows predict the tail's tokens: the tail sits at positions
        # start..start+n_rows-1 of the expanded prompt, so its predictors
        # are the n_rows rows from start - 1.
        rows = logits[start - 1 : start - 1 + n_rows]
        write_log_probs(rows, out_dir, rows.shape[0])
        return int(rows.shape[0])

    def generated(
        image_path: str,
        template_ids: list[int],
        max_new_tokens: int,
        seed: int,
        sampling: dict,
        allow_eos: bool,
        out_dir: Path,
    ):
        """Sample under the request's sampling record; the stored rows are the
        raw pre-processor logits. ``allow_eos`` shapes the decode gate exactly
        as the contract states: false demands exactly ``max_new_tokens`` ids
        (EOS suppressed via min_new_tokens), true accepts what the sampler
        emitted before EOS, at most that many.
        """
        _, gen, _, prompt_len = encode(image_path, template_ids, None)
        set_seed(seed)
        with torch.no_grad():
            out = model.generate(
                **gen,
                do_sample=True,
                temperature=sampling["temperature"],
                top_k=sampling["top_k"],
                top_p=sampling["top_p"],
                min_p=sampling["min_p"],
                repetition_penalty=1.0,
                min_new_tokens=0 if allow_eos else max_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
                logits_processor=[
                    NoRepeatNGramLogitsProcessor(NGRAM_SIZE, NGRAM_WINDOW, NGRAM_WHITELIST)
                ],
            )
        emitted = out.sequences[0, prompt_len:].tolist()
        write_log_probs(torch.cat(list(out.logits), dim=0), out_dir, len(emitted))
        text = processor.tokenizer.decode(emitted, skip_special_tokens=True)
        return emitted, text

    @route(SCORE)
    def score(body: dict) -> dict:
        image_path = body["attachments"][IMAGE]
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if body["mode"] == PREFILL:
            transcript = [int(token) for token in body["input_ids"]]
            if not transcript:
                fail("prefill over an empty transcript: the frozen ids are missing")
            n_rows = teacher_forced(image_path, transcript, int(body.get("n_rows", 0)), out_dir)
            consumed, output_ids, text = transcript, [], None
        else:
            # sampling is carried by prefill too and ignored there; decode
            # applies every field exactly as given
            template_ids = raw_template_ids_from(body)
            output_ids, text = generated(
                image_path,
                template_ids,
                int(body["max_new_tokens"]),
                int(body.get("seed", DEFAULT_SEED)),
                resolve_sampling(body),
                bool(body.get("allow_eos", False)),
                out_dir,
            )
            # The stored ids are the frozen template plus the generated ids:
            # the pre-expansion sequence, so the replay of this file carries
            # exactly one image placeholder.
            consumed = [*template_ids, *output_ids]
            n_rows = len(output_ids)
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

    @route(INFO)
    def info_reply(_body: dict) -> dict:
        return info

    class Handler(BaseHTTPRequestHandler):
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
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                self._json(HTTPStatus.OK, handler(body))
            except Exception as failure:
                self._error(HTTPStatus.INTERNAL_SERVER_ERROR, type(failure).__name__, str(failure))

        def log_request(self, code="-", size="-"):
            if str(code) != str(HTTPStatus.OK.value):
                super().log_request(code, size)

        def _json(self, status: HTTPStatus, payload: dict):
            body = json.dumps(payload).encode(UTF8)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: HTTPStatus, error: str, detail: str):
            self._json(status, {"error": error, "detail": detail})

    HTTPServer.request_queue_size = BACKLOG
    HTTPServer((HOST, args.port), Handler).serve_forever()


def fail(message: str):
    raise RuntimeError(message)


if __name__ == "__main__":
    main()
