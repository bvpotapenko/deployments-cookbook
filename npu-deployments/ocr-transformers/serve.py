#!/usr/bin/env python3
"""DeepSeek-OCR-2 serving of DEPLOYMENT_CONTRACT.md, one flat file, no kot imports.

    ./kot_serve.sh --port P --model /path/to/DeepSeek-OCR-2 --device cpu|cuda|npu

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
  vocabulary (129280), whatever dtype the weights run in, so every row's
  logsumexp is 0 and arrays align by token id. The checkpoint's own forward
  also casts its logits to float, and we do not rely on that alone;
* decode generates from the image template only (the sent ids are the frozen
  transcript, which decode does not use), applying the request's sampling
  object exactly as given, seed planted immediately before the call, and the
  stored logits are the raw ones before any warper (output_logits=True gives
  those); min_new_tokens pins exactly max_new_tokens rows. The sampling
  object carries six fields now: the contract's four (temperature, top_p,
  top_k, min_p; generate() takes all four natively) plus the two the
  Sampling record grew later, whose neutral defaults are "what ran before
  the file existed" (kot/store/schema.py) - the current kot client sends all
  six on every decode. repetition_penalty goes to generate() as given (1.0
  adds no warper there, so the neutral default computes exactly as the
  golden did, whose requests predate the field); presence_penalty has no
  knob in transformers 4.46.3's generate(), so a non-neutral value is an
  error naming it, never silently ignored, and the neutral 0.0 simply
  applies nothing - exactly what the value asks for;
* dtype: the weights run in bfloat16 on cuda and npu and float32 on cpu
  (KOT_DTYPE overrides, a typo fails loudly); the dtype the log-probs were
  produced in is reported in /kot/info and kot casts to float32 on its side.

The npu stack, added over the cuda deployment this ports:

* torch_npu is imported inside main(), gated on --device npu, before any
  tensor is placed: the import registers the accelerator backend.
* attention defaults to eager on npu, KOT_ATTN overrides. eager is the
  only MLA attention the vanilla checkpoint's LLM half implements (its
  ATTENTION_CLASSES carries eager and flash only and it never declares
  _supports_sdpa, so transformers 4.46.3 refuses an explicit sdpa request
  outright); the SAM vision encoder calls
  F.scaled_dot_product_attention unconditionally on every platform, so
  the env var switches the LLM half only and the image path stays the
  same on cuda and npu.
* the Tensor.cuda() redirect the cpu path already needs also carries the
  checkpoint's own .cuda() calls to npu;
* /kot/info reports the stack the process can read (torch, the torch_npu
  plugin, the driver and toolkit version files) and leaves empty what it
  cannot, never inventing one.

The image path mirrors the checkpoint's own infer() exactly, both branches:

* single view (both sides <= 768 px): the image is padded to 1024x1024 and
  encoded as one 16*16+1 = 257-token block, with a zero crop tensor;
* crop view (either side > 768 px): the checkpoint's own dynamic_preprocess
  picks a 2..6-crop grid of 768x768 tiles, the token block grows by
  144 * width_crops * height_crops tokens, and the forward receives the crop
  stack beside the padded global view, with images_spatial_crop = [w, h].

log_probs are written row-streamed through numpy.lib.format.open_memmap: the
file opens at its final shape and fills chunk by chunk, so a long unit never
holds a whole distribution in host memory.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import metadata
from itertools import repeat
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, set_seed

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
BASE_SIZE = 1024
MAX_SINGLE_VIEW = 768
CROP_QUERIES = math.ceil((MAX_SINGLE_VIEW // PATCH_SIZE) / DOWNSAMPLE)
NORM = (0.5, 0.5, 0.5)
RGB_MAX = 255
IMAGE_IDS = [IMAGE_TOKEN_ID] * (
    math.ceil((BASE_SIZE // PATCH_SIZE) / DOWNSAMPLE) ** 2 + 1
)
WRITE_CHUNK = 64  # rows of log-softmax per streaming write
ROUTES: dict = {}

# Ascend version files, as the CANN image and the driver mount lay them out.
# Read best-effort; a file that is not there reports "".
DRIVER_VERSION_FILE = "/usr/local/Ascend/driver/version.info"
TOOLKIT_VERSION_FILE = "/usr/local/Ascend/ascend-toolkit/version.info"
TOOLKIT_LATEST_DIR = "/usr/local/Ascend/ascend-toolkit/latest"

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

# The six the current kot client sends on every decode (Sampling.model_dump,
# kot/store/schema.py): the contract's four plus the two the Sampling record
# grew later, with the neutral defaults that ran before the file existed.
SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
)


def resolve_sampling(body: dict) -> dict:
    """The request's sampling object in generate() terms, every field applied
    exactly as given. Prefill carries the object too and samples nothing, so
    only the decode path resolves it; a field this engine cannot apply is an
    error naming it, never a silently ignored value. Missing fields fall back
    to their neutral values, so a bare request still serves.

    presence_penalty is the one field generate() has no knob for in
    transformers 4.46.3: passing it there would warn and silently ignore,
    so a non-neutral value is refused here and the neutral 0.0 applies
    nothing - which is exactly what the value asks for.
    """
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
            "unsupported sampling value: presence_penalty must be 0 (transformers "
            "4.46.3's generate() has no presence_penalty knob, and a value this "
            "engine cannot apply is an error, never ignored)"
        )
    return resolved


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
    if device.startswith("npu"):
        # Registers the npu backend; must precede every tensor placement.
        import torch_npu  # noqa: F401

        dtype = torch.bfloat16
        # eager: the only MLA attention the vanilla checkpoint implements
        # (its ATTENTION_CLASSES has no sdpa entry, and transformers 4.46.3
        # refuses an explicit sdpa request for a class without
        # _supports_sdpa). KOT_ATTN overrides for experiments.
        attention = os.environ.get("KOT_ATTN", "eager")
    elif device.startswith("cuda"):
        dtype = torch.bfloat16
        attention = "sdpa"
    else:
        dtype = torch.float32
        attention = "eager"
    if os.environ.get("KOT_DTYPE"):
        dtype = getattr(torch, os.environ["KOT_DTYPE"])

    if not (Path(args.model) / "config.json").is_file():
        fail(f"{args.model} holds no config.json; this deployment loads a checkpoint directory")
    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=dtype,
        _attn_implementation=attention,
        trust_remote_code=True,
        use_safetensors=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = model.eval().to(device)

    # The checkpoint's remote code calls Tensor.cuda(); on any other device,
    # redirect it so the same code runs there.
    if not device.startswith("cuda"):
        torch.Tensor.cuda = lambda tensor, *_args, **_kwargs: tensor.to(device)

    remote = sys.modules[type(model).__module__]
    vocab_size = int(model.config.vocab_size)

    parameter = next(model.parameters())
    info = {
        "contract_version": 1,
        "model_id": Path(args.model).name,
        "model_revision": getattr(model.config, "_commit_hash", None) or "",
        "tokenizer_id": str(tokenizer.name_or_path),
        "engine": f"transformers {metadata.version('transformers')}",
        "dtype": str(parameter.dtype).removeprefix("torch."),
        "device": str(parameter.device),
        "vocab_size": vocab_size,
        "torch_version": torch.__version__,
        "plugin_version": metadata.version("torch_npu") if device.startswith("npu") else "",
        "driver_version": ascend_version(DRIVER_VERSION_FILE),
        "toolkit_version": ascend_version(TOOLKIT_VERSION_FILE, TOOLKIT_LATEST_DIR),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[ocr-deepseek-ocr-2] serving on {HOST}:{args.port}: {info}", flush=True)

    def views(image_path: str):
        """The image as the checkpoint's own infer() encodes it.

        Returns the images list for the forward, the spatial-crop counts, and
        the image-token block: the padded global view alone for a small image,
        or global view plus a 2..6-tile crop grid for a large one, with the
        token block growing by 144 tokens per tile. The zero crop tensor is
        how the forward recognises "no tiles".
        """
        from PIL import Image, ImageOps

        image = Image.open(image_path).convert("RGB")
        transform = remote.BasicImageTransform(mean=NORM, std=NORM, normalize=True)
        pad_color = tuple(int(channel * RGB_MAX) for channel in NORM)
        global_view = transform(
            ImageOps.pad(image, (BASE_SIZE, BASE_SIZE), color=pad_color)
        ).to(dtype).unsqueeze(0)
        if image.size[0] <= MAX_SINGLE_VIEW and image.size[1] <= MAX_SINGLE_VIEW:
            crop = torch.zeros((1, 3, BASE_SIZE, BASE_SIZE))
            return [(crop.to(device), global_view.to(device))], [[1, 1]], list(IMAGE_IDS)
        tiles, (width_num, height_num) = remote.dynamic_preprocess(image)
        patches = torch.stack([transform(tile) for tile in tiles]).to(dtype)
        tokens = [
            *IMAGE_IDS,
            *([IMAGE_TOKEN_ID] * (CROP_QUERIES * width_num)) * (CROP_QUERIES * height_num),
        ]
        return (
            [(patches.to(device), global_view.to(device))],
            [[width_num, height_num]],
            tokens,
        )

    def template(raw_ids: list[int], image_tokens: list[int]) -> tuple[list[int], list[bool]]:
        """The expanded prompt template, and which of its positions are image.

        The raw ids the request carried (the version's frozen template) hold
        one image placeholder; this replaces it with the block the image
        expands to, so the deployment consumes ids and never tokenizes.
        """
        at = raw_ids.index(IMAGE_TOKEN_ID)
        expanded = [*raw_ids[:at], *image_tokens, *raw_ids[at + 1:]]
        return (
            expanded,
            [
                *repeat(False, at),
                *repeat(True, len(image_tokens)),
                *repeat(False, len(expanded) - at - len(image_tokens)),
            ],
        )

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
        image_list, spatial_crop, image_tokens = views(image_path)
        tpl, mask = template(raw_ids, image_tokens)
        ids = torch.tensor([[*tpl, *(continuation or [])]], dtype=torch.long, device=device)
        shared = {
            "images": image_list,
            "images_spatial_crop": torch.tensor(spatial_crop, dtype=torch.long),
        }
        forward = {
            **shared,
            "input_ids": ids,
            "images_seq_mask": torch.tensor(
                [mask + [False] * (ids.shape[1] - len(tpl))], dtype=torch.bool, device=device
            ),
            "use_cache": False,
        }
        generate = {
            **shared,
            "inputs": ids[:, : len(tpl)],
            "images_seq_mask": torch.tensor([mask], dtype=torch.bool, device=device),
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
                repetition_penalty=sampling["repetition_penalty"],
                min_new_tokens=0 if allow_eos else max_new_tokens,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
                logits_processor=[
                    NoRepeatNGramLogitsProcessor(NGRAM_SIZE, NGRAM_WINDOW, NGRAM_WHITELIST)
                ],
            )
        emitted = out.sequences[0, prompt_len:].tolist()
        write_log_probs(torch.cat(list(out.logits), dim=0), out_dir, len(emitted))
        text = tokenizer.decode(emitted, skip_special_tokens=True)
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
