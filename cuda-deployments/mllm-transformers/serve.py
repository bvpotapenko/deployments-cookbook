"""The mllm transformers deployment: Qwen3.8-27B as the first multimodal-LLM target.

Serves the kot capture contract over three unit kinds - text, image, video -
with every media-processing parameter PINNED IN THIS FILE and reported through
/kot/info, because a candidate capture must run the identical frame sampling
and pixel budgets or its speed and correctness numbers compare nothing.

The pinned video pipeline (the checkpoint's own defaults, from the Qwen
cookbooks and the checkpoint's processor configs):
  - frame sampling: fps=2.0, do_sample_frames=True (frames extracted from
    the container at 2 per second, then grouped into temporal patches)
  - pixel budgets: the checkpoint's video_preprocessor_config.json
    (longest_edge 25165824, shortest_edge 4096) and preprocessor_config.json
    for images (16777216 / 65536); patch_size 16, temporal_patch_size 2,
    merge_size 2 -- all read from the checkpoint, reported, never invented
  - instruct mode: the chat template renders with enable_thinking=False
    (the non-thinking mode; its recommended sampling lives in the store's
    sampling yaml and arrives per request)

The scoring contract follows the OCR family's frozen-inputs pattern, adapted
to videos: the frozen inputs carry no ids (the prompt is built around the
frames), so the deployment renders the pinned chat prompt itself - the raw
form (one video placeholder) is what a capture stores, and a replay prefill
re-renders it, asserts the stored head matches, and scores the stored tail
(n_rows). The decode's rows are the raw logits generate samples from, each
step's taken by a forward hook before any processor and moved to host at
once; at every 256 rows the hook runs their log-softmax on the card and
writes them into log_probs.npy, opened at max_new_tokens rows, so the host
holds one block, never the unit. A video replay runs in calls through one
cache: the expanded prompt with the video, then the tail in chunks of
KOT_TF_REPLAY_CHUNK tokens, so the working memory of a call stays bounded;
every row still sees the video, the prompt and every tail id before the one
it predicts. The rows of each call go into the file as the call returns. The
chunk size is part of the numerics and /kot/info reports it. The lm_head of
a prefill runs one block of rows at a time. allow_eos
shapes a video decode's stop as the contract states: false, the default, is exactly
max_new_tokens ids with EOS suppressed through min_new_tokens, and the rows
are still taken before that suppression; true stops at EOS, at most that many,
and the EOS id stays in output_ids with its row.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import urllib.parse
from http import HTTPStatus
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoModelForMultimodalLM, AutoProcessor

UTF8 = "utf-8"
INFO = "/kot/info"
SCORE = "/kot/score"
INFER = "/kot/infer"  # the speed surface: decode over the attachment, files named back
CHAT = "/v1/chat/completions"
HOST = "0.0.0.0"
BACKLOG = 64
CONTRACT_VERSION = 1

IMAGE = "image"
VIDEO = "video"
PREFILL = "prefill"
DECODE = "decode"

# The pinned video prompt: what every video unit asks, in the checkpoint's
# chat format. The deployment renders it; the frozen inputs carry no ids.
VIDEO_PROMPT = "Describe what you see in the video."

# The runtime media knobs this deployment pins. The pixel budgets and the
# patch geometry come from the checkpoint's own configs; fps and frame
# sampling are the cookbook defaults. cap_pixels_per_frame adopts the
# reference implementation's (qwen-vl-utils) per-frame cap. At 1080p it binds
# below 32 frames (22 frames cost 12100 tokens without it, 7920 with it); a
# 30 s clip costs 11700 either way. A candidate deployment must report the
# same values or its capture compares nothing.
VIDEO_FPS = 2.0
VIDEO_DO_SAMPLE_FRAMES = True
CAP_PIXELS_PER_FRAME = True

# Rows per log-softmax block. At vocabulary 248320 a block is 0.24 GiB of float32.
BLOCK_ROWS = 256

# Tokens per replay call after the prompt call. The working memory of a call
# grows with its tokens: one call over a 12.3k-token video prompt plus an
# 8192-token tail does not fit on card 0. The size is part of the numerics:
# the float order changes at every call edge, so rows produced at one size
# need not be bit-identical to rows produced at another. /kot/info reports it.
# The env override is for experiments; captures run at the default.
REPLAY_CHUNK = int(os.environ.get("KOT_TF_REPLAY_CHUNK", "2048"))


def compact_rows(path: Path, rows: int) -> None:
    """Cut log_probs.npy to its first rows: copied block by block into a new
    file at the final shape, which then replaces the old one."""
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


def nvidia_driver() -> str:
    """The kernel driver's version, as the driver itself reports it; empty when unreadable."""
    try:
        first_line = Path("/proc/driver/nvidia/version").read_text().splitlines()[0]
    except (OSError, IndexError):
        return ""
    match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", first_line)
    return match.group(1) if match else ""


def snapshot_identity(checkpoint: Path) -> tuple[str, str]:
    """(model id, revision): a hub snapshot's path names both, and its directory
    is named by its commit. Any other directory gives its name and no revision."""
    path = checkpoint.resolve()
    repo = path.parent.parent.name
    if path.parent.name == "snapshots" and repo.startswith("models--") and re.fullmatch(r"[0-9a-f]{40}", path.name):
        return repo.removeprefix("models--").replace("--", "/"), path.name
    return checkpoint.name, ""


def pinned_media_params(checkpoint: Path, processor) -> dict:
    """The media processing this deployment runs, resolved from the checkpoint.

    Everything here is reported through /kot/info so a capture's info.yaml
    records the exact frame sampling and pixel budgets a candidate must
    reproduce.
    """
    image_cfg = json.loads((checkpoint / "preprocessor_config.json").read_text())
    video_cfg = json.loads((checkpoint / "video_preprocessor_config.json").read_text())
    return {
        "video_fps": VIDEO_FPS,
        "video_do_sample_frames": VIDEO_DO_SAMPLE_FRAMES,
        "video_cap_pixels_per_frame": CAP_PIXELS_PER_FRAME,
        "video_prompt": VIDEO_PROMPT,
        "video_size_limits": video_cfg["size"],
        "image_size_limits": image_cfg["size"],
        "patch_size": video_cfg["patch_size"],
        "temporal_patch_size": video_cfg["temporal_patch_size"],
        "merge_size": video_cfg["merge_size"],
        "enable_thinking": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if REPLAY_CHUNK < 2:
        # A call of one token takes the decode path, not the prefill path.
        raise SystemExit(f"KOT_TF_REPLAY_CHUNK is {REPLAY_CHUNK}; a replay chunk needs at least 2 tokens")

    checkpoint = Path(args.model)
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForMultimodalLM.from_pretrained(
        checkpoint, dtype="bfloat16", device_map="auto"
    )
    model.eval()
    # The native config keeps the language model's fields under text_config.
    text_config = getattr(model.config, "text_config", None) or model.config
    vocab_size = int(getattr(model.config, "vocab_size", 0) or text_config.vocab_size)
    devices = sorted({str(parameter.device) for parameter in model.parameters()})

    model_id, model_revision = snapshot_identity(checkpoint)
    info = {
        "contract_version": CONTRACT_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": str(checkpoint),
        "engine": f"transformers {transformers.__version__}",
        "dtype": "bfloat16",
        "device": "+".join(str(d) for d in devices),
        "vocab_size": vocab_size,
        "torch_version": torch.__version__,
        "plugin_version": "",
        "driver_version": nvidia_driver(),
        "toolkit_version": torch.version.cuda or "",
        "media": pinned_media_params(checkpoint, processor),
        # The replay's chunk size is part of the numerics: the float order
        # changes at every call edge, so rows produced at one size need not
        # be bit-identical to rows produced at another. Every capture carries its own.
        "replay_chunk_tokens": REPLAY_CHUNK,
        "started_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    print(f"[mllm-transformers] serving: {json.dumps(info)}", flush=True)

    from http.server import BaseHTTPRequestHandler, HTTPServer

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != INFO:
                self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)
                return
            self._json(HTTPStatus.OK, info)

        def do_POST(self):
            if self.path not in (SCORE, INFER, CHAT):
                self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)
                return
            try:
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
                )
                if self.path == CHAT:
                    text = chat_completion(body)
                    if body.get("stream"):
                        self._stream(text)
                    else:
                        self._json(HTTPStatus.OK, chat_response(text))
                else:
                    self._json(
                        HTTPStatus.OK,
                        kot_infer(body) if self.path == INFER else score(body),
                    )
            except ChatRequestError as failure:
                self._error(HTTPStatus.BAD_REQUEST, "invalid_request", str(failure))
            except Exception as failure:
                self._error(
                    HTTPStatus.INTERNAL_SERVER_ERROR, type(failure).__name__, str(failure)
                )

        def log_request(self, code="-", size="-"):
            if str(code) != str(HTTPStatus.OK.value):
                super().log_request(code, size)

        def _json(self, status, payload):
            blob = json.dumps(payload).encode(UTF8)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(blob)))
            self.end_headers()
            self.wfile.write(blob)

        def _stream(self, text):
            completion_id = f"chatcmpl-{os.urandom(12).hex()}"
            created = int(__import__("time").time())
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()

            def write(delta, finish_reason=None):
                payload = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": info["model_id"],
                    "choices": [{
                        "index": 0,
                        "delta": delta,
                        "finish_reason": finish_reason,
                    }],
                }
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode(UTF8))
                self.wfile.flush()

            write({"role": "assistant", "content": ""})
            if text:
                write({"content": text})
            write({}, "stop")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

        def _error(self, status, error, detail):
            self._json(status, {"error": error, "detail": detail})

    def score(body: dict) -> dict:
        attachments = body.get("attachments") or {}
        path = attachments.get(VIDEO)
        if path:
            if body["mode"] == DECODE:
                return _video_decode(body, path)
            return _video_prefill(body, path, [int(t) for t in body["input_ids"]])
        if attachments:
            raise RuntimeError(
                "unsupported attachment kind(s): " + ", ".join(sorted(set(attachments) - {VIDEO}))
            )
        ids = [int(token) for token in body["input_ids"]]
        if not ids:
            raise RuntimeError("empty input_ids: the version's frozen ids are missing")
        if body["mode"] == DECODE:
            return _decode(body, ids)
        return _prefill(body, ids)

    def render_video(path: str, prompt: str = VIDEO_PROMPT):
        """Render the pinned video pipeline, with an optional chat prompt."""
        text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video", "video": path},
                {"type": "text", "text": prompt},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        # The raw form: the rendered string tokenized by the tokenizer only -
        # ONE video placeholder. apply_chat_template(tokenize=True) runs the
        # full processor in 5.22.1 and returns the expanded ids.
        raw = processor.tokenizer(text, add_special_tokens=False)["input_ids"]
        video = processor(
            text=[text],
            videos=[[path]],
            fps=VIDEO_FPS,
            do_sample_frames=VIDEO_DO_SAMPLE_FRAMES,
            cap_pixels_per_frame=CAP_PIXELS_PER_FRAME,
            return_tensors="pt",
        )
        expanded = video["input_ids"][0].tolist()
        media = {
            key: value.to(model.device)
            for key, value in video.items()
            if key != "input_ids"
        }
        return raw, expanded, media

    class ChatRequestError(ValueError):
        pass

    def _video_path_from_chat(part: dict) -> str:
        video_url = part.get("video_url")
        if not isinstance(video_url, dict) or not video_url.get("url"):
            raise ChatRequestError("video_url must contain a url")
        url = str(video_url["url"])
        if url.startswith("file://"):
            return urllib.parse.unquote(urllib.parse.urlparse(url).path)
        if url.startswith("/"):
            return url
        raise ChatRequestError(
            "video_url must be a local absolute path or file:// URL; remote URLs are unsupported"
        )

    def _chat_video_and_text(body: dict) -> tuple[str, str]:
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise ChatRequestError("messages must be a non-empty list")
        text_parts: list[str] = []
        video_path = None
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                text_parts.append(content)
                continue
            if not isinstance(content, list):
                raise ChatRequestError("the user message content must be text and video parts")
            for part in content:
                if not isinstance(part, dict):
                    raise ChatRequestError("message content parts must be objects")
                kind = part.get("type")
                if kind == "text":
                    text_parts.append(str(part.get("text", "")))
                elif kind == "video_url":
                    if video_path is not None:
                        raise ChatRequestError("exactly one video attachment is supported")
                    video_path = _video_path_from_chat(part)
                else:
                    raise ChatRequestError(f"unsupported message content type: {kind!r}")
        text = "".join(text_parts).strip()
        if video_path is None:
            raise ChatRequestError("the request has no video attachment")
        if not text:
            raise ChatRequestError("the request has no text prompt")
        return video_path, text

    def _chat_max_tokens(body: dict) -> int:
        value = body.get("max_completion_tokens", body.get("max_tokens", 512))
        if value is None:
            value = 512
        try:
            value = int(value)
        except (TypeError, ValueError):
            raise ChatRequestError("max_completion_tokens must be an integer") from None
        if value < 1:
            raise ChatRequestError("max_completion_tokens must be at least 1")
        return value

    def chat_completion(body: dict) -> str:
        """The OpenAI surface uses the same video render and generate path as kot_infer."""
        path, text = _chat_video_and_text(body)
        _raw, expanded, media = render_video(path, text)
        torch.manual_seed(int(body.get("seed", 42)))
        with torch.no_grad():
            output = model.generate(
                input_ids=torch.tensor([expanded], device=model.device),
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                max_new_tokens=_chat_max_tokens(body),
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                logits_to_keep=1,
                **media,
            )
        generated = output[0][len(expanded):].tolist()
        return processor.tokenizer.decode(generated, skip_special_tokens=True)

    def chat_response(text: str) -> dict:
        return {
            "id": f"chatcmpl-{os.urandom(12).hex()}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": info["model_id"],
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
        }

    def _open_rows(out_dir: Path, count: int) -> np.ndarray:
        """log_probs.npy at its final shape, a memmap to fill block by block."""
        out_dir.mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(
            out_dir / "log_probs.npy", mode="w+", dtype=np.float32, shape=(count, vocab_size)
        )

    def _write_rows(rows: np.ndarray, first: int, count: int, logits_of) -> int:
        """Write count rows from row first, block by block: logits_of(start, stop)
        gives rows start..stop-1 of the batch as raw logits on a card, the
        log-softmax runs there, and each finished block is copied to host and
        into the file, so a whole unit's rows never exist at once on a card.
        Returns the next row to write."""
        with torch.no_grad():
            for start in range(0, count, BLOCK_ROWS):
                stop = min(start + BLOCK_ROWS, count)
                # One expression: no name keeps a block alive, so its memory is back
                # in the allocator for the next one, without an empty_cache.
                rows[first + start:first + stop] = torch.log_softmax(
                    logits_of(start, stop).float(), dim=-1
                )[:, :vocab_size].cpu().numpy()
        return first + count

    def _save_ids(out_dir: Path, ids: list[int]) -> None:
        np.save(out_dir / "input_ids.npy", np.asarray(ids, dtype=np.int32), allow_pickle=False)

    def _store_rows(out_dir: Path, count: int, logits_of, ids: list[int]) -> None:
        """Write count log_probs.npy rows and the ids in one pass."""
        rows = _open_rows(out_dir, count)
        _write_rows(rows, 0, count, logits_of)
        rows.flush()
        _save_ids(out_dir, ids)

    def _head(hidden: torch.Tensor):
        """logits_of for _write_rows over last hidden states: the model's own
        lm_head, one block at a time. At vocabulary 248320 the logits of an
        8192-row tail are 3.8 GiB in bfloat16, too much beside the weights."""
        return lambda start, stop: model.lm_head(hidden[start:stop])

    def _hidden(**inputs) -> torch.Tensor:
        """The last hidden state at each position: the lm_head's input, as the
        model's own forward computes it."""
        with torch.no_grad():
            return model.model(**inputs).last_hidden_state[0]

    def _video_decode(body: dict, path: str) -> dict:
        raw, expanded, media = render_video(path)
        seed = int(body.get("seed", 42))
        torch.manual_seed(seed)
        sampling = body.get("sampling") or {}
        if float(sampling.get("presence_penalty", 0.0)):
            raise RuntimeError(
                "presence_penalty is unsupported on the transformers engine: "
                "it is an OpenAI-surface knob (vllm and sglang apply it)"
            )
        max_new_tokens = int(body["max_new_tokens"])
        allow_eos = bool(body.get("allow_eos", False))
        out_dir = Path(body["out_dir"])
        # At the cap shape: allow_eos may stop early, and the file is then
        # compacted to the rows produced.
        rows = _open_rows(out_dir, max_new_tokens)
        kept: list[torch.Tensor] = []
        written = 0

        def write_kept():
            # The blocks start at row 0, hold 256 rows and run on one card,
            # so the bytes do not depend on when a block is written.
            nonlocal written
            written = _write_rows(
                rows, written, len(kept),
                lambda start, stop: torch.vstack(kept[start:stop]).to(model.device),
            )
            kept.clear()

        def keep_row(_module, _args, output):
            # The row generate samples from: the last position's logits as
            # float32, before any processor. It goes to host at once; with
            # output_logits generate keeps every row on card 0, 7.6 GiB at 8192.
            # The host holds at most one block of rows, 0.24 GiB.
            kept.append(output.logits[0, -1].to(device="cpu", dtype=torch.float32))
            if len(kept) == BLOCK_ROWS:
                write_kept()

        hook = model.register_forward_hook(keep_row)
        try:
            with torch.no_grad():
                sequences = model.generate(
                    input_ids=torch.tensor([expanded], device=model.device),
                    do_sample=True,
                    temperature=float(sampling.get("temperature", 1.0) or 1.0),
                    top_p=float(sampling.get("top_p", 0.95) or 0.95),
                    top_k=int(sampling.get("top_k", 20) or 20),
                    min_new_tokens=0 if allow_eos else max_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True,
                    # generate reads only each step's last-position logits; without
                    # this the first forward materialises the whole prompt's
                    # [seq, 248320] logits and OOMs card 0 beside the weights.
                    logits_to_keep=1,
                    **media,
                )
        finally:
            hook.remove()
        write_kept()
        generated = sequences[0][len(expanded):].tolist()
        if written != len(generated):
            raise RuntimeError(f"{written} forward rows for {len(generated)} generated ids")
        rows.flush()
        del rows
        if written < max_new_tokens:
            compact_rows(out_dir / "log_probs.npy", written)
        _save_ids(out_dir, [*raw, *generated])
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": len(generated),
            "vocab_size": vocab_size,
            "output_ids": generated,
        }

    def _video_prefill(body: dict, path: str, ids: list[int]) -> dict:
        raw, expanded, media = render_video(path)
        if ids[: len(raw)] != raw:
            fail = ids[:8]
            raise RuntimeError(
                f"the stored ids' head is not this deployment's raw video template: {fail}"
            )
        tail = ids[len(raw):]
        n_rows = int(body.get("n_rows", 0))
        if n_rows != len(tail):
            raise RuntimeError(f"n_rows {n_rows} != the stored tail's length {len(tail)}")
        out_dir = Path(body["out_dir"])
        rows = _open_rows(out_dir, n_rows)
        # Row t predicts tail[t]. Call 1 is the decode's own first call: the
        # expanded prompt with the video inputs, the processor's own mask and
        # type ids. No cache goes in, so the model starts a fresh one. The
        # model keeps rope_deltas as state; with no earlier value and an
        # empty cache it computes them from this video. Its last position
        # gives row 0.
        model.model.rope_deltas = None
        with torch.no_grad():
            out = model.model(
                input_ids=torch.tensor([expanded], dtype=torch.long, device=model.device),
                use_cache=True,
                **media,
            )
        cache = out.past_key_values
        written = _write_rows(rows, 0, min(1, n_rows), _head(out.last_hidden_state[0, -1:]))
        del out
        # Then the tail, chunk by chunk, through the same cache: each chunk
        # attends to the video, the prompt and every earlier chunk. A chunk
        # call takes only its ids and the cache. No attention_mask: a
        # chunk-length mask restarts the 3D-rope positions and hides every key
        # after the first chunk-length ones, and neither raises an error.
        # The position of tail[t-1] gives row t; the last tail id's row is
        # not stored.
        for start in range(0, len(tail), REPLAY_CHUNK):
            chunk = tail[start:start + REPLAY_CHUNK]
            with torch.no_grad():
                out = model.model(
                    input_ids=torch.tensor([chunk], dtype=torch.long, device=model.device),
                    past_key_values=cache,
                    use_cache=True,
                )
            written = _write_rows(
                rows, written, min(len(chunk), n_rows - written), _head(out.last_hidden_state[0])
            )
            del out
        if written != n_rows or cache.get_seq_length() != len(expanded) + len(tail):
            raise RuntimeError(
                f"the replay wrote {written} rows of {n_rows} over "
                f"{cache.get_seq_length()} cached positions of {len(expanded) + len(tail)}"
            )
        rows.flush()
        _save_ids(out_dir, ids)
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": n_rows,
            "vocab_size": vocab_size,
            "output_ids": [],
        }

    def kot_infer(body: dict) -> dict:
        """The speed surface: one video decode, the description written into
        out_dir, the files named back. Only the clock reads what lands here."""
        path = (body.get("attachments") or {}).get(VIDEO)
        if not path:
            raise RuntimeError("the speed request carries no video attachment")
        raw, expanded, media = render_video(path)
        torch.manual_seed(int(body.get("seed", 42)))
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor([expanded], device=model.device),
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                max_new_tokens=512,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                logits_to_keep=1,
                **media,
            )
        generated = out[0][len(expanded):].tolist()
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        (out_dir / "text.txt").write_text(text, encoding=UTF8)
        return {"prompt_id": body["prompt_id"], "files": ["text.txt"]}

    def _text_hidden(ids: list[int]) -> torch.Tensor:
        return _hidden(input_ids=torch.tensor([ids], dtype=torch.long, device=model.device))

    def _decode(body: dict, ids: list[int]) -> dict:
        seed = int(body.get("seed", 42))
        torch.manual_seed(seed)
        sampling = body.get("sampling") or {}
        if float(sampling.get("presence_penalty", 0.0)):
            raise RuntimeError(
                "presence_penalty is unsupported on the transformers engine: "
                "it is an OpenAI-surface knob (vllm and sglang apply it)"
            )
        generated = model.generate(
            input_ids=torch.tensor([ids], dtype=torch.long, device=model.device),
            do_sample=True,
            temperature=float(sampling.get("temperature", 0.7) or 0.7),
            top_p=float(sampling.get("top_p", 0.8) or 0.8),
            top_k=int(sampling.get("top_k", 20) or 20),
            max_new_tokens=int(body["max_new_tokens"]),
            pad_token_id=processor.tokenizer.eos_token_id,
        )[0][len(ids):].tolist()
        hidden = _text_hidden([*ids, *generated])[len(ids) - 1 : len(ids) - 1 + len(generated)]
        out_dir = Path(body["out_dir"])
        _store_rows(out_dir, len(generated), _head(hidden), [*ids, *generated])
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": len(generated),
            "vocab_size": vocab_size,
            "output_ids": generated,
        }

    def _prefill(body: dict, ids: list[int]) -> dict:
        n_rows = int(body.get("n_rows", 0))
        if n_rows < 1:
            raise RuntimeError("the mllm prefill stores the decode's rows: set n_rows")
        hidden = _text_hidden(ids)[len(ids) - 1 - n_rows : len(ids) - 1]
        out_dir = Path(body["out_dir"])
        _store_rows(out_dir, n_rows, _head(hidden), ids)
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": n_rows,
            "vocab_size": vocab_size,
            "output_ids": [],
        }

    HTTPServer.request_queue_size = BACKLOG
    HTTPServer((HOST, int(args.port)), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
