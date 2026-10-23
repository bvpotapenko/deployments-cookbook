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
(n_rows). The decode's rows come from generate's own per-step logits
(output_logits), the raw pre-warper distribution.
"""

from __future__ import annotations

import argparse
import json
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
# reference implementation's (qwen-vl-utils) per-frame cap: without it the
# transformers video path costs ~12000 tokens per 30 s clip, with it the
# worst clip so far cost 7920. A candidate deployment must report the
# same values or its capture compares nothing.
VIDEO_FPS = 2.0
VIDEO_DO_SAMPLE_FRAMES = True
CAP_PIXELS_PER_FRAME = True


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

    checkpoint = Path(args.model)
    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForMultimodalLM.from_pretrained(
        checkpoint, dtype="bfloat16", device_map="auto"
    )
    model.eval()
    # The native config keeps the language model's fields under text_config.
    text_config = getattr(model.config, "text_config", None) or model.config
    vocab_size = int(getattr(model.config, "vocab_size", 0) or text_config.vocab_size)
    devices = sorted({str(device) for device in model.hf_device_map.values()})

    info = {
        "contract_version": CONTRACT_VERSION,
        "model_id": checkpoint.name,
        "model_revision": "",
        "tokenizer_id": str(checkpoint),
        "engine": f"transformers {transformers.__version__}",
        "dtype": "bfloat16",
        "device": "+".join(str(d) for d in devices),
        "vocab_size": vocab_size,
        "media": pinned_media_params(checkpoint, processor),
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
            if self.path not in (SCORE, INFER):
                self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)
                return
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            )
            try:
                self._json(HTTPStatus.OK, kot_infer(body) if self.path == INFER else score(body))
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

    def render_video(path: str):
        """The pinned prompt rendered two ways: the raw ids (one video
        placeholder, what the capture stores) and the expanded forward inputs
        (the processor's per-frame segments, deterministic in the video and
        the pinned fps)."""
        text = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video", "video": path},
                {"type": "text", "text": VIDEO_PROMPT},
            ]}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        raw = processor.apply_chat_template(
            [{"role": "user", "content": [
                {"type": "video", "video": path},
                {"type": "text", "text": VIDEO_PROMPT},
            ]}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )[0]
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

    def _store_rows(out_dir: Path, rows: torch.Tensor, ids: list[int]) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "log_probs.npy", rows.cpu().numpy(), allow_pickle=False)
        np.save(out_dir / "input_ids.npy", np.asarray(ids, dtype=np.int32), allow_pickle=False)

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
        with torch.no_grad():
            out = model.generate(
                input_ids=torch.tensor([expanded], device=model.device),
                attention_mask=torch.ones(len(expanded), dtype=torch.long, device=model.device).unsqueeze(0),
                do_sample=True,
                temperature=float(sampling.get("temperature", 1.0) or 1.0),
                top_p=float(sampling.get("top_p", 0.95) or 0.95),
                top_k=int(sampling.get("top_k", 20) or 20),
                max_new_tokens=int(body["max_new_tokens"]),
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                return_dict_in_generate=True,
                output_logits=True,
                # generate reads only each step's last-position logits; without
                # this the first forward materialises the whole prompt's
                # [seq, 248320] logits and OOMs card 0 beside the weights.
                logits_to_keep=1,
                **media,
            )
        generated = out.sequences[0][len(expanded):].tolist()
        rows = torch.log_softmax(torch.cat(list(out.logits)).float(), dim=-1)[:, :vocab_size]
        out_dir = Path(body["out_dir"])
        _store_rows(out_dir, rows, [*raw, *generated])
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
        full = torch.tensor([expanded + tail], dtype=torch.long, device=model.device)
        with torch.no_grad():
            # logits_to_keep pins the lm_head to the last n_rows+1 positions:
            # at vocab 248320 the full-sequence logits would not fit beside
            # the weights and the KV cache. The first n_rows of the block are
            # the tail's predictors (row t predicts position t+1).
            block = model(**media, input_ids=full, logits_to_keep=n_rows + 1).logits[0]
        rows = torch.log_softmax(block[:n_rows].float(), dim=-1)[:, :vocab_size]
        out_dir = Path(body["out_dir"])
        _store_rows(out_dir, rows, ids)
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
                attention_mask=torch.ones(1, len(expanded), dtype=torch.long, device=model.device),
                do_sample=True, temperature=1.0, top_p=0.95, top_k=20,
                max_new_tokens=512,
                pad_token_id=processor.tokenizer.eos_token_id,
                use_cache=True,
                logits_to_keep=1,
            )
        generated = out[0][len(expanded):].tolist()
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        text = processor.tokenizer.decode(generated, skip_special_tokens=True)
        (out_dir / "text.txt").write_text(text, encoding=UTF8)
        return {"prompt_id": body["prompt_id"], "files": ["text.txt"]}

    def _forward(ids: list[int]) -> torch.Tensor:
        tokens = torch.tensor([ids], dtype=torch.long, device=model.device)
        with torch.no_grad():
            return model(input_ids=tokens).logits[0]

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
        logits = _forward([*ids, *generated])
        rows = torch.log_softmax(logits[len(ids) - 1 : len(ids) - 1 + len(generated)].float(), dim=-1)
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "log_probs.npy", rows[:, :vocab_size].cpu().numpy(), allow_pickle=False)
        np.save(
            out_dir / "input_ids.npy",
            np.asarray([*ids, *generated], dtype=np.int32),
            allow_pickle=False,
        )
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
        logits = _forward(ids)
        rows = torch.log_softmax(
            logits[len(ids) - 1 - n_rows : len(ids) - 1].float(), dim=-1
        )
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "log_probs.npy", rows[:, :vocab_size].cpu().numpy(), allow_pickle=False)
        np.save(out_dir / "input_ids.npy", np.asarray(ids, dtype=np.int32), allow_pickle=False)
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
