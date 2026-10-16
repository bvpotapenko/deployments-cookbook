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

The scoring contract follows the OCR family's frozen-inputs pattern: the
input version's freeze renders the chat template to ids once (instruct
rendering), the request carries those ids plus the media attachment, and a
replay prefill stores only the rows over the generated ids (n_rows).
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
HOST = "0.0.0.0"
BACKLOG = 64
CONTRACT_VERSION = 1

IMAGE = "image"
VIDEO = "video"

# The runtime media knobs this deployment pins. The pixel budgets and the
# patch geometry come from the checkpoint's own configs; fps and frame
# sampling are the cookbook defaults. A candidate deployment must report the
# same values or its capture compares nothing.
VIDEO_FPS = 2.0
VIDEO_DO_SAMPLE_FRAMES = True


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
    vocab_size = int(model.config.vocab_size)
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
            if self.path != SCORE:
                self._error(HTTPStatus.NOT_FOUND, "not_found", self.path)
                return
            body = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}"
            )
            try:
                self._json(HTTPStatus.OK, score(body))
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
        if attachments:
            raise RuntimeError(
                "media scoring lands with the mllm family contract; this boot "
                "pins and reports the media parameters the contract will use"
            )
        ids = [int(token) for token in body["input_ids"]]
        if not ids:
            raise RuntimeError("empty input_ids: the version's frozen ids are missing")
        if body["mode"] == "decode":
            return _decode(body, ids)
        return _prefill(body, ids)

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
