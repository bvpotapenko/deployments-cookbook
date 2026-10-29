#!/usr/bin/env python3
"""The kot deployment of openvla/openvla-7b on transformers.

    serve.py --port P --model /path/to/openvla-7b --device cuda

GET /kot/info. POST /kot/infer reads the kot observation (.npz) the request
names, predicts one action and writes it as actions.npy, float32 [1, 1, 7]:
OpenVLA's chunk is one action of seven dimensions (x, y, z, roll, pitch, yaw,
gripper). POST /kot/score writes the log-probs of the action tokens, for a
comparison of distributions instead of actions:

  decode   builds the prompt from the observation's task, decodes the 7
           greedy tokens, writes one row per token and the ids prompt + tokens
  prefill  takes the ids as sent (a decode's stored ids), re-encodes the
           observation's frame, and writes the teacher-forced rows: n - 1,
           or the last n_rows when the request names them

Both read the frame from the observation; seed and sampling are not read,
because OpenVLA decodes greedily, as its reference does. There is no chat
route. One robot, one card: the server is a single-threaded http.server, so
requests wait on the socket and each one runs alone on the card.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from importlib import metadata
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from openvla import PROMPT, Policy

HOST = "127.0.0.1"
INFO = "/kot/info"
INFER = "/kot/infer"
SCORE = "/kot/score"
ACTIONS = "actions.npy"
LOG_PROBS = "log_probs.npy"
INPUT_IDS = "input_ids.npy"
# The dataset whose statistics turn the model's normalised actions into robot
# units. openvla-7b carries 25; bridge_orig (WidowX, BridgeData V2) is the one
# its model card uses. Pinned here: the contract's three flags carry no model
# setting.
UNNORM_KEY = "bridge_orig"
# Observation keys that are not cameras. OpenVLA reads the prompt and one
# camera; it takes no proprioceptive state and samples no noise.
NOT_CAMERAS = {"state", "noise", "prompt"}


def log(message: str) -> None:
    print(f"[vla-openvla-tf] {message}", flush=True)


def read_observation(path: str) -> tuple[Image.Image, str]:
    """The task and the one camera of a kot observation. A camera is float32
    [3, H, W], 1.0 = full scale, as kot's freeze writes it."""
    with np.load(path, allow_pickle=False) as observation:
        cameras = [key for key in observation.files if key not in NOT_CAMERAS]
        if len(cameras) != 1:
            raise ValueError(f"OpenVLA reads one camera; {path} holds {cameras}")
        pixels = observation[cameras[0]]
        task = str(observation["prompt"])
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError(f"camera {cameras[0]!r} must be [3, H, W], got {pixels.shape}")
    # Back to the 8-bit RGB frame the model's processor takes: round to the
    # nearest level (ties to even), then saturate. A photo's levels come back
    # exact. kot's gradient planes 2 and 3 carry half and quarter levels, up
    # to 255.75/255: those round, and the top ones saturate at 255.
    levels = np.clip(np.rint(pixels.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(levels), task


def identity(model_dir: Path) -> tuple[str, str]:
    """Model id and revision. A hub snapshot directory is named by its commit;
    any other directory gives its name and no revision."""
    path = model_dir.resolve()
    repo = path.parent.parent.name
    if path.parent.name == "snapshots" and repo.startswith("models--"):
        return repo.removeprefix("models--").replace("--", "/"), path.name
    return path.name, ""


def nvidia_driver() -> str:
    """The kernel driver's version, as the driver itself reports it; empty when unreadable."""
    try:
        first_line = Path("/proc/driver/nvidia/version").read_text().splitlines()[0]
    except (OSError, IndexError):
        return ""
    match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", first_line)
    return match.group(1) if match else ""


class Handler(BaseHTTPRequestHandler):
    """Routes for one server; the policy and the info dict live on self.server."""

    def do_GET(self) -> None:
        if self.path == INFO:
            self.reply(200, self.server.info)
        else:
            self.reply(404, {"error": "not_found", "detail": self.path})

    def do_POST(self) -> None:
        route = {INFER: self.infer, SCORE: self.score}.get(self.path)
        if route is None:
            self.reply(404, {"error": "not_found", "detail": self.path})
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            # A decode answer is a JSON-lines stream; one line is a valid one.
            stream = self.path == SCORE and body.get("mode") == "decode"
            self.reply(200, route(body), "application/x-ndjson" if stream else "application/json")
        except Exception as failure:
            oom = isinstance(failure, torch.OutOfMemoryError)
            error = "out_of_memory" if oom else type(failure).__name__
            log(f"{self.path} failed: {error}: {failure}")
            self.reply(500, {"error": error, "detail": str(failure)})

    def infer(self, body: dict) -> dict:
        frame, task = read_observation(body["attachments"]["observation"])
        started = time.perf_counter()
        action, tokens = self.server.policy(frame, task)
        elapsed_ms = (time.perf_counter() - started) * 1000
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / ACTIONS, action.astype(np.float32).reshape(1, 1, -1), allow_pickle=False)
        log(f"{body['prompt_id']}: {elapsed_ms:.1f} ms, tokens {tokens}, action {np.round(action, 4).tolist()}")
        return {"prompt_id": body["prompt_id"], "files": [ACTIONS]}

    def score(self, body: dict) -> dict:
        policy = self.server.policy
        frame, task = read_observation(body["attachments"]["observation"])
        started = time.perf_counter()
        if body["mode"] == "decode":
            if int(body["max_new_tokens"]) != policy.action_dim:
                raise ValueError(f"OpenVLA decodes {policy.action_dim} tokens; max_new_tokens is {body['max_new_tokens']}")
            prompt = policy.prompt_ids(task)
            logits = policy.decode(frame, prompt)
            output_ids = logits.argmax(dim=-1).tolist()
            ids = prompt + output_ids
        elif body["mode"] == "prefill":
            ids = [int(token) for token in body["input_ids"]]  # verbatim: no template, no BOS added
            logits = policy.prefill(frame, ids)
            n_rows = int(body.get("n_rows") or 0)
            logits = logits[-n_rows:] if n_rows else logits
            output_ids = []
        else:
            raise ValueError(f"unknown mode {body['mode']!r}")
        rows = torch.log_softmax(logits.float(), dim=-1).cpu().numpy()  # float32, full 32064 columns
        elapsed_ms = (time.perf_counter() - started) * 1000
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / LOG_PROBS, rows, allow_pickle=False)
        np.save(out_dir / INPUT_IDS, np.asarray(ids, dtype=np.int32), allow_pickle=False)
        log(f"{body['prompt_id']} {body['mode']}: {elapsed_ms:.1f} ms, {len(rows)} rows, tokens {output_ids}")
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": LOG_PROBS, "input_ids": INPUT_IDS},
            "n_positions": len(rows),
            "vocab_size": rows.shape[1],
            "output_ids": output_ids,
        }

    def reply(self, status: int, payload: dict, content_type: str = "application/json") -> None:
        blob = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(blob)))
        self.end_headers()
        self.wfile.write(blob)

    def log_message(self, *args) -> None:
        """Silent: each route logs one line per request, failures included."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    policy = Policy(args.model, args.device, UNNORM_KEY)
    # The first call pays for CUDA context, cuBLAS handles and kernel choice;
    # /kot/info answers only after it, so no timed request pays for it.
    policy(Image.new("RGB", (224, 224)), "warm up")

    model_id, revision = identity(args.model)
    info = {
        "contract_version": 1,
        "model_id": model_id,
        "model_revision": revision,
        "engine": f"transformers {metadata.version('transformers')}",
        "dtype": str(policy.dtype).removeprefix("torch."),
        "device": str(next(policy.model.parameters()).device),
        "vocab_size": policy.model.language_model.config.vocab_size,
        "torch_version": torch.__version__,
        "plugin_version": "",
        "driver_version": nvidia_driver(),
        "toolkit_version": torch.version.cuda or "",
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "unnorm_key": UNNORM_KEY,
        "prompt": PROMPT,
    }
    server = HTTPServer((HOST, args.port), Handler)
    server.policy, server.info = policy, info
    log(f"serving on {HOST}:{args.port}: {json.dumps(info)}")
    server.serve_forever()


if __name__ == "__main__":
    main()
