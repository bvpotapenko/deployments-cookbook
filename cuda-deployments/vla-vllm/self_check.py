#!/usr/bin/env python3
"""Run the VLA deployment contract checks.

    python3 self_check.py --deployment DIR --model PATH --device cuda

The check starts serve.sh, creates deterministic synthetic observations, and
checks /kot/info, /kot/infer, decode scoring, and teacher-forced prefill.
It intentionally uses only stdlib plus numpy so an agent can run it before
installing model-specific tooling in the deployment environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

HOST = "127.0.0.1"
LOAD_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 900
STOP_GRACE_S = 15
ACTION_DIM = 7
SEED = 42
failed: list[str] = []


def check(name: str, run):
    try:
        problems, detail, value = run()
    except urllib.error.HTTPError as error:
        problems = [f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"]
        detail, value = "", None
    except Exception as error:  # every failure is a readable FAIL line
        problems = [f"{type(error).__name__}: {error}"]
        detail, value = "", None
    if problems:
        failed.append(name)
    line = f"FAIL {name}: {'; '.join(problems)}" if problems else f"PASS {name}: {detail}"
    print(line, flush=True)
    return None if problems else value


def make_observation(path: Path, inverted: bool = False) -> None:
    height, width = 224, 224
    y, x = np.indices((height, width), dtype=np.float32)
    image = np.stack((x / width, y / height, (x + y) / (width + height)), axis=0)
    if inverted:
        image = 1.0 - image
    np.savez(path, prompt=np.asarray("pick up the red block"), image=image.astype(np.float32))


def wait_info(process: subprocess.Popen, port: int):
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/kot/info", timeout=10) as response:
                info = json.loads(response.read())
            return [], f"model {info.get('model_id')}, device {info.get('device')}", info
        except OSError:
            time.sleep(1)
    detail = f"serve.sh exited with code {process.returncode}" if process.poll() is not None else ""
    return [detail or f"no answer within {LOAD_TIMEOUT_S} s"], "", None


def post(port: int, route: str, payload: dict) -> bytes:
    request = urllib.request.Request(
        f"http://{HOST}:{port}{route}",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return response.read()


def score_answer(port: int, payload: dict):
    body = post(port, "/kot/score", payload)
    lines = [json.loads(line) for line in body.splitlines() if line.strip()]
    results = [line for line in lines if isinstance(line, dict) and "block" not in line]
    if not results or any(not isinstance(line, dict) or "error" in line for line in lines):
        return [f"score returned an error: {body[:500]!r}"], "", None
    return [], f"{len(lines) - len(results)} block line(s)", results[-1]


def output_arrays(payload: dict, reply: dict, root: Path, info: dict):
    named = reply.get("files") or {}
    paths = [Path(payload["out_dir"], named.get(key, "")) for key in ("log_probs", "input_ids")]
    if not all(path.is_file() for path in paths):
        return [f"named output files are missing: {named}"], "", None
    log_probs, input_ids = (np.load(path, allow_pickle=False) for path in paths)
    problems = []
    if log_probs.ndim != 2 or not np.issubdtype(log_probs.dtype, np.floating):
        problems.append(f"log_probs must be floating [rows,vocab], got {log_probs.shape} {log_probs.dtype}")
    if input_ids.ndim != 1 or not np.issubdtype(input_ids.dtype, np.integer):
        problems.append(f"input_ids must be integer [tokens], got {input_ids.shape} {input_ids.dtype}")
    if not np.isfinite(log_probs).all():
        problems.append("log_probs contain NaN or inf")
    if log_probs.ndim == 2 and not np.allclose(np.logaddexp.reduce(log_probs, axis=1), 0, atol=1e-3):
        problems.append("log_probs rows are not normalized")
    vocab = int(info.get("vocab_size") or 0)
    if log_probs.ndim == 2 and log_probs.shape[1] != vocab:
        problems.append(f"vocabulary width is {log_probs.shape[1]}, expected {vocab}")
    decode = payload["mode"] == "decode"
    output_ids = [int(token) for token in reply.get("output_ids") or []]
    expected_rows = len(output_ids) if decode else int(payload["n_rows"])
    if log_probs.ndim == 2 and log_probs.shape[0] != expected_rows:
        problems.append(f"row count is {log_probs.shape[0]}, expected {expected_rows}")
    if reply.get("n_positions") != expected_rows:
        problems.append(f"n_positions is {reply.get('n_positions')}, expected {expected_rows}")
    stored = input_ids.tolist()
    if decode and stored[-len(output_ids):] != output_ids:
        problems.append("stored input_ids do not end with decoded output_ids")
    if not decode and stored != payload["input_ids"]:
        problems.append("prefill changed the input_ids")
    return problems, f"actions/log_probs contract checked in {root}", (stored, expected_rows, log_probs)


def run_checks(port: int, root: Path, info: dict, observation: Path, inverted: Path):
    infer = {
        "prompt_id": "self_check_infer",
        "out_dir": str(root / "infer"),
        "attachments": {"observation": str(observation)},
    }
    reply = check("infer answer", lambda: ([], "JSON response", json.loads(post(port, "/kot/infer", infer))))
    if reply:
        action_path = root / "infer" / (reply.get("files") or ["actions.npy"])[0]
        check("infer file", lambda: ([], f"actions {np.load(action_path).shape}", _check_action(action_path)))

    decode = {
        "prompt_id": "self_check_decode", "input_ids": [], "mode": "decode",
        "out_dir": str(root / "decode"), "attachments": {"observation": str(observation)},
        "max_new_tokens": ACTION_DIM, "seed": SEED, "sampling": {}, "allow_eos": False, "n_rows": 0,
    }
    decoded = check("decode answer", lambda: score_answer(port, decode))
    stored = None
    if decoded:
        checked = check("decode files", lambda: output_arrays(decode, decoded, root, info))
        if checked:
            stored, count, rows = checked
            if rows.shape[0] == len(decoded.get("output_ids") or []):
                expected = np.asarray(decoded["output_ids"])
                check("decode argmax", lambda: ([], "decoded ids are row argmax", _check_argmax(rows, expected)))
            prefill = dict(decode, input_ids=stored, mode="prefill", out_dir=str(root / "prefill"), max_new_tokens=0, n_rows=count)
            replay = check("prefill answer", lambda: score_answer(port, prefill))
            if replay:
                check("prefill files", lambda: output_arrays(prefill, replay, root, info))

    changed = dict(decode, prompt_id="self_check_image", out_dir=str(root / "inverted"), attachments={"observation": str(inverted)})
    other = check("image sensitivity", lambda: score_answer(port, changed))
    if other and decoded:
        a = np.load(root / "decode" / "log_probs.npy", allow_pickle=False)
        b = np.load(root / "inverted" / "log_probs.npy", allow_pickle=False)
        check("image reaches model", lambda: ([], f"max |log_prob diff| {np.max(np.abs(a - b)):.6g}", _check_different(a, b)))


def _check_action(path: Path):
    actions = np.load(path, allow_pickle=False)
    if actions.shape != (1, 1, ACTION_DIM) or actions.dtype != np.float32 or not np.isfinite(actions).all():
        return [f"actions must be float32 [1,1,7], got {actions.shape} {actions.dtype}"], "", None
    return [], "float32 [1,1,7] finite", actions


def _check_argmax(rows: np.ndarray, ids: np.ndarray):
    if not np.array_equal(rows.argmax(axis=1), ids):
        return ["decoded ids are not row argmax"], "", None
    return [], "", True


def _check_different(first: np.ndarray, second: np.ndarray):
    if first.shape != second.shape or float(np.max(np.abs(first - second))) <= 1e-7:
        return ["different images produced identical score rows"], "", None
    return [], "", True


def stop(process: subprocess.Popen) -> None:
    for signum, grace in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, signum)
            for _ in range(grace * 5):
                process.poll()
                os.killpg(process.pid, 0)
                time.sleep(0.2)
        except (ProcessLookupError, PermissionError):
            return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()
    root, process = Path(tempfile.mkdtemp(prefix="kot-vla-self-check-")).resolve(), None
    try:
        observation, inverted = root / "observation.npz", root / "inverted.npz"
        make_observation(observation)
        make_observation(inverted, inverted=True)
        with socket.socket() as probe:
            probe.bind((HOST, 0))
            port = probe.getsockname()[1]
        argv = [str(Path(args.deployment).resolve() / "serve.sh"), "--port", str(port), "--model", os.path.abspath(args.model), "--device", args.device]
        with open(root / "serve.log", "wb") as log:
            process = check("start serve.sh", lambda: ([], " ".join(argv), subprocess.Popen(argv, cwd=args.deployment, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env={**os.environ, "KOT_OUT_MOUNTS": str(root)})))
        info = process and check("GET /kot/info", lambda: wait_info(process, port))
        if info:
            run_checks(port, root, info, observation, inverted)
    finally:
        if process:
            stop(process)
        if failed:
            tail = (root / "serve.log").read_text(errors="replace").splitlines()[-40:]
            print(f"--- the last {len(tail)} lines of deployment output ---", *tail, sep="\n")
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
