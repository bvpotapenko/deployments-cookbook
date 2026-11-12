#!/usr/bin/env python3
"""Check one MLLM deployment against the kot contract.
python3 self_check.py --deployment DIR --model PATH --device npu --video PATH
"""
import argparse
import base64
import json
import os
import signal
import socket
import subprocess
import sys
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
MAX_NEW_TOKENS = 16
SEED = 42
failed = []

def check(name, run):
    try:
        problems, detail, value = run()
    except urllib.error.HTTPError as error:
        problems = [f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"]
        detail, value = "", None
    except Exception as error:
        problems = [f"{type(error).__name__}: {error}"]
        detail, value = "", None
    if problems:
        failed.append(name)
    line = f"FAIL {name}: {'; '.join(problems)}" if problems else f"PASS {name}: {detail}"
    print(line, flush=True)
    return None if problems else value

def wait_info(process, port):
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/kot/info", timeout=10) as response:
                info = json.loads(response.read())
            keys = ("contract_version", "model_id", "engine", "device", "vocab_size")
            return [], ", ".join(f"{key} {info.get(key)}" for key in keys), info
        except OSError:
            time.sleep(1)
    detail = f"serve.sh exited with code {process.returncode}" if process.poll() is not None else ""
    return [detail or f"no answer within {LOAD_TIMEOUT_S} s"], "", None

def post(port, path, payload):
    request = urllib.request.Request(
        f"http://{HOST}:{port}{path}",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return response.read()

def score_answer(port, payload):
    body = post(port, "/kot/score", payload)
    lines = [json.loads(line) for line in body.splitlines() if line.strip()]
    if payload["mode"] == "prefill":
        lines = [json.loads(body)]
    results = [line for line in lines if "block" not in line]
    if not results or any("error" in line for line in lines):
        return [f"score returned an error: {body[:500]!r}"], "", None
    return [], f"{len(lines) - len(results)} block line(s)", results[-1]

def vocab_size(model):
    config = json.loads((Path(model) / "config.json").read_text())
    blocks = (config, config.get("text_config") or {}, config.get("language_config") or {})
    return next(block["vocab_size"] for block in blocks if "vocab_size" in block)

def arrays(payload, reply, model, info):
    named = reply.get("files") or {}
    paths = [Path(payload["out_dir"], named.get(key, "")) for key in ("log_probs", "input_ids")]
    if not all(path.is_file() for path in paths):
        return [f"named files are not on disk: {named}"], "", None
    log_probs, input_ids = (np.load(path, mmap_mode="r", allow_pickle=False) for path in paths)
    problems = []
    if log_probs.ndim != 2 or not np.issubdtype(log_probs.dtype, np.floating):
        problems.append(f"log_probs has invalid shape or dtype: {log_probs.shape} {log_probs.dtype}")
    if input_ids.ndim != 1 or not np.issubdtype(input_ids.dtype, np.integer):
        problems.append(f"input_ids has invalid shape or dtype: {input_ids.shape} {input_ids.dtype}")
    if not np.isfinite(log_probs).all():
        problems.append("log_probs contain NaN or inf")
    if log_probs.shape[1] != info.get("vocab_size") or log_probs.shape[1] != vocab_size(model):
        problems.append(f"vocabulary width is {log_probs.shape[1]}, expected {info.get('vocab_size')}")
    output_ids = [int(token) for token in reply.get("output_ids") or []]
    expected = len(output_ids) if payload["mode"] == "decode" else int(payload["n_rows"])
    if log_probs.shape[0] != expected or reply.get("n_positions") != expected:
        problems.append(f"row count is {log_probs.shape[0]}, expected {expected}")
    stored = input_ids.tolist()
    if payload["mode"] == "decode" and stored[-len(output_ids):] != output_ids:
        problems.append("stored input_ids do not end with output_ids")
    if payload["mode"] == "prefill" and stored != payload["input_ids"]:
        problems.append("prefill input_ids changed")
    return problems, f"log_probs {log_probs.dtype} {list(log_probs.shape)}, finite", (stored, len(output_ids))

def chat(port, payload, stream):
    body = post(port, "/v1/chat/completions", {**payload, "stream": stream})
    if not stream:
        reply = json.loads(body)
        return ([], "JSON completion", 1) if reply.get("choices") else ([f"no choices: {body[:300]!r}"], "", None)
    events = [line[5:].strip() for line in body.decode().splitlines() if line.startswith("data:")]
    if not events or events[-1] != "[DONE]":
        return ["stream does not end with data: [DONE]"], "", None
    chunks = [json.loads(event) for event in events[:-1]]
    if not chunks or any("error" in chunk for chunk in chunks):
        return ["stream has no valid completion chunks"], "", None
    return [], f"{len(chunks)} chunks, data: [DONE]", 1

def run_checks(port, root, info, model, video, sampling):
    attachments = {"video": str(Path(video).resolve())}
    decode = {
        "prompt_id": "self_check",
        "input_ids": [],
        "mode": "decode",
        "out_dir": str(root / "decode"),
        "attachments": attachments,
        "max_new_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        "sampling": sampling,
        "allow_eos": True,
        "n_rows": 0,
    }
    reply = check("decode answer", lambda: score_answer(port, decode))
    decoded = reply and check("decode files", lambda: arrays(decode, reply, model, info))
    if decoded:
        prefill = dict(decode, input_ids=decoded[0], mode="prefill", out_dir=str(root / "prefill"), n_rows=decoded[1])
        replay = check("prefill answer", lambda: score_answer(port, prefill))
        if replay:
            check("prefill files", lambda: arrays(prefill, replay, model, info))
    video_url = "file://" + str(Path(video).resolve())
    payload = {
        "model": info.get("model_id"),
        "messages": [{"role": "user", "content": [
            {"type": "video_url", "video_url": {"url": video_url}},
            {"type": "text", "text": "Describe what you see in the video."},
        ]}],
        "max_completion_tokens": MAX_NEW_TOKENS,
        "seed": SEED,
        **{key: sampling[key] for key in ("temperature", "top_p", "top_k") if key in sampling},
    }
    for stream in (False, True):
        check(f"chat, stream {str(stream).lower()}", lambda: chat(port, payload, stream))

def stop(process):
    for signum, grace in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, signum)
            for _ in range(grace * 5):
                process.poll()
                os.killpg(process.pid, 0)
                time.sleep(0.2)
        except (ProcessLookupError, PermissionError):
            return

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--video", required=True)
    args = parser.parse_args()
    root, process = Path(tempfile.mkdtemp(prefix="kot-mllm-self-check-")).resolve(), None
    try:
        with socket.socket() as probe:
            probe.bind((HOST, 0))
            port = probe.getsockname()[1]
        argv = [str(Path(args.deployment).resolve() / "serve.sh"), "--port", str(port), "--model", os.path.abspath(args.model), "--device", args.device]
        with open(root / "serve.log", "wb") as log:
            process = check("start serve.sh", lambda: ([], " ".join(argv), subprocess.Popen(argv, cwd=args.deployment, stdin=subprocess.DEVNULL, stdout=log, stderr=log, start_new_session=True, env={**os.environ, "KOT_OUT_MOUNTS": str(root)})))
        info = process and check("GET /kot/info", lambda: wait_info(process, port))
        if info:
            sampling = json.loads(os.environ.get("SAMPLING") or '{"temperature": 1.0, "top_p": 0.95, "top_k": 20}')
            run_checks(port, root, info, args.model, args.video, sampling)
    finally:
        if process:
            stop(process)
        if failed:
            tail = (root / "serve.log").read_text(errors="replace").splitlines()[-40:]
            print(f"--- the last {len(tail)} lines of the deployment output ---", *tail, sep="\n")
        import shutil
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
