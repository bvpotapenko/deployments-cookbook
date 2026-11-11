#!/usr/bin/env python3
"""Self-check of the kot deployment contract for an OCR deployment.

    python3 self_check.py --deployment DIR --model PATH --device npu

"""
import argparse, base64, json, os, shutil, signal, socket, struct, subprocess, sys, tempfile, time
import urllib.error, urllib.request, zlib
from pathlib import Path

import numpy as np

HOST = "127.0.0.1"
LOAD_TIMEOUT_S = 900  # the contract's startup timeout
REQUEST_TIMEOUT_S = 600  # the longest silence one request may keep
STOP_GRACE_S = 15  # SIGTERM, then SIGKILL, as the benchmark stops a deployment
MAX_NEW_TOKENS, SEED = 16, 42
DIGITS = [11415, 29671, 29647, 23497, 31183, 31215, 29257, 31727, 31695, 31599]  # 1234567890, 3x5
failed = []

def check(name, run):
    """Run one check and print its line. The check's value, or None when it failed."""
    try:
        problems, detail, value = run()
    except urllib.error.HTTPError as error:
        problems = [f"HTTP {error.code}: {error.read().decode(errors='replace')[:500]}"]
    except Exception as error:  # every failure is a FAIL line, never a traceback
        problems = [f"{type(error).__name__}: {error}"]
    if problems:
        failed.append(name)
    line = f"FAIL {name}: {'; '.join(problems)}" if problems else f"PASS {name}: {detail}"
    print(line, flush=True)
    return None if problems else value

def write_png(path):
    """Black digits 1234567890 on white, 704x144 px, 8-bit gray: the same bytes every run."""
    bits = [np.array([glyph >> (14 - i) & 1 for i in range(15)]).reshape(5, 3) for glyph in DIGITS]
    line = np.pad(np.hstack([np.pad(glyph, ((0, 0), (0, 1))) for glyph in bits]), 2)
    pixels = ((1 - np.kron(line, np.ones((16, 16), int))) * 255).astype(np.uint8)
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)  # filter type 0, then the row
    header = struct.pack(">IIBBBBB", pixels.shape[1], pixels.shape[0], 8, 0, 0, 0, 0)  # 8-bit gray
    png = b"\x89PNG\r\n\x1a\n"
    for tag, data in ((b"IHDR", header), (b"IDAT", zlib.compress(raw, 9)), (b"IEND", b"")):
        png += struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    path.write_bytes(png)

def wait_info(process, port):
    """GET /kot/info until it answers. Fail at once when serve.sh exits."""
    deadline = time.monotonic() + LOAD_TIMEOUT_S
    while process.poll() is None and time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://{HOST}:{port}/kot/info", timeout=10) as response:
                info = json.loads(response.read())
            keys = ("contract_version", "model_id", "engine", "device", "vocab_size")
            return [], ", ".join(f"{key} {info.get(key)}" for key in keys), info
        except OSError:  # not listening yet, or not ready yet
            time.sleep(1)
    exited = f"serve.sh exited with code {process.returncode}" if process.poll() is not None else ""
    return [exited or f"no answer within {LOAD_TIMEOUT_S} s"], "", None

def post(port, path, payload):
    request = urllib.request.Request(f"http://{HOST}:{port}{path}", json.dumps(payload).encode(),
                                     {"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S) as response:
        return response.read()

def answer(port, payload):
    """Decode answers NDJSON: block lines, then the result line. Prefill answers one object."""
    body, prefill = post(port, "/kot/score", payload), payload["mode"] == "prefill"
    try:
        lines = [json.loads(x) for x in ([body] if prefill else body.splitlines()) if x.strip()]
    except ValueError:
        shape = "one JSON object (NDJSON is for decode only)" if prefill else "JSON lines"
        return [f"the body is not {shape}: {body[:300]!r}"], "", None
    results = [line for line in lines if isinstance(line, dict) and "block" not in line]
    if not results or any(not isinstance(line, dict) or "error" in line for line in lines):
        return [f"an error line, or no result line: {body[:300]!r}"], "", None
    detail = f"{len(lines) - len(results)} block line(s), then the result line"
    return [], "one JSON object" if prefill else detail, results[-1]

def arrays(payload, reply, model, info):
    """The files the reply names: on disk; dtype, rank, finiteness, width; the row rule."""
    named = reply.get("files") or {}
    paths = [Path(payload["out_dir"], named.get(key, "")) for key in ("log_probs", "input_ids")]
    if not all(path.is_file() for path in paths):
        return [f"files.log_probs and files.input_ids are not both on disk: {named}"], "", None
    log_probs, input_ids = (np.load(path, mmap_mode="r", allow_pickle=False) for path in paths)
    if log_probs.ndim != 2 or not np.issubdtype(log_probs.dtype, np.floating) \
            or input_ids.ndim != 1 or not np.issubdtype(input_ids.dtype, np.integer):
        return [f"log_probs {log_probs.dtype} {log_probs.shape} and input_ids {input_ids.dtype} "
                f"{input_ids.shape} are not float [rows, vocab] and integer [n]"], "", None
    config = json.loads((Path(model) / "config.json").read_text())  # config.vocab_size: the width
    blocks = (config, config.get("text_config") or {}, config.get("language_config") or {})
    vocab = next((block["vocab_size"] for block in blocks if "vocab_size" in block), None)
    count, stored, decode = log_probs.shape[0], input_ids.tolist(), payload["mode"] == "decode"
    output_ids = [int(token) for token in reply.get("output_ids") or []]
    expected = len(output_ids) if decode else payload["n_rows"]
    problems = [] if np.isfinite(log_probs).all() else ["log_probs hold NaN or inf"]
    if not log_probs.shape[1] == info.get("vocab_size") == vocab:
        problems.append(f"rows are {log_probs.shape[1]} wide, /kot/info vocab_size is "
                        f"{info.get('vocab_size')}, config.json vocab_size is {vocab}")
    if not 0 < count < len(stored) or count != expected or reply.get("n_positions") != count:
        problems.append(f"{count} rows for {len(stored)} stored ids and n_positions "
                        f"{reply.get('n_positions')}; {expected} rows expected")
    if decode and stored[len(stored) - len(output_ids):] != output_ids:
        problems.append("the stored input_ids do not end with output_ids: store prompt + generated")
    if not decode and stored != payload["input_ids"]:
        problems.append("the stored input_ids are not the ids sent: consume them verbatim")
    return problems, f"log_probs {log_probs.dtype} {list(log_probs.shape)}, finite", (stored, count)

def chat(port, payload, stream):
    """One completion, read to the end of the response: a response that never ends fails."""
    body = post(port, "/v1/chat/completions", {**payload, "stream": stream})
    if not stream:
        reply = json.loads(body)
        ok = isinstance(reply, dict) and isinstance(reply.get("choices"), list)
        return [] if ok else [f"no completion with choices: {body[:300]!r}"], "JSON completion", 1
    events = [line[5:].strip() for line in body.decode().splitlines() if line.startswith("data:")]
    if not events or events[-1] != "[DONE]":
        return [f"the stream does not end with data: [DONE] ({len(events)} data lines)"], "", None
    chunks = [json.loads(event) for event in events[:-1]]
    if not chunks or any(not isinstance(chunk, dict) or "error" in chunk for chunk in chunks):
        return [f"no completion chunk before [DONE], or an error chunk: {events[:3]}"], "", None
    return [], f"{len(chunks)} chunks, data: [DONE], then the response ended", 1

def run_checks(port, root, info, model):
    """The decode, its replay as a prefill, then the chat route without and with streaming."""
    sampling, image = json.loads(os.environ.get("SAMPLING") or "{}"), root / "digits.png"
    write_png(image)
    decode = dict(prompt_id="self_check", input_ids=[], mode="decode", out_dir=str(root / "decode"),
                  attachments={"image": str(image)}, max_new_tokens=MAX_NEW_TOKENS, seed=SEED,
                  sampling=sampling, allow_eos=True, n_rows=0)
    reply = check("decode answer", lambda: answer(port, decode))
    decoded = reply and check("decode files", lambda: arrays(decode, reply, model, info))
    if decoded:  # the replay: the stored ids verbatim, n_rows the generated ones, the same image
        prefill = dict(decode, input_ids=decoded[0], mode="prefill", out_dir=str(root / "prefill"),
                       max_new_tokens=0, sampling={}, allow_eos=False, n_rows=decoded[1])
        replay = check("prefill answer", lambda: answer(port, prefill))
        if replay:
            check("prefill files", lambda: arrays(prefill, replay, model, info))
    url = "data:image/png;base64," + base64.b64encode(image.read_bytes()).decode()
    parts = [dict(type="text", text="Free OCR."), dict(type="image_url", image_url=dict(url=url))]
    fields = {key: sampling[key] for key in ("temperature", "top_p", "top_k") if key in sampling}
    payload = dict(model=info.get("model_id"), messages=[dict(role="user", content=parts)],
                   max_completion_tokens=MAX_NEW_TOKENS, seed=SEED, **fields)
    for stream in (False, True):
        check(f"chat, stream {str(stream).lower()}", lambda: chat(port, payload, stream))

def stop(process):
    """SIGTERM the whole process group, then SIGKILL what is left."""
    for signum, grace in ((signal.SIGTERM, STOP_GRACE_S), (signal.SIGKILL, 5)):
        try:
            os.killpg(process.pid, signum)
            for _ in range(grace * 5):
                process.poll()  # reap the leader, so a group of the dead reads as gone
                os.killpg(process.pid, 0)
                time.sleep(0.2)
        except (ProcessLookupError, PermissionError):  # gone; macOS answers EPERM for zombies
            return

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for flag in ("--deployment", "--model", "--device"):
        parser.add_argument(flag, required=True)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))  # so that the finally below runs
    root, process = Path(tempfile.mkdtemp(prefix="kot-self-check-")).resolve(), None
    try:
        with socket.socket() as probe:  # a loopback port nothing listens on now
            probe.bind((HOST, 0))
            port = probe.getsockname()[1]
        argv = [str(Path(args.deployment).resolve() / "serve.sh"), "--port", str(port),
                "--model", os.path.abspath(args.model), "--device", args.device]
        with open(root / "serve.log", "wb") as log:
            process = check("start serve.sh", lambda: ([], " ".join(argv), subprocess.Popen(
                argv, cwd=args.deployment, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                start_new_session=True, env={**os.environ, "KOT_OUT_MOUNTS": str(root)})))
        info = process and check("GET /kot/info", lambda: wait_info(process, port))
        if info:
            run_checks(port, root, info, args.model)
    except KeyboardInterrupt:
        check("interrupted", lambda: (["Ctrl-C"], "", None))
    finally:
        if process:
            stop(process)
        if failed:
            tail = (root / "serve.log").read_text(errors="replace").splitlines()[-40:]
            print(f"--- the last {len(tail)} lines of the deployment's output ---", *tail, sep="\n")
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
