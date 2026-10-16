#!/usr/bin/env bash
# smoke.sh -- run the contract's verification by hand, no kot, no docker.
#
#   ./smoke.sh /path/to/DeepSeek-OCR-2 [device]
#
# Starts kot_serve.sh on a free port, then checks:
#   1. /kot/info answers with the engine's own facts;
#   2. a teacher-forced prefill twice over one image: shape, dtype, logsumexp,
#      byte-identity (determinism under the engine's sampling and kernels);
#   3. a decode under each EOS gate (suppressed: exactly max_new_tokens rows;
#      natural: stop at EOS under the cap), each twice for byte-identity;
#   4. the chat surface the speed harness drives: a non-streaming probe
#      and a streaming request whose SSE chunks assemble into a transcription.
set -euo pipefail

MODEL="${1:?usage: smoke.sh /path/to/DeepSeek-OCR-2 [device]}"
DEVICE="${2:-cuda}"
PORT="${PORT:-8798}"
RUN="$(mktemp -d /tmp/kot-ocr-sglang-smoke-XXXX)"

if [ -z "${IMAGE:-}" ]; then
  for candidate in \
    /Users/bd/projects/kot/kot_store/inputs/images/v1/headline.png \
    /workspace/kot/kot_store/inputs/images/v1/headline.png \
    /workspace/kot_store/inputs/images/v1/headline.png; do
    if [ -f "$candidate" ]; then IMAGE="$candidate"; break; fi
  done
fi
if [ -z "${IMAGE:-}" ]; then
  echo "no fixture image: set IMAGE=/path/to/png (any image; >768px exercises crops)" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

# The interpreter for the checks: explicit override, the OCR venv, then PATH.
if [ -n "${PY:-}" ]; then :
elif [ -n "${KOT_SGLANG_PYTHON:-}" ]; then PY="$KOT_SGLANG_PYTHON"
elif [ -x /root/kotvenv-sglang-ocr/bin/python ]; then PY=/root/kotvenv-sglang-ocr/bin/python
else PY="$(command -v python3)"; fi

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 1800); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -5 "$RUN/serve.log"; exit 1; }
  sleep 1
done
echo "== /kot/info"; cat "$RUN/info.json"; echo

# The transcript's ids, frozen once by the checkpoint's own tokenizer
# (LlamaTokenizerFast, no remote code); see ocr-vllm's smoke for why a
# constant: loading it here would drag the checkpoint's remote imports into
# whatever venv runs this script.
# The version's frozen template ids, hand-verified against the checkpoint's
# own renderer: [BOS][<image>]"\nFree OCR.".
TEMPLATE="[0, 128815, 201, 21431, 126041, 16]"
IDS="[124438, 28, 13429, 56480, 25608, 39011, 2346, 20678, 15, 2700, 651, 8608]"
TEMPLATE_AND_IDS="${TEMPLATE%]}, ${IDS#[}"
TRANSCRIPT_LEN=12

score() { # tag mode allow_eos max_new
  # A decode sends the version's frozen template ids; a prefill sends the
  # template plus the fixed transcript and trims the rows to the
  # transcript's own (n_rows) - the shape the OCR replay stores.
  # no -f: a 500's body carries the deployment's own error and must be read;
  # --max-time so a hang cannot outlive the window.
  local ids="$TEMPLATE" n_rows=""
  if [ "$2" = "prefill" ]; then
    ids="$TEMPLATE_AND_IDS"
    n_rows=', "n_rows": '"$TRANSCRIPT_LEN"
  fi
  curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/kot/score" -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$1\", \"input_ids\": $ids, \"mode\": \"$2\", \"max_new_tokens\": $4,
         \"allow_eos\": $3$n_rows, \"out_dir\": \"$RUN/$1\", \"seed\": 42,
         \"sampling\": {\"temperature\": 0.0, \"top_p\": 1.0, \"top_k\": 0, \"min_p\": 0.0},
         \"attachments\": {\"image\": \"$IMAGE\"}}" >"$RUN/$1.json"
}

score p1 prefill false 4
score p2 prefill false 4
score d1 decode false 4
score d2 decode false 4
score e1 decode true 64
score e2 decode true 64

# ------------------------------------------------------------ 2026-10-12
# The capture's replay round-trip (STEP 1 -> STEP 2), which no smoke drove
# before this date: decode over a >768px image (the crop-tile path), then
# prefill with exactly the ids file the decode wrote - the request shape a
# capture replays. The regression pin: the stored ids carry ONE image
# placeholder, never the expanded block.
"$PY" - "$RUN" "$IMAGE" <<'PYEOF'
import sys
from pathlib import Path

from PIL import Image

run, image = Path(sys.argv[1]), sys.argv[2]
Image.open(image).resize((1024, 1024)).save(run / "big.png")
print("big.png: 1024x1024 written (crop-tile path)")
PYEOF

IMAGE="$RUN/big.png" score rt1 decode false 32
IMAGE="$RUN/big.png" score rt2 decode false 32

"$PY" - "$RUN" "$MODEL" <<'PYEOF'
import json
import sys
from pathlib import Path

import numpy as np
from transformers import LlamaTokenizerFast

run, model = Path(sys.argv[1]), sys.argv[2]
ids = np.asarray(np.load(run / "rt1" / "input_ids.npy"), dtype=int)
tokenizer = LlamaTokenizerFast.from_pretrained(model)
img_id = tokenizer.convert_tokens_to_ids("<image>")
count = int((ids == img_id).sum())
assert count == 1, f"stored decode ids carry {count} image placeholders; the raw form carries exactly one"
assert ids[:6].tolist() == [0, 128815, 201, 21431, 126041, 16], ids[:6]
for tag in ("rt_replay", "rt_replay2"):
    body = {
        "prompt_id": tag, "input_ids": ids.tolist(), "mode": "prefill",
        "max_new_tokens": 4, "allow_eos": False,
        "n_rows": int(len(ids) - 6),
        "out_dir": str(run / tag), "seed": 42,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
        "attachments": {"image": str(run / "big.png")},
    }
    (run / f"{tag}.json").write_text(json.dumps(body))
print(f"rt ids: {len(ids)} long, exactly one image placeholder")
PYEOF

curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/kot/score" \
  -H 'Content-Type: application/json' -d @"$RUN/rt_replay.json" > "$RUN/rt_replay.reply"
curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/kot/score" \
  -H 'Content-Type: application/json' -d @"$RUN/rt_replay2.json" > "$RUN/rt_replay2.reply"

"$PY" - "$RUN" <<'PYEOF'
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
blobs = []
for tag in ("rt_replay", "rt_replay2"):
    log_probs = np.load(run / tag / "log_probs.npy")
    ids = np.load(run / tag / "input_ids.npy")
    assert log_probs.shape[0] == len(ids) - 6, (tag, log_probs.shape, len(ids))
    assert np.isfinite(log_probs).all(), tag
    blobs.append(log_probs.tobytes())
print("STEP1->STEP2 replay: rows == the generated ids' rows, finite, byte-identical:",
      "PASS" if blobs[0] == blobs[1] else "FAIL")
PYEOF

"$PY" - "$RUN" <<'EOF'
import json, sys
import numpy as np

run = sys.argv[1]

def check(tag, mode):
    body = json.load(open(f"{run}/{tag}.json"))
    if "error" in body:
        print(f"{tag}: SERVER ERROR {body['error']}: {str(body.get('detail'))[:400]}")
        return None, None
    lp = np.load(f"{run}/{tag}/log_probs.npy")
    ids = np.load(f"{run}/{tag}/input_ids.npy")
    dev = float(np.abs(np.logaddexp.reduce(lp, axis=-1)).max())
    ok = lp.dtype == np.float32 and lp.flags["C_CONTIGUOUS"] and dev < 1e-3 \
        and bool(np.isfinite(lp).all()) and lp.shape[1] == body["vocab_size"]
    print(f"{tag}: shape={lp.shape} dtype={lp.dtype} max|logsumexp|={dev:.2e} {'PASS' if ok else 'FAIL'}")
    if mode == "prefill":
        # the stored ids are the template plus the transcript; the rows are
        # the transcript's own, trimmed by n_rows
        rows_ok = lp.shape[0] == body["n_positions"] == ids.shape[0] - 6
        print(f"{tag}: rows == n_positions == the transcript's rows {'PASS' if rows_ok else 'FAIL'}")
    else:
        out = body["output_ids"]
        tail = ids.tolist()[-len(out):] if out else []
        rows_ok = lp.shape[0] == len(out) == body["n_positions"] and tail == out
        print(f"{tag}: rows == output_ids == n_positions, ids end with them {'PASS' if rows_ok else 'FAIL'}")
    return lp.tobytes(), body["output_ids"]

b1 = check("p1", "prefill"); b2 = check("p2", "prefill")
print("prefill byte-identical:", "PASS" if b1 is not None and b1 == b2 else "FAIL")
a1, o1 = check("d1", "decode"); a2, o2 = check("d2", "decode")
print("decode (allow_eos=false, exactly max_new_tokens):",
      "PASS" if a1 is not None and a1 == a2 and o1 == o2 and len(o1) == 4 else "FAIL")
c1, k1 = check("e1", "decode"); c2, k2 = check("e2", "decode")
print("decode (allow_eos=true, natural stop, byte-identical):",
      "PASS" if c1 is not None and c1 == c2 and k1 == k2 and 0 < len(k1) <= 64 else "FAIL")
EOF

echo "== chat surface (what the speed harness drives)"
"$PY" - "$PORT" "$IMAGE" <<'EOF'
import base64, json, sys, urllib.request

port, image = sys.argv[1], sys.argv[2]
uri = "data:image/png;base64," + base64.b64encode(open(image, "rb").read()).decode()
messages = [{"role": "user", "content": [
    {"type": "text", "text": "Convert the document to markdown."},
    {"type": "image_url", "image_url": {"url": uri}},
]}]

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"messages": messages, "max_completion_tokens": 1}).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=600) as r:
    status = r.status
    probe = json.loads(r.read())
choice = probe["choices"][0]
print("probe (non-streaming, 1 token):",
      "PASS" if status == 200 and choice.get("finish_reason") else "FAIL",
      "| finish_reason:", choice.get("finish_reason"))

req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps({"messages": messages, "stream": True, "seed": 42,
                     "temperature": 0.0, "top_p": 1.0, "top_k": 0}).encode(),
    headers={"Content-Type": "application/json"},
)
text, chunks = "", 0
with urllib.request.urlopen(req, timeout=1800) as r:
    for raw in r:
        line = raw.decode().strip()
        if not line.startswith("data:"):
            continue
        payload = line.removeprefix("data:").strip()
        if payload == "[DONE]":
            break
        delta = json.loads(payload)["choices"][0].get("delta", {})
        text += delta.get("content") or ""
        chunks += 1
print(f"stream: {chunks} chunks, {len(text)} chars:",
      "PASS" if chunks > 1 and len(text) > 20 else "FAIL")
print("transcribed head:", repr(text[:120]))
EOF
