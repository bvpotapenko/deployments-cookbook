#!/usr/bin/env bash
# smoke.sh -- run the contract's verification by hand, no kot, no docker.
#
#   ./smoke.sh /path/to/DeepSeek-OCR-2 [device]
#
# Starts kot_serve.sh on a free port (host mode: through the docker wrap, so
# the first run builds the image first), teacher-forces one transcript over
# one image twice, decodes twice under each EOS gate (suppressed: exactly
# max_new_tokens rows; natural: stop at EOS under the cap), replays a
# >768px decode's stored ids through prefill (the capture's round-trip),
# checks shape, dtype, logsumexp and byte-identity, and prints PASS/FAIL
# lines.
set -euo pipefail

MODEL="${1:?usage: smoke.sh /path/to/DeepSeek-OCR-2 [device]}"
DEVICE="${2:-npu}"
PORT="${PORT:-8767}"
RUN="$(mktemp -d /tmp/kot-ocr-smoke-XXXX)"

# The fixture image: the frozen headline from a kot store, found by trying the
# known layouts, or named outright with IMAGE=/path/to.png. Any image works;
# one wider or taller than 768 px exercises the crop path.
if [ -z "${IMAGE:-}" ]; then
  for candidate in \
    "$HOME/kot_store/inputs/images/v1/headline.png" \
    /Users/bd/projects/kot/kot_store/inputs/images/v1/headline.png \
    /workspace/kot/kot_store/inputs/images/v1/headline.png; do
    if [ -f "$candidate" ]; then IMAGE="$candidate"; break; fi
  done
fi
if [ -z "${IMAGE:-}" ]; then
  echo "no fixture image: set IMAGE=/path/to.png (any image; >768px exercises crops)" >&2
  exit 2
fi

# Host mode: the server runs inside the wrap container, which mounts /tmp and
# the model dir but not the store the fixture came from. Copy the image into
# the mounted run dir so the request's path exists inside the container too.
cp "$IMAGE" "$RUN/fixture.png"
IMAGE="$RUN/fixture.png"

# The verification interpreter needs numpy only (the transcript below is
# hardcoded ids, and the crop fixture below is written with stdlib python).
if [ -n "${PY:-}" ]; then :
elif [ -n "${KOT_OCR_PY:-}" ]; then PY="$KOT_OCR_PY"
elif [ -x /root/kotvenv-ocr/bin/python ]; then PY=/root/kotvenv-ocr/bin/python
elif [ -x /root/kotvenv/bin/python ]; then PY=/root/kotvenv/bin/python
elif [ -x "$HOME/kot/.venv/bin/python" ]; then PY="$HOME/kot/.venv/bin/python"
else PY="$(command -v python3)"; fi

# Host mode: the server runs inside the wrap container, and /kot/score writes
# into $RUN under /tmp -- mount /tmp into the container at the same path.
export KOT_OUT_MOUNTS="${KOT_OUT_MOUNTS:-/tmp}"

HERE="$(cd "$(dirname "$0")" && pwd)"

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

# First run builds the image first; 900 s is the same budget the capture gives.
for _ in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -5 "$RUN/serve.log"; exit 1; }
  sleep 1
done
echo "== /kot/info"; cat "$RUN/info.json"; echo

# The version's frozen template ids, hand-verified against the checkpoint's
# own renderer: [BOS][<image>]"\nFree OCR.". The transcript's ids are ordinary
# hardcoded ids: the checks below are shape, dtype, logsumexp and
# byte-identity, never content. IDS keeps the list's brackets because the
# splice below strips them.
TEMPLATE="[0, 128815, 201, 21431, 126041, 16]"
IDS="[3376, 25, 2546, 1526, 2641, 11, 2794, 1917, 190, 330, 1850, 596]"
TRANSCRIPT_LEN=$(( $(echo "$IDS" | tr -cd ',' | wc -c | tr -d ' ') + 1 ))
TEMPLATE_AND_IDS="${TEMPLATE%]}, ${IDS#[}"

score() { # tag mode allow_eos max_new
  # A decode sends the version's frozen template ids; a prefill sends the
  # template plus the transcript and trims the rows to the transcript's
  # own (n_rows) - the shape the OCR replay stores.
  local ids="$TEMPLATE" n_rows="" reply
  if [ "$2" = "prefill" ]; then
    ids="$TEMPLATE_AND_IDS"
    n_rows=', "n_rows": '"$TRANSCRIPT_LEN"
  fi
  # No curl -f: it would eat the server's error body and die silently, and
  # the server reports failures as JSON, so read them.
  reply=$(curl -s -X POST "http://127.0.0.1:$PORT/kot/score" -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$1\", \"input_ids\": $ids, \"mode\": \"$2\", \"max_new_tokens\": $4,
         \"allow_eos\": $3$n_rows, \"out_dir\": \"$RUN/$1\", \"seed\": 42,
         \"sampling\": {\"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0.0, \"repetition_penalty\": 1.0, \"presence_penalty\": 0.0},
         \"attachments\": {\"image\": \"$IMAGE\"}}") \
    || { echo "score $1: request failed (curl rc=$?)" >&2; exit 1; }
  printf "%s" "$reply" > "$RUN/$1.json"
  case "$reply" in
    *'"prompt_id"'*) ;;
    *) echo "score $1: server refused: $reply" >&2; exit 1 ;;
  esac
}

score p1 prefill false 4
score p2 prefill false 4
score d1 decode false 4
score d2 decode false 4
score e1 decode true 32
score e2 decode true 32

# ------------------------------------------------------------- capture replay
# The capture's replay round-trip (STEP 1 -> STEP 2): decode over a >768px
# image (the crop-tile path), then prefill with exactly the ids file the
# decode wrote - the request shape a capture replays. The regression pin:
# the stored ids carry ONE image placeholder, never the expanded block.
# The image is written with stdlib python (zlib + struct): no PIL needed,
# and the checks are mechanics, never OCR content.
"$PY" - "$RUN" <<'PYEOF'
import struct
import sys
import zlib
from pathlib import Path

width = height = 1024
row = bytes(range(256)) * (width * 3 // 256 + 1)
raw = b"".join(b"\x00" + row[: width * 3] for _ in range(height))


def chunk(tag, data):
    return (
        struct.pack(">I", len(data))
        + tag
        + data
        + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
    )


png = b"\x89PNG\r\n\x1a\n"
png += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
png += chunk(b"IDAT", zlib.compress(raw))
png += chunk(b"IEND", b"")
(Path(sys.argv[1]) / "big.png").write_bytes(png)
print("big.png: 1024x1024 written (crop-tile path)")
PYEOF

IMAGE="$RUN/big.png" score rt1 decode false 32
IMAGE="$RUN/big.png" score rt2 decode false 32

"$PY" - "$RUN" <<'PYEOF'
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
ids = np.asarray(np.load(run / "rt1" / "input_ids.npy"), dtype=int)
count = int((ids == 128815).sum())  # the deployment's own IMAGE_TOKEN_ID
assert count == 1, f"stored decode ids carry {count} image placeholders; the raw form carries exactly one"
assert ids[:6].tolist() == [0, 128815, 201, 21431, 126041, 16], ids[:6]
for tag in ("rt_replay", "rt_replay2"):
    body = {
        "prompt_id": tag, "input_ids": ids.tolist(), "mode": "prefill",
        "max_new_tokens": 4, "allow_eos": False,
        "n_rows": int(len(ids) - 6),
        "out_dir": str(run / tag), "seed": 42,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                     "repetition_penalty": 1.0, "presence_penalty": 0.0},
        "attachments": {"image": str(run / "big.png")},
    }
    (run / f"{tag}.json").write_text(json.dumps(body))
print(f"rt ids: {len(ids)} long, exactly one image placeholder")
PYEOF

post_replay() { # tag
  local reply
  reply=$(curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/kot/score" \
    -H 'Content-Type: application/json' -d @"$RUN/$1.json") \
    || { echo "replay $1: request failed (curl rc=$?)" >&2; exit 1; }
  printf "%s" "$reply" > "$RUN/$1.reply"
  case "$reply" in
    *'"prompt_id"'*) ;;
    *) echo "replay $1: server refused: $reply" >&2; exit 1 ;;
  esac
}

post_replay rt_replay
post_replay rt_replay2

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
    lp = np.load(f"{run}/{tag}/log_probs.npy")
    ids = np.load(f"{run}/{tag}/input_ids.npy")
    dev = float(np.abs(np.logaddexp.reduce(lp, axis=-1)).max())
    ok = lp.dtype == np.float32 and lp.flags["C_CONTIGUOUS"] and dev < 1e-3 \
        and bool(np.isfinite(lp).all()) and lp.shape[1] == body["vocab_size"]
    print(f"{tag}: shape={lp.shape} dtype={lp.dtype} max|logsumexp|={dev:.2e} {'PASS' if ok else 'FAIL'}")
    if mode == "prefill":
        rows_ok = lp.shape[0] == body["n_positions"] == ids.shape[0] - 6
        print(f"{tag}: rows == len(transcript) - 1 {'PASS' if rows_ok else 'FAIL'}")
    else:
        tail = ids.tolist()[-len(body["output_ids"]):]
        rows_ok = lp.shape[0] == len(body["output_ids"]) and tail == body["output_ids"]
        print(f"{tag}: rows == output_ids, ids end with them {'PASS' if rows_ok else 'FAIL'}")
    return lp.tobytes(), body["output_ids"]

b1 = check("p1", "prefill"); b2 = check("p2", "prefill")
print("prefill byte-identical:", "PASS" if b1 == b2 else "FAIL")
a1, o1 = check("d1", "decode"); a2, o2 = check("d2", "decode")
print("decode (allow_eos=false, exactly max_new_tokens):",
      "PASS" if a1 == a2 and o1 == o2 and len(o1) == 4 else "FAIL")
c1, k1 = check("e1", "decode"); c2, k2 = check("e2", "decode")
print("decode (allow_eos=true, natural stop, byte-identical):",
      "PASS" if c1 == c2 and k1 == k2 and 0 < len(k1) <= 32 else "FAIL")
EOF
