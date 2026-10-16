#!/usr/bin/env bash
# smoke.sh -- run the contract's verification by hand, no kot, no docker.
#
#   ./smoke.sh /path/to/DeepSeek-OCR-2 [device]
#
# Starts kot_serve.sh on a free port, teacher-forces one transcript over one
# image twice, decodes twice under each EOS gate (suppressed: exactly
# max_new_tokens rows; natural: stop at EOS under the cap), checks shape,
# dtype, logsumexp and byte-identity, and prints PASS/FAIL lines.
set -euo pipefail

MODEL="${1:?usage: smoke.sh /path/to/DeepSeek-OCR-2 [device]}"
DEVICE="${2:-cpu}"
PORT="${PORT:-8767}"
RUN="$(mktemp -d /tmp/kot-ocr-smoke-XXXX)"
TEXT="${TEXT:-Breaking: Local Terminal Assistant Achieves Bit-Exact Log Probabilities}"

# The fixture image: the frozen headline from a kot store, found by trying the
# known layouts, or named outright with IMAGE=/path/to.png. Any image works;
# one wider or taller than 768 px exercises the crop path.
if [ -z "${IMAGE:-}" ]; then
  for candidate in \
    /Users/bd/projects/kot/kot_store/inputs/images/v1/headline.png \
    /workspace/kot/kot_store/inputs/images/v1/headline.png; do
    if [ -f "$candidate" ]; then IMAGE="$candidate"; break; fi
  done
fi
if [ -z "${IMAGE:-}" ]; then
  echo "no fixture image: set IMAGE=/path/to.png (any image; >768px exercises crops)" >&2
  exit 2
fi

# The interpreter, resolved the way kot_serve.sh resolves it.
if [ -n "${PY:-}" ]; then :
elif [ -n "${KOT_OCR_PY:-}" ]; then PY="$KOT_OCR_PY"
elif [ -x /root/kotvenv-ocr/bin/python ]; then PY=/root/kotvenv-ocr/bin/python
elif [ -x /root/kotvenv/bin/python ]; then PY=/root/kotvenv/bin/python
else PY="$(command -v python3)"; fi

HERE="$(cd "$(dirname "$0")" && pwd)"

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -5 "$RUN/serve.log"; exit 1; }
  sleep 1
done
echo "== /kot/info"; cat "$RUN/info.json"; echo

# The version's frozen template ids, hand-verified against the checkpoint's
# own renderer: [BOS][<image>]"\nFree OCR.".
# The transcript's ids, tokenized the way this venv does: no special tokens.
TEMPLATE="[0, 128815, 201, 21431, 126041, 16]"
IDS="$("$PY" - "$MODEL" "$TEXT" <<'EOF'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1], trust_remote_code=True)
print(tok.encode(sys.argv[2], add_special_tokens=False)[:12])
EOF
)"
TRANSCRIPT_LEN=$(( $(echo "$IDS" | tr -cd ',' | wc -c | tr -d ' ') + 1 ))
TEMPLATE_AND_IDS="${TEMPLATE%]}, ${IDS#[}"

score() { # tag mode allow_eos max_new
  # A decode sends the version's frozen template ids; a prefill sends the
  # template plus the transcript and trims the rows to the transcript's
  # own (n_rows) - the shape the OCR replay stores.
  local ids="$TEMPLATE" n_rows=""
  if [ "$2" = "prefill" ]; then
    ids="$TEMPLATE_AND_IDS"
    n_rows=', "n_rows": '"$TRANSCRIPT_LEN"
  fi
  curl -sf -X POST "http://127.0.0.1:$PORT/kot/score" -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$1\", \"input_ids\": $ids, \"mode\": \"$2\", \"max_new_tokens\": $4,
         \"allow_eos\": $3$n_rows, \"out_dir\": \"$RUN/$1\", \"seed\": 42,
         \"sampling\": {\"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0.0},
         \"attachments\": {\"image\": \"$IMAGE\"}}" >"$RUN/$1.json"
}

score p1 prefill false 4
score p2 prefill false 4
score d1 decode false 4
score d2 decode false 4
score e1 decode true 32
score e2 decode true 32

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
