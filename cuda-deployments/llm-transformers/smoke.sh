#!/usr/bin/env bash
# smoke.sh -- run the contract's verification by hand, no kot, no docker.
#
#   ./smoke.sh /path/to/weights [device]
#
# Starts kot_serve.sh on a free port, scores one prefill and one decode twice,
# checks shape, dtype, logsumexp and byte-identity, and prints PASS/FAIL lines.
set -euo pipefail

MODEL="${1:?usage: smoke.sh /path/to/weights [device]}"
DEVICE="${2:-cpu}"
PORT="${PORT:-8747}"
RUN="$(mktemp -d /tmp/kot-llm-smoke-XXXX)"
# The smoke's ids come from the model's own tokenizer, so any vocabulary
# serves: a fixed id list breaks the next model whose vocab is smaller.
TEXT="${TEXT:-Summarize the history of the transistor in two sentences.}"

if [ -n "${KOT_PYTHON:-}" ]; then PY="$KOT_PYTHON"
elif [ -x /root/kotvenv/bin/python ]; then PY=/root/kotvenv/bin/python
else PY="$(command -v python3)"
fi
IDS="$("$PY" - "$MODEL" "$TEXT" <<'EOF'
import sys
from transformers import AutoTokenizer

print(AutoTokenizer.from_pretrained(sys.argv[1]).encode(sys.argv[2]))
EOF
)"

HERE="$(cd "$(dirname "$0")" && pwd)"

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 600); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -5 "$RUN/serve.log"; exit 1; }
  sleep 1
done
echo "== /kot/info"; cat "$RUN/info.json"; echo

score() { # tag mode
  curl -sf -X POST "http://127.0.0.1:$PORT/kot/score" -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$1\", \"input_ids\": $IDS, \"mode\": \"$2\", \"max_new_tokens\": 5,
         \"out_dir\": \"$RUN/$1\", \"seed\": 42, \"sampling\": {\"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0.0}}" >"$RUN/$1.json"
}

score p1 prefill
score p2 prefill
score d1 decode
score d2 decode

IDS_N=$(( $(echo "$IDS" | tr -cd ',' | wc -c | tr -d ' ') + 1 ))
python3 - "$RUN" "$IDS" "$IDS_N" <<'EOF'
import json, sys
import numpy as np

run, sent_ids, n_ids = sys.argv[1], json.loads(sys.argv[2]), int(sys.argv[3])

def check(tag, expect_rows, mode):
    body = json.load(open(f"{run}/{tag}.json"))
    lp = np.load(f"{run}/{tag}/log_probs.npy")
    ids = np.load(f"{run}/{tag}/input_ids.npy")
    dev = float(np.abs(np.logaddexp.reduce(lp, axis=-1)).max())
    ok = lp.shape == (expect_rows, body["vocab_size"]) and lp.dtype == np.float32 \
        and lp.flags["C_CONTIGUOUS"] and dev < 1e-3 and bool(np.isfinite(lp).all())
    print(f"{tag}: shape={lp.shape} dtype={lp.dtype} max|logsumexp|={dev:.2e} {'PASS' if ok else 'FAIL'}")
    if mode == "prefill":
        print(f"{tag}: ids verbatim {'PASS' if ids.tolist() == sent_ids else 'FAIL'}")
    else:
        tail = ids.tolist()[-len(body["output_ids"]):]
        rows_ok = lp.shape[0] == len(body["output_ids"]) and tail == body["output_ids"]
        print(f"{tag}: rows==output_ids, ids end with them {'PASS' if rows_ok else 'FAIL'}")
    return lp.tobytes(), body["output_ids"]

b1 = check("p1", n_ids - 1, "prefill"); b2 = check("p2", n_ids - 1, "prefill")
print("prefill byte-identical:", "PASS" if b1 == b2 else "FAIL")
a1, o1 = check("d1", 5, "decode"); a2, o2 = check("d2", 5, "decode")
print("decode byte-identical:", "PASS" if a1 == a2 and o1 == o2 else "FAIL")
EOF
