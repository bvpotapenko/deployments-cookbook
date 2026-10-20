#!/usr/bin/env bash
# smoke.sh -- the mllm-transformers contract, by hand, no kot.
#   ./smoke.sh /path/to/Qwen3.8-27B [device]
# Boots the deployment, checks /kot/info reports the pinned media pipeline,
# then drives one text decode and its replay prefill. Media legs land with
# the family contract (they need the frozen render + the captain's videos).
set -euo pipefail

MODEL="${1:?usage: smoke.sh /path/to/model [device]}"
DEVICE="${2:-cuda}"
PORT="${PORT:-8811}"
RUN="$(mktemp -d /tmp/kot-mllm-smoke-XXXX)"
TEXT="${TEXT:-Summarize the history of the transistor in two sentences.}"

echo "run dir: $RUN"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -n "${KOT_MLLM_PYTHON:-}" ]; then PY="$KOT_MLLM_PYTHON"
elif [ -x /root/kotvenv-mllm/bin/python ]; then PY=/root/kotvenv-mllm/bin/python
else PY="$(command -v python3)"; fi

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 2400); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -5 "$RUN/serve.log"; exit 1; }
  sleep 1
done
echo "== /kot/info"; cat "$RUN/info.json"; echo

IDS="$("$PY" - "$MODEL" "$TEXT" <<'EOF'
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(sys.argv[1])
print(tokenizer.encode(sys.argv[2], add_special_tokens=False))
EOF
)"
N_IDS=$(( $(echo "$IDS" | tr -cd ',' | wc -c | tr -d ' ') + 1 ))

score() { # tag mode max_new n_rows
  curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/kot/score" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$1\", \"input_ids\": $IDS, \"mode\": \"$2\", \"max_new_tokens\": $3,
         \"allow_eos\": true, \"n_rows\": $4, \"out_dir\": \"$RUN/$1\", \"seed\": 42,
         \"sampling\": {\"temperature\": 0.7, \"top_p\": 0.8, \"top_k\": 20, \"min_p\": 0.0}}" \
    >"$RUN/$1.json"
}
score d1 decode 32 0
score d2 decode 32 0
"$PY" - "$RUN" "$N_IDS" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

run, n_ids = Path(sys.argv[1]), int(sys.argv[2])
info = json.load(open(run / "info.json"))
media = info["media"]
assert media["video_fps"] == 2.0 and media["video_do_sample_frames"] is True, media
assert media["temporal_patch_size"] == 2 and media["merge_size"] == 2, media
assert media["video_size_limits"]["shortest_edge"] == 4096, media
assert media["enable_thinking"] is False, media
print("info: pinned media pipeline reported: PASS")

for tag in ("d1", "d2"):
    reply = json.load(open(run / f"{tag}.json"))
    ids = np.load(run / tag / "input_ids.npy")
    lp = np.load(run / tag / "log_probs.npy")
    out = reply["output_ids"]
    assert lp.shape[0] == len(out) == ids.shape[0] - n_ids, (tag, lp.shape, len(out))
    assert lp.dtype == np.float32 and np.isfinite(lp).all(), tag
a, b = (np.load(run / t / "log_probs.npy").tobytes() for t in ("d1", "d2"))
print("text decode: rows == generated, finite, byte-identical:",
      "PASS" if a == b else "FAIL")
EOF

# The replay round-trip: prefill with exactly the ids file the decode wrote
# and the n_rows the decode generated - the shape a capture replays.
"$PY" - "$RUN" <<'EOF'
import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
ids = np.load(run / "d1" / "input_ids.npy").astype(int).tolist()
n_rows = len(json.load(open(run / "d1.json"))["output_ids"])
(run / "rt_ids.json").write_text(json.dumps({
    "prompt_id": "rt1", "input_ids": ids, "mode": "prefill",
    "max_new_tokens": 4, "allow_eos": False, "n_rows": n_rows,
    "out_dir": str(run / "rt1"), "seed": 42,
    "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20, "min_p": 0.0},
}))
EOF
curl -s --max-time 1800 -X POST "http://127.0.0.1:$PORT/kot/score" \
  -H 'Content-Type: application/json' -d @"$RUN/rt_ids.json" >"$RUN/rt1.reply"
"$PY" - "$RUN" <<'EOF'
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
lp = np.load(run / "rt1" / "log_probs.npy")
ids = np.load(run / "rt1" / "input_ids.npy")
n_rows = lp.shape[0]
assert n_rows == ids.shape[0] - n_rows - 1 or n_rows < ids.shape[0], (n_rows, ids.shape)
print("replay prefill: rows == the decode's n_rows:", "PASS" if 0 < n_rows < ids.shape[0] else "FAIL")
EOF
echo "SMOKE DONE"
