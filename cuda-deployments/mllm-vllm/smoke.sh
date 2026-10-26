#!/usr/bin/env bash
# smoke.sh -- the mllm-vllm contract, by hand, no kot.
#   ./smoke.sh /path/to/Qwen3.8-27B [device]
# Boots the deployment (the first boot compiles; minutes), then:
#   1. /kot/info answers with the engine's own facts;
#   2. a video decode twice: shape, dtype, logsumexp, byte-identity, the
#      stored ids = the raw rendered template (one video placeholder) + the
#      generated ids;
#   3. the replay prefill twice over the stored ids: byte-identity;
#   4. the teacher-forced acceptance: the tf reference golden's stored ids
#      for the same unit replayed here, the rows diffed against the
#      golden's rows (median renyi_inf reported; a wrong expansion or a
#      wrong frame selection reads orders above the band);
#   5. the speed surface (/kot/infer) over one video.
set -euo pipefail

MODEL="${1:?usage: smoke.sh /path/to/Qwen3.8-27B [device]}"
DEVICE="${2:-cuda}"
PORT="${PORT:-8802}"
RUN="$(mktemp -d /tmp/kot-mllm-vllm-smoke-XXXX)"
VIDEO="${VIDEO:-/workspace/kot/kot_store/inputs/videos/v1/video_430.mp4}"
REF="${REF:-/workspace/kot/kot_store/goldens/qwen3-8-27b-rtx5090-transformers-ref}"

echo "run dir: $RUN"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -n "${KOT_PYTHON:-}" ]; then PY="$KOT_PYTHON"
else PY="$(command -v python3)"; fi

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 3600); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -5 "$RUN/serve.log"; exit 1; }
  sleep 2
done
echo "== /kot/info"; cat "$RUN/info.json"; echo

score() { # tag mode max_new n_rows ids
  local ids="[]" n_rows=""
  if [ -n "${5:-}" ]; then ids="$5"; n_rows=", \"n_rows\": $4"; fi
  curl -s --max-time 3600 -X POST "http://127.0.0.1:$PORT/kot/score" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$1\", \"input_ids\": $ids, \"mode\": \"$2\", \"max_new_tokens\": $3,
         \"allow_eos\": true$n_rows, \"out_dir\": \"$RUN/$1\", \"seed\": 42,
         \"attachments\": {\"video\": \"$VIDEO\"},
         \"sampling\": {\"temperature\": 1.0, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0.0}}" \
    >"$RUN/$1.json"
}

score d1 decode 64
score d2 decode 64

"$PY" - "$RUN" "$MODEL" <<'EOF'
import json, sys
from pathlib import Path

import numpy as np

run, model = Path(sys.argv[1]), sys.argv[2]
video_token = json.load(open(f"{model}/config.json"))["video_token_id"]
blobs = []
for tag in ("d1", "d2"):
    reply = json.load(open(run / f"{tag}.json"))
    ids = np.load(run / tag / "input_ids.npy")
    lp = np.load(run / tag / "log_probs.npy")
    out = reply["output_ids"]
    assert lp.shape[0] == len(out) == reply["n_positions"], (tag, lp.shape, len(out))
    assert lp.dtype == np.float32 and np.isfinite(lp).all(), tag
    assert lp.shape[1] == reply["vocab_size"] == 248320, tag
    assert int((ids == video_token).sum()) == 1, \
        (tag, "the stored ids must carry exactly one video placeholder (the raw form)")
    blobs.append(lp.tobytes())
print("video decode: rows == generated, one raw placeholder, finite, byte-identical:",
      "PASS" if blobs[0] == blobs[1] else "FAIL")
EOF

REPLAY_IDS="$("$PY" - "$RUN" <<'EOF'
import sys
from pathlib import Path
import numpy as np
run = Path(sys.argv[1])
print(np.load(run / "d1" / "input_ids.npy").astype(int).tolist())
EOF
)"
N_ROWS="$("$PY" -c "import json; print(len(json.load(open('$RUN/d1.json'))['output_ids']))")"
"$PY" - "$RUN" "$REPLAY_IDS" "$N_ROWS" <<'EOF'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
ids = json.loads(sys.argv[2])
n_rows = int(sys.argv[3])
for tag in ("r1", "r2"):
    (run / f"{tag}.json").write_text(json.dumps({
        "prompt_id": tag, "input_ids": ids, "mode": "prefill",
        "max_new_tokens": 4, "allow_eos": False, "n_rows": n_rows,
        "out_dir": str(run / tag), "seed": 42,
        "attachments": {"video": "/workspace/kot/kot_store/inputs/videos/v1/video_430.mp4"},
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
    }))
print(f"replay bodies: stored ids {len(ids)}, n_rows {n_rows}")
EOF
score r1 prefill 4 "$N_ROWS" "$REPLAY_IDS"
score r2 prefill 4 "$N_ROWS" "$REPLAY_IDS"

"$PY" - "$RUN" <<'EOF'
import json, sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
blobs = []
for tag in ("r1", "r2"):
    reply = json.load(open(run / f"{tag}.json"))
    lp = np.load(run / tag / "log_probs.npy")
    ids = np.load(run / tag / "input_ids.npy")
    assert np.isfinite(lp).all(), tag
    assert lp.shape[0] == reply["n_positions"] > 0, (tag, lp.shape, reply["n_positions"])
    assert lp.shape[0] < len(ids), (tag, "the replay stores the tail's rows, not the full length")
    blobs.append(lp.tobytes())
print("video replay: rows == the tail's n_rows, finite, byte-identical:",
      "PASS" if blobs[0] == blobs[1] else "FAIL")
EOF

# The teacher-forced acceptance: the tf reference golden's stored ids for the
# same unit replayed here; the rows diffed against the golden's rows. The
# median renyi_inf is REPORTED: the cross-engine band 0.4-0.7 is an
# expectation, not a gate; a wrong expansion or a wrong frame selection
# reads orders above it.
"$PY" - "$RUN" "$REF" "$VIDEO" <<'EOF'
import json, sys
from pathlib import Path

import numpy as np

run, ref, video = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
unit = Path(video).stem
ref_ids = np.load(ref / "capture_0" / unit / "input_ids.npy").astype(int).tolist()
decode_ids = np.load(ref / "capture_0" / unit / "decode_ids.npy")
n_rows = len(decode_ids)
body = {
    "prompt_id": "parity", "input_ids": ref_ids, "mode": "prefill",
    "max_new_tokens": 4, "allow_eos": False, "n_rows": n_rows,
    "out_dir": str(run / "parity"), "seed": 42,
    "attachments": {"video": video},
    "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0},
}
(run / "parity.json").write_text(json.dumps(body))
print(f"parity body: ref ids {len(ref_ids)}, n_rows {n_rows}")
EOF
curl -s --max-time 3600 -X POST "http://127.0.0.1:$PORT/kot/score" \
  -H 'Content-Type: application/json' -d @"$RUN/parity.json" >"$RUN/parity.reply"

"$PY" - "$RUN" <<'EOF'
import json, sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
ref = Path("/workspace/kot/kot_store/goldens/qwen3-8-27b-rtx5090-transformers-ref")
reply = json.load(open(run / "parity.reply"))
lp = np.load(run / "parity" / "log_probs.npy")
assert np.isfinite(lp).all(), "parity rows not finite"
golden = np.load(ref / "capture_0" / "video_430" / "log_probs.npy")
n = min(len(golden), len(lp))
renyi = np.max(np.abs(golden[:n] - lp[:n]), axis=-1)
flips = int((golden[:n].argmax(1) != lp[:n].argmax(1)).sum())
print(f"parity vs tf golden video_430: rows {n}, flips {flips} ({100*flips/n:.2f}%), "
      f"median renyi_inf {np.median(renyi):.4f}, max {renyi.max():.4f}")
EOF

# The speed surface: one video decode through /kot/infer.
curl -s --max-time 3600 -X POST "http://127.0.0.1:$PORT/kot/infer" \
  -H 'Content-Type: application/json' \
  -d "{\"prompt_id\": \"speed_probe\", \"out_dir\": \"$RUN/speed_probe\",
       \"attachments\": {\"video\": \"$VIDEO\"}}" >"$RUN/infer.reply"
"$PY" - "$RUN" <<'EOF'
import json, sys
from pathlib import Path

run = Path(sys.argv[1])
reply = json.load(open(run / "infer.reply"))
assert reply.get("files") == ["text.txt"], reply
text = (run / "speed_probe" / "text.txt").read_text()
assert text.strip(), "empty speed output"
print(f"kot/infer: text.txt written, {len(text)} chars: PASS")
print("SMOKE DONE")
EOF
