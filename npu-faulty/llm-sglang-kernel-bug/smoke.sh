#!/usr/bin/env bash
# smoke.sh -- run the contract's verification by hand, no kot.
#
#   ./smoke.sh /path/to/weights [device]
#
# Starts kot_serve.sh on a free port, scores one prefill and one decode twice,
# probes the streaming chat surface, checks shape, dtype, logsumexp and
# byte-identity, and prints PASS/FAIL lines. On the Ascend host the server
# comes up through the docker wrap.
set -euo pipefail

MODEL="${1:?usage: smoke.sh /path/to/weights [device]}"
DEVICE="${2:-npu}"
PORT="${PORT:-8749}"
RUN="$(mktemp -d /tmp/kot-npu-sglang-smoke-XXXX)"
IDS='[151644, 872, 3838, 374, 1140, 30, 151645, 198]'

HERE="$(cd "$(dirname "$0")" && pwd)"

# /kot/score must write where this host can read: the smoke's own out_dir.
export KOT_OUT_MOUNTS="/tmp"

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true' EXIT

for _ in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -30 "$RUN/serve.log"; exit 1; }
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

# The chat surface the speed harness drives: one streamed completion. The
# probe asserts the SSE shape; a silent non-stream reply fails here.
STREAM="$RUN/stream.sse"
curl -sfN -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
  -d '{"model": "probe", "messages": [{"role": "user", "content": "Say hello."}],
       "max_completion_tokens": 8, "seed": 7, "temperature": 1.0, "stream": true}' \
  >"$STREAM" 2>/dev/null
CHUNKS="$(grep -c '^data: {' "$STREAM" || true)"
if grep -q '^\[DONE\]' "$STREAM" || grep -q 'data: \[DONE\]' "$STREAM"; then DONE=yes; else DONE=no; fi
echo "== stream: $CHUNKS chunks, [DONE]=$DONE (need > 0 / yes)"

# The host interpreter's numpy may be broken (the Ascend server's is); uv is
# the convention anyway. Fall back to plain python3 where uv is absent. An
# array, not a quoted string: "$VERIFY" on a multi-word value is one command
# name, and the verifier silently never ran that way.
VERIFY=(python3)
if command -v uv >/dev/null 2>&1; then VERIFY=(uv run --with numpy python3); fi

"${VERIFY[@]}" - "$RUN" <<'EOF'
import json, sys
import numpy as np

run = sys.argv[1]
info = json.load(open(f"{run}/info.json"))

def check(tag, expect_rows, mode):
    body = json.load(open(f"{run}/{tag}.json"))
    lp = np.load(f"{run}/{tag}/log_probs.npy")
    ids = np.load(f"{run}/{tag}/input_ids.npy")
    dev = float(np.abs(np.logaddexp.reduce(lp, axis=-1)).max())
    # The stash reads the raw pre-sampling logits, so every entry must be
    # finite: -inf at exact token columns was the stash-hook bug wearing a
    # "known limit" disguise, and tolerating it let garbage pass the smoke.
    finite_ok = bool(np.isfinite(lp).all())
    ok = lp.shape == (expect_rows, info["vocab_size"]) and lp.dtype == np.float32 \
        and lp.flags["C_CONTIGUOUS"] and dev < 1e-3 and finite_ok
    print(f"{tag}: shape={lp.shape} dtype={lp.dtype} max|logsumexp|={dev:.2e} {'PASS' if ok else 'FAIL'}")
    if mode == "prefill":
        verbatim = ids.tolist() == [151644, 872, 3838, 374, 1140, 30, 151645, 198]
        print(f"{tag}: ids verbatim {'PASS' if verbatim else 'FAIL'}")
    else:
        tail = ids.tolist()[-len(body["output_ids"]):]
        rows_ok = lp.shape[0] == len(body["output_ids"]) and tail == body["output_ids"]
        print(f"{tag}: rows==output_ids, ids end with them {'PASS' if rows_ok else 'FAIL'}")
    return lp.tobytes(), body["output_ids"]

b1 = check("p1", 7, "prefill"); b2 = check("p2", 7, "prefill")
print("prefill byte-identical:", "PASS" if b1 == b2 else "FAIL")
a1, o1 = check("d1", 5, "decode"); a2, o2 = check("d2", 5, "decode")
print("decode byte-identical:", "PASS" if a1 == a2 and o1 == o2 else "FAIL")
EOF
