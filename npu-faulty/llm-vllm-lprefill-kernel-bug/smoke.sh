#!/usr/bin/env bash
# smoke.sh -- run the contract's verification by hand, no kot.
#
#   ./smoke.sh /path/to/weights [device]
#
# Starts kot_serve.sh on a free port, waits for /kot/info, scores one prefill
# and one decode twice, checks shape, dtype, logsumexp ~ 0, verbatim ids,
# byte-identity, and probes the streaming chat endpoint the speed harness
# drives. Prints PASS/FAIL lines. On the Ascend host the server comes up
# through the docker wrap; on a machine with torch_npu and vllm provisioned,
# kot_serve.sh serves directly.
set -uo pipefail

MODEL="${1:?usage: smoke.sh /path/to/weights [device]}"
DEVICE="${2:-npu}"
PORT="${PORT:-8749}"
RUN="$(mktemp -d /tmp/kot-npu-vllm-smoke-XXXX)"
IDS='[151644, 872, 3838, 374, 1140, 30, 151645, 198]'

HERE="$(cd "$(dirname "$0")" && pwd)"

# /kot/score must write where this host can read: the smoke's own out_dir.
export KOT_OUT_MOUNTS="/tmp"

"$HERE/kot_serve.sh" --port "$PORT" --model "$MODEL" --device "$DEVICE" >"$RUN/serve.log" 2>&1 &
SERVE_PID=$!
trap 'kill "$SERVE_PID" 2>/dev/null || true; sleep 1; docker ps -q --filter name=bd-vllm-serve | xargs -r docker rm -f >/dev/null 2>&1 || true' EXIT

for _ in $(seq 1 900); do
  curl -sf "http://127.0.0.1:$PORT/kot/info" >"$RUN/info.json" 2>/dev/null && break
  kill -0 "$SERVE_PID" 2>/dev/null || { tail -30 "$RUN/serve.log"; exit 1; }
  sleep 1
done
echo "== /kot/info"; cat "$RUN/info.json"; echo

score() { # tag mode
  curl -sf --max-time 600 -X POST "http://127.0.0.1:$PORT/kot/score" -H 'Content-Type: application/json' \
    -d "{\"prompt_id\": \"$1\", \"input_ids\": $IDS, \"mode\": \"$2\", \"max_new_tokens\": 5,
         \"out_dir\": \"$RUN/$1\", \"seed\": 42, \"sampling\": {\"temperature\": 0.6, \"top_p\": 0.95, \"top_k\": 20, \"min_p\": 0.0}}" >"$RUN/$1.json"
}

score p1 prefill || { echo "prefill request failed"; tail -30 "$RUN/serve.log"; exit 1; }
score p2 prefill
score d1 decode || { echo "decode request failed"; tail -30 "$RUN/serve.log"; exit 1; }
score d2 decode

# The host interpreter's numpy may be broken; uv is the convention anyway.
VERIFY="python3"
if command -v uv >/dev/null 2>&1; then VERIFY="uv run --with numpy python3"; fi

"$VERIFY" - "$RUN" <<'EOF'
import json, sys
import numpy as np

run = sys.argv[1]

def check(tag, expect_rows, mode):
    body = json.load(open(f"{run}/{tag}.json"))
    lp = np.load(f"{run}/{tag}/log_probs.npy")
    ids = np.load(f"{run}/{tag}/input_ids.npy")
    dev = float(np.abs(np.logaddexp.reduce(lp, axis=-1)).max())
    ok = lp.shape == (expect_rows, body["vocab_size"]) and lp.dtype == np.float32 \
        and bool(lp.flags["C_CONTIGUOUS"]) and dev < 1e-3 and bool(np.isfinite(lp).all())
    print(f"{tag}: shape={lp.shape} dtype={lp.dtype} max|logsumexp|={dev:.2e} {'PASS' if ok else 'FAIL'}")
    if mode == "prefill":
        print(f"{tag}: ids verbatim {'PASS' if ids.tolist() == [151644, 872, 3838, 374, 1140, 30, 151645, 198] else 'FAIL'}")
    else:
        tail = ids.tolist()[-len(body["output_ids"]):]
        rows_ok = lp.shape[0] == len(body["output_ids"]) == 5 and tail == body["output_ids"]
        print(f"{tag}: rows==output_ids==max_new_tokens, ids end with them {'PASS' if rows_ok else 'FAIL'}")
    return lp.tobytes(), body["output_ids"]

b1 = check("p1", 7, "prefill"); b2 = check("p2", 7, "prefill")
print("prefill byte-identical:", "PASS" if b1 == b2 else "FAIL")
a1, o1 = check("d1", 5, "decode"); a2, o2 = check("d2", 5, "decode")
print("decode byte-identical:", "PASS" if a1 == a2 and o1 == o2 else "FAIL")
EOF

echo "== /v1/chat/completions stream probe"
STREAM=$(curl -sfN --max-time 120 -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d '{"model": "smoke", "messages": [{"role": "user", "content": "Say hello in one word."}],
       "max_completion_tokens": 8, "temperature": 1.0, "seed": 42, "stream": true}' | head -50)
CHUNKS=$(printf '%s' "$STREAM" | grep -c '^data: ' || true)
DONE=$(printf '%s' "$STREAM" | grep -c '^\[DONE\]\|^data: \[DONE\]' || true)
CONTENT=$(printf '%s' "$STREAM" | grep '^data: ' | head -50 | python3 -c "
import json, sys
text = ''
for line in sys.stdin:
    payload = line[5:].strip()
    if payload == '[DONE]':
        continue
    try:
        delta = json.loads(payload)['choices'][0].get('delta', {})
    except Exception:
        continue
    text += delta.get('content') or ''
print(repr(text))
" 2>/dev/null || echo "unparsed")
echo "stream chunks: $CHUNKS, done marker: $DONE, content: $CONTENT"
if [ "$CHUNKS" -ge 2 ] && [ "$DONE" -ge 1 ]; then echo "streaming chat: PASS"; else echo "streaming chat: FAIL"; fi

echo "smoke dir: $RUN"
