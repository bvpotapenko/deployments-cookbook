#!/usr/bin/env bash
# kot deployment: llm family, vllm engine.
#
#   ./kot_serve.sh --port <P> --model <weights dir> --device <torch device>
#
# Self-provisioning: if the ambient python cannot import the pinned vllm, a
# venv is created in a writable scratch location and filled OFFLINE from
# ./wheels (the graded container has no network). The wheel set is
# self-contained: vllm 0.11.0 with its own torch 2.8.0 (cu128,
# Blackwell-capable) and CUDA libraries, so it runs on any host with a CUDA
# 12.x driver regardless of the image's own torch.
set -euo pipefail

VLLM_PIN="0.11.0"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PORT="" MODEL="" DEVICE=""
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="${2:?}"; shift 2 ;;
    --port=*) PORT="${1#*=}"; shift ;;
    --model) MODEL="${2:?}"; shift 2 ;;
    --model=*) MODEL="${1#*=}"; shift ;;
    --device) DEVICE="${2:?}"; shift 2 ;;
    --device=*) DEVICE="${1#*=}"; shift ;;
    *) echo "kot_serve.sh: unknown argument '$1' (expected --port --model --device)" >&2; exit 2 ;;
  esac
done
: "${PORT:?--port is required}" "${MODEL:?--model is required}" "${DEVICE:?--device is required}"

# The speed harness opens hundreds of connections at once; vLLM adds its ZMQ
# sockets and shm handles on top. The default 1024 fd limit kills the engine
# core mid-benchmark with "Too many open files".
ULIMIT_SOFT=$(ulimit -Sn) ULIMIT_HARD=$(ulimit -Hn)
if [ "$ULIMIT_HARD" -gt "$ULIMIT_SOFT" ] 2>/dev/null; then
  ulimit -n "$ULIMIT_HARD" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
fi

find_python() {
  if [ -n "${KOT_PYTHON:-}" ]; then
    printf '%s' "$KOT_PYTHON"
    return 0
  fi
  if python3 -c "import vllm" >/dev/null 2>&1; then
    printf '%s' "python3"
    return 0
  fi
  local venv="${KOT_VENV:-/scratch/kot-vllm-venv}"
  if ! mkdir -p "$(dirname "$venv")" 2>/dev/null || ! mkdir -p "$venv" 2>/dev/null; then
    venv="/tmp/kot-vllm-venv"
    mkdir -p "$venv"
  fi
  if [ ! -x "$venv/bin/python" ]; then
    echo "[kot-vllm] creating venv at $venv" >&2
    python3 -m venv "$venv"
  fi
  if ! "$venv/bin/python" -c "import vllm" >/dev/null 2>&1; then
    echo "[kot-vllm] installing vllm $VLLM_PIN offline from $DIR/wheels" >&2
    "$venv/bin/pip" install --no-index --no-cache-dir \
      --find-links "$DIR/wheels" "vllm==$VLLM_PIN" >&2
  fi
  printf '%s' "$venv/bin/python"
}

PY="$(find_python)"
echo "[kot-vllm] python: $PY" >&2
"$PY" -c "import vllm; assert vllm.__version__ == '$VLLM_PIN', vllm.__version__" >&2

exec "$PY" "$DIR/kot_serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
