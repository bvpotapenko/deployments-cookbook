#!/usr/bin/env bash
# kot deployment: ocr family (DeepSeek-OCR-2), vllm engine.
#
#   ./kot_serve.sh --port <P> --model <weights dir> --device <torch device>
#
# Self-provisioning: if the ambient python cannot import the pinned vllm, a
# venv is created in a writable scratch location and filled OFFLINE from
# ./wheels (the graded container has no network). The wheel set is
# self-contained: vllm 0.30.0 with its own torch (cu128,
# Blackwell-capable) and CUDA libraries, so it runs on any host with a CUDA
# 12.x driver regardless of the image's own torch. On a networked test host,
# pre-provision a venv and point KOT_PYTHON at it instead.
set -euo pipefail

VLLM_PIN="0.32.0"
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
    venv="/tmp/kot-vllm-ocr-venv"
    mkdir -p "$venv"
  fi
  if [ ! -x "$venv/bin/python" ]; then
    echo "[kot-vllm-mllm] creating venv at $venv with CPython 3.12 (the wheel set is cp312)" >&2
    if command -v uv >/dev/null 2>&1; then
      uv python install 3.12 && uv venv --seed --python 3.12 "$venv"
    elif command -v python3.12 >/dev/null 2>&1; then
      python3.12 -m venv "$venv"
    else
      echo "[kot-vllm-mllm] no CPython 3.12 on this host: the offline wheels install on no" \
           "other interpreter. Install uv, or point KOT_PYTHON at a 3.12 interpreter with vllm." >&2
      exit 1
    fi
  fi
  if ! "$venv/bin/python" -c "import vllm" >/dev/null 2>&1; then
    echo "[kot-vllm-mllm] installing vllm $VLLM_PIN offline from $DIR/wheels" >&2
    "$venv/bin/pip" install --no-index --no-cache-dir \
      --find-links "$DIR/wheels" "vllm==$VLLM_PIN" >&2
  fi
  printf '%s' "$venv/bin/python"
}

PY="$(find_python)"
echo "[kot-vllm-mllm] python: $PY" >&2
"$PY" -c "import vllm; assert vllm.__version__ == '$VLLM_PIN', vllm.__version__" >&2

# The serve log is a file: block-buffered stdout would swallow diagnostics
# while the process hangs. The faulthandler dump and the phase prints must
# reach the log as they happen.
export PYTHONUNBUFFERED=1

exec "$PY" "$DIR/kot_serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
