#!/usr/bin/env bash
# Launch the model on Ascend, using the local engine or the deployment container.
set -euo pipefail

VLLM_PIN="0.33.0"
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
    *) echo "serve.sh: unknown argument '$1' (expected --port --model --device)" >&2; exit 2 ;;
  esac
done
: "${PORT:?--port is required}" "${MODEL:?--model is required}" "${DEVICE:?--device is required}"

# A missing weights dir must fail here, loudly: passed on, vllm reads it as a
# hub repo id and dies inside huggingface_hub with a confusing validation
# error instead of naming the problem.
if [ ! -d "$MODEL" ]; then
  echo "[serve-vllm] model directory not found: $MODEL" >&2
  exit 1
fi

# The load-test client opens hundreds of connections at once; vLLM adds its ZMQ
# sockets and shm handles on top. The default 1024 fd limit kills the engine
# core mid-benchmark with "Too many open files".
ULIMIT_SOFT=$(ulimit -Sn) ULIMIT_HARD=$(ulimit -Hn)
if [ "$ULIMIT_HARD" -gt "$ULIMIT_SOFT" ] 2>/dev/null; then
  ulimit -n "$ULIMIT_HARD" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
fi

find_python() {
  if [ -n "${SERVE_PYTHON:-}" ]; then
    printf '%s' "$SERVE_PYTHON"
    return 0
  fi
  if python3 -c "import vllm" >/dev/null 2>&1; then
    printf '%s' "python3"
    return 0
  fi
  local venv="${SERVE_VENV:-/scratch/serve-vllm-venv}"
  if ! mkdir -p "$(dirname "$venv")" 2>/dev/null || ! mkdir -p "$venv" 2>/dev/null; then
    venv="/tmp/serve-vllm-venv"
    mkdir -p "$venv"
  fi
  if [ ! -x "$venv/bin/python" ]; then
    echo "[serve-vllm] creating venv at $venv with CPython 3.12 (the line the run book documents)" >&2
    if command -v uv >/dev/null 2>&1; then
      uv python install 3.12 && uv venv --seed --python 3.12 "$venv"
    elif command -v python3.12 >/dev/null 2>&1; then
      python3.12 -m venv "$venv"
    else
      echo "[serve-vllm] no CPython 3.12 on this host. Install uv, or point SERVE_PYTHON at a" \
           "3.12 interpreter with vllm." >&2
      exit 1
    fi
  fi
  if ! "$venv/bin/python" -c "import vllm" >/dev/null 2>&1; then
    echo "[serve-vllm] installing vllm $VLLM_PIN offline from $DIR/wheels" >&2
    "$venv/bin/pip" install --no-index --no-cache-dir \
      --find-links "$DIR/wheels" "vllm==$VLLM_PIN" >&2
  fi
  printf '%s' "$venv/bin/python"
}

PY="$(find_python)"
echo "[serve-vllm] python: $PY" >&2
"$PY" -c "import vllm; assert vllm.__version__ == '$VLLM_PIN', vllm.__version__" >&2

exec "$PY" "$DIR/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
