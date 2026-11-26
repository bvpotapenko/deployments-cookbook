#!/usr/bin/env bash
# Launch the model on Ascend, using the local engine or the deployment container.
set -euo pipefail

VLLM_PIN="0.33.0"
# The run book's video pair, on top of vllm: the PyTorch cu130 torchvision
# build and the closure's torchcodec.
TORCHVISION_PIN="0.29.0+cu130"
TORCHCODEC_PIN="0.17.0"
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
    echo "[serve-vllm-mllm] creating venv at $venv with CPython 3.12 (the wheel set is cp312)" >&2
    if command -v uv >/dev/null 2>&1; then
      uv python install 3.12 && uv venv --seed --python 3.12 "$venv"
    elif command -v python3.12 >/dev/null 2>&1; then
      python3.12 -m venv "$venv"
    else
      echo "[serve-vllm-mllm] no CPython 3.12 on this host: the offline wheels install on no" \
           "other interpreter. Install uv, or point SERVE_PYTHON at a 3.12 interpreter with vllm." >&2
      exit 1
    fi
  fi
  if ! "$venv/bin/python" -c "import vllm; from importlib.metadata import version as v; assert (v('torchvision'), v('torchcodec')) == ('$TORCHVISION_PIN', '$TORCHCODEC_PIN')" >/dev/null 2>&1; then
    echo "[serve-vllm-mllm] installing vllm $VLLM_PIN, torchvision $TORCHVISION_PIN, torchcodec $TORCHCODEC_PIN offline from $DIR/wheels" >&2
    "$venv/bin/pip" install --no-index --no-cache-dir \
      --find-links "$DIR/wheels" "vllm==$VLLM_PIN" \
      "torchvision==$TORCHVISION_PIN" "torchcodec==$TORCHCODEC_PIN" >&2
  fi
  printf '%s' "$venv/bin/python"
}

PY="$(find_python)"
echo "[serve-vllm-mllm] python: $PY" >&2
"$PY" -c "import vllm; assert vllm.__version__ == '$VLLM_PIN', vllm.__version__" >&2

# The serve log is a file: block-buffered stdout would swallow diagnostics
# while the process hangs. The faulthandler dump and the phase prints must
# reach the log as they happen.
export PYTHONUNBUFFERED=1

# The V2 model runner: tail-only prompt scoring (prompt_logprob_token_ids +
# prompt_logprob_start), no top-k and no full-prompt pinned buffer - the
# V1 floor that OOMs this box does not exist on this path. Default on;
# VLLM_USE_V2_MODEL_RUNNER=0 boots the V1 path for a comparison run.
export VLLM_USE_V2_MODEL_RUNNER="${VLLM_USE_V2_MODEL_RUNNER:-1}"

# vLLM's per-request decoded-media limit fails a 1080p clip longer than about
# 57 s, on chat only: /score has no such limit. 0 disables it.
export VLLM_MAX_MEDIA_REQUEST_DECODE_BYTES=0

exec "$PY" "$DIR/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
