#!/usr/bin/env bash
# kot_serve.sh -- boot the mllm transformers deployment.
#   ./kot_serve.sh --port P --model PATH --device DEVICE
set -euo pipefail

PORT=""
MODEL=""
DEVICE="cuda"
while [ $# -gt 0 ]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "usage: kot_serve.sh --port P --model PATH --device DEVICE" >&2; exit 2 ;;
  esac
done
[ -n "$PORT" ] && [ -n "$MODEL" ] || { echo "usage: kot_serve.sh --port P --model PATH --device DEVICE" >&2; exit 2; }

HERE="$(cd "$(dirname "$0")" && pwd)"

# The allocator's own suggestion against fragmentation: the 52G of weights
# over two 32G cards leaves little slack, and a fragmented pool OOMs a
# request that would otherwise fit.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# The interpreter: explicit override, then the provisioned venv, then PATH.
if [ -n "${KOT_MLLM_PYTHON:-}" ]; then PY="$KOT_MLLM_PYTHON"
elif [ -x /root/kotvenv-mllm/bin/python ]; then PY=/root/kotvenv-mllm/bin/python
elif [ -x /root/kotvenv/bin/python ]; then PY=/root/kotvenv/bin/python
else PY="$(command -v python3)"
fi

exec "$PY" "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
