#!/usr/bin/env bash
# kot_serve.sh -- start the OpenVLA transformers deployment.
#   ./kot_serve.sh --port P --model PATH --device DEVICE
set -euo pipefail

PORT=""
MODEL=""
DEVICE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [[ -z "$PORT" || -z "$MODEL" || -z "$DEVICE" ]]; then
  echo "usage: kot_serve.sh --port P --model PATH --device DEVICE" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

# The interpreter by absolute path: the process that starts us may run in a
# Python environment that is not ours. KOT_VLA_PYTHON overrides it.
PY="${KOT_VLA_PYTHON:-/root/kotvenv-vla/bin/python}"

exec "$PY" "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
