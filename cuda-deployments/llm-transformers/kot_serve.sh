#!/usr/bin/env bash
# kot_serve.sh -- port, model, device: the whole contract surface this script takes.
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

# When the weights were handed over as a huggingface home (the capture mounts
# its cache at --model), point the hub at it so the snapshot resolves offline.
if [[ -d "$MODEL/hub" ]]; then
  export HF_HOME="$MODEL"
  export HF_HUB_OFFLINE=1
fi

exec /root/kotvenv/bin/python "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
