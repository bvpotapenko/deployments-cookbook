#!/usr/bin/env bash
# Launch the model on Ascend, using the local engine or the deployment container.
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
  echo "usage: serve.sh --port P --model PATH --device DEVICE" >&2
  exit 2
fi

HERE="$(cd "$(dirname "$0")" && pwd)"

# The interpreter is named explicitly, with PATH only as a late fallback: the
# process that starts us may run inside an environment that is not ours, and
# the stack needs the pinned transformers 5.22.1 venv on a provisioned host.
# In the container the image carries the family stack and PATH
# resolves to it.
if [ -n "${SERVE_OCR_PY:-}" ]; then
  PY="$SERVE_OCR_PY"
elif [ -x /root/serve-venv-ocr/bin/python ]; then
  PY=/root/serve-venv-ocr/bin/python
elif [ -x /root/serve-venv/bin/python ]; then
  PY=/root/serve-venv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY=python3
fi

# The working directory is the model dir, as in the reference era: the
# runner's out_dir resolves against it when sent relative. serve.py imports
# its sibling builder modules through its own directory on sys.path.
cd "$MODEL"

exec "$PY" "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
