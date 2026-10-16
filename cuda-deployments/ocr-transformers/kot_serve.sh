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

# The interpreter is named explicitly, with PATH only as a late fallback: the
# process that starts us may run inside an environment that is not ours, and
# the OCR stack needs transformers 4.46.3 (the checkpoint's remote code breaks
# on 5.x), so a dedicated venv wins on a provisioned host. In the graded
# container the image carries the family stack and PATH resolves to it.
if [ -n "${KOT_OCR_PY:-}" ]; then
  PY="$KOT_OCR_PY"
elif [ -x /root/kotvenv-ocr/bin/python ]; then
  PY=/root/kotvenv-ocr/bin/python
elif [ -x /root/kotvenv/bin/python ]; then
  PY=/root/kotvenv/bin/python
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
else
  PY=python3
fi

# The checkpoint's remote code imports from its own directory.
cd "$MODEL"

exec "$PY" "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
