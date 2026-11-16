#!/usr/bin/env bash
# serve.sh -- port, model, device: the whole contract surface this script takes.
#
# Runs inside the graded container (the image carries torch_npu, the Ascend
# driver and the CANN toolkit), so serve.py runs directly. The interpreter is
# named by absolute path, as the contract asks; the first one that imports
# both torch_npu and transformers wins. The CANN environment is sourced
# defensively: the image sets most of it already.
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

# The harness opens thousands of connections before its first request; the
# docker default soft nofile (1024) makes accept() fail with EMFILE past the
# first thousand. A process may raise its own soft limit to the hard limit.
ulimit -n 65536 2>/dev/null || true

# The CANN environment registers the Ascend backends and libraries.
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

PY=""
for candidate in "${KOT_OCR_PY:-}" /usr/local/python3.12.16/bin/python3 \
                 /root/kotvenv-ocr/bin/python /root/kotvenv/bin/python; do
  if [[ -x "$candidate" ]] && "$candidate" -c "import torch_npu, transformers" >/dev/null 2>&1; then
    PY="$candidate"
    break
  fi
done
if [[ -z "$PY" ]]; then
  for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 \
       && "$candidate" -c "import torch_npu, transformers" >/dev/null 2>&1; then
      PY="$(command -v "$candidate")"
      break
    fi
  done
fi
if [[ -z "$PY" ]]; then
  echo "[serve] no interpreter with torch_npu and transformers" >&2
  exit 3
fi

# The working directory is the model dir: a runner's out_dir resolves against
# it when sent relative. serve.py imports its sibling modules through its own
# directory on sys.path.
cd "$MODEL"
exec "$PY" "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
