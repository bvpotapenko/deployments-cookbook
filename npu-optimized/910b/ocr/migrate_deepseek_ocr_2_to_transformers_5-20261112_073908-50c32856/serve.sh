#!/usr/bin/env bash
# serve.sh -- the whole contract surface: port, model, device.
#
# Started from the deployment directory (read-only at grading):
#   ./serve.sh --port P --model PATH --device npu
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

# The accelerator stack: the CANN toolkit environment must be sourced before
# any tensor is placed (torch_npu needs libhccl & co.).
if [[ "$DEVICE" == npu* ]]; then
  # shellcheck disable=SC1091
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
fi

# The harness opens thousands of connections before its first request.
ulimit -n 65536 2>/dev/null || true

# A writable scratch home for anything HF-level that wants to write.
export HF_HOME="${HF_HOME:-/scratch/hf}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
mkdir -p "$HF_HOME" 2>/dev/null || true

# The deployment's own package sits beside this script; the container's
# PYTHONPATH (e.g. /workspace/kot) is preserved after it.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# The interpreter is named by absolute path: the process that starts this
# script may run inside a different Python environment.
PY=/usr/local/python3.12.16/bin/python3
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi

# bash execs python, so SIGTERM reaches the server directly.
exec "$PY" "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
