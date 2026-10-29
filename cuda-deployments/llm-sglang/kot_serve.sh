#!/usr/bin/env bash
# Start the SGLang deployment of the kot contract.
#
#   ./kot_serve.sh --port P --model /path/to/weights --device cpu|cuda
#
# One process serves everything: this script launches kot_server.py, which
# embeds sglang.Engine in-process and answers /kot/info, /kot/score,
# /v1/completions and /v1/chat/completions. The contract's one-server rule
# lives here: the score path and the speed harness drive the same engine.
#
# The engine is sglang 0.5.22. On a CUDA host it expects to find sglang,
# sglang-kernel and triton already installed in the interpreter named by
# KOT_SGLANG_PYTHON (or on PATH); see run-book.md for the exact wheel set.
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
  echo "usage: kot_serve.sh --port P --model PATH --device cpu|cuda" >&2
  exit 2
fi
if [[ ! -d "$MODEL" ]]; then
  echo "model directory not found: $MODEL" >&2
  exit 1
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# The interpreter that has sglang. The CUDA host names its engine venv with
# KOT_SGLANG_PYTHON.
PYTHON_CANDIDATES=(
  "${KOT_SGLANG_PYTHON:-}"
  "$HERE/.venv/bin/python"
  "$HERE/.venv-cuda/bin/python"
  "$(command -v python3 || true)"
)
PYTHON=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
  if [[ -n "$candidate" && -x "$candidate" ]] && "$candidate" -c "import sglang" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo "no interpreter with sglang found; set KOT_SGLANG_PYTHON" >&2
  exit 1
fi
echo "[kot_serve] engine python: $PYTHON"
"$PYTHON" -c "import sglang, torch; print(f'[kot_serve] sglang {sglang.__version__}, torch {torch.__version__}')"

# The engine holds CUDA + zmq + one fd per queued connection; a soft limit of
# 1024 runs out under the harness's connection warmup (EMFILE mid-serve).
ulimit -n 65535 2>/dev/null || ulimit -n "$(ulimit -Hn)"

# sglang's scheduler subprocesses run with a curated environment that drops
# PYTHONPATH and everything custom, so the patch layer travels inside the
# engine venv instead: site.py imports it from site-packages in every spawned
# process, whatever the environment says. A foreign sitecustomize is refused;
# ours is refreshed on every start.
SITE_PACKAGES="$("$PYTHON" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
if [[ -f "$SITE_PACKAGES/sitecustomize.py" ]] && ! grep -q "_kot_patched" "$SITE_PACKAGES/sitecustomize.py"; then
  echo "sitecustomize.py already exists in $SITE_PACKAGES and is not ours" >&2
  exit 1
fi
# Two delivery routes for the patch layer, because both can be defeated on
# their own: this image's distro sitecustomize.py sits in the stdlib dir,
# which precedes any venv's site-packages on sys.path, so site.py may import
# the distro's file instead of ours; and sglang's scheduler subprocesses get
# a curated environment, so PYTHONPATH does not reach them. A .pth file in
# the venv's site-packages is executed by site.py in every process whatever
# the environment says -- that is the route that always fires.
cp "$HERE/sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py"
cp "$HERE/sitecustomize.py" "$SITE_PACKAGES/kot_sglang_patch.py"
echo "import kot_sglang_patch" > "$SITE_PACKAGES/kot_sglang_patch.pth"
echo "[kot_serve] patch layer installed into $SITE_PACKAGES (module + .pth)"

# DeepGEMM's JIT asserts at startup without NVCC >= 12.9 (serving hosts carry
# no toolkit); the triton kernels cover these GPUs. Set before any sglang
# process spawns.
export SGLANG_ENABLE_JIT_DEEPGEMM=false

# The batch-invariant TRITON persistent matmul is the fallback and
# keeps determinism without DeepGEMM. Set before any sglang
# process spawns.
export SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0

# sglang's async asserts make the embedding (and other layers) check their
# inputs against real bounds and name the layer when one fails, at no cost
# when nothing trips. The diagnosis path for unknown walls: the default
# device-side assert names neither the input nor the layer.
export SGLANG_ENABLE_ASYNC_ASSERT=1

# The weights are local and the graded container has no network; the engine
# must never reach for the hub either.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# sitecustomize.py patches the scheduler processes; it must be importable
# before sglang starts, hence PYTHONPATH ahead of anything else.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# Where the full-vocab log-softmax stashes land. /scratch is the contract's
# writable directory; fall back to a temp dir when it does not exist. The
# patch layer resolves the same default in the scheduler processes, whose
# environment sglang replaces.
if [[ -d /scratch && -w /scratch ]]; then
  export KOT_SGLANG_STASH_DIR="/scratch/kot-sglang-stash"
else
  export KOT_SGLANG_STASH_DIR="/tmp/kot-sglang-stash"
fi
mkdir -p "$KOT_SGLANG_STASH_DIR"
echo "[kot_serve] stash dir: $KOT_SGLANG_STASH_DIR"

# Pre-sampling rows come out of the engine in float32 whatever the weights
# run in; the patch recompute is documented in sitecustomize.py.
export KOT_SGLANG_FULL_LOGPROBS=1

exec "$PYTHON" "$HERE/kot_server.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
