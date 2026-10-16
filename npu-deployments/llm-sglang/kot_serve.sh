#!/usr/bin/env bash
# kot_serve.sh -- port, model, device: the whole contract surface this script takes.
#
# Two ways this script is started:
#
# * inside the graded container (gather built it from the CANN base image with
#   torch_npu): provision the engine stack offline from ./wheels into the
#   container's site-packages and run kot_server.py directly. A container
#   cannot start another one.
# * on a bare Ascend host (a golden capture with --no-container, or a human):
#   the host interpreter has no torch_npu, so wrap docker run with the Ascend
#   driver mounts and device nodes, mirroring kot.platforms' ascend-910b row,
#   and re-enter this script inside the container.
#
# The image carries no serving engine (the contract): sglang and its NPU
# kernels ride in ./wheels and are installed at start, offline, into the
# writable container. Startup budget is 900 s; install + engine load fit.
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
if [[ ! -d "$MODEL" ]]; then
  echo "model directory not found: $MODEL" >&2
  exit 1
fi

HERE="$(cd "$(dirname "$0")" && pwd)"
WHEELS="$HERE/wheels"
if [[ ! -d "$WHEELS" ]]; then
  echo "wheels directory not found: $WHEELS (run fetch_wheels.sh)" >&2
  exit 1
fi

# The speed harness's warmup opens thousands of pooled connections in one
# burst; the docker default soft nofile (1024) makes accept() fail with
# EMFILE past the first thousand. A process may raise its own soft limit.
ulimit -n 65536 2>/dev/null || true

if ! python3 -c "import torch_npu" >/dev/null 2>&1; then
  # ---------------------------------------------------------------- host mode
  IMAGE="${KOT_NPU_IMAGE:-kot-llm-ascend-910b:latest}"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[kot_serve] building $IMAGE from $HERE/Dockerfile (first run only)" >&2
    docker build -t "$IMAGE" "$HERE" >&2
  fi

  # Weights are mounted at their own path so the server sees the same tree. An
  # HF home (a cache with hub/) points HF_HOME at itself; a plain checkpoint
  # gets a writable scratch home so nothing writes into the read-only mount.
  DOCKER_ARGS=(--rm --privileged --network host --ulimit nofile=65536:65536)
  DOCKER_ARGS+=(-v "${MODEL}:${MODEL}:ro")
  if [[ -d "$MODEL/hub" ]]; then
    DOCKER_ARGS+=(-e "HF_HOME=${MODEL}" -e HF_HUB_OFFLINE=1)
  else
    DOCKER_ARGS+=(-e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1)
  fi

  # Every directory in KOT_OUT_MOUNTS (colon separated) is mounted rw at the
  # same path: /kot/score writes exactly where the host-side runner reads.
  for dir in ${KOT_OUT_MOUNTS:-}; do
    [[ -d "$dir" ]] || { echo "KOT_OUT_MOUNTS entry is not a directory: $dir" >&2; exit 2; }
    DOCKER_ARGS+=(-v "${dir}:${dir}")
  done

  # Visible ordinals: the platform's variable when set, else card 0.
  VIS="${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0}}"
  DOCKER_ARGS+=(-e "ASCEND_VISIBLE_DEVICES=${VIS}" -e "ASCEND_RT_VISIBLE_DEVICES=${VIS}")
  DOCKER_ARGS+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  for index in ${VIS//,/ }; do
    DOCKER_ARGS+=(--device "/dev/davinci${index}")
  done
  for mount in \
    /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
    /etc/ascend_install.info:/etc/ascend_install.info:ro \
    /var/log/npu/:/usr/slog:rw \
    /usr/local/dcmi:/usr/local/dcmi:ro \
    /usr/local/sbin:/usr/local/sbin:ro \
    /etc/hccn.conf:/etc/hccn.conf:ro
  do
    if [[ -e "${mount%%:*}" ]]; then
      DOCKER_ARGS+=(-v "$mount")
    fi
  done

  # The deployment dir rides along read-only at the path kot's own capture
  # uses; the inner run re-enters this script, which now finds torch_npu and
  # takes the direct path below. Host network: the server binds 127.0.0.1
  # inside the container and the host runner reaches the same loopback -- a
  # published port would forward to the container's ethernet address, where
  # nothing listens. bash execs the inner bash, so SIGTERM reaches the server.
  DOCKER_ARGS+=(-v "${HERE}:/deployment:ro" -w /deployment)

  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    exec bash /deployment/kot_serve.sh --port '"$PORT"' --model '"$MODEL"' --device '"$DEVICE"'
  '
fi

# ----------------------------------------------------------- graded / direct
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true

# The engine stack is installed offline from ./wheels into the container's
# site-packages: the image carries torch + torch_npu + transformers, the
# contract forbids a serving engine in the image, and the scheduler
# subprocesses get a curated environment from sglang -- so the stack (and the
# sitecustomize patch layer) must live where site.py finds it in every
# process, whatever the environment says.
SITE_PACKAGES="$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")"
# Beside what it proves, NOT in /tmp: host mode mounts the host's /tmp into
# the container (KOT_OUT_MOUNTS), where a marker would outlive the container
# and make a fresh container skip its install.
MARKER="$SITE_PACKAGES/kot-sglang-provisioned"

if [[ ! -f "$MARKER" ]]; then
  echo "[kot_serve] installing engine stack from $WHEELS (offline)" >&2
  # The sglang wheel matching the recorded commit (a rerun of fetch_wheels.sh
  # after main moved can leave two behind); fall back to the newest wheel.
  # triton-ascend is deliberately NOT installed: the ascend attention backend
  # runs torch_npu's fused attention, not triton kernels, and triton-ascend
  # pins a heavy tail (numpy 1.26.4 downgrade, scipy/psutil/pytest pins,
  # upstream triton) the engine does not use. The wheel stays in wheels/ if a
  # missing-import error ever proves otherwise. The memfabric pair is part of
  # the CANN 9.1.0 component set.
  short="$(cut -c1-9 "$WHEELS/SGLANG_COMMIT.txt" 2>/dev/null || true)"
  SGLANG_WHEEL="$(ls "$WHEELS"/sglang-*g${short}*.whl 2>/dev/null | head -1 || true)"
  [[ -n "$SGLANG_WHEEL" ]] || SGLANG_WHEEL="$(ls "$WHEELS"/sglang-*.whl | tail -1)"
  # The sglang wheel's own metadata resolves the rest of its dependency
  # closure from find-links; the NPU-specific packages are named explicitly.
  python3 -m pip install --no-cache-dir --break-system-packages \
    --no-index --find-links "$WHEELS" \
    "$SGLANG_WHEEL" \
    "memfabric-hybrid==1.1.4" "memfabric-zbal==1.1.3" >&2
  # sglang hard-imports triton in its common utils; on Ascend the triton
  # module comes from triton-ascend. Installed --no-deps: its metadata pins
  # a heavy tail (numpy 1.26.4 downgrade, pytest, upstream triton) the
  # engine does not use -- only pybind11 and filelock are actually imported
  # (verified: import sglang + import triton with exactly this set).
  python3 -m pip install --no-cache-dir --break-system-packages --no-index --no-deps \
    "$WHEELS"/triton_ascend-3.2.2-*.whl "$WHEELS"/pybind11-*.whl "$WHEELS"/filelock-*.whl >&2
  # deep_ep and attentions load fine, so they install. sgl_kernel_npu
  # installs too -- its python modules carry the triton-ascend kernels the
  # ascend model path runs (e.g. qwen3's fused QKV+RMSNorm+RoPE) -- but its
  # C++ .so needs a CANN newer than this image, so the load below is made
  # non-fatal: the registered torch.ops are absent, and any call to one
  # fails loudly at its call site.
  python3 -m pip install --no-cache-dir --break-system-packages --no-index --no-deps \
    "$WHEELS"/deep_ep-*.whl "$WHEELS"/attentions-*.whl "$WHEELS"/sgl_kernel_npu-*.whl >&2
  INIT="$SITE_PACKAGES/sgl_kernel_npu/__init__.py"
  if [[ -f "$INIT" ]] && ! grep -q "except OSError" "$INIT"; then
    python3 - "$INIT" <<'PYEOF'
import pathlib
import re
import sys

path = pathlib.Path(sys.argv[1])
text = path.read_text(encoding="utf-8")
patched = re.sub(
    r"^_load_sgl_kernel_npu\(\)$",
    "try:\n"
    "    _load_sgl_kernel_npu()\n"
    "except OSError as failure:\n"
    "    import warnings\n"
    "\n"
    "    warnings.warn(f\"sgl_kernel_npu C++ kernels unavailable: {failure}\")",
    text,
    flags=re.MULTILINE,
)
path.write_text(patched, encoding="utf-8")
print("[kot_serve] sgl_kernel_npu .so load made non-fatal", flush=True)
PYEOF
  fi
  # The CANN custom operators are .run self-installers, not wheels; they
  # register kernels in the toolkit's opp directory (writable container FS).
  # Names inside the release zips: CANN-custom_ops-none-linux.<arch>.run and
  # cann-ops-transformer_custom_linux-<arch>.run. The deployment dir mounts
  # read-only at grading: the files arrive executable and must not be
  # chmod'd in place; run them through bash instead.
  OPP_DIR="/usr/local/Ascend/ascend-toolkit/latest/opp"
  if compgen -G "$WHEELS/*custom_ops*.run" >/dev/null; then
    for installer in "$WHEELS"/*custom_ops*.run "$WHEELS"/*ops-transformer*.run; do
      [[ -e "$installer" ]] || continue
      bash "$installer" --install-path="$OPP_DIR" >&2 || {
        echo "[kot_serve] WARNING: $(basename "$installer") install failed; continuing" >&2
      }
    done
  fi
  if compgen -G "$WHEELS/custom_ops-*cp312*.whl" >/dev/null; then
    python3 -m pip install --no-cache-dir --break-system-packages --no-index --no-deps \
      "$WHEELS"/custom_ops-*cp312*.whl >&2
  fi
  touch "$MARKER"
fi

# The patch layer is refreshed on every start, inside or outside the marker:
# the installed copy would otherwise outlive the deployment's sitecustomize.py
# across warm restarts (the marker gate skips all provisioning). The guard
# stays: a foreign sitecustomize.py is refused, ours is overwritten.
if [[ -f "$SITE_PACKAGES/sitecustomize.py" ]] && ! grep -q "_kot_patched" "$SITE_PACKAGES/sitecustomize.py" 2>/dev/null; then
  echo "sitecustomize.py already exists in $SITE_PACKAGES and is not ours" >&2
  exit 1
fi
cp "$HERE/sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py"
echo "[kot_serve] patch layer installed into $SITE_PACKAGES"

# The weights are local and the graded container has no network; the engine
# must never reach for the hub either.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# sitecustomize.py patches the scheduler processes; belt and braces ahead of
# anything else, though the site-packages copy is what the scheduler sees.
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"

# Where the full-vocab log-softmax stashes land. /scratch is the contract's
# writable directory; fall back to a temp dir when it does not exist. The
# patch layer resolves the same default inside the scheduler processes, whose
# environment sglang replaces.
if [[ -d /scratch && -w /scratch ]]; then
  export KOT_SGLANG_STASH_DIR="/scratch/kot-sglang-stash"
else
  export KOT_SGLANG_STASH_DIR="/tmp/kot-sglang-stash"
fi
mkdir -p "$KOT_SGLANG_STASH_DIR"
echo "[kot_serve] stash dir: $KOT_SGLANG_STASH_DIR"

# Pre-sampling rows come out of the engine in float32 whatever the weights run
# in; the patch recompute is documented in sitecustomize.py.
export KOT_SGLANG_FULL_LOGPROBS=1

exec python3 "$HERE/kot_server.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
