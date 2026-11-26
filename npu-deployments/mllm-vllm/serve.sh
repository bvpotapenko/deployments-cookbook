#!/usr/bin/env bash
# Launch the model on Ascend, using the local engine or the deployment container.
set -euo pipefail

VLLM_PIN="0.30.0"
VLLM_ASCEND_PIN="0.30.0"
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

# The documented one-card profile must cover the long video contract. Keep
# these defaults in the launcher so host wrapping and direct container starts
# pass the same values into the engine. Both remain explicit overrides.
export SERVE_VLLM_GPU_MEM="${SERVE_VLLM_GPU_MEM:-0.99}"
export SERVE_VLLM_MAX_MODEL_LEN="${SERVE_VLLM_MAX_MODEL_LEN:-24576}"
export SERVE_VLLM_REQUEST_TIMEOUT="${SERVE_VLLM_REQUEST_TIMEOUT:-3300}"

# The load-test client opens thousands of pooled connections at once; vLLM adds
# its ZMQ sockets and shm handles on top. The docker default soft limit (1024)
# kills the engine core mid-benchmark with "Too many open files". A process
# may raise its own soft limit to the hard limit.
ULIMIT_SOFT=$(ulimit -Sn) ULIMIT_HARD=$(ulimit -Hn)
if [ "$ULIMIT_HARD" -gt "$ULIMIT_SOFT" ] 2>/dev/null; then
  ulimit -n "$ULIMIT_HARD" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
fi

# Configure the image runtime before probing torch_npu. The CANN setup can
# leave the ATB libraries out of LD_LIBRARY_PATH even though they are
# installed in the image. Keep the runtime path explicit so torch_npu's
# libop_plugin_atb.so is loadable by the EngineCore child.
source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
ATB_HOME_PATH="${ATB_HOME_PATH:-/usr/local/Ascend/nnal/atb/latest/atb/cxx_abi_1}"
if [[ -d "$ATB_HOME_PATH/lib" ]]; then
  export ATB_HOME_PATH
  export LD_LIBRARY_PATH="$ATB_HOME_PATH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

if ! python3 -c "import torch_npu" >/dev/null 2>&1; then
  # Bare Ascend host: the platform stack lives in the container, not here.
  IMAGE="${SERVE_NPU_IMAGE:-serve-mllm-ascend-910b:latest}"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[serve-vllm] building $IMAGE from $DIR/Dockerfile (first run only)" >&2
    docker build -t "$IMAGE" "$DIR" >&2
  fi

  # Weights are mounted at their own path so the inner script sees the same
  # tree. An HF home (a cache with hub/) points HF_HOME at itself; a plain
  # checkpoint gets a writable scratch home so nothing writes into the
  # read-only mount.
  DOCKER_ARGS=(--rm --privileged --network host --ulimit nofile=65536:65536)
  DOCKER_ARGS+=(-v "${MODEL}:${MODEL}:ro")
  if [[ -d "$MODEL/hub" ]]; then
    DOCKER_ARGS+=(-e "HF_HOME=${MODEL}" -e HF_HUB_OFFLINE=1)
  else
    DOCKER_ARGS+=(-e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1)
  fi

  # Every directory in SERVE_OUT_MOUNTS (colon separated) is mounted rw at the
  # same path: /score writes exactly where the host-side runner will
  # read. Point it at the output directory.
  for dir in ${SERVE_OUT_MOUNTS:-}; do
    [[ -d "$dir" ]] || { echo "SERVE_OUT_MOUNTS entry is not a directory: $dir" >&2; exit 2; }
    DOCKER_ARGS+=(-v "${dir}:${dir}")
  done

  # Visible ordinals: the platform's variable when set, else card 0.
  VIS="${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-0}}"
  DOCKER_ARGS+=(-e "ASCEND_VISIBLE_DEVICES=${VIS}" -e "ASCEND_RT_VISIBLE_DEVICES=${VIS}")
  DOCKER_ARGS+=(--device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc)
  for index in ${VIS//,/ }; do
    DOCKER_ARGS+=("--device" "/dev/davinci${index}")
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

  # The deployment dir rides along read-only (wheels/ included, so the venv
  # builds offline inside too). Host networking: the server binds 127.0.0.1
  # inside the container and the host-side runner reaches it on the same
  # loopback -- a published port forwards to the container's ethernet
  # address, where nothing listens. Re-enter this script inside: it now sees
  # torch_npu and takes one of the direct paths. bash execs python, so
  # SIGTERM reaches the server. A deterministic name lets a supervisor
  # clean up an orphaned wrap after the client dies.
  DOCKER_ARGS+=("--name=bd-vllm-serve-$$")
  DOCKER_ARGS+=(-v "${DIR}:/deployment:ro" -w /deployment)

  DOCKER_ARGS+=(-e "SERVE_UID=$(id -u)")

  # SERVE_VENV set on the host: forward it and mount the venv at the same
  # path, so a previously provisioned interpreter is reused instead of
  # rebuilt (diagnostics; offline installation leaves it unset and provisions
  # offline into /scratch from wheels/).
  if [ -n "${SERVE_VENV:-}" ]; then
    DOCKER_ARGS+=(-e "SERVE_VENV=${SERVE_VENV}")
    if [ -d "$SERVE_VENV" ]; then
      DOCKER_ARGS+=(-v "${SERVE_VENV}:${SERVE_VENV}")
    fi
  fi

  # Forward engine tuning knobs into the image. Without this, a host-mode
  # launch silently falls back to serve.py defaults because the inner
  # shell has a fresh environment.
  for variable in \
    SERVE_VLLM_GPU_MEM \
    SERVE_VLLM_MAX_MODEL_LEN \
    SERVE_VLLM_MAX_BATCHED_TOKENS \
    SERVE_VLLM_MAX_NUM_SEQS \
    SERVE_VLLM_REPLAY_SEGMENT_ROWS \
    SERVE_VLLM_REQUEST_TIMEOUT \
    SERVE_VLLM_MOE_BACKEND \
    SERVE_VLLM_DTYPE \
    SERVE_VLLM_COMPILATION
  do
    if [[ -n "${!variable:-}" ]]; then
      DOCKER_ARGS+=(-e "$variable=${!variable}")
    fi
  done
  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    exec bash /deployment/serve.sh --port '"$PORT"' --model '"$MODEL"' --device '"$DEVICE"'
  '
fi

find_python() {
  if [ -n "${SERVE_PYTHON:-}" ]; then
    printf '%s' "$SERVE_PYTHON"
    return 0
  fi
  if python3 -c "import vllm, torch_npu" >/dev/null 2>&1; then
    printf '%s' "python3"
    return 0
  fi
  # The image carries the platform torch and family stack but no engine.
  # /scratch is writable in the container; /tmp is the human fallback.
  local venv="${SERVE_VENV:-/scratch/serve-vllm-venv}"
  if ! mkdir -p "$(dirname "$venv")" 2>/dev/null || ! mkdir -p "$venv" 2>/dev/null; then
    venv="/tmp/serve-vllm-venv"
    mkdir -p "$venv"
  fi
  if ! "$venv/bin/python" -c "import vllm, torch_npu" >/dev/null 2>&1; then
    echo "[serve-vllm] provisioning venv at $venv offline from $DIR/wheels" >&2
    if [ ! -x "$venv/bin/python" ]; then
      if [ -x "$DIR/bin/uv" ]; then
        "$DIR/bin/uv" venv --system-site-packages --python python3 "$venv" >&2
      else
        python3 -m venv --system-site-packages "$venv"
      fi
    fi
    # vllm first, then the pinned closure, all with --no-deps: the wheel set
    # IS the verified environment (its metadata graph has conflicts pip would
    # refuse to resolve -- scipy 1.13.1 beside numpy 2.3.5 -- that the
    # runtime never touches, and resolution would fight them forever).
    # Excluded from the batch: torch and torchvision (the image's system
    # copies satisfy the pins), torch_npu (the image carries the same 2.10.0.post7
    # line), and vllm itself (installed just above).
    PINS=$(grep -E '^[a-zA-Z0-9].*==' "$DIR/wheels-requirements.txt" \
      | grep -vE '^(torch==|torchvision==|torch_npu==|torchcodec==|vllm==)')
    if [ -x "$DIR/bin/uv" ]; then
      # Vendored uv: seconds instead of minutes, no network touched.
      "$DIR/bin/uv" pip install --python "$venv/bin/python" --offline --no-cache \
        --no-deps --find-links "$DIR/wheels" "vllm==$VLLM_PIN" >&2 || exit 1
      echo "$PINS" | xargs "$DIR/bin/uv" pip install \
        --python "$venv/bin/python" --offline --no-cache --no-deps \
        --find-links "$DIR/wheels" >&2 || exit 1
    else
      "$venv/bin/pip" install --no-index --no-cache-dir --no-deps \
        --find-links "$DIR/wheels" "vllm==$VLLM_PIN" >&2 || exit 1
      echo "$PINS" | xargs "$venv/bin/pip" install --no-index --no-cache-dir --no-deps \
          --find-links "$DIR/wheels" >&2 || exit 1
    fi
    "$venv/bin/python" -c "import vllm, torch_npu" >&2 || exit 1
  fi
  printf '%s' "$venv/bin/python"
}

PY="$(find_python)"
echo "[serve-vllm] python: $PY" >&2
"$PY" "$DIR/patch_vllm_ascend_logprob.py" >&2
"$PY" -c "import vllm, torch_npu; assert vllm.__version__.startswith('$VLLM_PIN'), vllm.__version__" >&2

exec "$PY" "$DIR/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
