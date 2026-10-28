#!/usr/bin/env bash
# kot deployment: llm family, vllm engine, ascend-910b platform.
#
#   ./kot_serve.sh --port <P> --model <weights dir> --device npu
#
# Three ways this script runs, chosen by what the ambient python can import:
#
# * ambient python has vllm AND torch_npu: serve directly (a container where
#   the engine is already provisioned).
# * ambient python has torch_npu but no vllm (the graded container: the image
#   carries the platform torch and the family model stack, no engine): build
#   a venv in a writable scratch location and install OFFLINE from ./wheels
#   (the graded container has no network), then serve.
# * no torch_npu at all (a bare Ascend host: the host interpreter is
#   torch-free by design, or a human's Mac): docker-wrap the image built from
#   ./Dockerfile with the Ascend device nodes and driver mounts, host
#   networking, and a raised nofile, and re-enter this script inside the
#   container -- a container cannot start another one.
set -euo pipefail

VLLM_PIN="0.23.0"
VLLM_ASCEND_PIN="0.23.0"
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
    *) echo "kot_serve.sh: unknown argument '$1' (expected --port --model --device)" >&2; exit 2 ;;
  esac
done
: "${PORT:?--port is required}" "${MODEL:?--model is required}" "${DEVICE:?--device is required}"

# The speed harness opens thousands of pooled connections at once; vLLM adds
# its ZMQ sockets and shm handles on top. The docker default soft limit (1024)
# kills the engine core mid-benchmark with "Too many open files". A process
# may raise its own soft limit to the hard limit.
ULIMIT_SOFT=$(ulimit -Sn) ULIMIT_HARD=$(ulimit -Hn)
if [ "$ULIMIT_HARD" -gt "$ULIMIT_SOFT" ] 2>/dev/null; then
  ulimit -n "$ULIMIT_HARD" 2>/dev/null || ulimit -n 65536 2>/dev/null || true
fi

if ! python3 -c "import torch_npu" >/dev/null 2>&1; then
  # Bare Ascend host: the platform stack lives in the container, not here.
  IMAGE="${KOT_NPU_IMAGE:-kot-llm-ascend-910b:latest}"
  if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "[kot-vllm] building $IMAGE from $DIR/Dockerfile (first run only)" >&2
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

  # Every directory in KOT_OUT_MOUNTS (colon separated) is mounted rw at the
  # same path: /kot/score writes exactly where the host-side runner will
  # read. Point it at the capture tree; never at a store that holds goldens.
  for dir in ${KOT_OUT_MOUNTS:-}; do
    [[ -d "$dir" ]] || { echo "KOT_OUT_MOUNTS entry is not a directory: $dir" >&2; exit 2; }
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

  # The capture tree on mounted paths is written by BOTH sides: the server
  # (root in here) and the host-side runner (the invoking user). The server
  # hands ownership of every out_dir it writes back to the runner via this
  # uid, so the runner's own atomic writes never hit a root-owned directory.
  # torch_npu cannot import as a non-root uid (driver files are root-only),
  # so the server itself must stay root.
  DOCKER_ARGS+=(-e "KOT_UID=$(id -u)")

  # KOT_VENV set on the host: forward it and mount the venv at the same
  # path, so a previously provisioned interpreter is reused instead of
  # rebuilt (diagnostics; the graded path leaves it unset and provisions
  # offline into /scratch from wheels/).
  if [ -n "${KOT_VENV:-}" ]; then
    DOCKER_ARGS+=(-e "KOT_VENV=${KOT_VENV}")
    if [ -d "$KOT_VENV" ]; then
      DOCKER_ARGS+=(-v "${KOT_VENV}:${KOT_VENV}")
    fi
  fi
  exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c '
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    exec bash /deployment/kot_serve.sh --port '"$PORT"' --model '"$MODEL"' --device '"$DEVICE"'
  '
fi

find_python() {
  if [ -n "${KOT_PYTHON:-}" ]; then
    printf '%s' "$KOT_PYTHON"
    return 0
  fi
  if python3 -c "import vllm, torch_npu" >/dev/null 2>&1; then
    printf '%s' "python3"
    return 0
  fi
  # The image carries the platform torch and family stack but no engine.
  # /scratch is writable at grading; /tmp is the human fallback.
  local venv="${KOT_VENV:-/scratch/kot-vllm-venv}"
  if ! mkdir -p "$(dirname "$venv")" 2>/dev/null || ! mkdir -p "$venv" 2>/dev/null; then
    venv="/tmp/kot-vllm-venv"
    mkdir -p "$venv"
  fi
  if ! "$venv/bin/python" -c "import vllm, torch_npu" >/dev/null 2>&1; then
    echo "[kot-vllm] provisioning venv at $venv offline from $DIR/wheels" >&2
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
    # copies satisfy the pins), torch_npu .post2 (the venv carries the
    # vllm-ascend-pinned .post4), and vllm itself (installed just above).
    PINS=$(grep -E '^[a-zA-Z0-9].*==' "$DIR/wheels-requirements.txt" \
      | grep -vE '^(torch==|torchvision==|torch_npu==2\.10\.0\.post2|vllm==)')
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

source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
PY="$(find_python)"
echo "[kot-vllm] python: $PY" >&2
"$PY" -c "import vllm, torch_npu; assert vllm.__version__.startswith('$VLLM_PIN'), vllm.__version__" >&2

exec "$PY" "$DIR/kot_serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
