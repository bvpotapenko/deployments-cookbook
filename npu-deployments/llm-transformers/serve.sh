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

# The load-test client's auto warmup opens hundreds of pooled connections in one
# burst; the docker default soft nofile (1024) makes accept() fail with EMFILE
# past the first thousand and the accept loop wedges with a full backlog. A
# process may raise its own soft limit to the hard limit.
ulimit -n 65536 2>/dev/null || true

if python3 -c "import torch_npu" >/dev/null 2>&1; then
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
  exec python3 "$HERE/serve.py" --port "$PORT" --model "$MODEL" --device "$DEVICE"
fi

IMAGE="${SERVE_NPU_IMAGE:-serve-llm-ascend-910b:latest}"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "[serve] building $IMAGE from $HERE/Dockerfile (first run only)" >&2
  docker build -t "$IMAGE" "$HERE" >&2
fi

# Weights are mounted at their own path so serve.py sees the same tree. An HF
# home (a cache with hub/) points HF_HOME at itself; a plain checkpoint gets a
# writable scratch home so nothing tries to write into the read-only mount.
DOCKER_ARGS=(--rm --privileged --network host --ulimit nofile=65536:65536)
DOCKER_ARGS+=(-v "${MODEL}:${MODEL}:ro")
if [[ -d "$MODEL/hub" ]]; then
  DOCKER_ARGS+=(-e "HF_HOME=${MODEL}" -e HF_HUB_OFFLINE=1)
else
  DOCKER_ARGS+=(-e HF_HOME=/tmp/hf -e HF_HUB_OFFLINE=1)
fi

# Every directory in SERVE_OUT_MOUNTS (colon separated) is mounted rw at the same
# path: /score writes exactly where the host-side runner will read. Point
# it at the output directory.
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

# The deployment dir rides along read-only, so serve.py is edited here, not
# baked into the image. bash execs python, so SIGTERM reaches the server.
# Host network: the server binds 127.0.0.1 inside the container, and the
# host-side runner reaches it on the same loopback -- a published port would
# forward to the container's ethernet address, where nothing listens.
DOCKER_ARGS+=(-v "${HERE}:/deployment:ro" -w /deployment)

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE" bash -c '
  source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
  exec python3 serve.py --port '"$PORT"' --model '"$MODEL"' --device '"$DEVICE"'
'
