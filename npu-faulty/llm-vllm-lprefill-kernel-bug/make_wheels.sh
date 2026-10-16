#!/usr/bin/env bash
# Rebuild the offline wheel closure for this deployment (wheels/).
#
# The closure is the pip resolution of the engine pair (vllm 0.23.0 +
# vllm-ascend==0.23.0) inside the graded image's python 3.12 against its
# system site packages, verified on the ascend-910b box; plus the three
# platform wheels the Dockerfile bakes in. kot_serve.sh installs the engine
# offline from wheels/ into /scratch at first start; the Dockerfile COPYs
# the same directory for the image build.
#
#   ./make_wheels.sh [output-dir]      # rerun until its two counts match
#
# Notes:
# - pypi.org stalls mid-download on the box's network; the huaweicloud
#   mirrors are domestic and fast. triton-ascend / memfabric / memcache
#   exist only on the ascend mirror.
# - vllm itself is the 266MB pypi aarch64 wheel; fetch it separately with
#   curl -sfC - --retry if pip's download stalls (it is just a pinned name
#   in wheels-requirements.txt).
# - arctic-inference==0.1.1 ships no wheel: pip builds it from the sdist
#   (one rerun of this script picks the built wheel up).
# - per-package --no-deps over exact pins: one pass, no resolution,
#   resumable (pip skips files already in $OUT). Do NOT point pip at the
#   whole reqs file: full-graph resolution with this many pins stalls and
#   fights its own conflicts.
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$DIR/wheels-requirements.txt"
OUT="${1:-$DIR/wheels}"
mkdir -p "$OUT"

grep -E '^[a-zA-Z0-9].*==' "$REQ" | tr '[:upper:]' '[:lower:]' |
  xargs -P 8 -n 1 -I{} pip download {} --no-deps \
    -d "$OUT" \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --only-binary=:all: --no-cache-dir -q

echo "wheels in $OUT: $(ls "$OUT" | wc -l)"
echo "pinned:        $(grep -cE '^[a-zA-Z0-9].*==' "$REQ")"
echo "rerun this script until the two counts match (transient mirror timeouts)."
