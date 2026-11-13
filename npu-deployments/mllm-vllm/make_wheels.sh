#!/usr/bin/env bash
# Rebuild the offline wheel closure for this deployment (wheels/).
#
# The closure is the pip resolution of the engine pair (vllm==0.30.0 +
# vllm-ascend==0.30.0) inside the graded image's python 3.12 against its
# system site packages, verified on the ascend-910b box; plus the five
# platform wheels the Dockerfile bakes in. serve.sh installs the engine
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
# - arctic_inference==0.1.1 ships only an sdist and is absent from the
#   configured mirrors. The special-case below fetches that pinned sdist from
#   PyPI, verifies it, and builds the wheel locally.
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
  grep -v '^arctic_inference==' |
  xargs -P 8 -n 1 -I{} pip download {} --no-deps \
    -d "$OUT" \
    -i https://mirrors.huaweicloud.com/repository/pypi/simple \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    --only-binary=:all: --no-cache-dir -q

# The configured mirrors do not publish arctic_inference 0.1.1. Fetch the
# small source archive directly, verify its immutable PyPI digest, and build
# its wheel without consulting an index or resolving optional dependencies.
ARCTIC_SRC_DIR="$OUT/.arctic-src"
ARCTIC_SRC="$ARCTIC_SRC_DIR/arctic_inference-0.1.1.tar.gz"
ARCTIC_URL="https://files.pythonhosted.org/packages/2a/d2/3bcebda042d08cfa0f35a2bbc6dd168d60fea60b6678824665c7a3665418/arctic_inference-0.1.1.tar.gz"
ARCTIC_SHA256="36857e2fbe3ead34821a04be98dfab95086050c1e38199c8fdfe308234fdcf63"
mkdir -p "$ARCTIC_SRC_DIR"
if ! compgen -G "$OUT/arctic_inference-0.1.1-*.whl" >/dev/null; then
  if [[ ! -s "$ARCTIC_SRC" ]]; then
    curl -sfL --retry 20 --retry-all-errors -o "$ARCTIC_SRC" "$ARCTIC_URL"
  fi
  echo "$ARCTIC_SHA256  $ARCTIC_SRC" | sha256sum -c -
  python3 -m pip wheel "$ARCTIC_SRC" --no-deps --no-cache-dir \
    --index-url https://pypi.org/simple \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    -w "$OUT"
fi
rm -rf "$ARCTIC_SRC_DIR"

# The Dockerfile and first-start venv both depend on this platform set. Do
# not report a usable closure when a large or architecture-specific wheel was
# skipped by a mirror or was never copied into the wheelhouse.
for pattern in \
  'torch-2.10.0-*.whl' \
  'torch_npu-2.10.0.post7-*.whl' \
  'torchvision-0.25.0-*.whl' \
  'torchaudio-2.10.0-*.whl' \
  'torchcodec-0.17.0-*.whl' \
  'arctic_inference-0.1.1-*.whl'
do
  if ! compgen -G "$OUT/$pattern" >/dev/null; then
    echo "missing required offline wheel: $pattern" >&2
    exit 1
  fi
done

echo "wheels in $OUT: $(ls "$OUT" | wc -l)"
echo "pinned:        $(grep -cE '^[a-zA-Z0-9].*==' "$REQ")"
echo "rerun this script until the two counts match (transient mirror timeouts)."
