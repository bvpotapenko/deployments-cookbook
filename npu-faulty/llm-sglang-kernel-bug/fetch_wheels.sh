#!/usr/bin/env bash
# fetch_wheels.sh -- fill ./wheels with the sglang NPU stack, offline-installable.
#
# Run by hand on the ascend-910b host (or any aarch64 box with network), from
# this directory. The box's network stalls mid-download on big pypi wheels,
# so every big file goes through `curl -sfC - --retry`, which resumes.
#
#   ./fetch_wheels.sh
#
# Produces:
#   wheels/sglang-<ver>-py3-none-any.whl     built from sglang-src/ at SGLANG_COMMIT
#   wheels/triton_ascend-3.2.2-*.whl         the Ascend triton
#   wheels/sgl_kernel_npu-*.whl, deep_ep-*, attentions-*.whl
#   wheels/*custom_ops*.run, *ops-transformer*.run   CANN custom-op installers
#   wheels/<dependency closure of sglang's npu pyproject minus the image stack>
#   sglang-src/                              the vendored source (readable)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
WHEELS="$HERE/wheels"
IMAGE="${KOT_NPU_IMAGE:-kot-llm-ascend-910b:latest}"
KERNEL_TAG="${KERNEL_TAG:-2026.9.0.post1}"

# Pin sglang at fetch time: reuse the recorded commit on reruns (main moves
# every day; one pinned build for the whole deployment), override by env.
SGLANG_COMMIT="${SGLANG_COMMIT:-$(cat "$WHEELS/SGLANG_COMMIT.txt" 2>/dev/null || curl -sfL https://api.github.com/repos/sgl-project/sglang/commits/main | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])')}"

mkdir -p "$WHEELS"
echo "== sglang @ $SGLANG_COMMIT"
if [[ ! -d "$HERE/sglang-src/.git" ]]; then
  git clone --depth 1 https://github.com/sgl-project/sglang "$HERE/sglang-src"
fi
git -C "$HERE/sglang-src" fetch --depth 1 origin "$SGLANG_COMMIT"
git -C "$HERE/sglang-src" checkout --force "$SGLANG_COMMIT"
echo "$SGLANG_COMMIT" >"$WHEELS/SGLANG_COMMIT.txt"

# The vendored pyproject pins memfabric-zbal==1.1.2, which pypi does not
# publish for cp312/aarch64 (available: 1.1.3, 1.2.x). The sglang docs'
# CANN 9.1.0 component set mandates 1.1.3, so the pin moves to the documented
# pairing; kot_serve.sh installs the same version explicitly.
#
# outlines and timm are dropped: outlines is only imported by the
# constrained-generation backends (never used on the score/speed paths) and
# pulls an unpinned torch; timm is only imported by deepseek_vl2 (a vision
# model) and pulls torchvision. Without them the resolver stays off the
# image's own torch stack entirely.
sed -i 's/memfabric-zbal==1\.1\.2/memfabric-zbal==1.1.3/' "$HERE/sglang-src/python/pyproject_npu.toml"
sed -i '/"outlines==/d; /"timm==/d' "$HERE/sglang-src/python/pyproject_npu.toml"

echo "== building the sglang wheel inside $IMAGE"
docker run --rm -v "$HERE":/dep -w /dep "$IMAGE" bash -c '
  set -euo pipefail
  cd /dep/sglang-src/python
  rm -f pyproject.toml
  cp pyproject_npu.toml pyproject.toml
  pip wheel --no-cache-dir --no-deps -w /dep/wheels .
'

echo "== triton-ascend 3.2.2 (aarch64, cp312)"
TRITON_URL="https://github.com/triton-lang/triton-ascend/releases/download/v3.2.2/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
TRITON_WHL="$WHEELS/triton_ascend-3.2.2-cp312-cp312-manylinux_2_27_aarch64.manylinux_2_28_aarch64.whl"
if [[ ! -s "$TRITON_WHL" ]]; then
  curl -sfLC - --retry 20 --retry-all-errors -o "$TRITON_WHL" "$TRITON_URL"
fi

echo "== sgl-kernel-npu $KERNEL_TAG (cann9.1.0, 910b, aarch64)"
base="https://github.com/sgl-project/sgl-kernel-npu/releases/download/$KERNEL_TAG"
for name in \
  "custom-ops-$KERNEL_TAG-torch2.10.0-cann9.1.0-910b-aarch64.zip" \
  "ops-transformer-$KERNEL_TAG-torch2.10.0-cann9.1.0-910b-aarch64.zip" \
  "sgl-kernel-npu-$KERNEL_TAG-torch2.10.0-py312-cann9.1.0-910b-aarch64.zip"
do
  [[ -s "$WHEELS/$name" ]] || curl -sfLC - --retry 20 --retry-all-errors -o "$WHEELS/$name" "$base/$name"
done
unzip -o -j "$WHEELS/sgl-kernel-npu-$KERNEL_TAG-torch2.10.0-py312-cann9.1.0-910b-aarch64.zip" -d "$WHEELS"
unzip -o -j "$WHEELS/custom-ops-$KERNEL_TAG-torch2.10.0-cann9.1.0-910b-aarch64.zip" -d "$WHEELS"
unzip -o -j "$WHEELS/ops-transformer-$KERNEL_TAG-torch2.10.0-cann9.1.0-910b-aarch64.zip" -d "$WHEELS"

echo "== dependency closure (resolved inside the image: torch/torch_npu/numpy are satisfied there)"
docker run --rm -v "$HERE":/dep -w /dep "$IMAGE" bash -c '
  set -euo pipefail
  export PIP_BREAK_SYSTEM_PACKAGES=1 PIP_RETRIES=20 PIP_TIMEOUT=60
  pip download --no-cache-dir -d /dep/wheels /dep/sglang-src/python/ \
    || echo "pip download hit a stall; fetch the named wheel by curl -C - and rerun"
'

echo "== wheels/ contents"
# Keep exactly one sglang wheel (the pinned build): main moved between fetches
# once and two wheels would both match the install glob. The torch/torch-
# vision/torchaudio wheels the resolver pulled are the image's own stack
# (pip download ignores installed dists); at install time the requirement is
# satisfied by the image, so they only dead-weight the handed-back directory.
cd "$WHEELS"
ls sglang-*.whl | grep -v "g${SGLANG_COMMIT:0:9}" | xargs -r rm -f || true
rm -f torch-*.whl torchvision-*.whl torchaudio-*.whl
ls -la "$WHEELS"
echo "Done. Sizes matter only for the first fetch; the deployment installs offline."
