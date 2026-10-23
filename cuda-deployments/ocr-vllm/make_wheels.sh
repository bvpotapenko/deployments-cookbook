#!/usr/bin/env bash
# Rebuild the offline wheel closure for this deployment (wheels/).
#
#   ./make_wheels.sh
#
# The closure is vllm 0.32.0 plus its torch and CUDA libs (linux x86_64,
# python 3.12). The exact pins live in wheels-requirements.txt; regenerate
# them from a verified provisioned venv with:
#   /root/kotvenv-vllm-ocr/bin/pip freeze > wheels-requirements.txt
# Then `./kot_serve.sh` installs offline from wheels/ on first start.
# Per-package --no-deps over exact pins: one pass, no resolution, resumable
# (pip skips files already in $OUT).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$DIR/wheels-requirements.txt"
OUT="${1:-$DIR/wheels}"
mkdir -p "$OUT"

grep -E '^[a-zA-Z0-9].*==' "$REQ" | tr '[:upper:]' '[:lower:]' |
  xargs -P 8 -n 1 -I{} pip download {} --no-deps \
    -d "$OUT" \
    --python-version 3.12 --abi cp312 --abi abi3 --implementation cp \
    --platform manylinux2014_x86_64 --platform manylinux1_x86_64 \
    --platform manylinux_2_17_x86_64 --platform manylinux_2_26_x86_64 \
    --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 \
    --platform manylinux_2_39_x86_64 \
    --only-binary=:all: --no-cache-dir

echo "wheels in $OUT: $(ls "$OUT" | wc -l)"
echo "pinned:        $(grep -cE '^[a-zA-Z0-9].*==' "$REQ")"
echo "rerun this script until the two counts match (transient pypi timeouts)."
