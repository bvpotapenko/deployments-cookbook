#!/usr/bin/env bash
# Rebuild the offline wheel closure for this deployment (wheels/).
#
# The closure was once built and verified (4.7GB, 147 packages, vllm 0.11.0 +
# torch 2.8.0 + cu128 CUDA libs, linux x86_64, python 3.11) but the rented
# box holding it was retired before the download home landed. Rebuilding is
# one command on any linux x86_64 host with network:
#
#   ./make_wheels.sh
#
# Then `./kot_serve.sh` installs offline from wheels/ on first start.
# Pins are in wheels-requirements.txt (note the transformers==4.56.2 pin:
# 2026-era transformers 5.x breaks this vllm; see shortcuts).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$DIR/wheels-requirements.txt"
OUT="${1:-$DIR/wheels}"
mkdir -p "$OUT"

# Per-package --no-deps over exact pins: one pass, no resolution, resumable
# (pip skips files already in $OUT). Do NOT point pip at the whole reqs file:
# full-graph resolution with 147 pins stalls and fights its own conflicts.
grep -E '^[a-zA-Z0-9].*==' "$REQ" | tr '[:upper:]' '[:lower:]' |
  xargs -P 8 -n 1 -I{} pip download {} --no-deps \
    -d "$OUT" \
    --python-version 3.11 --abi cp311 --abi abi3 --implementation cp \
    --platform manylinux2014_x86_64 --platform manylinux1_x86_64 \
    --platform manylinux_2_17_x86_64 --platform manylinux_2_26_x86_64 \
    --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64 \
    --platform manylinux_2_39_x86_64 \
    --only-binary=:all: --no-cache-dir

echo "wheels in $OUT: $(ls "$OUT" | wc -l)"
echo "pinned:        $(grep -cE '^[a-zA-Z0-9].*==' "$REQ")"
echo "rerun this script until the two counts match (transient pypi timeouts)."
