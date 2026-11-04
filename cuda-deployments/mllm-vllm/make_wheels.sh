#!/usr/bin/env bash
# Rebuild the offline wheel closure for this deployment (wheels/).
#
# The closure is rebuilt whenever the pinned vllm line moves: run this on any
# linux x86_64 host with network, then `./kot_serve.sh` installs offline from
# wheels/ on first start (an offline host has no network).
#
#   ./make_wheels.sh
#
# Pins are in wheels-requirements.txt (compiled with uv pip compile for
# CPython 3.12, --python-platform x86_64-manylinux_2_31; the vllm wheel is
# cp38-abi3 and the torch wheel is cp312, so the download flags must carry both).
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REQ="$DIR/wheels-requirements.txt"
OUT="${1:-$DIR/wheels}"
mkdir -p "$OUT"

# pip matches these tags exactly, not by glibc order: a wheel tagged only
# manylinux_2_18 needs its own flag.
FLAGS=(--no-deps -d "$OUT"
  --python-version 3.12 --abi cp312 --abi abi3 --implementation cp
  --platform manylinux2014_x86_64 --platform manylinux1_x86_64
  --platform manylinux_2_17_x86_64 --platform manylinux_2_18_x86_64
  --platform manylinux_2_25_x86_64 --platform manylinux_2_26_x86_64
  --platform manylinux_2_27_x86_64 --platform manylinux_2_28_x86_64
  --platform manylinux_2_31_x86_64 --platform manylinux_2_39_x86_64
  --only-binary=:all: --no-cache-dir)

# A failed pin must not stop the script: the counts below are how a person
# knows to rerun it.
# Per-package --no-deps over exact pins: one pass, no resolution, resumable
# (pip skips files already in $OUT). Do NOT point pip at the whole reqs file:
# full-graph resolution with hundreds of pins stalls and fights its own
# conflicts.
grep -E '^[a-zA-Z0-9].*==' "$REQ" | tr '[:upper:]' '[:lower:]' | grep -v '^torchvision==' |
  xargs -P 8 -n 1 -I{} python3 -m pip download {} "${FLAGS[@]}" || true

# torchvision is the build the run book installs: the PyTorch cu130 index, not
# PyPI's. Only this pin reads that index; torch stays PyPI's.
python3 -m pip download "$(grep '^torchvision==' "$REQ")" "${FLAGS[@]}" \
  --index-url https://download.pytorch.org/whl/cu130 || true

echo "wheels in $OUT: $(ls "$OUT" | wc -l)"
echo "pinned:        $(grep -cE '^[a-zA-Z0-9].*==' "$REQ")"
echo "rerun this script until the two counts match (transient pypi timeouts)."
