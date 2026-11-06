#!/usr/bin/env python3
"""Write kot's VLA input version v1_openvla from v1, for this deployment.

    make_v1_openvla.py --root $KOT_STORE

v1 describes a pi0-style robot: 3 cameras, action_dim 32, chunk_size 50.
OpenVLA reads one camera and decodes 7 tokens, and kot asks a VLA unit for
action_dim tokens. v1_openvla keeps every field of v1 except three, per
observation: cameras (the first camera of v1 only; freeze draws a camera
from its index, so its image stays the same), action_dim 7 and chunk_size 1.

Reads  <root>/inputs/vla_inputs/v1/observations.json
Writes <root>/inputs/vla_inputs/v1_openvla/observations.json
An existing v1_openvla directory stops the script: a version never changes.
Freeze it with kot afterwards; the run book gives the commands.
"""

import argparse
import json
import os
import sys
from pathlib import Path

SOURCE = "v1"
TARGET = "v1_openvla"
OBSERVATIONS = "observations.json"
ACTION_DIM = 7  # one token per dimension: x, y, z, roll, pitch, yaw, gripper
CHUNK_SIZE = 1  # OpenVLA predicts one action per step
CHANNELS = 3  # serve.py reads a camera as [3, H, W]


def converted(observation: dict) -> dict:
    """One v1 observation with OpenVLA's three fields; every other field as in v1."""
    if not observation["cameras"]:
        raise ValueError(f"{observation['id']}: v1 names no camera")
    if observation["image_shape"][0] != CHANNELS:
        raise ValueError(f"{observation['id']}: serve.py reads [3, H, W]; v1 has {observation['image_shape']}")
    return {
        **observation,
        "cameras": observation["cameras"][:1],
        "action_dim": ACTION_DIM,
        "chunk_size": CHUNK_SIZE,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = os.environ.get("KOT_STORE", "")
    parser.add_argument("--root", type=Path, default=root, required=not root, help="the kot store; default $KOT_STORE")
    args = parser.parse_args()

    versions = args.root / "inputs" / "vla_inputs"
    source = versions / SOURCE / OBSERVATIONS
    document = json.loads(source.read_text(encoding="utf-8"))
    observations = [converted(observation) for observation in document["observations"]]

    target = versions / TARGET
    try:
        target.mkdir()
    except FileExistsError:
        sys.exit(f"{target} exists: a version never changes, and nothing was written")
    with open(target / OBSERVATIONS, "x", encoding="utf-8") as out:
        json.dump({**document, "observations": observations}, out, indent=2, ensure_ascii=False)
        out.write("\n")

    print(f"read  {source}")
    for old, new in zip(document["observations"], observations):
        changes = [f"{key} {old[key]} -> {new[key]}" for key in new if new[key] != old[key]]
        print(f"{new['id']}: {'; '.join(changes) or 'no change'}")
    print(f"wrote {target / OBSERVATIONS}")


if __name__ == "__main__":
    main()
