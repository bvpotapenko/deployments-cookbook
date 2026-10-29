#!/usr/bin/env python3
"""Turn a photo and a task into the kot observation file /kot/infer reads.

    make_observation.py photo.jpg "pick up the spoon" obs.npz

The .npz holds what kot's freeze writes for a VLA step: `prompt`, the task as
a string, and one camera, here named `image`, as float32 [3, H, W] in [0, 1].
It holds no `state` and no `noise`: OpenVLA reads neither.
"""

import sys

import numpy as np
from PIL import Image

if len(sys.argv) != 4:
    sys.exit(__doc__)
photo, task, out = sys.argv[1:]
pixels = np.asarray(Image.open(photo).convert("RGB"), dtype=np.float32).transpose(2, 0, 1) / 255
np.savez(out, prompt=np.asarray(task), image=pixels)
