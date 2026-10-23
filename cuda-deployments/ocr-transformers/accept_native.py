"""Per-unit acceptance of the native 5.20.2 engine against the 4.46.3 golden.

For every unit: do the two captures hold the same ids (same teacher-forced
sequence), how many argmax positions flip, and the renyi_inf distance per
position, with kot's own metric functions. Run on the box:

    /root/kotvenv-run/bin/python accept_native.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/kot")
from kot.eval.distance import linf_per_position, top1_flip_per_position

GOLDEN = Path("/workspace/kot/kot_store/goldens/deepseek-ocr-2-rtx5090-transformers/capture_0")
CANDIDATE = Path("/workspace/kot/kot_store/candidates/ocr-native-5.20.2/capture_0")

print(f"{'unit':12s} {'rows':>5s} {'ids_eq':>7s} {'flips':>6s} {'flip%':>7s} {'renyi_med':>9s} {'renyi_max':>9s} {'first_id_div':>12s}")
all_rows = []
for unit in sorted(p.name for p in CANDIDATE.iterdir() if p.is_dir()):
    g_ids = np.load(GOLDEN / unit / "input_ids.npy")
    c_ids = np.load(CANDIDATE / unit / "input_ids.npy")
    g_rows = np.load(GOLDEN / unit / "log_probs.npy")
    c_rows = np.load(CANDIDATE / unit / "log_probs.npy")
    ids_eq = g_ids.shape == c_ids.shape and bool((g_ids == c_ids).all())
    first_div = -1
    if not ids_eq:
        common = min(len(g_ids), len(c_ids))
        differing = np.nonzero(g_ids[:common] != c_ids[:common])[0]
        first_div = int(differing[0]) if len(differing) else -1
    n = min(len(g_rows), len(c_rows))
    flips = top1_flip_per_position(g_rows[:n], c_rows[:n])
    renyi = linf_per_position(g_rows[:n], c_rows[:n])
    all_rows.append(n)
    print(
        f"{unit:12s} {n:5d} {str(ids_eq):>7s} {int(flips.sum()):6d} "
        f"{100 * flips.mean():6.2f}% {np.median(renyi):9.4f} {renyi.max():9.4f} {first_div:12d}"
    )
print(f"total rows compared: {sum(all_rows)} over {len(all_rows)} units")
