"""Teacher-forced acceptance: the native engine over the 4.46.3 reference's
own ids. For every golden unit, the golden's stored ids go to the native
server as one prefill (n_rows = the decode tail), and the returned rows are
compared with the golden's stored rows on IDENTICAL contexts: argmax flips
and renyi_inf with kot's own functions. A server must be running on PORT.

    /root/kotvenv-run/bin/python probe_reference.py 8791
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, "/workspace/kot")
from kot.eval.distance import linf_per_position, top1_flip_per_position

GOLDEN = Path("/workspace/kot/kot_store/goldens/deepseek-ocr-2-rtx5090-transformers/capture_0")
IMAGE_DIR = "/workspace/kot/kot_store/inputs/images/v2"
RUN = Path("/tmp/accept_reference")


def post(port, body):
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/kot/score",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as reply:
        return json.load(reply)


def main():
    port = int(sys.argv[1])
    RUN.mkdir(exist_ok=True)
    print(f"{'unit':12s} {'rows':>5s} {'rows_eq':>8s} {'flips':>6s} {'flip%':>7s} {'renyi_med':>9s} {'renyi_max':>9s}")
    worst_renyi = 0.0
    for unit in sorted(p.name for p in GOLDEN.iterdir() if p.is_dir()):
        g_ids = np.load(GOLDEN / unit / "input_ids.npy")
        g_rows = np.load(GOLDEN / unit / "log_probs.npy")
        n_rows = len(g_ids) - 6
        body = {
            "prompt_id": f"ref_{unit}",
            "input_ids": g_ids.tolist(),
            "mode": "prefill",
            "max_new_tokens": 4,
            "allow_eos": False,
            "n_rows": n_rows,
            "out_dir": str(RUN / unit),
            "seed": 42,
            "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0},
            "attachments": {"image": f"{IMAGE_DIR}/{unit}.png"},
        }
        post(port, body)
        c_rows = np.load(RUN / unit / "log_probs.npy")
        rows_eq = g_rows.shape == c_rows.shape and bool((g_rows == c_rows).all())
        n = min(len(g_rows), len(c_rows))
        flips = top1_flip_per_position(g_rows[:n], c_rows[:n])
        renyi = linf_per_position(g_rows[:n], c_rows[:n])
        worst_renyi = max(worst_renyi, float(np.median(renyi)))
        print(
            f"{unit:12s} {n:5d} {str(rows_eq):>8s} {int(flips.sum()):6d} "
            f"{100 * flips.mean():6.2f}% {np.median(renyi):9.4f} {renyi.max():9.4f}"
        )
    print(f"worst per-unit median renyi_inf: {worst_renyi:.4f} (cross-engine band 0.4-0.7)")


if __name__ == "__main__":
    main()
