#!/usr/bin/env python3
"""Native transformers 5.20.2 processor for DeepSeek-OCR-2.

build_processor(dir) wraps the checkpoint's own LlamaTokenizerFast and the
native PIL image processor into the native DeepseekOcr2Processor. The knobs
reproduce the 4.46.3 reference server's image path (serve.py) exactly:

* global view: bicubic resize so the longest side is 1024, centered pad to
  1024x1024 with color 127, rescale by 1/255, normalize with mean=std=0.5;
* small image (both sides <= 768 px): no tiles, one 257-token image block
  (16x16 = 256 queries + 1 view separator), pixel_values_local absent;
* large image (either side > 768 px): the tile grid of the checkpoint's own
  dynamic_preprocess (2..6 tiles of 768x768, closest aspect ratio), each tile
  costing 144 queries, so the block is 257 + 144 * tiles.

The tokenizer is loaded as LlamaTokenizerFast directly. AutoTokenizer is not
used: it follows config.json's auto_map into the checkpoint's remote code.

Values the checkpoint's processor_config.json pins (image_mean, image_std,
patch_size, downsample_ratio, candidate_resolutions, image_token) are read
from that file and honored. The tile knobs are not pinned there; they come
from the remote dynamic_preprocess defaults (min_num=2, max_num=6,
image_size=768), which the reference calls with defaults.

Output contract of DeepseekOcr2Model.forward (what serve.py wires):

    pixel_values        [batch, 3, 1024, 1024]
    pixel_values_local  [total_tiles, 3, 768, 768], or None when no image
                        in the batch has tiles
    num_local_patches   list[int], tile count per image, e.g. [0] or [6]
    input_ids           the <image> placeholder repeated 257 + 144 * tiles
                        times per image, token id 128815; the forward's
                        placeholder mask asserts token count == feature rows

Per image the vision tower emits [tiles, global, separator] feature rows, in
that order, so a plain run of 128815 ids matches position by position.

The gate (_main) is torch-free: it checks shapes on the numpy arrays the PIL
backend returns with return_tensors=None. Where torch exists, pass
return_tensors="pt"; the shapes are the same. Run on the box:

    python native_processor.py --dir /path/to/DeepSeek-OCR-2 --image page.png
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from transformers import LlamaTokenizerFast
from transformers.models.deepseek_ocr2.image_processing_pil_deepseek_ocr2 import (
    DeepseekOcr2ImageProcessorPil,
)
from transformers.models.deepseek_ocr2.processing_deepseek_ocr2 import DeepseekOcr2Processor

IMAGE_TOKEN_ID = 128815
GLOBAL_SIZE = 1024  # the padded global view, reference BASE_SIZE
MAX_SINGLE_VIEW = 768  # tile size; both sides <= this means no tiles
MIN_TILES = 2  # remote dynamic_preprocess(min_num=2, ...)
MAX_TILES = 6  # remote dynamic_preprocess(..., max_num=6)
PATCH_SIZE = 16
DOWNSAMPLE = 4
NORM = (0.5, 0.5, 0.5)
PAD_COLOR = (127, 127, 127)  # int(0.5 * 255) per channel, as the reference pads
TEMPLATE = "<|User|>:\n<image>\nFree OCR."  # the checkpoint's plain sft format
GLOBAL_QUERIES = (GLOBAL_SIZE // PATCH_SIZE // DOWNSAMPLE) ** 2  # 256
TILE_QUERIES = (MAX_SINGLE_VIEW // PATCH_SIZE // DOWNSAMPLE) ** 2  # 144


def build_processor(dir):
    """Build the native DeepseekOcr2Processor for the checkpoint at dir."""
    checkpoint = Path(dir)
    pins = {}
    pinned = checkpoint / "processor_config.json"
    if pinned.is_file():
        pins = json.loads(pinned.read_text())

    # candidate_resolutions pins the global view. This checkpoint pins one
    # square resolution, and the 257-token block math needs the square.
    resolutions = pins.get("candidate_resolutions", [[GLOBAL_SIZE, GLOBAL_SIZE]])
    width, height = resolutions[0]
    if width != height:
        raise ValueError(f"candidate_resolutions {resolutions} is not square: the global view must be square")

    # LlamaTokenizerFast directly: never AutoTokenizer, which follows
    # config.json's auto_map back into the checkpoint's remote code.
    tokenizer = LlamaTokenizerFast.from_pretrained(str(checkpoint))

    image_processor = DeepseekOcr2ImageProcessorPil(
        size={"height": height, "width": width},
        image_mean=list(pins.get("image_mean", NORM)),
        image_std=list(pins.get("image_std", NORM)),
        do_normalize=bool(pins.get("normalize", True)),
        crop_to_patches=True,
        min_patches=MIN_TILES,
        max_patches=MAX_TILES,
        tile_size=MAX_SINGLE_VIEW,
        background_color=list(PAD_COLOR),
    )

    processor = DeepseekOcr2Processor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        patch_size=int(pins.get("patch_size", PATCH_SIZE)),
        downsample_ratio=int(pins.get("downsample_ratio", DOWNSAMPLE)),
    )

    expected_token = pins.get("image_token", "<image>")
    if processor.image_token != expected_token:
        raise ValueError(f"image token is {processor.image_token!r}, the checkpoint pins {expected_token!r}")
    if processor.image_token_id != IMAGE_TOKEN_ID:
        raise ValueError(f"<image> maps to id {processor.image_token_id}, expected {IMAGE_TOKEN_ID}")
    return processor


def _oracle_grid(width, height):
    """The tile grid the checkpoint's own dynamic_preprocess picks.

    Candidate grids are the (columns, rows) with MIN_TILES <= columns * rows
    <= MAX_TILES. Pick the grid whose aspect ratio is closest to the image's.
    On a tie, keep the later grid only if it covers more than half the image
    area. No two candidate grids share an aspect ratio, so ties need the
    image to sit exactly between two grid ratios.
    """
    aspect = width / height
    area = width * height
    grids = sorted(
        (w, h)
        for w in range(1, MAX_TILES + 1)
        for h in range(1, MAX_TILES + 1)
        if MIN_TILES <= w * h <= MAX_TILES
    )
    best, best_diff = (1, 1), float("inf")
    for grid in grids:
        diff = abs(aspect - grid[0] / grid[1])
        if diff < best_diff or (diff == best_diff and area > 0.5 * MAX_SINGLE_VIEW**2 * grid[0] * grid[1]):
            best, best_diff = grid, diff
    return best


def _flat_ids(value):
    """The batch's single input_ids row as a plain list of ints."""
    row = value[0] if isinstance(value, list) else value.tolist()[0]
    return row.tolist() if hasattr(row, "tolist") else list(row)


def _per_image_shapes(value):
    """Shapes of the per-image arrays a pixel_values field holds."""
    if value is None:
        return []
    return [tuple(arr.shape) for arr in value]


def _run_one(processor, image, label):
    """Run one PIL image through the processor; print PASS/FAIL lines.

    Returns the list of failed check names.
    """
    out = processor(text=[TEMPLATE], images=[image], return_tensors=None)
    ids = _flat_ids(out["input_ids"])
    block = sum(1 for token in ids if token == IMAGE_TOKEN_ID)
    width, height = image.size
    tiles = 0 if max(image.size) <= MAX_SINGLE_VIEW else math.prod(_oracle_grid(width, height))
    grid = "single view" if not tiles else "{}x{} ({} tiles)".format(*_oracle_grid(width, height), tiles)
    expected_block = GLOBAL_QUERIES + 1 + TILE_QUERIES * tiles

    counts = out["num_local_patches"]
    counts = counts.tolist() if hasattr(counts, "tolist") else list(counts)
    local_shapes = _per_image_shapes(out.get("pixel_values_local"))
    global_shapes = _per_image_shapes(out.get("pixel_values"))
    expected_local = [] if not tiles else [(3, MAX_SINGLE_VIEW, MAX_SINGLE_VIEW)] * tiles

    checks = [
        (f"{label}: grid {grid}", True, ""),
        (
            f"{label}: {expected_block} image tokens of id {IMAGE_TOKEN_ID}",
            block == expected_block,
            f"got {block}",
        ),
        (
            f"{label}: num_local_patches == [{tiles}]",
            counts == [tiles],
            f"got {counts}",
        ),
        (
            f"{label}: pixel_values [1, 3, {GLOBAL_SIZE}, {GLOBAL_SIZE}]",
            global_shapes == [(3, GLOBAL_SIZE, GLOBAL_SIZE)],
            f"got {global_shapes}",
        ),
        (
            f"{label}: pixel_values_local {expected_local or 'absent'}",
            local_shapes == expected_local,
            f"got {local_shapes or 'absent'}",
        ),
    ]
    failed = []
    for name, ok, detail in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f" ({detail})" if not ok else ""))
        if not ok:
            failed.append(name)
    return failed


def _main(argv=None):
    parser = argparse.ArgumentParser(description="Gate for build_processor: token counts and forward shapes.")
    parser.add_argument("--dir", required=True, help="DeepSeek-OCR-2 checkpoint dir")
    parser.add_argument("--image", default=None, help="optional png run through the same checks")
    args = parser.parse_args(argv)

    import transformers
    from PIL import Image

    print(f"transformers {transformers.__version__}")
    processor = build_processor(args.dir)
    image_processor = processor.image_processor
    print(f"tokenizer {type(processor.tokenizer).__name__}, image token {processor.image_token!r} -> {processor.image_token_id}")
    print(
        "image processor: size={} mean={} std={} normalize={} crop_to_patches={} "
        "min_patches={} max_patches={} tile_size={} background_color={}".format(
            dict(image_processor.size),
            image_processor.image_mean,
            image_processor.image_std,
            image_processor.do_normalize,
            image_processor.crop_to_patches,
            image_processor.min_patches,
            image_processor.max_patches,
            image_processor.tile_size,
            image_processor.background_color,
        )
    )
    print(f"processor: patch_size={processor.patch_size} downsample_ratio={processor.downsample_ratio}")

    # (a) and (b): generated images, no fixtures.
    small = Image.new("RGB", (640, 640), (202, 52, 43))
    large = Image.new("RGB", (1600, 1200), (43, 57, 84))
    failed = _run_one(processor, small, "640x640 (a)")
    failed += _run_one(processor, large, "1600x1200 (b)")

    if args.image:
        with Image.open(args.image) as opened:
            extra = opened.convert("RGB")
        failed += _run_one(processor, extra, f"{args.image} {extra.size[0]}x{extra.size[1]}")

    if failed:
        print(f"FAIL: {len(failed)} check(s) failed")
        return 1
    print("PASS: all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
