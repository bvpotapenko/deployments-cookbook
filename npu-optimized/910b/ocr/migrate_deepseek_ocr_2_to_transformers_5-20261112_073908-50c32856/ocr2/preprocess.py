"""Image preprocessing and prompt assembly, verbatim from the working
example deployment (which mirrors the checkpoint's own infer()):

* single view (both sides <= 768 px): the image is padded to 1024x1024 and
  encoded as one 16*16+1 = 257-token block, with a zero crop tensor;
* crop view (either side > 768 px): dynamic_preprocess picks a 2..6-crop grid
  of 768x768 tiles, the token block grows by 144 tokens per tile;
* the prompt template expands the single <image> placeholder of the frozen
  template ids into that block, consuming every other id verbatim.
"""
import math
from itertools import repeat

import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGE_TOKEN_ID = 128815
NORM = (0.5, 0.5, 0.5)
RGB_MAX = 255
BASE_SIZE = 1024
MAX_SINGLE_VIEW = 768
PATCH_SIZE = 16
DOWNSAMPLE = 4
CROP_QUERIES = math.ceil((MAX_SINGLE_VIEW // PATCH_SIZE) / DOWNSAMPLE)
IMAGE_IDS = [IMAGE_TOKEN_ID] * (
    math.ceil((BASE_SIZE // PATCH_SIZE) / DOWNSAMPLE) ** 2 + 1
)


class BasicImageTransform:
    def __init__(self, mean=NORM, std=NORM, normalize=True):
        self.mean = mean
        self.std = std
        pipeline = [transforms.ToTensor()]
        if normalize:
            pipeline.append(transforms.Normalize(mean=mean, std=std))
        self.transform = transforms.Compose(pipeline)

    def __call__(self, x):
        return self.transform(x)


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=2, max_num=6, image_size=768, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_aspect_ratio


def views(pil_image, device, dtype):
    """The image as the forward encodes it.

    Returns ((patches_or_None, global_view), (width_num, height_num),
    image_tokens).  patches is None for the single-view case (the zero crop
    tensor the checkpoint fed the forward), else the 768px tile stack.
    """
    image = pil_image.convert("RGB")
    transform = BasicImageTransform()
    pad_color = tuple(int(channel * RGB_MAX) for channel in NORM)
    global_view = transform(
        ImageOps.pad(image, (BASE_SIZE, BASE_SIZE), color=pad_color)
    ).to(dtype).unsqueeze(0)
    if image.size[0] <= MAX_SINGLE_VIEW and image.size[1] <= MAX_SINGLE_VIEW:
        return (None, global_view.to(device)), (1, 1), list(IMAGE_IDS)
    tiles, (width_num, height_num) = dynamic_preprocess(image)
    patches = torch.stack([transform(tile) for tile in tiles]).to(dtype)
    tokens = [
        *IMAGE_IDS,
        *([IMAGE_TOKEN_ID] * (CROP_QUERIES * width_num)) * (CROP_QUERIES * height_num),
    ]
    return (patches.to(device), global_view.to(device)), (width_num, height_num), tokens


def load_image(path):
    return Image.open(path).convert("RGB")


def expand_template(raw_ids, image_tokens):
    """Expand the single image placeholder; every other id verbatim.

    Returns (expanded_ids, image_position_mask).
    """
    at = raw_ids.index(IMAGE_TOKEN_ID)
    expanded = [*raw_ids[:at], *image_tokens, *raw_ids[at + 1:]]
    mask = [
        *repeat(False, at),
        *repeat(True, len(image_tokens)),
        *repeat(False, len(expanded) - at - len(image_tokens)),
    ]
    return expanded, mask
