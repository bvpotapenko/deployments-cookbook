"""SAM vision rel-position fix for DeepSeek-OCR-2 on the Ascend 910B.

The stock ``DeepseekOcr2SamVisionAttention.get_rel_pos`` builds its index
tensors on the CPU (``torch.arange`` with no device) and then gathers from a
device tensor with them: 24 blocking host-to-device copies per vision chunk,
~12 ms each on this card - seconds per batch, and the single largest cost in
the image path.

The gathered rel-position table depends only on the layer's (frozen) rel-pos
parameters and the (q_size, k_size) shape pair - never on the image data.
This module replaces the method with one that (a) builds the coordinate
tensors on the parameters' device and (b) caches the gathered table per
(shape pair) on the attention module. The values are bit-identical to the
stock computation; only where the arithmetic runs and how often changes.
"""

from __future__ import annotations

import torch


def install_fast_rel_pos(model):
    """Patch every SAM attention module's get_rel_pos with the cached one."""
    sam = model.model.vision_tower.sam_encoder
    patched = 0
    for layer in sam.layers:
        attention = layer.attn
        if hasattr(attention, "get_rel_pos"):
            attention.get_rel_pos = _cached_get_rel_pos.__get__(attention, type(attention))
            patched += 1
    return patched


def _cached_get_rel_pos(self, q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    cache = getattr(self, "_rel_pos_cache", None)
    if cache is None:
        cache = self._rel_pos_cache = {}
    # the table is per (rel_pos parameter, shape pair): a module calls this
    # twice per forward, once for the height axis and once for the width
    key = (id(rel_pos), q_size, k_size)
    if key in cache:
        return cache[key]

    device = rel_pos.device
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    rel_pos_resized = torch.nn.functional.interpolate(
        rel_pos.reshape(1, rel_pos.shape[0], -1).transpose(1, 2),
        size=max_rel_dist,
        mode="linear",
    )
    rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)

    q_coords = torch.arange(q_size, device=device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)

    gathered = rel_pos_resized[relative_coords.long()]
    cache[key] = gathered
    return gathered
