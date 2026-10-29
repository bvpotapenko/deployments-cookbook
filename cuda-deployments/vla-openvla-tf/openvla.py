"""OpenVLA-7B, inference only: one RGB frame and one task in, one 7-DoF action
out, or the log-probs of the 7 action tokens.

The network is the one the checkpoint's own modeling_prismatic.py builds:

    frame -> resize 224x224 bicubic -> two normalisations, channel-stacked: [6, 224, 224]
      channels 0-2 -> DINOv2 ViT-L/14   -> 256 patches x 1024 --+
      channels 3-5 -> SigLIP SO400M/14  -> 256 patches x 1152 --+-> concat: 256 x 2176
      projector MLP 2176 -> 8704 -> 4096 -> 4096               -> 256 image tokens
    "<s>In: What action should the robot take to {task}?\\nOut:" + token 29871
      the 256 image tokens go in right after <s> -> Llama-2-7B -> 7 greedy tokens
    token -> bin (256 bins over [-1, 1]) -> unnormalise with one dataset's q01/q99

Why not the checkpoint's own code: it was written for transformers 4.40.1 and
breaks on 5.20. Its _supports_sdpa property and its tie_weights() override
both fail at construction, its class has no generate(), and once patched past
those, generate() hands it an empty cache on the first step: it then keeps
only the last prompt token, never sees the image, and returns an action
without an error. This file builds the same modules - timm for the two vision
towers, transformers' LlamaForCausalLM for the language model - loads the same
safetensors into them with every key accounted for, and decodes the 7 tokens
in a loop of its own.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image
from safetensors import safe_open
from torch import nn
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

# The prompt OpenVLA was trained on. The task is lower-cased, as the reference
# deploy script (openvla vla-scripts/deploy.py) does.
PROMPT = "In: What action should the robot take to {task}?\nOut:"
# predict_action appends token 29871 ("▁", an empty piece) when the prompt does
# not end with it: the training sequences had it right after "Out:".
EMPTY_TOKEN = 29871


class OpenVLA(nn.Module):
    """The modules, named as the checkpoint names them, so its keys load as they are."""

    def __init__(self, config: dict) -> None:
        super().__init__()
        if not config["use_fused_vision_backbone"]:
            raise ValueError("this file builds openvla-7b's fused DINOv2 + SigLIP backbone")
        towers = zip(config["timm_model_ids"], config["image_sizes"], config["timm_override_act_layers"])
        self.vision_backbone = nn.ModuleDict({
            name: timm.create_model(model_id, pretrained=False, num_classes=0, img_size=size, act_layer=act)
            for name, (model_id, size, act) in zip(("featurizer", "fused_featurizer"), towers)
        })
        vision_dim = sum(tower.embed_dim for tower in self.vision_backbone.values())
        # The checkpoint states only 4 Llama fields; the rest are LlamaConfig's
        # defaults, which are Llama-2-7B's geometry (rms_norm_eps 1e-6, rope_theta
        # 10000), in 4.40.1 and in 5.20.2 alike.
        text = LlamaConfig(**config["text_config"], attn_implementation="sdpa")
        self.projector = nn.ModuleDict({
            "fc1": nn.Linear(vision_dim, 4 * vision_dim),
            "fc2": nn.Linear(4 * vision_dim, text.hidden_size),
            "fc3": nn.Linear(text.hidden_size, text.hidden_size),
        })
        self.language_model = LlamaForCausalLM(text)

    def image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[1, 6, 224, 224] -> [1, 256, 4096]. Each tower reads its 3 channels and
        gives the patches of its second-to-last block; the two sets sit side by side."""
        halves = torch.split(pixel_values, [3, 3], dim=1)
        patches = [
            tower.get_intermediate_layers(half, n={len(tower.blocks) - 2})[0]
            for tower, half in zip(self.vision_backbone.values(), halves)
        ]
        x = torch.cat(patches, dim=2)
        x = nn.functional.gelu(self.projector["fc1"](x))
        x = nn.functional.gelu(self.projector["fc2"](x))
        return self.projector["fc3"](x)

    def embed(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """[<s>, 256 image tokens, the rest of the ids] as one embedding sequence."""
        text = self.language_model.get_input_embeddings()(input_ids)
        return torch.cat([text[:, :1], self.image_tokens(pixel_values), text[:, 1:]], dim=1)

    @torch.inference_mode()
    def decode(self, pixel_values: torch.Tensor, input_ids: torch.Tensor, n_tokens: int) -> torch.Tensor:
        """Greedy decoding: one forward over [<s>, image, prompt], then one token
        at a time from the KV cache, each the argmax of all logits. Returns each
        step's logits, [n_tokens, vocab]; their argmax are the tokens."""
        out = self.language_model(inputs_embeds=self.embed(pixel_values, input_ids), use_cache=True)
        logits = [out.logits[:, -1]]
        for _ in range(n_tokens - 1):
            token = logits[-1].argmax(dim=-1, keepdim=True)
            out = self.language_model(input_ids=token, past_key_values=out.past_key_values, use_cache=True)
            logits.append(out.logits[:, -1])
        return torch.cat(logits)

    @torch.inference_mode()
    def prefill(self, pixel_values: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Teacher forcing over [<s>, image, ids[1:]]: the logits that predict
        ids[1:], [n - 1, vocab]. Each id is predicted at the position before it;
        for ids[1] that is the last image token, so the rows start there."""
        logits = self.language_model(inputs_embeds=self.embed(pixel_values, input_ids), use_cache=False).logits[0]
        n_image = logits.shape[0] - input_ids.shape[1]
        return logits[n_image:-1]


def build(model_dir: Path, config: dict, device: torch.device, dtype: torch.dtype) -> OpenVLA:
    """Construct the modules on the device in the weights' dtype, then copy every
    checkpoint tensor in. A tensor without a parameter, a parameter without a
    tensor, or a shape that differs stops the load."""
    torch.set_default_dtype(dtype)
    with device:
        model = OpenVLA(config)
    torch.set_default_dtype(torch.float32)
    params = model.state_dict()
    loaded = set()
    for shard in sorted(model_dir.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as weights:
            for key in weights.keys():
                # The remote code renamed timm's LayerScale.gamma to scale_factor:
                # transformers 4.x renamed every *gamma* key it loaded.
                name = key.replace(".scale_factor", ".gamma")
                tensor = weights.get_tensor(key)
                if name not in params or params[name].shape != tensor.shape:
                    raise RuntimeError(f"checkpoint tensor {key} {tuple(tensor.shape)} fits no parameter")
                params[name].copy_(tensor)
                loaded.add(name)
    missing = sorted(set(params) - loaded)
    if missing:
        raise RuntimeError(f"{len(missing)} parameters have no checkpoint tensor: {missing[:3]} ...")
    return model.eval()


class Policy:
    """Frame + task -> action, with the checkpoint's own preprocessing, prompt and action statistics."""

    def __init__(self, model_dir: Path, device: str, unnorm_key: str) -> None:
        config = json.loads((model_dir / "config.json").read_text())
        self.dtype = getattr(torch, config["torch_dtype"])  # bfloat16: the weights' own precision
        self.device = torch.device(device)
        self.model = build(model_dir, config, self.device, self.dtype)
        # The concrete class: AutoTokenizer reads config.json, finds remote code
        # and stops to ask whether to run it.
        self.tokenizer = LlamaTokenizer.from_pretrained(model_dir)

        # The image processor's parameters, as the checkpoint stores them. The
        # reference processor takes its tvf_* fields from this file, so its
        # normalisation runs with the stored, bf16-rounded means (0.484375, not
        # 0.485). This code reproduces "resize-naive": no letterbox, crop = size.
        processor = json.loads((model_dir / "preprocessor_config.json").read_text())
        crops = [crop["output_size"] for crop in processor["tvf_crop_params"]]
        self.resize = processor["tvf_resize_params"]
        if processor["image_resize_strategy"] != "resize-naive" or crops != [r["size"] for r in self.resize]:
            raise ValueError(f"unsupported image processor: {processor['image_resize_strategy']}")
        self.mean = [np.array(n["mean"], dtype=np.float32) for n in processor["tvf_normalize_params"]]
        self.std = [np.array(n["std"], dtype=np.float32) for n in processor["tvf_normalize_params"]]

        # Token -> action. The action tokens are the last 256 ids of the
        # unpadded vocabulary (32064 rows minus the 64 of padding = 32000).
        stats = config["norm_stats"][unnorm_key]["action"]
        self.low, self.high = np.array(stats["q01"]), np.array(stats["q99"])
        self.action_dim = len(self.low)  # 7: one token per dimension
        self.mask = np.array(stats.get("mask", np.ones_like(self.low, dtype=bool)))
        edges = np.linspace(-1, 1, config["n_action_bins"])
        self.bin_centers = (edges[:-1] + edges[1:]) / 2.0
        self.vocab_size = config["text_config"]["vocab_size"] - config["pad_to_multiple_of"]

    def __call__(self, frame: Image.Image, task: str) -> tuple[np.ndarray, list[int]]:
        """The action, float64 [7], and the 7 tokens it was decoded from."""
        tokens = self.decode(frame, self.prompt_ids(task)).argmax(dim=-1).tolist()
        return self.unnormalize(tokens), tokens

    def decode(self, frame: Image.Image, ids: list[int]) -> torch.Tensor:
        """The logits of the greedy steps after the prompt ids, one per action dimension."""
        return self.model.decode(self.pixels(frame), self.tensor(ids), self.action_dim)

    def prefill(self, frame: Image.Image, ids: list[int]) -> torch.Tensor:
        """The teacher-forced logits that predict ids[1:]."""
        return self.model.prefill(self.pixels(frame), self.tensor(ids))

    def pixels(self, frame: Image.Image) -> torch.Tensor:
        """[1, 6, H, W]: per tower, PIL resize, /255 and normalise in float32;
        stacked on channels, then cast to the model's dtype."""
        frame = frame.convert("RGB")
        planes = []
        for resize, mean, std in zip(self.resize, self.mean, self.std):
            height, width = resize["size"]
            resized = frame.resize((width, height), Image.Resampling(resize["interpolation"]))
            planes.append((np.asarray(resized, dtype=np.float32) / 255 - mean) / std)
        pixel_values = torch.from_numpy(np.concatenate(planes, axis=2)).permute(2, 0, 1)[None]
        return pixel_values.to(self.device, self.dtype)

    def prompt_ids(self, task: str) -> list[int]:
        """<s> first (the tokenizer adds it), EMPTY_TOKEN last."""
        ids = self.tokenizer(PROMPT.format(task=task.lower()))["input_ids"]
        if ids[-1] != EMPTY_TOKEN:
            ids.append(EMPTY_TOKEN)
        return ids

    def tensor(self, ids: list[int]) -> torch.Tensor:
        return torch.tensor([ids], device=self.device)

    def unnormalize(self, tokens: list[int]) -> np.ndarray:
        """predict_action's arithmetic: token -> bin centre in [-1, 1] -> q01..q99
        for the dimensions the mask names; the rest (bridge_orig: the gripper)
        stay in [-1, 1]."""
        bins = np.clip(self.vocab_size - np.array(tokens) - 1, 0, len(self.bin_centers) - 1)
        normalized = self.bin_centers[bins]
        return np.where(self.mask, 0.5 * (normalized + 1) * (self.high - self.low) + self.low, normalized)
