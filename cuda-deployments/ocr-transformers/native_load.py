"""Native weight loader for DeepSeek-OCR-2 on transformers 5.20.2.

load_native(dir, cfg) returns (model, report). The config comes from
native_config.build_config; from_pretrained receives it with the explicit
class, so config.json's model_type "deepseek_vl_v2" is never parsed and the
native deepseek_ocr2 architecture is used.

Transformers 5.20.2 registers a deepseek_ocr2 conversion mapping
(conversion_mapping.py) whose source keys are this checkpoint's own names
(model.sam_model.*, model.qwen2_model.*, model.projector.*, model.layers.*,
lm_head, view_seperator). Applied to the checkpoint it is 2577 pure renames
plus two packing operations over each MoE layer's 64 experts: gate_proj and
up_proj stacked into gate_up_proj, the expert list merged on dim 0. Both
operations only rearrange values; none does arithmetic. The checkpoint is
never written to disk in any other layout.

The weights load in bfloat16, the reference dtype.

Gates before any scoring, all recorded in the report:
* strict load: 0 missing keys, 0 unexpected keys;
* bitwise: every checkpoint tensor is compared with the loaded model through
  an independent rename table, verified against the registered mapping
  during recon (595 direct renames, 22 packed expert tensors, 617 native
  tensors over 2707 checkpoint tensors). A single mismatch fails the load;
* provenance: file sha256s, hub revision, model class, transformers version.

Gate (run on the box, CPU, no serving):

    python native_load.py --dir /path/to/DeepSeek-OCR-2
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
import sys
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from transformers import DeepseekOcr2ForConditionalGeneration

REVISION = "aaa02f3811945a91062062994c5c4a3f4c0af2b0"

# The independent rename oracle: source key -> native key, verified key by
# key against the registered deepseek_ocr2 mapping and the meta skeleton.
_RENAME = [
    (r"sam_model\.blocks\.(\d+)\.norm1\.", r"vision_tower.sam_encoder.layers.\1.layer_norm1."),
    (r"sam_model\.blocks\.(\d+)\.norm2\.", r"vision_tower.sam_encoder.layers.\1.layer_norm2."),
    (r"sam_model\.blocks\.(\d+)\.attn\.", r"vision_tower.sam_encoder.layers.\1.attn."),
    (r"sam_model\.blocks\.(\d+)\.mlp\.", r"vision_tower.sam_encoder.layers.\1.mlp."),
    (r"sam_model\.patch_embed\.proj\.", r"vision_tower.sam_encoder.patch_embed.projection."),
    (r"sam_model\.pos_embed", r"vision_tower.sam_encoder.pos_embed"),
    (r"sam_model\.neck\.0\.", r"vision_tower.sam_encoder.neck.conv1."),
    (r"sam_model\.neck\.1\.", r"vision_tower.sam_encoder.neck.layer_norm1."),
    (r"sam_model\.neck\.2\.", r"vision_tower.sam_encoder.neck.conv2."),
    (r"sam_model\.neck\.3\.", r"vision_tower.sam_encoder.neck.layer_norm2."),
    (r"sam_model\.net_2\.", r"vision_tower.sam_encoder.proj.conv1."),
    (r"sam_model\.net_3\.", r"vision_tower.sam_encoder.proj.conv2."),
    (r"qwen2_model\.model\.model\.layers\.", r"vision_tower.vision_encoder.layers."),
    (r"qwen2_model\.model\.model\.norm\.", r"vision_tower.vision_encoder.norm."),
    (r"qwen2_model\.query_768\.", r"vision_tower.query_768_resolution."),
    (r"qwen2_model\.query_1024\.", r"vision_tower.query_1024_resolution."),
    (r"projector\.layers\.", r"multi_modal_projector."),
    (r"view_seperator", r"view_separator"),
    (r"(^|model\.)embed_tokens\.", r"\1language_model.embed_tokens."),
    (r"(^|model\.)layers\.", r"\1language_model.layers."),
    (r"(^|model\.)norm\.", r"\1language_model.norm."),
]

_EXPERT_KEY = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


def _renamed(key: str) -> str:
    for pattern, repl in _RENAME:
        key = re.sub(pattern, repl, key)
    return key


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _revision(dir) -> str:
    trees = glob.glob(str(Path(dir) / ".cache" / "huggingface" / "trees" / "*.json"))
    return Path(trees[0]).stem if trees else ""


def _safetensors_path(dir) -> Path:
    files = sorted(Path(dir).glob("*.safetensors"))
    if len(files) != 1:
        raise RuntimeError(f"expected exactly one safetensors file in {dir}, found {len(files)}")
    return files[0]


def load_native(dir, cfg):
    """Build the native model from the original checkpoint dir; never touch
    the files on it. cfg carries dtype and _attn_implementation (the two
    fields from_pretrained would take from its own arguments)."""
    path = Path(dir)
    kwargs = {"config": cfg, "use_safetensors": True, "output_loading_info": True}
    if getattr(cfg, "dtype", None) is not None:
        kwargs["dtype"] = cfg.dtype
    if getattr(cfg, "_attn_implementation", None) is not None:
        kwargs["attn_implementation"] = cfg._attn_implementation
    model, loading = DeepseekOcr2ForConditionalGeneration.from_pretrained(str(path), **kwargs)

    report = {
        "model_class": type(model).__name__,
        "transformers_version": transformers.__version__,
        "revision": _revision(path) or REVISION,
        "file_hashes": {
            "config.json": _sha256(path / "config.json"),
            _safetensors_path(path).name: _sha256(_safetensors_path(path)),
        },
        "missing_keys": sorted(loading["missing_keys"]),
        "unexpected_keys": sorted(loading["unexpected_keys"]),
        "mismatched_keys": sorted(loading.get("mismatched_keys", [])),
        "error_msgs": list(loading.get("error_msgs", [])),
        "dtype": str(kwargs.get("dtype", "default")).removeprefix("torch."),
    }
    _gate_strict_load(report)
    _gate_bitwise(model, path, report)
    return model, report


def _gate_strict_load(report) -> None:
    problems = report["missing_keys"] + report["unexpected_keys"] + report["mismatched_keys"]
    if problems or report["error_msgs"]:
        raise RuntimeError(
            "strict-load gate failed: "
            f"missing={report['missing_keys']} unexpected={report['unexpected_keys']} "
            f"mismatched={report['mismatched_keys']} errors={report['error_msgs']}"
        )


def _gate_bitwise(model, path: Path, report) -> None:
    """Every checkpoint tensor, compared with the loaded model.

    Direct keys compare through the rename table. Expert keys collect per
    layer and compare as the packed tensors the native model holds: gate_up
    is gate stacked on up per expert (dim 0), the experts merged on dim 0.
    """
    state = model.state_dict()
    checked = mismatched = 0
    experts: dict = {}
    with safe_open(_safetensors_path(path), framework="pt", device="cpu") as checkpoint:
        keys = list(checkpoint.keys())
        for key in keys:
            native = _renamed(key)
            expert = _EXPERT_KEY.match(native)
            if expert:
                layer, index, kind = expert.group(1), int(expert.group(2)), expert.group(3)
                experts.setdefault((layer, kind), {})[index] = checkpoint.get_tensor(key)
                continue
            if native not in state:
                raise RuntimeError(f"bitwise gate: {key} renamed to {native}, absent from the model")
            if not torch.equal(checkpoint.get_tensor(key), state[native]):
                print(f"BITWISE MISMATCH: {key} -> {native}")
                mismatched += 1
            checked += 1
        for (layer, kind) in sorted(experts):
            if kind == "up_proj":
                continue  # packed together with gate_proj
            if kind == "down_proj":
                target = f"model.language_model.layers.{layer}.mlp.experts.down_proj"
                expected = torch.stack([experts[(layer, kind)][i] for i in sorted(experts[(layer, kind)])])
            else:
                gate, up = experts[(layer, "gate_proj")], experts[(layer, "up_proj")]
                target = f"model.language_model.layers.{layer}.mlp.experts.gate_up_proj"
                expected = torch.stack([torch.cat([gate[i], up[i]], dim=0) for i in sorted(gate)])
            if target not in state:
                raise RuntimeError(f"bitwise gate: packed {target} absent from the model")
            if not torch.equal(expected, state[target]):
                print(f"BITWISE MISMATCH: {target} (packed)")
                mismatched += 1
            checked += 1
    report["checkpoint_tensor_count"] = len(keys)
    report["native_tensor_count"] = len(state)
    report["bitwise_checked"] = checked
    report["bitwise_mismatched"] = mismatched
    if mismatched:
        raise RuntimeError(f"bitwise gate failed: {mismatched} of {checked} tensors differ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load gate for the native DeepSeek-OCR-2 loader.")
    parser.add_argument("--dir", required=True, help="checkpoint dir holding config.json and the safetensors")
    args = parser.parse_args()

    from native_config import build_config

    cfg = build_config(args.dir)
    cfg.dtype = torch.bfloat16
    cfg._attn_implementation = "sdpa"
    model, report = load_native(args.dir, cfg)
    print(json.dumps(report, indent=1))
    print("LOAD GATE PASSED")
