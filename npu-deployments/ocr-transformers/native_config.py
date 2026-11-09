"""Build a native DeepSeek-OCR-2 config from an original DeepSeek-VL2 checkpoint.

The checkpoint's config.json uses the DeepSeek-VL2 schema (model_type
"deepseek_vl_v2"). Transformers 5.x cannot load it: from_pretrained re-reads
config.json and fails on the model type. This module constructs the native
config classes directly from the original file.

Two values in the original file are decoys. They disagree with the weights.
The comments at the two overrides below tell that story.

Gate (run on the box):

    python native_config.py --dir <checkpoint_dir> --community <community_config.json>

The community config is a reference for the diff only. It is never a source
for the build.
"""

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from transformers import (
    DeepseekOcr2Config,
    DeepseekOcr2SamVisionConfig,
    DeepseekOcr2TextConfig,
    DeepseekOcr2VisionConfig,
    DeepseekOcr2VisionEncoderConfig,
)


def build_config(dir) -> DeepseekOcr2Config:
    """Build the native config from `<dir>/config.json`.

    `dir` holds the original DeepSeek-VL2-schema config.
    """
    raw = json.loads((Path(dir) / "config.json").read_text())
    _assert_original_schema(raw)
    _assert_language_copies_agree(raw)
    return DeepseekOcr2Config(
        vision_config=DeepseekOcr2VisionConfig(
            sam_config=_build_sam_config(raw["vision_config"]),
            encoder_config=_build_encoder_config(raw["vision_config"]),
        ),
        text_config=_build_text_config(raw["language_config"]),
        image_token_id=128815,
        tie_word_embeddings=False,
    )


def _assert_original_schema(raw):
    if raw.get("model_type") != "deepseek_vl_v2":
        raise ValueError(
            "expected the original DeepSeek-VL2 schema (model_type deepseek_vl_v2), "
            f"got {raw.get('model_type')!r}"
        )


def _assert_language_copies_agree(raw):
    # The original file stores the language settings twice: nested under
    # language_config, and lifted to the top level. The build reads only the
    # nested block. Every shared field must agree in both places.
    # architectures, auto_map and torch_dtype are exempt: at the top level
    # they describe the whole VL model, nested they describe the language
    # model. Today the two copies share 25 fields and agree on all of them.
    nested = raw["language_config"]
    exempt = {"architectures", "auto_map", "torch_dtype"}
    shared = [key for key in nested if key not in exempt and key in raw]
    clashes = [key for key in shared if nested[key] != raw[key]]
    if clashes:
        details = "; ".join(
            f"{key}: nested={nested[key]!r} top-level={raw[key]!r}" for key in clashes
        )
        raise ValueError(f"language_config disagrees with its top-level copy: {details}")


def _build_text_config(lang) -> DeepseekOcr2TextConfig:
    # The original has no mlp_layer_types key. first_k_dense_replace says how
    # many layers keep a dense MLP. The remaining layers are sparse.
    dense_layers = lang["first_k_dense_replace"]
    total_layers = lang["num_hidden_layers"]
    mlp_layer_types = ["dense"] * dense_layers + ["sparse"] * (total_layers - dense_layers)
    # The original has no rope_scaling block. The language model runs plain
    # rope with theta 10000.0.
    return DeepseekOcr2TextConfig(
        vocab_size=lang["vocab_size"],
        hidden_size=lang["hidden_size"],
        intermediate_size=lang["intermediate_size"],
        num_hidden_layers=total_layers,
        num_attention_heads=lang["num_attention_heads"],
        num_key_value_heads=lang["num_key_value_heads"],
        moe_intermediate_size=lang["moe_intermediate_size"],
        n_routed_experts=lang["n_routed_experts"],
        n_shared_experts=lang["n_shared_experts"],
        num_experts_per_tok=lang["num_experts_per_tok"],
        n_group=lang["n_group"],
        topk_group=lang["topk_group"],
        topk_method=lang["topk_method"],
        max_position_embeddings=lang["max_position_embeddings"],
        bos_token_id=lang["bos_token_id"],
        eos_token_id=lang["eos_token_id"],
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
        mlp_layer_types=mlp_layer_types,
    )


def _build_sam_config(vision) -> DeepseekOcr2SamVisionConfig:
    sam = vision["width"]["sam_vit_b"]
    # Decoy 1: mlp_ratio. The file says 3.7362. int(768 * 3.7362) is 2869.
    # The weights are (3072, 768): sam_model.blocks.*.mlp.lin1.weight.
    # An explicit mlp_dim wins over mlp_ratio, so the file value stays but
    # never applies.
    # Decoy 2: downsample_channels. The file says [512, 1024]. The weights'
    # final conv is (896, 512, 3, 3): sam_model.net_3.weight. Force the
    # weight truth.
    return DeepseekOcr2SamVisionConfig(
        hidden_size=sam["width"],
        num_hidden_layers=sam["layers"],
        num_attention_heads=sam["heads"],
        global_attn_indexes=sam["global_attn_indexes"],
        image_size=vision["image_size"],
        mlp_ratio=vision["mlp_ratio"],
        mlp_dim=3072,
        downsample_channels=[512, 896],
        patch_size=16,
        output_channels=256,
        hidden_act="gelu",
        layer_norm_eps=1e-6,
        qkv_bias=True,
        use_abs_pos=True,
        use_rel_pos=True,
        window_size=14,
        initializer_range=1e-10,
    )


def _build_encoder_config(vision) -> DeepseekOcr2VisionEncoderConfig:
    # The original file carries no encoder settings. All values below except
    # hidden_size are the CustomQwen2Decoder defaults from the original
    # DeepSeek-OCR-2 code (deepencoderv2.py). hidden_size comes from the
    # file's width block.
    return DeepseekOcr2VisionEncoderConfig(
        hidden_size=vision["width"]["qwen2-0-5b"]["dim"],
        num_hidden_layers=24,
        num_attention_heads=14,
        num_key_value_heads=2,
        intermediate_size=4864,
        max_position_embeddings=131072,
        rms_norm_eps=1e-6,
        vocab_size=151936,
        hidden_act="silu",
        rope_parameters={"rope_theta": 1000000.0, "rope_type": "default"},
    )


_ABSENT = object()

# Known, inert deviations of the community reference from this build. The
# gate tolerates exactly these. Every other disagreement fails the gate.
_WHITELIST = {
    ("sam_config", "mlp_ratio"): (
        "community 4.0 vs original 3.7362; inert: the native config and model "
        "read mlp_dim, and mlp_dim is 3072 on both sides"
    ),
    ("encoder_config", "max_position_embeddings"): (
        "community kept the native class default 32768; the original encoder "
        "code runs 131072; the encoder sees at most 4096 patch positions "
        "(1024 / 16 = 64, 64 * 64), so both values give the same rope for "
        "every position in use"
    ),
}


def _plain(value):
    """Reduce a config value to plain data for comparison."""
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {field.name: _plain(getattr(value, field.name)) for field in dataclasses.fields(value)}
    if hasattr(value, "model_dump"):
        return _plain(value.model_dump())
    return value


def _matches(ours, expected):
    ours = _plain(ours)
    if isinstance(expected, dict) and isinstance(ours, dict):
        # The reference may list fewer keys than the config carries.
        return all(key in ours and ours[key] == value for key, value in expected.items())
    return ours == expected


def _field_names(cfg):
    names = []
    for cls in type(cfg).__mro__:
        for name in getattr(cls, "__annotations__", {}):
            if name not in names:
                names.append(name)
    return names


def _diff_against_community(label, built, community):
    """Diff one subconfig against the community reference.

    A field is checked when this build moves it away from the native class
    default. A field left at its default carries no decision. Community keys
    that are null or absent carry no value, so they are skipped.
    """
    default = type(built)()
    failures = []
    for name in _field_names(built):
        ours = getattr(built, name, _ABSENT)
        base = getattr(default, name, _ABSENT)
        if ours is _ABSENT or base is _ABSENT:
            continue
        if _plain(ours) == _plain(base):
            continue
        if name not in community or community[name] is None:
            print(f"{label}.{name}: community has no value; skipped")
            continue
        expected = community[name]
        if (label, name) in _WHITELIST:
            status = f"WHITELISTED: {_WHITELIST[(label, name)]}"
        elif _matches(ours, expected):
            status = "OK"
        else:
            status = "MISMATCH"
            failures.append(f"{label}.{name}")
        print(f"{label}.{name}: expected={expected!r} actual={_plain(ours)!r} {status}")
    return failures


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the native DeepSeek-OCR-2 config and diff it against the community reference."
    )
    parser.add_argument("--dir", required=True, help="checkpoint dir holding the original config.json")
    parser.add_argument("--community", required=True, help="community reference config.json")
    args = parser.parse_args()

    config = build_config(args.dir)
    community = json.loads(Path(args.community).read_text())

    failures = []
    failures += _diff_against_community("text_config", config.text_config, community["text_config"])
    failures += _diff_against_community(
        "sam_config",
        config.vision_config.sam_config,
        community["vision_config"]["sam_config"],
    )
    failures += _diff_against_community(
        "encoder_config",
        config.vision_config.encoder_config,
        community["vision_config"]["encoder_config"],
    )

    if failures:
        print(f"GATE FAILED: {len(failures)} mismatching field(s): {', '.join(failures)}")
        sys.exit(1)
    print("GATE PASSED")
