"""DeepSeek-OCR-2 top-level model, as a transformers 5.22.1 PreTrainedModel.

Structure and parameter names mirror the checkpoint's DeepseekOCR2ForCausalLM
exactly (model.embed_tokens, model.layers.N..., model.sam_model,
model.qwen2_model, model.projector.layers, model.view_seperator, lm_head), so
`Ocr2ForCausalLM.from_pretrained(<model dir>, ...)` loads
model-00001-of-000001.safetensors with no key remapping and the weights are
read from --model only.

The forward is split the way the original DeepseekOCR2Model.forward ran it:

* embed_tokens lookup, then the vision features masked_scatter_()-ed into the
  image-token positions (same in-place op the checkpoint used);
* the language stack (ocr2.language) over the embeds;
* lm_head, logits cast to float as the checkpoint's forward did.
"""
import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel

from .language import DeepseekV2DecoderLayer, RopeTable, build_causal_mask
from .qwen2_encoder import Qwen2Decoder2Encoder
from .sam import build_sam_vit_b

LANGUAGE_DEFAULTS = {
    "vocab_size": 129280,
    "hidden_size": 1280,
    "intermediate_size": 6848,
    "moe_intermediate_size": 896,
    "num_hidden_layers": 12,
    "num_attention_heads": 10,
    "num_key_value_heads": 10,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "first_k_dense_replace": 1,
    "routed_scaling_factor": 1.0,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "max_position_embeddings": 8192,
    "attention_dropout": 0.0,
}


class Ocr2Config(PretrainedConfig):
    model_type = "DeepseekOCR2"

    def __init__(self, **kwargs):
        self.language = {}
        for key, default in LANGUAGE_DEFAULTS.items():
            value = kwargs.get(key, default)
            setattr(self, key, value)
            self.language[key] = value
        super().__init__(**kwargs)

    def language_params(self):
        return self.language


class Projector(nn.Module):
    """The checkpoint's MlpProjector for projector_type="linear": one Linear.

    Kept under attribute `layers` so the checkpoint keys
    model.projector.layers.weight / .bias load unchanged.
    """

    def __init__(self, input_dim, n_embed):
        super().__init__()
        self.layers = nn.Linear(input_dim, n_embed)

    def forward(self, x):
        return self.layers(x)


class Ocr2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        params = config.language_params()
        self.embed_tokens = nn.Embedding(params["vocab_size"], params["hidden_size"])
        self.layers = nn.ModuleList(
            DeepseekV2DecoderLayer(layer_idx, params)
            for layer_idx in range(params["num_hidden_layers"])
        )
        from .language import DeepseekV2RMSNorm
        self.norm = DeepseekV2RMSNorm(params["hidden_size"], eps=params["rms_norm_eps"])
        self.sam_model = build_sam_vit_b()
        self.qwen2_model = Qwen2Decoder2Encoder(
            decoder_layer=24, hidden_dimension=896, num_attention_heads=14,
            num_key_value_heads=2, intermediate_size=4864,
        )
        self.projector = Projector(896, params["hidden_size"])
        self.view_seperator = nn.Parameter(torch.randn(params["hidden_size"]))
        self.rope = None  # built lazily on the device the model lands on

    def _rope(self, device):
        params = self._language_params
        if self.rope is None or self.rope.inv_freq.device != device:
            self.rope = RopeTable(
                params["hidden_size"] // params["num_attention_heads"],
                params["rope_theta"], params["max_position_embeddings"] + 1024, device,
            )
        return self.rope

    def refresh_buffers(self):
        """Rebuild the non-persistent inv_freq of the qwen2 encoder.

        transformers' from_pretrained materializes modules on the meta device
        and only fills checkpoint keys there, so non-persistent buffers arrive
        uninitialized; their values are constants of the architecture and are
        recomputed here exactly as __init__ built them.
        """
        inner = self.qwen2_model.model.model
        head_dim = inner.layers[0].self_attn.head_dim
        rope_theta = 1000000.0
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        inner.inv_freq = inv_freq.to(inner.layers[0].input_layernorm.weight.device)
        for layer in inner.layers:
            layer.self_attn.inv_freq = inner.inv_freq

    def language_forward(self, inputs_embeds, position_ids, mask4d, kv_cache=None):
        """The DeepseekV2Model.forward op sequence over prepared embeds."""
        params = self._language_params
        cos, sin = self._rope(inputs_embeds.device).cos_sin(position_ids, inputs_embeds.dtype)
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, mask4d, kv_cache)
        return self.norm(hidden_states)

    def embed_with_vision(self, input_ids, images_seq_mask, features):
        """embed_tokens lookup + masked_scatter_ of vision features, as the
        checkpoint's DeepseekOCR2Model.forward did (per batch row)."""
        inputs_embeds = self.embed_tokens(input_ids)
        for row, (mask_row, feats) in enumerate(zip(images_seq_mask, features)):
            inputs_embeds[row].masked_scatter_(mask_row.unsqueeze(-1), feats)
        return inputs_embeds

    def _sam_chunks(self, stack, size):
        """SAM over fixed-size chunks of a batched image stack."""
        if stack.shape[0] <= size:
            return self.sam_model(stack)
        return torch.cat([self.sam_model(stack[i:i + size]) for i in range(0, stack.shape[0], size)], dim=0)

    def encode_vision(self, image_batch):
        """The checkpoint's vision path, batched.

        image_batch: list of (patches_or_None, global_view) per image, where
        patches is [n_tiles, 3, 768, 768] (None/0 tiles for small images) and
        global_view is [1, 3, 1024, 1024].  Tiles are stacked across the batch
        into one SAM call and globals into another (the per-item math is
        unchanged: every op is row/token-local), then each image's features
        are [its tiles' rows..., its global rows..., view_seperator] exactly
        as torch.cat([local, global, sep]) ordered them.

        A batch of one keeps the exact input shapes the reference forward
        used (per-request tile count, single global view).  Larger batches
        pad the stacks to fixed buckets (extra zero rows, whose outputs are
        sliced away before any real row is touched) so the NPU kernels see a
        small set of stable shapes instead of recompiling per batch.
        Returns a list of [n_image_tokens, hidden] bf16 tensors.
        """
        device = self.view_seperator.device
        exact = len(image_batch) == 1
        all_patches = []
        tile_counts = []
        for patches, _ in image_batch:
            count = 0 if patches is None else patches.shape[0]
            if count:
                all_patches.append(patches)
            tile_counts.append(count)
        globals_list = [g for _, g in image_batch]
        if not exact:
            bucket = 1
            while bucket < len(globals_list):
                bucket *= 2
            while len(globals_list) < bucket:
                globals_list.append(torch.zeros(1, 3, 1024, 1024, dtype=globals_list[0].dtype, device=device))
        globals_stack = torch.cat(globals_list, dim=0)
        local_features = None
        total_tiles = sum(tile_counts)
        if all_patches:
            patches_stack = torch.cat(all_patches, dim=0)
            if not exact:
                pad_to = max(8, (total_tiles + 7) // 8 * 8)
                if patches_stack.shape[0] < pad_to:
                    pad = torch.zeros(pad_to - patches_stack.shape[0], *patches_stack.shape[1:],
                                      dtype=patches_stack.dtype, device=device)
                    patches_stack = torch.cat([patches_stack, pad], dim=0)
            # SAM's global-attention blocks grow quadratically with the batch,
            # so the stack is encoded in fixed-size chunks; the batch dim is
            # independent per item, so the per-image results are unchanged.
            local_features = self.projector(self.qwen2_model(self._sam_chunks(patches_stack, 8)))
        global_features = self.projector(self.qwen2_model(self._sam_chunks(globals_stack, 8)))
        n_dim = global_features.shape[-1]
        global_flat = global_features.view(-1, n_dim)
        sep = self.view_seperator[None, :]
        features = []
        offset = 0
        for row, count in enumerate(tile_counts):
            parts = []
            if count:
                lf = local_features.view(-1, n_dim)
                parts.append(lf[offset * 144: (offset + count) * 144])
                offset += count
            parts.append(global_flat[row * 256:(row + 1) * 256])
            parts.append(sep)
            features.append(torch.cat(parts, dim=0))
        return features


class Ocr2PreTrainedModel(PreTrainedModel):
    config_class = Ocr2Config
    base_model_prefix = "model"
    _no_split_modules = ["DeepseekV2DecoderLayer"]

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)


class Ocr2ForCausalLM(Ocr2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = Ocr2Model(config)
        self.model._language_params = config.language_params()
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def head(self, hidden_states):
        """lm_head then .float(), the checkpoint's forward order."""
        return self.lm_head(hidden_states).float()

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None,
                position_ids=None, images_seq_mask=None, image_features=None,
                past_kv=None, use_cache=False):
        """One forward over prepared embeds (vision features already computed
        and scattered).  The engine drives prefill/decode through this."""
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)
        bsz, q_len = inputs_embeds.shape[:2]
        past_len = 0 if not past_kv else past_kv[0][0].shape[-2]
        kv_len = past_len + q_len
        if position_ids is None:
            position_ids = torch.arange(past_len, kv_len, device=inputs_embeds.device)
            position_ids = position_ids.unsqueeze(0).expand(bsz, -1)
        mask4d = build_causal_mask(
            bsz, q_len, kv_len, inputs_embeds.dtype, inputs_embeds.device,
            past_len, pad2d=attention_mask if attention_mask is not None and attention_mask.dim() == 2 else None,
        )
        hidden = self.model.language_forward(inputs_embeds, position_ids, mask4d, past_kv if use_cache else None)
        return hidden
