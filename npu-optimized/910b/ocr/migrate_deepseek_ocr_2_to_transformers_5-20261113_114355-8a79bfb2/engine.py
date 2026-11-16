"""Batched generation engine for DeepSeek-OCR-2 on one accelerator.

One forward path for every endpoint. The engine owns the model and a single
worker thread; requests are admitted in waves and decoded together with an
explicitly masked, right-aligned KV cache:

* per image, the processor produces the padded 1024x1024 global view, the
  768x768 crop tiles (when a side exceeds 768 px) and the tile count - the
  same call for every request shape. Rows carry their own pixel tensors; the
  engine batches them through the model's own `get_image_features`, which
  takes the per-image tile counts and returns each image's feature block
  (local tiles, then the 256 global features, then the view separator) - so a
  batch with tile counts [2, 6] keeps image 0's rows from its two tiles and
  image 1's rows from its six;
* prompts are left-padded to one length and expanded over their single
  image placeholder, so every row's tokens sit right-aligned in the cache
  window and an explicit attention mask states where each row's real tokens
  are; position ids are per row and start at 0 for its first real token;
* decode appends one token per row per step; finished rows are dropped from
  the batch by index-selecting the cache's rows;
* sampling applies, per row and exactly as transformers does: the
  min_new_tokens EOS ban, presence/frequency/repetition penalties (a no-op
  when neutral), the no-repeat 30-gram rule over the last 90 tokens with the
  two table tokens whitelisted, temperature, top_k, top_p, min_p, then a
  seeded `torch.multinomial` per row - so a request's draws depend only on
  its own seed, never on what else is in the batch. The distributions the
  score route stores are the raw logits before all of that, exactly as
  `output_logits=True` would have produced;
* rows produce their tokens through callbacks, from the engine thread.
"""

from __future__ import annotations

import queue
import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

# DeepSeek's own anti-repetition rule for OCR outputs (their repo's run
# scripts and sglang's native defaults agree on the values): a token that
# would complete an n-gram that already occurred within the lookback window
# is banned from sampling, except the table tokens, so structure may repeat.
NGRAM_SIZE = 30
NGRAM_WINDOW = 90
NGRAM_WHITELIST = (128821, 128822)  # <td>, </td>

SAMPLING_FIELDS = (
    "temperature",
    "top_p",
    "top_k",
    "min_p",
    "repetition_penalty",
    "presence_penalty",
    "frequency_penalty",
)

NEUTRAL_SAMPLING = {
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
}

NEG_INF = float("-inf")


def install_dense_experts():
    """One MoE compute path for every endpoint and batch size.

    Replaces the native experts module's per-expert Python loop - which
    launches hundreds of small kernels and synchronizes on a `nonzero()` per
    layer - with two batched GEMMs over all experts, weighting each token's
    unselected experts by exactly 0.0:

    * every expert's gate/up/down computation is the same linear algebra the
      loop performs, on the same bf16 inputs with fp32-accumulating GEMM
      kernels;
    * the token's routed output is the sum over its selected experts with
      the same router weights - accumulated in fp32 in one deterministic
      reduction instead of one bf16 index_add per expert, so a sum differs
      from the loop only in final-bit rounding;
    * both endpoints and every batch size run this one path.

    Idempotent, and applied at engine construction so the model that serves
    is the model that was validated.
    """
    from transformers.models.deepseek_ocr2.modeling_deepseek_ocr2 import (
        DeepseekOcr2TextExperts,
    )

    if getattr(DeepseekOcr2TextExperts, "_kot_dense_installed", False):
        return

    def dense_forward(self, hidden_states, top_k_index, top_k_weights):
        tokens = hidden_states.shape[0]
        final = torch.empty_like(hidden_states)
        # A long prefill would materialize every expert's output for every
        # token at once (tens of GB); the expert math is per token, so big
        # prefills stream through in token chunks instead.
        chunk = tokens if tokens <= 1024 else 1024
        for start in range(0, tokens, chunk):
            stop = start + chunk
            piece = hidden_states[start:stop]
            width = piece.shape[0]
            repeated = piece.unsqueeze(0).expand(
                self.num_experts, width, self.hidden_dim)
            gate_up = torch.bmm(repeated, self.gate_up_proj.transpose(1, 2))
            gate, up = gate_up.chunk(2, dim=-1)
            routed = torch.bmm(self.act_fn(gate) * up, self.down_proj.transpose(1, 2))
            weights = torch.zeros(
                width, self.num_experts, dtype=torch.float32, device=routed.device)
            weights.scatter_(1, top_k_index[start:stop], top_k_weights[start:stop])
            final[start:stop] = (
                routed.to(torch.float32) * weights.t().unsqueeze(-1)).sum(dim=0
            ).to(hidden_states.dtype)
        return final

    DeepseekOcr2TextExperts.forward = dense_forward
    DeepseekOcr2TextExperts._kot_dense_installed = True


class KotStaticLayer:
    """One transformer layer's KV cache: a preallocated buffer mutated in
    place, whose update returns only the filled view.

    `DynamicCache` re-concatenates the whole cache on every step - hundreds
    of megabytes copied per token at batch 16 - so the wave's cache window is
    preallocated once and written at the running position. The filled length
    is tracked as a plain Python int (the kv length is a shape, never a
    device value), so no update or mask build ever synchronizes.
    """

    is_sliding = False
    is_croppable = False
    is_compileable = False
    is_initialized = False

    def __init__(self, max_cache_len):
        self.max_cache_len = int(max_cache_len)
        self.keys = None
        self.values = None
        self.seen = 0
        self.dtype = None
        self.device = None

    def allocate(self, batch, heads, head_dim, dtype, device):
        """Reserve the layer's buffer outside any inference-mode region, so
        in-place updates stay legal wherever the forward runs."""
        self.dtype, self.device = dtype, device
        self.keys = torch.zeros(
            (batch, heads, self.max_cache_len, head_dim), dtype=dtype, device=device)
        self.values = torch.zeros(
            (batch, heads, self.max_cache_len, head_dim), dtype=dtype, device=device)
        self.is_initialized = True

    def lazy_initialization(self, key_states, value_states):
        batch, heads, _, head_dim = key_states.shape
        self.allocate(batch, heads, head_dim, key_states.dtype, key_states.device)

    def update(self, key_states, value_states, *args, **kwargs):
        if not self.is_initialized:
            self.lazy_initialization(key_states, value_states)
        kv_length = key_states.shape[-2]
        end = self.seen + kv_length
        position = torch.arange(self.seen, end, device=self.device)
        self.keys.index_copy_(2, position, key_states)
        self.values.index_copy_(2, position, value_states)
        self.seen = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]

    def get_seq_length(self):
        return self.seen if self.is_initialized else 0

    def get_max_length(self):
        return self.max_cache_len

    def get_mask_sizes(self, query_length):
        return self.seen + query_length, 0

    def batch_select_indices(self, indices):
        if self.is_initialized and self.seen > 0:
            self.keys = self.keys[indices, ...]
            self.values = self.values[indices, ...]

    def reset(self):
        self.keys = self.values = None
        self.is_initialized = False
        self.seen = 0


class KotStaticCache:
    """The wave's KV cache: `KotStaticLayer` per decoder layer, through the
    interface transformers' attention and mask utilities call on a cache."""

    def __init__(self, num_layers, max_cache_len):
        self.layers = [KotStaticLayer(max_cache_len) for _ in range(num_layers)]

    def allocate(self, batch, heads, head_dim, dtype, device):
        for layer in self.layers:
            layer.allocate(batch, heads, head_dim, dtype, device)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        return self.layers[layer_idx].update(key_states, value_states)

    def get_seq_length(self, layer_idx=0):
        return self.layers[layer_idx].get_seq_length()

    def get_max_length(self):
        return self.layers[0].get_max_length()

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self.layers[layer_idx].get_mask_sizes(query_length)

    def batch_select_indices(self, indices):
        for layer in self.layers:
            layer.batch_select_indices(indices)

    def get_query_offset(self, layer_idx=0):
        """The decode query sits after everything seen - the causal mask's
        q_offset is exactly the cache length."""
        return self.get_seq_length(layer_idx)


def resolve_sampling(sampling: dict | None) -> dict:
    """The request's sampling record, every field applied exactly as given.
    Missing fields fall back to their neutral values."""
    sampling = sampling or {}
    unknown = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unknown:
        raise ValueError("unsupported sampling field(s): " + ", ".join(unknown))
    resolved = {}
    for field in SAMPLING_FIELDS:
        if field in sampling:
            value = sampling[field]
            resolved[field] = int(value) if field == "top_k" else float(value)
        else:
            resolved[field] = NEUTRAL_SAMPLING[field]
    return resolved


class NoRepeatNGramRule:
    """DeepSeek's rule, in the same shape the reference server applies it."""

    def __init__(self, size=NGRAM_SIZE, window=NGRAM_WINDOW, whitelist=NGRAM_WHITELIST):
        self.size = size
        self.window = window
        self.prefix_len = size - 1
        self.whitelist = frozenset(whitelist)

    def banned(self, ids: np.ndarray):
        """Token ids that would complete an already-seen n-gram."""
        length = ids.shape[0]
        if length < self.size:
            return ()
        start = max(0, length - self.window)
        end = length - self.size + 1
        if end <= start:
            return ()
        prefix = ids[length - self.prefix_len:]
        windows = np.lib.stride_tricks.sliding_window_view(ids, self.prefix_len)[start:end]
        matched = (windows == prefix).all(axis=1)
        if not matched.any():
            return ()
        banned = ids[np.nonzero(matched)[0] + start + self.prefix_len]
        return tuple(int(t) for t in banned if int(t) not in self.whitelist)


class Row:
    """One sequence in a wave. All ids here are real (no padding)."""

    __slots__ = (
        "ids", "raw_ids", "generated", "generator", "sampling", "max_new",
        "min_new", "allow_eos", "emit", "capture", "pixel_values",
        "pixel_values_local", "tiles", "image_rows", "prompt_len", "done",
        "ngram", "on_done",
    )

    def __init__(self):
        self.ids = None            # list[int]: the expanded prompt, then generated ids
        self.generated = []         # list[int]
        self.generator = None      # torch.Generator on the engine device
        self.sampling = dict(NEUTRAL_SAMPLING)
        self.max_new = 16
        self.min_new = 0
        self.allow_eos = True
        self.emit = None            # callable(token_id) or None
        self.capture = None        # callable(fp32 raw logits [vocab]) or None
        self.pixel_values = None
        self.pixel_values_local = None
        self.tiles = 0
        self.image_rows = 0        # feature rows the image expands to
        self.prompt_len = 0
        self.done = False
        self.ngram = NoRepeatNGramRule()
        self.on_done = None


class EngineRequest:
    """What the HTTP layer submits: one row's worth of generation, or a
    teacher-forced prefill over a stored sequence."""

    __slots__ = ("kind", "row", "n_rows", "prefill_ids", "prefill_features",
                 "logits_sink", "event", "error")

    def __init__(self, kind: str):
        self.kind = kind           # "decode" | "prefill"
        self.row = None             # decode: the Row
        self.n_rows = 0             # prefill: rows to store
        self.prefill_ids = None     # prefill: template ids + transcript, verbatim
        self.prefill_features = None  # prefill: (pixel_values, local, tiles)
        self.logits_sink = None    # prefill: callable(chunk_fp32_tensor) per write chunk
        self.event = threading.Event()
        self.error = None

    def finish(self, error=None):
        self.error = error
        self.event.set()




class DecodeGraph:
    """A replayable NPU graph for one decode-step shape.

    The host-side cost of a step - two thousand dispatcher calls through the
    eager model - is the decode bottleneck on this card, so each (batch,
    window) shape is captured once as an NPU graph and replayed per step:

    * the KV cache is the transformers StaticCache: preallocated buffers,
      written in place at a position that comes from a device-side counter,
      so the captured graph keeps writing at the advancing slot;
    * the attention mask is a prebuilt 4D bool buffer (the model returns an
      already-4D mask as-is, which skips the mask machinery in the graph);
      one column is switched on per step, outside the graph;
    * input ids and position ids are static buffers written per step.

    The same math runs as eager: the captured callable is the model's own
    forward, on the same weights, reading the same buffers the eager path
    would build.
    """

    def __init__(self, engine, batch, window):
        import torch as _torch
        from transformers import StaticCache
        self.engine = engine
        self.batch = batch
        self.window = window
        device = engine.device
        self.input_ids = _torch.zeros((batch, 1), dtype=_torch.long, device=device)
        self.position_ids = _torch.zeros((batch, 1), dtype=_torch.long, device=device)
        self.mask = _torch.zeros((batch, 1, 1, window), dtype=_torch.bool, device=device)
        self.cache = StaticCache(
            config=engine.model.config.text_config, max_cache_len=window)

        def forward():
            with _torch.no_grad():
                out = engine.model(
                    input_ids=self.input_ids,
                    attention_mask=self.mask,
                    position_ids=self.position_ids,
                    past_key_values=self.cache,
                    use_cache=True,
                )
            return out.logits[:, -1, :]

        # Warm up once outside capture (kernel selection and allocator), then
        # reset the cache counter so the captured graph starts from slot 0.
        forward()
        _torch.npu.synchronize()
        for layer in self.cache.layers:
            layer.cumulative_length.zero_()
        self.graph = _torch.npu.NPUGraph()
        with _torch.npu.graph(self.graph):
            self.logits = forward()
        _torch.npu.synchronize()

    def reset(self):
        for layer in self.cache.layers:
            layer.cumulative_length.zero_()

    def replay(self):
        self.graph.replay()
        return self.logits


class GenerationEngine:
    def __init__(self, model, device, dtype, *,
                 eos_token_id, pad_token_id=None, max_batch=16,
                 vision_images=4, vision_tiles=24, batch_window=0.02,
                 compile_model=False, use_graphs=True):
        install_dense_experts()
        self.model = model
        self.device = device
        self.dtype = dtype
        if compile_model:
            # One compiled callable for the language-model forward (decode and
            # prefill both trace through it; the vision tower is exercised
            # through its own eager call, which batching already amortizes).
            self.model = torch.compile(model, backend="aot_eager", dynamic=True)
        self.eos_token_id = int(eos_token_id)
        self.pad_token_id = int(pad_token_id if pad_token_id is not None else eos_token_id)
        self.max_batch = max_batch
        self.vision_images = vision_images
        self.vision_tiles = vision_tiles
        self.batch_window = batch_window
        self.use_graphs = bool(use_graphs) and str(device).startswith("npu")
        self.graph_batches = sorted({1, 2, 4, 8, max_batch, max_batch // 2, max_batch // 4} | {1, 2})
        self.graph_batches = [b for b in self.graph_batches if 0 < b <= max_batch]
        self.graph_cache = 8
        self.graph_sync = bool(int(__import__("os").environ.get("KOT_GRAPH_SYNC", 0)))
        self._graphs = {}
        self.vocab_size = int(model.config.text_config.vocab_size)
        self.num_layers = int(model.config.text_config.num_hidden_layers)
        self._requests = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._serve, name="engine", daemon=True)
        self._running = True
        self._thread.start()

    # ------------------------------------------------------------------ #
    # lifecycle                                                          #
    # ------------------------------------------------------------------ #

    def stop(self):
        self._running = False
        try:
            self._requests.put_nowait(None)
        except Exception:
            pass

    def submit(self, request: EngineRequest):
        self._requests.put(request)

    # ------------------------------------------------------------------ #
    # engine thread                                                      #
    # ------------------------------------------------------------------ #

    def _serve(self):
        while self._running:
            first = self._requests.get()
            if first is None:
                break
            wave = [first]
            deadline = time.monotonic() + self.batch_window
            while len(wave) < self.max_batch:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                try:
                    wave.append(self._requests.get(timeout=remaining))
                except queue.Empty:
                    break
            decodes = [request for request in wave if request.kind == "decode"]
            prefills = [request for request in wave if request.kind == "prefill"]
            try:
                if decodes:
                    self._run_wave(decodes)
                for request in prefills:
                    self._run_prefill(request)
            except Exception as error:  # noqa: BLE001 - a wave failure is reported, never fatal
                for request in wave:
                    if not request.event.is_set():
                        request.finish(error)

    # ------------------------------------------------------------------ #
    # shared forward pieces                                              #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def image_features(self, rows):
        """Per-image feature blocks, through the model's own batched vision
        call. Images are grouped so the SAM attention biases stay bounded;
        each group carries explicit per-image tile counts."""
        blocks = []
        index = 0
        total = len(rows)
        while index < total:
            group = [rows[index]]
            tiles = rows[index].tiles
            index += 1
            while index < total and len(group) < self.vision_images \
                    and tiles + rows[index].tiles <= self.vision_tiles:
                group.append(rows[index])
                tiles += rows[index].tiles
                index += 1
            pixel_values = torch.stack([row.pixel_values for row in group]).to(
                device=self.device, dtype=self.dtype)
            local = [row.pixel_values_local for row in group if row.pixel_values_local is not None]
            if local:
                pixel_values_local = torch.cat(local).to(device=self.device, dtype=self.dtype)
            else:
                pixel_values_local = None
            counts = [row.tiles for row in group]
            features = self.model.model.get_image_features(
                pixel_values=pixel_values,
                pixel_values_local=pixel_values_local,
                num_local_patches=counts,
            )
            for row, block in zip(group, features.pooler_output):
                row.image_rows = int(block.shape[0])
            blocks.extend(features.pooler_output)
        return blocks

    @staticmethod
    def expand_template(raw_ids: list[int], image_rows: int) -> list[int]:
        """Replace the sequence's single image placeholder with its block."""
        at = raw_ids.index(128815)
        return [*raw_ids[:at], *([128815] * image_rows), *raw_ids[at + 1:]]

    @torch.no_grad()
    def _prefill_rows(self, rows, cache, features):
        """Left-padded prefill of the wave's expanded prompts; returns the
        last-position logits [B, vocab] and fills `cache`."""
        prompts = [row.ids for row in rows]
        lengths = [len(ids) for ids in prompts]
        longest = max(lengths)
        batch = len(rows)
        input_ids = torch.full((batch, longest), self.pad_token_id, dtype=torch.long)
        attention = torch.zeros((batch, longest), dtype=torch.bool)
        for index, ids in enumerate(prompts):
            input_ids[index, longest - lengths[index]:] = torch.tensor(ids, dtype=torch.long)
            attention[index, longest - lengths[index]:] = True
        input_ids = input_ids.to(self.device)
        attention = attention.to(self.device)
        position_ids = (attention.long().cumsum(-1) - 1).clamp(min=0)
        packed = SimpleNamespace(pooler_output=features)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
            mm_encoder_outputs={"image": packed},
            logits_to_keep=1,
        )
        return out.logits[:, -1, :]

    @torch.no_grad()
    def _decode_step(self, rows, cache, last_tokens):
        """One decode step for the active rows; returns logits [B, vocab]."""
        seen = cache.get_seq_length()
        batch = len(rows)
        input_ids = torch.tensor(last_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
        attention = torch.zeros((batch, seen + 1), dtype=torch.bool)
        for index, row in enumerate(rows):
            real = row.prompt_len + len(row.generated) - 1
            attention[index, seen - real:] = True
        attention = attention.to(self.device)
        position_ids = torch.tensor(
            [row.prompt_len + len(row.generated) - 1 for row in rows],
            dtype=torch.long, device=self.device).unsqueeze(1)
        out = self.model(
            input_ids=input_ids,
            attention_mask=attention,
            position_ids=position_ids,
            past_key_values=cache,
            use_cache=True,
        )
        return out.logits[:, -1, :]

    # ------------------------------------------------------------------ #
    # sampling                                                           #
    # ------------------------------------------------------------------ #

    def _sample(self, logits, rows, want_raw):
        """Sample one token per row, applying the request's sampling record
        exactly as transformers applies it. Returns (tokens, raw); raw is the
        fp32 logits before any processing, only when asked for."""
        scores = logits.float()
        raw = scores.clone() if want_raw else None
        for index, row in enumerate(rows):
            settings = row.sampling
            if len(row.generated) < row.min_new:
                scores[index, self.eos_token_id] = NEG_INF
            if settings["presence_penalty"] != 0.0 or settings["frequency_penalty"] != 0.0:
                seen, counts = np.unique(np.asarray(row.ids), return_counts=True)
                seen_t = torch.as_tensor(seen, device=self.device)
                if settings["presence_penalty"] != 0.0:
                    scores[index, seen_t] -= settings["presence_penalty"]
                if settings["frequency_penalty"] != 0.0:
                    scores[index, seen_t] -= settings["frequency_penalty"] * torch.as_tensor(
                        counts, device=self.device, dtype=torch.float32)
            if settings["repetition_penalty"] != 1.0:
                seen_t = torch.as_tensor(
                    np.unique(np.asarray(row.ids)), device=self.device)
                values = scores[index, seen_t]
                scores[index, seen_t] = torch.where(
                    values < 0,
                    values * settings["repetition_penalty"],
                    values / settings["repetition_penalty"],
                )
        for index, row in enumerate(rows):
            banned = row.ngram.banned(np.asarray(row.ids, dtype=np.int64))
            if banned:
                scores[index, torch.as_tensor(list(banned), device=self.device)] = NEG_INF
        # warpers, in transformers' order: temperature, top_k, top_p, min_p
        temperatures = torch.tensor(
            [row.sampling["temperature"] for row in rows],
            dtype=torch.float32, device=self.device).unsqueeze(1)
        scores = scores / temperatures
        for index, row in enumerate(rows):
            settings = row.sampling
            if settings["top_k"] and settings["top_k"] > 0:
                top_k = min(settings["top_k"], scores.shape[-1])
                threshold = torch.topk(scores[index], top_k).values[-1]
                scores[index] = scores[index].masked_fill(scores[index] < threshold, NEG_INF)
            if settings["top_p"] < 1.0:
                row_scores = scores[index]
                sorted_scores, sorted_indices = torch.sort(row_scores, descending=False)
                cumulative = sorted_scores.softmax(dim=-1).cumsum(dim=-1)
                remove = cumulative <= (1.0 - settings["top_p"])
                remove[..., -1] = False  # min_tokens_to_keep == 1
                scores[index] = row_scores.masked_fill(
                    remove.scatter(0, sorted_indices, remove), NEG_INF)
            if settings["min_p"] > 0.0:
                row_scores = scores[index]
                probs = row_scores.softmax(dim=-1)
                remove = probs < settings["min_p"] * probs.amax()
                remove[probs.argmax()] = False
                scores[index] = row_scores.masked_fill(remove, NEG_INF)
        probs = torch.softmax(scores, dim=-1)
        draws = [torch.multinomial(probs[index], num_samples=1, generator=row.generator)
                 for index, row in enumerate(rows)]
        tokens = torch.stack(draws).flatten().tolist()
        return tokens, raw

    # ------------------------------------------------------------------ #
    # waves                                                              #
    # ------------------------------------------------------------------ #

    def _run_wave(self, requests):
        rows = [request.row for request in requests]
        try:
            features = self.image_features(rows)
            for row in rows:
                if not row.ids:
                    row.ids = self.expand_template(row.raw_ids, row.image_rows)
                    row.prompt_len = len(row.ids)
            window = max(row.prompt_len + row.max_new for row in rows)
            if self.use_graphs:
                self._graph_wave(rows, features, window)
            else:
                cache = KotStaticCache(self.num_layers, window)
                cache.allocate(
                    batch=len(rows),
                    heads=self.model.config.text_config.num_key_value_heads,
                    head_dim=(self.model.config.text_config.hidden_size
                              // self.model.config.text_config.num_attention_heads),
                    dtype=self.dtype, device=self.device)
                logits = self._prefill_rows(rows, cache, features)
                self._decode_loop(rows, cache, logits)
            for request in requests:
                if not request.event.is_set():
                    request.finish()
        except Exception as error:  # noqa: BLE001
            for request in requests:
                if not request.event.is_set():
                    request.finish(error)

    def _decode_loop(self, rows, cache, logits):
        """Eager decode: one forward per step over the surviving rows."""
        while rows:
            want_raw = any(row.capture is not None for row in rows)
            tokens, raw = self._sample(logits, rows, want_raw)
            survivors = []
            keep_index = []
            for index, row in enumerate(rows):
                if row.capture is not None and raw is not None:
                    row.capture(raw[index])
                token = tokens[index]
                row.generated.append(token)
                row.ids.append(token)
                if row.emit is not None:
                    row.emit(token)
                if token == self.eos_token_id and row.allow_eos and len(row.generated) > row.min_new:
                    row.done = True
                elif len(row.generated) >= row.max_new:
                    row.done = True
                if row.done:
                    if row.on_done is not None:
                        row.on_done()
                else:
                    survivors.append(row)
                    keep_index.append(index)
            if not survivors:
                break
            if len(survivors) < len(rows):
                cache.batch_select_indices(
                    torch.tensor(keep_index, device=self.device))
            logits = self._decode_step(survivors, cache,
                                       [row.generated[-1] for row in survivors])
            rows = survivors

    # ------------------------------------------------------------------ #
    # graph decode                                                       #
    # ------------------------------------------------------------------ #

    def _graph_for(self, batch, window):
        """The replayable decode graph for one (batch, window) shape, captured
        on first use and reused across waves."""
        import math
        bucket = int(math.ceil(window / 256.0) * 256)
        key = (batch, bucket)
        graph = self._graphs.get(key)
        if graph is None:
            graph = DecodeGraph(self, batch, bucket)
            if len(self._graphs) >= self.graph_cache:
                victim = next(iter(self._graphs))
                del self._graphs[victim]
                torch.npu.empty_cache()
            self._graphs[key] = graph
        return graph

    def _graph_wave(self, rows, features, window):
        """Decode a wave through replayed NPU graphs.

        The wave is padded up to the graph's batch with dummy rows (pad-token
        prompts, no image placeholder, finite but unused output), which keeps
        every replay shape-stable; a finished row is simply retired from the
        active list while its slot keeps decoding pad tokens. The prefill and
        every value the step reads stay exactly what the eager path computes.
        """
        batch_size = next((b for b in self.graph_batches if b >= len(rows)),
                          self.graph_batches[-1])
        if len(rows) > batch_size:
            batch_size = len(rows)
        graph = self._graph_for(batch_size, window)
        graph.reset()
        batch = graph.batch

        padded = list(rows)
        while len(padded) < batch:
            dummy = Row()
            dummy.raw_ids = [self.pad_token_id]
            dummy.ids = [self.pad_token_id]
            dummy.prompt_len = 1
            dummy.max_new = max(row.max_new for row in rows)
            dummy.allow_eos = True
            dummy.done = True  # never active; its slot only fills the shape
            padded.append(dummy)

        # prefill (eager) into the graph's cache; the dummies carry no image
        # placeholder, so the scatter still maps real rows to real features
        logits = self._prefill_rows(padded, graph.cache, features)
        filled = graph.cache.get_seq_length()

        # the 4D decode mask: each real row's region, bounded by what the cache
        # actually holds (the mask carries causality, so no slot at or after
        # the step being fed may be visible); one column is switched on per
        # step, outside the graph
        mask = graph.mask
        mask.fill_(False)
        for index, row in enumerate(padded):
            if index < len(rows):
                mask[index, 0, 0, filled - row.prompt_len:filled] = True
            else:
                mask[index, 0, 0, :filled] = True

        input_ids = graph.input_ids
        position_ids = graph.position_ids
        active = list(enumerate(rows))  # (slot, row)
        step = 0
        while active:
            want_raw = any(slot_row[1].capture is not None for slot_row in active)
            slots = torch.tensor([slot_row[0] for slot_row in active],
                                 dtype=torch.long, device=self.device)
            step_logits = logits.index_select(0, slots)
            tokens, raw = self._sample(step_logits,
                                       [slot_row[1] for slot_row in active],
                                       want_raw)
            survivors = []
            for position, (slot, row) in enumerate(active):
                if row.capture is not None and raw is not None:
                    row.capture(raw[position])
                token = tokens[position]
                row.generated.append(token)
                row.ids.append(token)
                if row.emit is not None:
                    row.emit(token)
                if token == self.eos_token_id and row.allow_eos and len(row.generated) > row.min_new:
                    row.done = True
                elif len(row.generated) >= row.max_new:
                    row.done = True
                if row.done:
                    if row.on_done is not None:
                        row.on_done()
                else:
                    survivors.append((slot, row))
            if not survivors:
                break
            # feed the next token per active row; finished slots feed the pad
            feed = [self.pad_token_id] * batch
            positions = [0] * batch
            for slot, row in survivors:
                feed[slot] = row.generated[-1]
                positions[slot] = row.prompt_len + len(row.generated) - 1
            input_ids.copy_(torch.tensor(feed, dtype=torch.long,
                                         device=self.device).unsqueeze(1))
            position_ids.copy_(torch.tensor(positions, dtype=torch.long,
                                            device=self.device).unsqueeze(1))
            slot = filled + step
            if slot < graph.window:
                mask[:, 0, 0, slot] = True
            if self.graph_sync:
                torch.npu.synchronize()
            logits = graph.replay()
            step += 1
            active = survivors

    # ------------------------------------------------------------------ #
    # teacher-forced prefill                                             #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def _run_prefill(self, request: EngineRequest):
        try:
            row = Row()
            row.pixel_values, row.pixel_values_local, row.tiles = request.prefill_features
            features = self.image_features([row])
            row.image_rows = features[0].shape[0]
            ids = self.expand_template(request.prefill_ids, row.image_rows)
            lengths = len(ids)
            input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
            attention = torch.ones((1, lengths), dtype=torch.bool, device=self.device)
            position_ids = torch.arange(lengths, device=self.device).unsqueeze(0)
            packed = SimpleNamespace(pooler_output=features)
            n_rows = request.n_rows
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention,
                position_ids=position_ids,
                mm_encoder_outputs={"image": packed},
                use_cache=False,
                logits_to_keep=n_rows + 1,
            )
            request.logits_sink(out.logits[0, :-1, :])
            request.finish()
        except Exception as error:  # noqa: BLE001
            request.finish(error)
