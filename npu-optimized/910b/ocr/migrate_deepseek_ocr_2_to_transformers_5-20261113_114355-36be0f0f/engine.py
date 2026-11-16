"""The batched generation engine: one code path for /kot/score and chat.

The engine thread is the only thread that touches the accelerator. Units
arrive preprocessed (image already encoded by the processor); the engine
groups pending units into one batch when they share sampling settings, runs
one batched vision + prefill + decode over left-padded rows, and streams
tokens out per row as they are produced.

Numerics rules this module implements exactly (see serve.py's docstring):

* the raw logits of every step are cast to float32 before any processing;
  those rows are what a scored decode writes (log-softmax of them);
* the sampling order is generate()'s own: min-new-tokens EOS suppression,
  the no-repeat n-gram rule, the temperature / top_k / top_p / min_p
  warpers, softmax, one multinomial per row;
* position ids follow generate(): cumsum(attention_mask)-1 at prefill, last
  position + 1 per generated token;
* finished rows are masked to the pad token and keep being forwarded, as
  generate() does, so a batch row's RNG stream is not disturbed by the
  others' finishing.

Nothing here is cached between requests: every unit re-encodes its image
through the same batched vision forward the next unit uses.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from collections import deque

import numpy as np
import torch
from types import SimpleNamespace
from transformers import set_seed
from transformers.cache_utils import StaticCache
from transformers.generation.logits_process import (
    MinPLogitsWarper,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)

IMAGE_TOKEN_ID = 128815
PAD_ID = 1
NEG_INF = float("-inf")

BATCH_MAX = int(os.environ.get("KOT_BATCH_MAX", "100"))
BATCH_TOKEN_CAP = int(os.environ.get("KOT_BATCH_TOKENS", "400000"))
BATCH_WAIT_S = float(os.environ.get("KOT_BATCH_WAIT", "0.5"))
FLUSH_EVERY = int(os.environ.get("KOT_FLUSH_EVERY", "16"))
LOGPROB_CHUNK = 32
VISION_GLOBAL_CHUNK = int(os.environ.get("KOT_VISION_GLOBAL_CHUNK", "8"))
VISION_TILE_CHUNK = int(os.environ.get("KOT_VISION_TILE_CHUNK", "16"))
TIMING = bool(os.environ.get("KOT_TIMING"))

NGRAM_SIZE = 30
NGRAM_WINDOW = 90
NGRAM_WHITELIST = (128821, 128822)


class Unit:
    """One request's generation work, from submit to streamed answer."""

    def __init__(self, prompt_ids, image, mode, max_new_tokens, seed, sampling,
                 allow_eos, need_logprobs, out_dir, prompt_id="", n_rows=0,
                 continuation=None, stream=False):
        self.prompt_ids = prompt_ids
        self.image = image
        self.mode = mode
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.sampling = sampling
        self.allow_eos = allow_eos
        self.need_logprobs = need_logprobs
        self.out_dir = out_dir
        self.prompt_id = prompt_id
        self.n_rows = n_rows
        self.continuation = continuation
        self.stream = stream
        self.enc = None
        self.events = queue.Queue()
        self.emitted = []
        self.result = None

    @property
    def group(self):
        """Units batch together only when their sampling matches exactly."""
        if self.need_logprobs:
            return ("solo", self.seed)
        return ("chat", json.dumps(self.sampling, sort_keys=True), self.seed)


class Engine:
    def __init__(self, model, processor, device, dtype, vocab_size, eos_id):
        self.model = model
        self.tokenizer = processor.tokenizer
        self.device = device
        self.dtype = dtype
        self.vocab_size = vocab_size
        self.eos_id = eos_id
        self.pending = deque()
        self.cv = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True, name="engine")
        self.thread.start()

    def enqueue(self, unit: Unit):
        with self.cv:
            self.pending.append(unit)
            self.cv.notify_all()

    # -- scheduler ---------------------------------------------------------

    def _loop(self):
        while True:
            with self.cv:
                while not self.pending:
                    self.cv.wait()
            if BATCH_WAIT_S > 0:
                time.sleep(BATCH_WAIT_S)
            batch = self._collect()
            if not batch:
                continue
            try:
                if batch[0].mode == "prefill":
                    self._run_prefill(batch[0])
                else:
                    self._run_decode(batch)
            except Exception as error:
                import traceback

                traceback.print_exc()
                for unit in batch:
                    unit.events.put(("error", f"{type(error).__name__}: {error}"))

    def _collect(self):
        with self.cv:
            batch = []
            positions = 0
            while self.pending and len(batch) < BATCH_MAX:
                unit = self.pending[0]
                if batch and (unit.group != batch[0].group):
                    break
                if unit.need_logprobs and batch:
                    break
                unit_positions = (unit.enc["block"] + 12 if unit.enc else 512) + unit.max_new_tokens
                if batch and positions + unit_positions > BATCH_TOKEN_CAP:
                    break
                self.pending.popleft()
                batch.append(unit)
                positions += unit_positions
                if unit.need_logprobs:
                    break
            return batch

    # -- helpers -----------------------------------------------------------

    def _forward(self, cache, ids, mask, position_ids, image_features=None,
                 logits_to_keep=0, use_cache=True):
        """One LM forward. Image features arrive precomputed (the native
        get_image_features output, per image, in batch order) through the
        forward's own mm_encoder_outputs door - the same door generate()
        uses - so the vision tower's memory is bounded by _vision_features'
        chunking and nothing else about the forward changes."""
        mm = None
        if image_features is not None:
            mm = {"image": SimpleNamespace(pooler_output=image_features)}
        return self.model(
            input_ids=ids,
            attention_mask=mask,
            position_ids=position_ids,
            mm_encoder_outputs=mm,
            use_cache=use_cache,
            past_key_values=cache,
            logits_to_keep=logits_to_keep,
        )

    def _vision_features(self, batch):
        """Per-image projected features through the native get_image_features,
        in bounded chunks.

        The chunking bounds the SAM encoder's transient activations (its
        global-attention layers scale as batch x 4096^2); it changes nothing
        about the computation: each chunk is the same call the native forward
        makes, and the per-image feature lists come back concatenated in
        batch order, exactly what the native scatter consumes. A batch of one
        is a chunk of one - the scored path and the chat path run the same
        code."""
        encs = [u.enc for u in batch]
        features = []
        index = 0
        while index < len(encs):
            chunk, globals_n, tiles_n = [], 0, 0
            while index < len(encs):
                tiles = encs[index]["tiles"]
                if chunk and (globals_n >= VISION_GLOBAL_CHUNK or tiles_n + tiles > VISION_TILE_CHUNK):
                    break
                chunk.append(encs[index])
                globals_n += 1
                tiles_n += tiles
                index += 1
            with torch.no_grad():
                pixels = torch.cat([e["pixel_values"] for e in chunk]).to(self.device, self.dtype)
                locals_ = [e["pixel_values_local"] for e in chunk if e["pixel_values_local"] is not None]
                local = torch.cat(locals_).to(self.device, self.dtype) if locals_ else None
                tiles_list = [e["tiles"] for e in chunk]
                out = self.model.model.get_image_features(pixels, local, tiles_list)
            features.extend(out.pooler_output)
        return features

    def _ngram_ban(self, all_ids):
        """The no-repeat rule as device ops: no per-token host sync.

        A token that would complete an n-gram already seen within the window
        is banned, the two table tags excepted - the same rule the reference
        applies per row."""
        total = all_ids.shape[1]
        if total < NGRAM_SIZE:
            return None
        prefix = all_ids[:, total - (NGRAM_SIZE - 1):]
        start = max(0, total - NGRAM_WINDOW)
        end = total - NGRAM_SIZE + 1
        if end <= start:
            return None
        windows = all_ids[:, start:end + NGRAM_SIZE - 1].unfold(1, NGRAM_SIZE, 1)
        match = (windows[:, :, :-1] == prefix.unsqueeze(1)).all(dim=-1)
        completions = windows[:, :, -1]
        scratch = torch.zeros(
            all_ids.shape[0], self.vocab_size + 1, dtype=torch.bool, device=all_ids.device
        )
        scratch.scatter_(1, completions.masked_fill(~match, self.vocab_size), True)
        ban = scratch[:, : self.vocab_size]
        ban[:, list(NGRAM_WHITELIST)] = False
        return ban

    def _warpers(self, sampling):
        chain = []
        if sampling["temperature"] != 1.0:
            chain.append(TemperatureLogitsWarper(sampling["temperature"]))
        if sampling["top_k"]:
            chain.append(TopKLogitsWarper(top_k=sampling["top_k"]))
        if sampling["top_p"] < 1.0:
            chain.append(TopPLogitsWarper(top_p=sampling["top_p"]))
        if sampling["min_p"] > 0.0:
            chain.append(MinPLogitsWarper(min_p=sampling["min_p"]))
        return chain

    def _row_text(self, ids):
        """The checkpoint's own decode: the stop token goes, table tags stay."""
        text = self.tokenizer.decode(ids, skip_special_tokens=False)
        eos_token = getattr(self.tokenizer, "eos_token", None)
        if eos_token:
            text = text.removesuffix(eos_token)
        return text

    @staticmethod
    def _open_logprobs(out_dir, rows, vocab):
        path = out_dir / "log_probs.npy"
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float32, shape=(rows, vocab))
        return path, array

    @staticmethod
    def _shrink_npy(path, rows, vocab):
        """Patch the header shape in place and truncate: a decode that ended
        early wrote fewer rows than the file was opened with."""
        with open(path, "r+b") as handle:
            head = handle.read(10)
            if head[:6] != b"\x93NUMPY" or head[6] != 1:
                return
            length = int.from_bytes(head[8:10], "little")
            data_offset = 10 + length
            text = "{'descr': '<f4', 'fortran_order': False, 'shape': (%d, %d), }" % (rows, vocab)
            encoded = text.encode("utf-8")
            if len(encoded) + 1 > length:
                return
            encoded += b" " * (length - len(encoded) - 1) + b"\n"
            handle.seek(10)
            handle.write(encoded)
            handle.truncate(data_offset + rows * vocab * 4)

    # -- teacher-forced prefill --------------------------------------------

    def _run_prefill(self, unit: Unit):
        device, dtype = self.device, self.dtype
        prompt = unit.enc["prompt"]
        tail = unit.continuation[len(unit.continuation) - unit.n_rows:]
        ids = torch.as_tensor([prompt + tail], dtype=torch.long, device=device)
        image_features = self._vision_features([unit])
        with torch.no_grad():
            out = self._forward(None, ids, torch.ones_like(ids),
                                torch.arange(ids.shape[1], device=device).unsqueeze(0),
                                image_features, logits_to_keep=unit.n_rows + 1, use_cache=False)
        rows = out.logits[0, : unit.n_rows]
        _, array = self._open_logprobs(unit.out_dir, unit.n_rows, self.vocab_size)
        block = 0
        for start in range(0, unit.n_rows, LOGPROB_CHUNK):
            stop = min(start + LOGPROB_CHUNK, unit.n_rows)
            chunk = torch.log_softmax(rows[start:stop].float(), dim=-1)
            array[start:stop] = chunk.cpu().numpy()
            block += 1
            unit.events.put(("block", (block, stop - start)))
        array.flush()
        del array
        np.save(unit.out_dir / "input_ids.npy",
                np.asarray(unit.continuation, dtype=np.int32), allow_pickle=False)
        unit.result = {"rows": unit.n_rows}
        unit.events.put(("done", {"reply": {
            "prompt_id": unit.prompt_id,
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": unit.n_rows,
            "vocab_size": self.vocab_size,
            "output_ids": [],
        }}))

    # -- batched decode ------------------------------------------------------

    def _run_decode(self, batch):
        device, dtype = self.device, self.dtype
        B = len(batch)
        prompts = [u.enc["prompt"] for u in batch]
        prompt_lens = [len(p) for p in prompts]
        maxlen = max(prompt_lens)
        ids = torch.full((B, maxlen), PAD_ID, dtype=torch.long, device=device)
        mask = torch.zeros((B, maxlen), dtype=torch.long, device=device)
        for i, prompt in enumerate(prompts):
            ids[i, maxlen - len(prompt):] = torch.as_tensor(prompt, dtype=torch.long, device=device)
            mask[i, maxlen - len(prompt):] = 1
        image_features = self._vision_features(batch)
        max_news = [u.max_new_tokens for u in batch]
        global_max = max(max_news)

        score_unit = batch[0] if batch[0].need_logprobs else None
        warpers = self._warpers(batch[0].sampling or
                                {"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0})

        logprobs_array = None
        logprobs_path = None
        rows_written = 0
        stash = []
        if score_unit is not None:
            logprobs_path, logprobs_array = self._open_logprobs(
                score_unit.out_dir, score_unit.max_new_tokens, self.vocab_size
            )

        def flush_logprobs():
            nonlocal rows_written
            if not stash:
                return
            logged = torch.log_softmax(torch.cat(stash, dim=0), dim=-1)
            stash.clear()
            count = logged.shape[0]
            logprobs_array[rows_written:rows_written + count] = logged.cpu().numpy()
            rows_written += count
            score_unit.events.put(("block", ((rows_written + LOGPROB_CHUNK - 1) // LOGPROB_CHUNK, count)))

        set_seed(batch[0].seed)
        cache = StaticCache(config=self.model.config, max_cache_len=maxlen + global_max)
        t_prefill0 = time.perf_counter()
        with torch.no_grad():
            out = self._forward(cache, ids, mask, (mask.cumsum(-1) - 1).masked_fill(mask == 0, 0),
                                image_features, logits_to_keep=1)
        logits = out.logits[:, -1]
        if TIMING:
            print(f"[engine] batch B={B} maxlen={maxlen} global_max={global_max} "
                  f"vision+prefill {time.perf_counter()-t_prefill0:.1f}s", flush=True)

        base_pos = torch.tensor(prompt_lens, dtype=torch.long, device=device)
        max_new_row = torch.tensor(max_news, dtype=torch.long, device=device)
        allow_eos_row = torch.tensor([u.allow_eos for u in batch], dtype=torch.bool, device=device)
        min_new_row = torch.where(allow_eos_row, torch.zeros_like(max_new_row), max_new_row)
        unfinished = torch.ones(B, dtype=torch.long, device=device)
        fin_step = torch.full((B,), -1, dtype=torch.long, device=device)
        ones_col = torch.ones((B, 1), dtype=torch.long, device=device)
        all_ids = ids
        cur_mask = mask
        prev_text_len = [0] * B
        stopped = False
        step = 0
        t_steps0 = time.perf_counter()
        t_last = t_steps0

        while step < global_max and not stopped:
            raw = logits.to(torch.float32)
            if score_unit is not None:
                # only while the scored row is still generating: past its
                # finish step the loop may still run to the flush boundary
                live = raw[fin_step < 0]
                if live.shape[0]:
                    stash.append(live.clone())
            scores = raw
            ban_eos = step < min_new_row
            scores[:, self.eos_id] = torch.where(ban_eos, torch.full_like(scores[:, self.eos_id], NEG_INF),
                                                 scores[:, self.eos_id])
            ban_mask = self._ngram_ban(all_ids)
            if ban_mask is not None:
                scores = scores.masked_fill_(ban_mask, NEG_INF)
            for warp in warpers:
                scores = warp(all_ids, scores)
            probs = torch.softmax(scores, dim=-1)
            tokens = torch.multinomial(probs, 1)

            active = unfinished.bool()
            done_now = (active & ((tokens[:, 0] == self.eos_id) & allow_eos_row)) \
                | (active & ((step + 1) >= max_new_row))
            tokens = torch.where(active.unsqueeze(1), tokens, torch.full_like(tokens, PAD_ID))
            unfinished = unfinished & (~done_now).long()
            fin_step = torch.where(fin_step >= 0, fin_step,
                                   torch.where(done_now, torch.full_like(fin_step, step), fin_step))
            all_ids = torch.cat([all_ids, tokens], dim=1)
            cur_mask = torch.cat([cur_mask, ones_col], dim=1)
            step += 1

            if step % FLUSH_EVERY == 0 or step >= global_max:
                cpu_ids = all_ids.cpu().tolist()
                fins = fin_step.cpu().tolist()
                stopped = all(fin >= 0 for fin in fins)
                if TIMING and step and step % 256 == 0:
                    now = time.perf_counter()
                    print(f"[engine] step {step}/{global_max} active "
                          f"{sum(1 for f in fins if f < 0)}/{B} "
                          f"({(now-t_last)*1000/256:.1f} ms/step)", flush=True)
                    t_last = now
                for i, unit in enumerate(batch):
                    count = fins[i] + 1 if fins[i] >= 0 else step
                    unit.emitted = cpu_ids[i][maxlen:maxlen + count]
                    if not unit.need_logprobs and unit.emitted:
                        text_now = self._row_text(unit.emitted)
                        delta = text_now[prev_text_len[i]:]
                        if delta:
                            unit.events.put(("delta", delta))
                        prev_text_len[i] = len(text_now)
                if score_unit is not None:
                    flush_logprobs()

            if step < global_max and not stopped:
                position_ids = (base_pos + step - 1).unsqueeze(1)
                with torch.no_grad():
                    out = self._forward(cache, tokens, cur_mask, position_ids, logits_to_keep=1)
                logits = out.logits[:, -1]

        if score_unit is not None:
            flush_logprobs()
            logprobs_array.flush()
            del logprobs_array
            if rows_written < score_unit.max_new_tokens:
                self._shrink_npy(logprobs_path, rows_written, self.vocab_size)
        if TIMING:
            print(f"[engine] decode done: {step} steps in {time.perf_counter()-t_steps0:.1f}s "
                  f"({(time.perf_counter()-t_steps0)/max(1,step)*1000:.1f} ms/step)", flush=True)

        for i, unit in enumerate(batch):
            emitted = unit.emitted
            text = self._row_text(emitted)
            if unit.mode == "chat":
                stopped_by_eos = bool(emitted) and emitted[-1] == self.eos_id
                unit.events.put(("done", {"text": text, "stop": stopped_by_eos}))
                continue
            np.save(unit.out_dir / "input_ids.npy",
                    np.asarray(unit.prompt_ids + emitted, dtype=np.int32), allow_pickle=False)
            (unit.out_dir / "text.txt").write_text(text, encoding="utf-8")
            unit.result = {"text": text, "rows": len(emitted)}
            unit.events.put(("done", {"reply": {
                "prompt_id": unit.prompt_id,
                "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
                "n_positions": len(emitted),
                "vocab_size": self.vocab_size,
                "output_ids": list(emitted),
            }, "text": text}))
