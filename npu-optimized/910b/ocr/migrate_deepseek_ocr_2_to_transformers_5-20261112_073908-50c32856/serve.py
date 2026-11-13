#!/usr/bin/env python3
"""DeepSeek-OCR-2 serving the kot deployment contract on transformers 5.22.1.

One process, one model load, three routes:

* GET  /kot/info            - provenance of the running process.
* POST /kot/score           - the correctness measurement: full-vocabulary
                              log-probabilities, written as .npy files,
                              never returned as tensors.  Decode streams
                              NDJSON block lines as rows are produced; the
                              replay prefill answers one JSON object.
* POST /v1/chat/completions - the OpenAI-shaped chat surface the speed
                              harness drives, streaming SSE.

The model is the vendored ocr2 package: the checkpoint's own architecture
ported to transformers 5.22.1 (its bundled remote code is 4.46-era), loaded
with from_pretrained from the --model weights and nothing else.  The same
preprocessing (views/template expansion), the same vision encoder, the same
language forward and the same sampling serve both /kot/score and the chat
endpoint; the endpoints differ only in how they collect results.

Score requests arrive one at a time and run alone at batch 1 through the
per-op path verified bit-identical against the transformers 4.46.3 reference
on this card.  Chat requests are served with continuous batching over the
same modules: left-padded rows, per-row positions, one batched KV cache, and
explicit tile counts and offsets in the batched vision encoder.

    ./serve.sh --port P --model /path/to/DeepSeek-OCR-2 --device npu
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from ocr2.model import Ocr2Config, Ocr2ForCausalLM  # noqa: E402
from ocr2.preprocess import (  # noqa: E402
    IMAGE_TOKEN_ID, expand_template, load_image, views,
)
from ocr2.sampling import Sampler  # noqa: E402

HOST = "127.0.0.1"
INFO = "/kot/info"
SCORE = "/kot/score"
CHAT = "/v1/chat/completions"
UTF8 = "utf-8"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 1024
PROMPT_TEXT = "Free OCR."
WRITE_CHUNK = 64
BLOCK_ROWS = 64
BACKLOG = 4096
MAX_BODY = 512 * 1024 * 1024

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")

DRIVER_VERSION_FILE = "/usr/local/Ascend/driver/version.info"
TOOLKIT_VERSION_FILE = "/usr/local/Ascend/ascend-toolkit/version.info"
TOOLKIT_LATEST_DIR = "/usr/local/Ascend/ascend-toolkit/latest"


def ascend_version(version_file, latest_dir=""):
    try:
        for line in Path(version_file).read_text(encoding=UTF8).splitlines():
            if line.startswith("Version="):
                return line.removeprefix("Version=")
    except OSError:
        pass
    if latest_dir:
        try:
            return Path(latest_dir).resolve().name
        except OSError:
            pass
    return ""


def snapshot_identity(model_path):
    """(model id, revision): a hub snapshot names both; a plain directory names itself."""
    path = Path(model_path).resolve()
    repo = path.parent.parent.name
    if path.parent.name == "snapshots" and repo.startswith("models--") and re.fullmatch(r"[0-9a-f]{40}", path.name):
        return repo.removeprefix("models--").replace("--", "/"), path.name
    return Path(model_path).name, ""


def resolve_sampling(body):
    """The request's sampling object, every field applied exactly as given.

    An absent field falls back to its neutral value; a field this engine
    cannot apply is an error naming it, never a silently ignored value.
    """
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ValueError("unsupported sampling field(s): " + ", ".join(unsupported))
    resolved = {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", 0)),
        "min_p": float(sampling.get("min_p", 0.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
        "presence_penalty": float(sampling.get("presence_penalty", 0.0)),
    }
    if resolved["presence_penalty"] != 0.0:
        raise ValueError(
            "unsupported sampling value: presence_penalty must be 0 (no engine knob "
            "applies it, and a value this deployment cannot apply is an error, never ignored)"
        )
    if resolved["temperature"] == 0.0:
        raise ValueError("temperature 0 is not a path this model runs")
    return resolved


class ChatSlot:
    _next_id = [0]

    def __init__(self, request, emit):
        ChatSlot._next_id[0] += 1
        self.id = ChatSlot._next_id[0]
        self.request = request
        self.emit = emit
        self.image = request["image"]
        self.instruction = request["instruction"]
        self.max_new_tokens = request["max_new_tokens"]
        self.sampling = request["sampling"]
        self.seed = request["seed"]
        # filled at admission
        self.expanded = None
        self.image_mask = None
        self.feats = None
        self.template_ids = None
        # filled during generation
        self.consumed = None
        self.emitted = []
        self.pending_text_ids = []
        self.sent_text = ""
        self.sampler = None
        self.finished = False
        self.finish_reason = None


class Engine:
    """Owns the NPU and the model.  One background thread runs everything.

    Score jobs run alone at batch 1 (the verified exact path).  Chat slots
    run through the same modules with continuous batching: admissions are
    prefilled in left-padded groups with the vision encoder batched across
    them (explicit tile counts and offsets per image), then every active
    slot decodes one token per engine step over one batched KV cache.
    """

    def __init__(self, model, tokenizer, device, dtype, max_batch):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.max_batch = max_batch
        self.vocab_size = int(model.config.vocab_size)
        self.eos_token_id = int(model.config.eos_token_id)
        self.pad_token_id = int(getattr(model.config, "pad_token_id", None) or 0)
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.score_jobs = deque()
        self.waiting = deque()
        self.active = []
        self.kv = []  # batched cache: per layer (k, v), each [B, H, T, D]
        self.preprocess_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="prep")
        self.thread = threading.Thread(target=self.run, name="engine", daemon=True)

    def start(self):
        self.thread.start()

    # ---- submission -------------------------------------------------------
    def submit_score(self, body, emit):
        job = {"body": body, "emit": emit}
        with self.lock:
            self.score_jobs.append(job)
        self.wake.set()

    def submit_chat(self, slot):
        with self.lock:
            self.waiting.append(slot)
        self.wake.set()

    # ---- shared prompt build ----------------------------------------------
    def build_prompt_ids(self, instruction):
        """[BOS, <image>, ids("\\n" + instruction)]: what the frozen template
        spells for the OCR prompt; the placeholder is expanded at admission."""
        text_ids = self.tokenizer.encode("\n" + instruction, add_special_tokens=False)
        return [int(self.tokenizer.bos_token_id), IMAGE_TOKEN_ID, *map(int, text_ids)]

    # ---- score paths (batch 1, exact) --------------------------------------
    def run_score(self, job):
        body = job["body"]
        emit = job["emit"]
        image_path = body["attachments"]["image"]
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if body.get("mode") == "prefill":
            consumed, output_ids, n_rows, text = self.score_prefill(image_path, body, out_dir)
        else:
            consumed, output_ids, n_rows, text = self.score_decode(image_path, body, out_dir, emit)
        np.save(out_dir / "input_ids.npy", np.asarray(consumed, dtype=np.int32), allow_pickle=False)
        if text is not None:
            (out_dir / "text.txt").write_text(text, encoding=UTF8)
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": n_rows,
            "vocab_size": self.vocab_size,
            "output_ids": list(output_ids),
        }

    def score_prefill(self, image_path, body, out_dir):
        incoming = [int(t) for t in body["input_ids"]]
        if not incoming:
            raise ValueError("prefill over an empty transcript: the frozen ids are missing")
        n_rows = int(body.get("n_rows", 0) or 0)
        if n_rows < 1:
            raise ValueError("the ocr prefill stores the decode's rows: set n_rows, got 0")
        if n_rows >= len(incoming):
            raise ValueError(f"n_rows {n_rows} needs at least {n_rows + 1} input ids")
        raw_ids, tail = incoming[: len(incoming) - n_rows], incoming[len(incoming) - n_rows:]
        pil = load_image(image_path)
        (patches, global_view), crop, tokens = views(pil, self.device, self.dtype)
        feats = self.model.model.encode_vision([(patches, global_view)])[0]
        expanded, mask = expand_template(raw_ids, tokens)
        start = len(expanded)
        ids = torch.tensor([[*expanded, *tail]], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([mask + [False] * len(tail)], dtype=torch.bool, device=self.device)
        embeds = self.model.model.embed_with_vision(ids, mask_t, [feats])
        hidden = self.model(inputs_embeds=embeds, use_cache=False)
        # The reference forward ran lm_head over the whole sequence; the GEMM
        # shape participates in the bf16 rounding, so keep it and slice after.
        rows = self.model.head(hidden)[0, start - 1: start - 1 + n_rows, :]
        self.write_rows(rows, out_dir, n_rows)
        return incoming, [], n_rows, None

    def score_decode(self, image_path, body, out_dir, emit):
        template_ids = [int(t) for t in body["input_ids"]]
        if not template_ids:
            template_ids = self.build_prompt_ids(PROMPT_TEXT)
        max_new_tokens = int(body["max_new_tokens"])
        seed = int(body.get("seed", DEFAULT_SEED))
        sampling = resolve_sampling(body)
        allow_eos = bool(body.get("allow_eos", False))
        pil = load_image(image_path)
        (patches, global_view), crop, tokens = views(pil, self.device, self.dtype)
        feats = self.model.model.encode_vision([(patches, global_view)])[0]
        expanded, mask = expand_template(template_ids, tokens)
        consumed = list(expanded)
        ids = torch.tensor([consumed], dtype=torch.long, device=self.device)
        mask_t = torch.tensor([mask], dtype=torch.bool, device=self.device)
        embeds = self.model.model.embed_with_vision(ids, mask_t, [feats])
        kv = []
        hidden = self.model(inputs_embeds=embeds, use_cache=True, past_kv=kv)
        # Row 0 is the last row of a full-sequence lm_head GEMM, exactly as
        # generate()'s first scores were produced.
        logits = self.model.head(hidden)[0, -1]
        sampler = Sampler(sampling, seed, self.eos_token_id)
        emitted = []
        buffer = []
        array = None
        if not allow_eos:
            # exactly max_new_tokens rows: open the file at its final shape and
            # fill it chunk by chunk as the rows are produced.
            array = np.lib.format.open_memmap(
                out_dir / "log_probs.npy", mode="w+", dtype=np.float32,
                shape=(max_new_tokens, self.vocab_size))
        length = ids.shape[1]
        min_new = 0 if allow_eos else max_new_tokens
        while len(emitted) < max_new_tokens:
            buffer.append(logits)
            if array is not None:
                if len(buffer) == WRITE_CHUNK:
                    self._fill(array, torch.stack(buffer), len(emitted) + 1 - len(buffer))
                    emit("block", len(emitted) + 1)
                    buffer = []
            elif (len(emitted) + 1) % BLOCK_ROWS == 0:
                emit("block", len(emitted) + 1)
            scores = sampler.process(logits, consumed, allow_eos, min_new, len(emitted))
            token = sampler.sample(scores)
            emitted.append(token)
            consumed.append(token)
            if allow_eos and token == self.eos_token_id:
                break
            if len(emitted) >= max_new_tokens:
                break
            next_ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
            embeds = self.model.model.embed_tokens(next_ids)
            hidden = self.model(
                inputs_embeds=embeds, use_cache=True, past_kv=kv,
                position_ids=torch.tensor([[length]], device=self.device))
            logits = self.model.head(hidden[:, -1:, :])[0, 0]
            length += 1
        if array is not None:
            if buffer:
                self._fill(array, torch.stack(buffer), len(emitted) - len(buffer))
            array.flush()
            del array
        else:
            self.write_rows(torch.stack(buffer), out_dir, len(emitted))
        text = self.tokenizer.decode(emitted, skip_special_tokens=True)
        return [*template_ids, *emitted], emitted, len(emitted), text

    def _fill(self, array, rows, start):
        stop = start + rows.shape[0]
        chunk = torch.log_softmax(rows[:, : self.vocab_size].float(), dim=-1)
        array[start:stop] = chunk.cpu().numpy()

    def write_rows(self, rows, out_dir, n_rows):
        """Stream log-softmax rows into log_probs.npy at its final shape."""
        array = np.lib.format.open_memmap(
            out_dir / "log_probs.npy", mode="w+", dtype=np.float32,
            shape=(n_rows, self.vocab_size))
        for start in range(0, n_rows, WRITE_CHUNK):
            self._fill(array, rows[start:start + WRITE_CHUNK], start)
        array.flush()
        del array

    # ---- chat paths (batched, same modules) --------------------------------
    def _fail_slot(self, slot, error):
        slot.finished = True
        slot.finish_reason = "error"
        slot.feats = None
        try:
            slot.emit("error", error)
        except Exception:
            pass

    def admit_slots(self):
        """Admit waiting slots up to max_batch: CPU preprocessing in parallel
        (the PIL/torch view pipeline releases the GIL), then ONE batched
        vision-encoder pass for the whole group (explicit tile counts and
        offsets per image inside encode_vision)."""
        candidates = []
        while self.waiting and len(self.active) + len(candidates) < self.max_batch:
            candidates.append(self.waiting.popleft())
        if not candidates:
            return
        prepped = []
        futures = [(slot, self.preprocess_pool.submit(views, slot.image, self.device, self.dtype))
                   for slot in candidates]
        for slot, future in futures:
            try:
                (patches, global_view), crop, tokens = future.result()
            except Exception as error:
                traceback.print_exc()
                self._fail_slot(slot, error)
                continue
            slot._image_batch = (patches, global_view)
            slot._image_tokens = tokens
            slot.template_ids = self.build_prompt_ids(slot.instruction)
            slot.expanded, slot.image_mask = expand_template(slot.template_ids, tokens)
            prepped.append(slot)
        if not prepped:
            return
        try:
            feats = self.model.model.encode_vision([s._image_batch for s in prepped])
        except Exception as error:
            traceback.print_exc()
            for slot in prepped:
                self._fail_slot(slot, error)
            return
        for slot, slot_feats in zip(prepped, feats):
            slot.feats = slot_feats
            slot._image_batch = None
            self.active.append(slot)

    def prefill_batch(self, slots):
        """Left-padded prefill for a group of freshly admitted slots."""
        lengths = [len(slot.expanded) for slot in slots]
        max_len = max(lengths)
        batch = len(slots)
        ids = torch.full((batch, max_len), self.pad_token_id, dtype=torch.long, device=self.device)
        pad2d = torch.zeros((batch, max_len), dtype=torch.long, device=self.device)
        for row, slot in enumerate(slots):
            n = lengths[row]
            ids[row, max_len - n:] = torch.tensor(slot.expanded, dtype=torch.long, device=self.device)
            pad2d[row, max_len - n:] = 1
        position_ids = pad2d.long().cumsum(-1) - 1
        position_ids = position_ids.masked_fill(pad2d == 0, 1)
        masks = []
        for row, slot in enumerate(slots):
            n = lengths[row]
            masks.append([False] * (max_len - n) + slot.image_mask)
        mask_t = torch.tensor(masks, dtype=torch.bool, device=self.device)
        embeds = self.model.model.embed_tokens(ids)
        for row, slot in enumerate(slots):
            embeds[row].masked_scatter_(mask_t[row].unsqueeze(-1), slot.feats)
        group_kv = []
        hidden = self.model(inputs_embeds=embeds, attention_mask=pad2d,
                            position_ids=position_ids, use_cache=True, past_kv=group_kv)
        self.merge_cache(group_kv, max_len)
        logits = self.model.head(hidden[:, -1:, :])[:, 0]  # [batch, vocab] fp32
        for row, slot in enumerate(slots):
            slot.consumed = list(slot.expanded)
            slot.sampler = Sampler(slot.sampling, slot.seed, self.eos_token_id)
            self.step_slot(slot, logits[row])

    def merge_cache(self, group_kv, group_len):
        """Append the new group's cache rows to the batched cache, left-padding
        whichever side is narrower so every row shares one width (the real
        tokens of every row end at the shared right edge)."""
        if not self.kv:
            self.kv = group_kv
            return
        cur = self.kv[0][0].shape[-2]
        width = max(cur, group_len)
        merged = []
        for (k0, v0), (k1, v1) in zip(self.kv, group_kv):
            if width > cur:
                pad = (0, 0, width - cur, 0)  # (D_l, D_r, T_l, T_r): pad time at the left
                k0 = torch.nn.functional.pad(k0, pad)
                v0 = torch.nn.functional.pad(v0, pad)
            if width > group_len:
                pad = (0, 0, width - group_len, 0)
                k1 = torch.nn.functional.pad(k1, pad)
                v1 = torch.nn.functional.pad(v1, pad)
            merged.append((torch.cat([k0, k1], dim=0), torch.cat([v0, v1], dim=0)))
        self.kv = merged

    def decode_step(self):
        self.compact_cache()
        slots = [s for s in self.active if not s.finished]
        if not slots:
            return
        batch = len(slots)
        total = self.kv[0][0].shape[-2] if self.kv else 0
        tokens = torch.tensor([[slot.emitted[-1]] for slot in slots],
                              dtype=torch.long, device=self.device)
        lengths = torch.tensor([len(slot.consumed) for slot in slots],
                               dtype=torch.long, device=self.device)
        position_ids = (lengths - 1).unsqueeze(1)
        # left padding: the real tokens of every row end at the right edge
        pad2d = (torch.arange(total + 1, device=self.device)[None, :]
                 >= (total + 1 - lengths)[:, None]).long()
        embeds = self.model.model.embed_tokens(tokens)
        hidden = self.model(inputs_embeds=embeds, attention_mask=pad2d,
                            position_ids=position_ids, use_cache=True, past_kv=self.kv)
        logits = self.model.head(hidden[:, -1:, :])[:, 0]
        for row, slot in enumerate(slots):
            self.step_slot(slot, logits[row])
        self.compact_cache()

    def step_slot(self, slot, logits):
        """Sample one token for one slot from its fp32 logits row."""
        try:
            scores = slot.sampler.process(logits, slot.consumed, True, 0, len(slot.emitted))
            token = slot.sampler.sample(scores)
        except Exception as error:
            self._fail_slot(slot, error)
            return
        slot.emitted.append(token)
        slot.consumed.append(token)
        self.push_text(slot, token)
        if token == self.eos_token_id:
            self.finish_slot(slot, "stop")
        elif len(slot.emitted) >= slot.max_new_tokens:
            self.finish_slot(slot, "length")

    def push_text(self, slot, token):
        """Incremental detokenization: flush when the pending ids decode to
        text without a replacement character (multi-byte characters complete
        within a few tokens)."""
        slot.pending_text_ids.append(token)
        try:
            text = self.tokenizer.decode(slot.pending_text_ids, skip_special_tokens=True)
        except Exception:
            text = ""
        if (text and "\ufffd" not in text) or len(slot.pending_text_ids) > 8:
            slot.pending_text_ids = []
            if text:
                slot.sent_text += text
                slot.emit("text", text)

    def finish_slot(self, slot, reason):
        if slot.pending_text_ids:
            try:
                text = self.tokenizer.decode(slot.pending_text_ids, skip_special_tokens=True)
            except Exception:
                text = ""
            slot.pending_text_ids = []
            if text:
                slot.sent_text += text
                slot.emit("text", text)
        slot.finished = True
        slot.finish_reason = reason
        slot.feats = None
        slot.emit("finish", reason)

    def compact_cache(self):
        """Drop cache rows of finished slots; keep the rest in order."""
        if all(not s.finished for s in self.active):
            return
        if self.kv:
            keep = torch.tensor([i for i, s in enumerate(self.active) if not s.finished],
                                dtype=torch.long, device=self.device)
            for layer in range(len(self.kv)):
                k, v = self.kv[layer]
                self.kv[layer] = (k.index_select(0, keep), v.index_select(0, keep))
        self.active = [s for s in self.active if not s.finished]

    # ---- thread loop -------------------------------------------------------
    def run(self):
        torch.set_grad_enabled(False)
        stats = {"step": 0, "t_vision": 0.0, "t_prefill": 0.0, "t_decode": 0.0, "last": time.time()}
        while True:
            with self.lock:
                job = self.score_jobs.popleft() if self.score_jobs else None
                idle = job is None and not self.waiting and not self.active
            if job is not None:
                try:
                    result = self.run_score(job)
                    job["emit"]("result", result)
                except Exception as error:
                    traceback.print_exc()
                    job["emit"]("error", error)
                continue
            if idle:
                # only sleep when there is nothing in flight: a decode step
                # must never wait behind the poll timeout
                self.wake.wait(timeout=0.05)
                self.wake.clear()
                continue
            t0 = time.perf_counter()
            self.admit_slots()
            t1 = time.perf_counter()
            stats["t_vision"] += t1 - t0
            fresh = [s for s in self.active if s.sampler is None and not s.finished]
            if fresh:
                try:
                    self.prefill_batch(fresh)
                except Exception as error:
                    traceback.print_exc()
                    for slot in self.active:
                        self._fail_slot(slot, error)
                    self.active = []
                    self.kv = []
                stats["t_prefill"] += time.perf_counter() - t1
            elif self.active:
                try:
                    self.decode_step()
                except Exception as error:
                    traceback.print_exc()
                    for slot in self.active:
                        self._fail_slot(slot, error)
                    self.active = []
                    self.kv = []
                stats["t_decode"] += time.perf_counter() - t1
                stats["step"] += 1
                if stats["step"] % 200 == 0:
                    now = time.time()
                    print(f"[engine] steps={stats['step']} batch={len(self.active)} "
                          f"vision={stats['t_vision']:.1f}s prefill={stats['t_prefill']:.1f}s "
                          f"decode={stats['t_decode']:.1f}s ({stats['t_decode'] / max(1, stats['step']) * 1000:.0f}ms/step) "
                          f"wall={now - stats['last']:.1f}s", flush=True)
                    stats["last"] = now


# ---------------------------------------------------------------------------
# HTTP layer: asyncio streams, keep-alive, chunked streaming responses.
# ---------------------------------------------------------------------------

class BadRequest(Exception):
    pass


def http_date():
    return time.strftime("%a, %d %b %Y %H:%M:%S GMT", time.gmtime())


async def read_request(reader, writer):
    """One HTTP/1.1 request: (method, path, headers, body) or None on EOF."""
    while True:
        line = await reader.readline()
        if not line:
            return None
        if line in (b"\r\n", b"\n"):
            continue
        break
    parts = line.decode("latin-1").strip().split()
    if len(parts) != 3:
        raise BadRequest("malformed request line")
    method, target, _version = parts
    headers = {}
    while True:
        header = await reader.readline()
        if header in (b"\r\n", b"\n", b""):
            break
        if len(header) > 65536:
            raise BadRequest("header line too long")
        name, _, value = header.decode("latin-1").partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length", 0) or 0)
    if length > MAX_BODY:
        raise BadRequest("body too large")
    if headers.get("expect", "").lower() == "100-continue":
        # the client waits for the interim response before sending the body
        writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")
        await writer.drain()
    body = b""
    if length:
        body = await reader.readexactly(length)
    return method, target, headers, body


class Responder:
    """Writes one response over the connection; supports chunked streaming."""

    def __init__(self, writer):
        self.writer = writer
        self.started = False
        self.chunked = False

    def _head(self, status, content_type, extra=(), chunked=False):
        reason = {
            200: "OK", 400: "Bad Request", 404: "Not Found",
            405: "Method Not Allowed", 500: "Internal Server Error",
        }.get(status, "OK")
        lines = [f"HTTP/1.1 {status} {reason}", f"Date: {http_date()}",
                 "Server: kot-ocr-deepseek-ocr-2", f"Content-Type: {content_type}"]
        if chunked:
            self.chunked = True
            lines.append("Transfer-Encoding: chunked")
        for item in extra:
            lines.append(item)
        lines.append("")
        lines.append("")
        self.writer.write(("\r\n".join(lines)).encode("latin-1"))
        self.started = True

    def json(self, status, payload):
        body = json.dumps(payload).encode(UTF8)
        self._head(status, "application/json", [f"Content-Length: {len(body)}"])
        self.writer.write(body)

    def continue_100(self):
        self.writer.write(b"HTTP/1.1 100 Continue\r\n\r\n")

    def chunk(self, data: bytes):
        if not data:
            return
        self.writer.write(b"%x\r\n" % len(data) + data + b"\r\n")

    def finish_chunks(self):
        if self.chunked:
            self.writer.write(b"0\r\n\r\n")


class Server:
    def __init__(self, engine, info, model_id, tokenizer):
        self.engine = engine
        self.info = info
        self.model_id = model_id
        self.tokenizer = tokenizer
        self.loop = None

    def from_engine(self, kind, payload, queue):
        """Thread-safe bridge: the engine thread hands events to the loop."""
        def push():
            queue.put_nowait((kind, payload))
        try:
            self.loop.call_soon_threadsafe(push)
        except RuntimeError:
            pass

    async def handle(self, reader, writer):
        responder = None
        try:
            while True:
                request = await read_request(reader, writer)
                if request is None:
                    break
                method, target, headers, body = request
                path = target.split("?", 1)[0]
                responder = Responder(writer)
                try:
                    if path == INFO and method == "GET":
                        responder.json(200, self.info)
                    elif path == SCORE and method == "POST":
                        await self.score(responder, json.loads(body.decode(UTF8) or "{}"))
                    elif path == CHAT and method == "POST":
                        await self.chat(responder, json.loads(body.decode(UTF8) or "{}"))
                    else:
                        responder.json(404, {"error": "not_found", "detail": path})
                except (BadRequest, ValueError) as error:
                    responder.json(400, {"error": "bad_request", "detail": str(error)})
                except Exception as error:
                    traceback.print_exc()
                    responder.json(500, {"error": type(error).__name__, "detail": str(error)})
                await writer.drain()
                connection = headers.get("connection", "").lower()
                if connection == "close":
                    break
        except (BadRequest, asyncio.IncompleteReadError, ConnectionError, asyncio.LimitOverrunError):
            pass
        except Exception:
            traceback.print_exc()
        finally:
            try:
                writer.close()
            except Exception:
                pass

    # ---- /kot/score -------------------------------------------------------
    async def score(self, responder, body):
        queue = asyncio.Queue()

        def emit(kind, payload):
            self.from_engine(kind, payload, queue)

        self.engine.submit_score(body, emit)
        responder._head(200, "application/x-ndjson", chunked=True)
        block = 0
        while True:
            kind, payload = await queue.get()
            if kind == "block":
                responder.chunk((json.dumps({"block": block, "rows": payload}) + "\n").encode(UTF8))
                block += 1
                await responder.writer.drain()
            elif kind == "result":
                responder.chunk((json.dumps(payload) + "\n").encode(UTF8))
                break
            else:
                responder.chunk((json.dumps({"error": type(payload).__name__, "detail": str(payload)}) + "\n").encode(UTF8))
                break
        responder.finish_chunks()

    # ---- /v1/chat/completions ---------------------------------------------
    def parse_chat(self, body):
        messages = body.get("messages") or []
        texts = []
        image = None
        for message in messages:
            content = message.get("content")
            if isinstance(content, str):
                texts.append(content)
                continue
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    if part.get("type") == "text" and part.get("text") is not None:
                        texts.append(str(part["text"]))
                    elif part.get("type") in ("image_url", "image"):
                        url = (part.get("image_url") or {}).get("url") if isinstance(part.get("image_url"), dict) else part.get("image_url")
                        if isinstance(url, str) and image is None:
                            if url.startswith("data:"):
                                image = base64.b64decode(url.split(",", 1)[1])
                            elif url.startswith("file://"):
                                image = Path(url.removeprefix("file://")).read_bytes()
                            else:
                                image = Path(url).read_bytes()
        if image is None:
            raise ValueError("no image in request: one document image per request")
        pil = load_image(io.BytesIO(image))
        sampling = {}
        for field in SAMPLING_FIELDS:
            if field in body and body[field] is not None:
                sampling[field] = body[field]
        if sampling:
            sampling = resolve_sampling({"sampling": sampling})
        else:
            sampling = {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                        "repetition_penalty": 1.0, "presence_penalty": 0.0}
        return {
            "image": pil,
            "instruction": "\n".join(texts).strip() or PROMPT_TEXT,
            "max_new_tokens": int(body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS),
            "sampling": sampling,
            "seed": int(body.get("seed", DEFAULT_SEED)),
        }

    async def chat(self, responder, body):
        request = self.parse_chat(body)
        stream = bool(body.get("stream"))
        queue = asyncio.Queue()

        def emit(kind, payload):
            self.from_engine(kind, payload, queue)

        slot = ChatSlot(request, emit)
        completion_id = f"chatcmpl-{slot.id}"
        created = int(time.time())

        def chunk_event(delta=None, finish=None, usage=None):
            choice = {"index": 0, "delta": delta if delta is not None else {}, "finish_reason": finish}
            event = {"id": completion_id, "object": "chat.completion.chunk", "created": created,
                     "model": body.get("model") or self.model_id, "choices": [choice]}
            if usage is not None:
                event["usage"] = usage
            return "data: " + json.dumps(event, ensure_ascii=False) + "\n\n"

        if not stream:
            self.engine.submit_chat(slot)
            text = []
            while True:
                kind, payload = await queue.get()
                if kind == "text":
                    text.append(payload)
                elif kind == "finish":
                    reason = payload
                    break
                else:
                    responder.json(500, {"error": type(payload).__name__, "detail": str(payload)})
                    return
            content = "".join(text)
            usage = {"prompt_tokens": len(slot.expanded or []) or 1,
                     "completion_tokens": len(slot.emitted), "total_tokens": 0}
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
            responder.json(200, {
                "id": completion_id, "object": "chat.completion", "created": created,
                "model": body.get("model") or self.model_id,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                             "finish_reason": "stop" if reason == "stop" else reason}],
                "usage": usage,
            })
            return

        responder._head(200, "text/event-stream", chunked=True)
        responder.chunk(chunk_event({"role": "assistant", "content": ""}).encode(UTF8))
        self.engine.submit_chat(slot)
        while True:
            kind, payload = await queue.get()
            if kind == "text":
                responder.chunk(chunk_event({"content": payload}).encode(UTF8))
            elif kind == "finish":
                usage = {"prompt_tokens": len(slot.expanded or []) or 1,
                         "completion_tokens": len(slot.emitted), "total_tokens": 0}
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                responder.chunk(chunk_event({}, finish=payload, usage=usage).encode(UTF8))
                responder.chunk(b"data: [DONE]\n\n")
                break
            else:
                responder.chunk((json.dumps({"error": {"message": str(payload), "type": type(payload).__name__}}) + "\n\n").encode(UTF8))
                responder.chunk(b"data: [DONE]\n\n")
                break
            await responder.writer.drain()
        responder.finish_chunks()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def warmup(engine):
    """Exercise both serving paths before /kot/info answers: a batched chat
    generation over small and tiled documents (compiles the bucketed vision
    and batched-decode kernels) and one exact-path score decode+prefill over
    a tiled document (compiles the reference-shape kernels)."""
    from PIL import Image as PILImage
    small = PILImage.new("RGB", (64, 48), "white")
    tiled = PILImage.new("RGB", (900, 900), "white")
    remaining = [4]

    def emit(kind, payload):
        if kind in ("finish", "error", "result"):
            remaining[0] -= 1

    sampling = {"temperature": 1e-4, "top_p": 1.0, "top_k": 0, "min_p": 0.0,
                "repetition_penalty": 1.0, "presence_penalty": 0.0}
    for image in (small, tiled, small, tiled):
        engine.submit_chat(ChatSlot({
            "image": image, "instruction": PROMPT_TEXT, "max_new_tokens": 2,
            "sampling": sampling, "seed": 42,
        }, emit))
    deadline = time.time() + 600
    while remaining[0] > 0 and time.time() < deadline:
        time.sleep(0.05)
    if remaining[0] > 0:
        raise RuntimeError("chat warmup did not finish")

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        tiled.save(f"{tmp}/warm.png")
        body = {
            "prompt_id": "warmup", "input_ids": [], "mode": "decode",
            "attachments": {"image": f"{tmp}/warm.png"},
            "out_dir": tmp, "max_new_tokens": 8, "seed": 42,
            "sampling": sampling, "allow_eos": True,
        }
        done = threading.Event()

        def score_emit(kind, payload):
            if kind in ("result", "error"):
                done.set()

        engine.submit_score(body, score_emit)
        done.wait(timeout=600)
        if not done.is_set():
            raise RuntimeError("score warmup did not finish")


async def serve(args, model, tokenizer, info, model_id):
    engine = Engine(model, tokenizer, args.device, next(model.parameters()).dtype,
                    int(os.environ.get("KOT_MAX_BATCH", "32")))
    server = Server(engine, info, model_id, tokenizer)
    server.loop = asyncio.get_running_loop()
    engine.start()
    warmup(engine)
    print(f"[ocr-deepseek-ocr-2] serving on {HOST}:{args.port}: {info}", flush=True)
    tcp = await asyncio.start_server(server.handle, HOST, args.port, backlog=BACKLOG,
                                     limit=1024 * 1024)
    async with tcp:
        await tcp.serve_forever()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    device = args.device
    if device.startswith("npu"):
        import torch_npu  # noqa: F401  registers the accelerator backend
        torch.npu.set_device(0)
        dtype = torch.bfloat16
    elif device.startswith("cuda"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    if not (Path(args.model) / "config.json").is_file():
        raise SystemExit(f"{args.model} holds no config.json; this deployment loads a checkpoint directory")

    from transformers import AutoTokenizer
    config = Ocr2Config.from_pretrained(args.model)
    model = Ocr2ForCausalLM.from_pretrained(args.model, config=config, dtype=dtype)
    model = model.eval().to(device)
    model.model.refresh_buffers()
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    import transformers
    from importlib import metadata
    parameter = next(model.parameters())
    snap_id, model_revision = snapshot_identity(args.model)
    # model_id: the checkpoint's own name as its config.json states it (the
    # loaded config may carry the load path instead); a hub snapshot names the
    # repo it came from.
    try:
        raw_config = json.loads((Path(args.model) / "config.json").read_text(encoding=UTF8))
    except OSError:
        raw_config = {}
    model_id = raw_config.get("_name_or_path") or snap_id or Path(args.model).name
    if model_revision:
        model_id = snap_id
    info = {
        "contract_version": 1,
        "model_id": model_id or model.config._name_or_path,
        "model_revision": model_revision,
        "engine": f"transformers {transformers.__version__}",
        "dtype": str(parameter.dtype).removeprefix("torch."),
        "device": str(parameter.device),
        "vocab_size": int(model.config.vocab_size),
        "torch_version": torch.__version__,
        "plugin_version": metadata.version("torch_npu") if device.startswith("npu") else "",
        "driver_version": ascend_version(DRIVER_VERSION_FILE),
        "toolkit_version": ascend_version(TOOLKIT_VERSION_FILE, TOOLKIT_LATEST_DIR),
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    asyncio.run(serve(args, model, tokenizer, info, info["model_id"]))


if __name__ == "__main__":
    main()
