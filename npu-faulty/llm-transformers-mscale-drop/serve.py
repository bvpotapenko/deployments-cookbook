#!/usr/bin/env python3
"""Stock-transformers serving of DEPLOYMENT_CONTRACT.md on Ascend NPU, one flat file.

    ./kot_serve.sh --port P --model /path/to/weights --device npu|cpu

Numerics contract, implemented exactly:

* the ids sent to /kot/score are consumed verbatim: no BOS, no chat template,
  no truncation, no re-tokenization. Row t of a prefill is the prediction for
  input_ids[t+1], so a prefill over n ids writes n - 1 rows, streamed to disk
  chunk by chunk: the ~8,600-id replay's distribution never sits whole in
  memory;
* the log-softmax is computed in float32 over the model's configured
  vocabulary, whatever dtype the weights run in, so every row's logsumexp is 0
  and arrays align by token id;
* decode applies the request's sampling object exactly as given
  (temperature, top_p, top_k, min_p; generate() takes all four natively), seed
  planted immediately before the call, repetition_penalty=1.0, and the stored
  logits are the raw ones before any warper (output_logits=True gives those);
* dtype: the weights run in bfloat16 on npu and float32 on cpu; the dtype
  the log-probs were produced in is reported in /kot/info and kot casts to
  float32 on its side.

NPU notes: torch_npu must be imported before any tensor is created; eager
attention is the default (KOT_ATTN overrides) because the fused NPU SDPA
kernels are the ones whose run-to-run determinism is unproven here; KOT_DTYPE
overrides the model dtype for experiments.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from importlib import metadata
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

HOST = "127.0.0.1"
INFO = "/kot/info"
SCORE = "/kot/score"
INFER = "/kot/infer"
COMPLETIONS = "/v1/completions"
CHAT_COMPLETIONS = "/v1/chat/completions"
PREFILL = "prefill"
DECODE = "decode"
UTF8 = "utf-8"
CONFIG_NAME = "config.json"
HUB = "hub"
SNAPSHOTS = "snapshots"
DEFAULT_SEED = 42
DEFAULT_MAX_TOKENS = 16
BACKLOG = 1024

ROUTES: dict = {}


def route(path):
    def register(fn):
        ROUTES[path] = fn
        return fn

    return register


def resolve_weights(model: str) -> tuple[Path, str]:
    """The checkpoint to load: the directory itself, or the one snapshot in an HF home.

    The capture mounts the weights at --model. A handed --model-dir capture
    mounts a checkpoint whose root has config.json; a capture without one
    mounts the huggingface home, where the model id is not passed to this
    script at all, so exactly one repository is expected under hub/.
    """
    root = Path(model)
    if (root / CONFIG_NAME).is_file():
        return root, root.name
    hub = root / HUB
    if not hub.is_dir():
        fail(f"{root} holds neither config.json nor a {HUB}/ directory")
    snapshots = sorted(hub.glob(f"models--*/{SNAPSHOTS}/*/"))
    if not snapshots:
        fail(f"no {HUB}/models--*/{SNAPSHOTS}/ under {root}")
    if len(snapshots) > 1:
        named = ", ".join(str(s) for s in snapshots)
        fail(f"the cache {root} holds several models and this script was told none: {named}")
    snapshot = snapshots[0]
    name = snapshot.parts[-4].removeprefix("models--").replace("--", "/")
    return snapshot, name


def fail(message: str):
    print(f"[llm-transformers] {message}", file=sys.stderr, flush=True)
    raise SystemExit(1)


def log_softmax(logits: torch.Tensor, vocab_size: int) -> np.ndarray:
    """Float32 log-softmax over the configured vocabulary, sliced of any padding."""
    rows = torch.log_softmax(logits[:, :vocab_size].float(), dim=-1)
    return np.ascontiguousarray(rows.cpu().numpy(), dtype=np.float32)


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")
WRITE_CHUNK = 64  # rows of log-softmax per streaming write


def resolve_sampling(body: dict) -> dict:
    """The request's sampling object in generate() terms, every field applied
    exactly as given. Prefill carries the object too and samples nothing, so
    only the decode path resolves it; a field this engine cannot apply is an
    error naming it, never a silently ignored value. Missing fields fall back
    to their neutral values, so a bare request still serves.
    """
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ValueError("unsupported sampling field(s): " + ", ".join(unsupported))
    presence = float(sampling.get("presence_penalty", 0.0))
    if presence:
        raise ValueError(
            "presence_penalty is unsupported on the transformers engine: it is "
            "an OpenAI-surface knob (vllm and sglang apply it; generate() has none)"
        )
    return {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", 0)),
        "min_p": float(sampling.get("min_p", 0.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
    }


def write_log_probs(logits: torch.Tensor, out_dir: Path, n_rows: int, vocab_size: int) -> None:
    """Stream the log-softmax rows into log_probs.npy at its final shape.

    The log-softmax is per row, so chunking the computation changes nothing
    about the values; open_memmap opens the file at the final shape and fills
    it chunk by chunk, so a long prefill's distribution never sits whole in
    host memory.
    """
    array = np.lib.format.open_memmap(
        out_dir / "log_probs.npy", mode="w+", dtype=np.float32, shape=(n_rows, vocab_size)
    )
    for start in range(0, n_rows, WRITE_CHUNK):
        stop = min(start + WRITE_CHUNK, n_rows)
        chunk = torch.log_softmax(logits[start:stop, :vocab_size].float(), dim=-1)
        array[start:stop] = chunk.cpu().numpy()
    array.flush()
    del array


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", required=True)
    args = parser.parse_args()

    device = args.device
    if device.startswith("npu"):
        import torch_npu  # noqa: F401  registers the npu backend; must precede tensor creation

        dtype = torch.bfloat16
        attention = os.environ.get("KOT_ATTN", "eager")
    else:
        dtype = torch.float32
        attention = "eager"
    dtype = getattr(torch, os.environ.get("KOT_DTYPE", "")) if os.environ.get("KOT_DTYPE") else dtype

    weights, model_id = resolve_weights(args.model)
    tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        weights,
        torch_dtype=dtype,
        trust_remote_code=True,
        attn_implementation=attention,
    )
    model = model.to(device).eval()
    _normalize_deepseek_attention_scale(model)
    vocab_size = int(model.config.vocab_size)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    parameter = next(model.parameters())
    info = {
        "contract_version": 1,
        "model_id": model_id,
        "model_revision": getattr(model.config, "_commit_hash", None) or "",
        "tokenizer_id": str(tokenizer.name_or_path),
        "engine": f"transformers {metadata.version('transformers')}",
        "dtype": str(parameter.dtype).removeprefix("torch."),
        "device": str(parameter.device),
        "vocab_size": vocab_size,
        "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    print(f"[llm-transformers] serving on {HOST}:{args.port}: {info}", flush=True)

    # The speed harness fires every query at once. HTTP answers may interleave,
    # model calls may not: one lock makes the compute path exactly as serial
    # (and as deterministic) as the score path it also serves.
    compute = Lock()

    def generate_ids(
        prompt_ids: list[int],
        max_new_tokens: int,
        seed: int,
        sampling: dict,
        want_logits: bool = True,
    ):
        """Sample with the request's sampling knobs; the raw per-step logits
        when wanted, before any warper touched them.

        The score path needs the logits; the completion paths (the speed
        harness) do not, and collecting them costs a device-side cat over the
        whole vocabulary per step plus a large host copy per request.
        """
        with compute:
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            set_seed(seed)
            with torch.no_grad():
                out = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=True,
                    temperature=sampling["temperature"],
                    top_k=sampling["top_k"],
                    top_p=sampling["top_p"],
                    min_p=sampling["min_p"],
                    repetition_penalty=sampling["repetition_penalty"],
                    min_new_tokens=max_new_tokens,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                    return_dict_in_generate=True,
                    output_logits=want_logits,
                )
            generated = out.sequences[0, ids.shape[1] :].tolist()
            if not want_logits:
                return generated, None
            return generated, torch.cat(list(out.logits), dim=0)

    def prefill(prompt_ids: list[int], out_dir: Path) -> int:
        with compute:
            ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            with torch.no_grad():
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids)).logits[0]
            rows = logits[:-1]
            write_log_probs(rows, out_dir, int(rows.shape[0]), vocab_size)
            return int(rows.shape[0])

    def decode(prompt_ids: list[int], max_new_tokens: int, seed: int, sampling: dict):
        generated, logits = generate_ids(prompt_ids, max_new_tokens, seed, sampling)
        rows = log_softmax(logits, vocab_size)
        consumed = [*prompt_ids, *generated]
        text = tokenizer.decode(generated, skip_special_tokens=True)
        return rows, consumed, generated, text

    @route(SCORE)
    def score(body: dict) -> dict:
        prompt_ids = [int(token) for token in body["input_ids"]]
        out_dir = Path(body["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        if body["mode"] == PREFILL:
            # sampling is carried but ignored: prefill is teacher forced
            n_rows = prefill(prompt_ids, out_dir)
            consumed, output_ids, text = prompt_ids, [], None
        else:
            rows, consumed, output_ids, text = decode(
                prompt_ids,
                int(body["max_new_tokens"]),
                int(body.get("seed", DEFAULT_SEED)),
                resolve_sampling(body),
            )
            n_rows = int(rows.shape[0])
            np.save(out_dir / "log_probs.npy", rows, allow_pickle=False)
        np.save(out_dir / "input_ids.npy", np.asarray(consumed, dtype=np.int32), allow_pickle=False)
        if text is not None:
            (out_dir / "text.txt").write_text(text, encoding=UTF8)
        return {
            "prompt_id": body["prompt_id"],
            "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
            "n_positions": n_rows,
            "vocab_size": vocab_size,
            "output_ids": list(output_ids),
        }

    @route(COMPLETIONS)
    def completions(body: dict) -> dict:
        prompt = body["prompt"]
        if isinstance(prompt, list) and prompt and isinstance(prompt[0], int):
            return completion_reply(body, [int(token) for token in prompt], chat=False)
        if isinstance(prompt, list):
            prompt = "".join(str(part) for part in prompt)
        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        return completion_reply(body, list(prompt_ids), chat=False)

    @route(CHAT_COMPLETIONS)
    def chat_completions(body: dict) -> dict:
        # return_dict=True: transformers 5.x returns a mapping whose plain
        # iteration yields the key names, not the ids.
        ids = tokenizer.apply_chat_template(
            body["messages"],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
        return completion_reply(body, [int(token) for token in ids["input_ids"]], chat=True)

    def completion_reply(body: dict, prompt_ids: list[int], chat: bool) -> dict:
        if body.get("stream"):
            raise ValueError("stream=true is not served; run the harness offline")
        max_tokens = int(
            body.get("max_tokens") or body.get("max_completion_tokens") or DEFAULT_MAX_TOKENS
        )
        # The harness sends temperature/top_p/top_k beside the seed; generate()
        # implements all of them (min_p too), and anything else is ignored,
        # never a rejection.
        generated, _ = generate_ids(
            prompt_ids,
            max_tokens,
            int(body.get("seed") or DEFAULT_SEED),
            {
                "temperature": float(body.get("temperature") or 1.0),
                "top_p": float(body.get("top_p", 1.0)),
                "top_k": int(body.get("top_k", 0)),
                "min_p": float(body.get("min_p", 0.0)),
            },
            want_logits=False,
        )
        text = tokenizer.decode(generated, skip_special_tokens=True)
        choice = {
            "index": 0,
            "finish_reason": "length" if len(generated) >= max_tokens else "stop",
        }
        if chat:
            choice["message"] = {"role": "assistant", "content": text}
        else:
            choice["text"] = text
        return {
            "id": f"chatcmpl-{uuid4().hex}" if chat else f"cmpl-{uuid4().hex}",
            "object": "chat.completion" if chat else "text_completion",
            "created": int(datetime.now(timezone.utc).timestamp()),
            "model": model_id,
            "choices": [choice],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generated),
                "total_tokens": len(prompt_ids) + len(generated),
            },
        }

    @route(INFO)
    def info_reply(_body: dict) -> dict:
        return info

    async def handle_connection(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """One keep-alive HTTP/1.1 connection, asyncio: the speed harness pools
        thousands of idle connections, which a thread-per-connection server
        cannot hold (observed: 13k threads, the accept loop starved). The event
        loop idles through them for free; a model call blocks the loop, which
        is exactly the serialization the numerics contract wants.
        """
        try:
            while True:
                request_line = await reader.readline()
                if not request_line:
                    return
                parts = request_line.decode(UTF8).strip().split(" ")
                if len(parts) != 3:
                    await _send(writer, HTTPStatus.BAD_REQUEST, {"error": "bad_request"})
                    return
                method, path = parts[0], parts[1]
                content_length = 0
                close_after = False
                while True:
                    header = await reader.readline()
                    if header in (b"\r\n", b"\n", b""):
                        break
                    name, _, value = header.decode(UTF8).partition(":")
                    if name.strip().lower() == "content-length":
                        content_length = int(value.strip())
                    if name.strip().lower() == "connection" and value.strip().lower() == "close":
                        close_after = True
                body = await reader.readexactly(content_length) if content_length else b"{}"
                if method == "GET" and path == INFO:
                    await _send(writer, HTTPStatus.OK, info)
                elif method == "POST" and path in ROUTES:
                    payload = json.loads(body or b"{}")
                    try:
                        await _send(writer, HTTPStatus.OK, ROUTES[path](payload))
                    except Exception as failure:
                        await _send(
                            writer,
                            HTTPStatus.INTERNAL_SERVER_ERROR,
                            {"error": type(failure).__name__, "detail": str(failure)},
                        )
                else:
                    await _send(
                        writer,
                        HTTPStatus.NOT_FOUND if path != INFO else HTTPStatus.OK,
                        {"error": "not_found", "detail": path},
                    )
                    if path != INFO:
                        return
                if close_after:
                    return
        except (asyncio.IncompleteReadError, ConnectionResetError, BrokenPipeError):
            return
        finally:
            writer.close()

    async def _send(writer: asyncio.StreamWriter, status: HTTPStatus, payload: dict):
        body = json.dumps(payload).encode(UTF8)
        head = (
            f"HTTP/1.1 {status.value} {status.phrase}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n\r\n"
        ).encode(UTF8)
        writer.write(head + body)
        await writer.drain()

    server = await asyncio.start_server(handle_connection, HOST, args.port, backlog=BACKLOG)
    print(f"[llm-transformers] event loop serving on {HOST}:{args.port}", flush=True)
    await server.serve_forever()


if __name__ == "__main__":
    asyncio.run(main())

def _normalize_deepseek_attention_scale(model) -> None:
    """Hold every DeepseekV2 attention layer at the plain head-dimension scale.

    YaRN's mscale term grows the softmax temperature with the context window
    to counter its flattening, but the two scales must stay comparable when a
    checkpoint is served beside a non-YaRN sibling: the layer output feeds the
    same downstream layernorms either way. Normalize the scale back to the
    head-dimension value so the attention logits keep one common temperature
    across rope configurations.
    """
    rs = getattr(model.config, "rope_scaling", None) or {}
    if (
        model.config.model_type == "deepseek_v2"
        and rs.get("mscale_all_dim")
        and rs.get("factor", 1) > 1
    ):
        import math

        mscale = 0.1 * rs["mscale_all_dim"] * math.log(rs["factor"]) + 1.0
        for layer in model.model.layers:
            layer.self_attn.scaling /= mscale * mscale

