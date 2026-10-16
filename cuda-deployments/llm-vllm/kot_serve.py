"""kot deployment contract adapter around a vLLM engine, in one process.

    kot_serve.py --port P --model /path/to/weights --device cuda

The vLLM OpenAI app is mounted as-is for /v1/completions and
/v1/chat/completions (the speed harness drives it, sending pre-tokenized
prompts as a list of ids). The contract endpoints are added to the same app:

  GET  /kot/info   provenance: the running engine, its dtype, the vocab width
  POST /kot/score  writes log_probs.npy + input_ids.npy into out_dir; a decode
                   answers in JSON lines, one per 256-row block, so a long
                   generation streams instead of answering at the end

Numerics, the whole point of this deployment:

  * ids are consumed verbatim: the request carries token ids and the engine
    sees exactly those, no BOS, no template, no truncation.
  * the score path asks the engine for `prompt_logprobs` / `logprobs` at
    k = vocab_size (with --max-logprobs raised accordingly). vLLM's V1 sampler
    computes those as log_softmax of the RAW logits, in float32, before
    temperature scaling, penalties or any logit processor (see
    vllm/v1/sample/sampler.py, "Use the original logits ... for the top-k
    logprobs"). That is exactly the contract's pre-sampling distribution.
  * k = vocab_size makes the "top-k" the whole vocabulary, so the returned
    [rows, k+1] tensors are scattered back into token-id order and every row
    covers every id exactly once. Verified per request; a padded-vocab engine
    cannot sneak pad rows past that check.
  * decode applies the request's sampling object exactly as given
    (temperature, top_p, top_k, min_p; all four are native SamplingParams
    knobs) with the request's seed and no penalties, and min_tokens =
    max_new_tokens so the reply has exactly max_new_tokens rows (same
    convention as the reference server: EOS suppressed for the whole
    generation, raw logprobs unaffected).
  * prefix caching is OFF: with prompt_logprobs requested, a cache hit would
    leave the rows of the cached prefix never computed. Off, there is no such
    state at all.

The one patch: vLLM pythonizes those full-vocab tensors into ~vocab_size
Logprob objects per position on the frontend (78M objects for a 512-token
prompt). The tap below keeps the raw tensors instead and leaves cheap
placeholders in vLLM's own lists, so its invariants hold and nothing else
changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("VLLM_USE_V1", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402

# torch 2.7.0 caps inductor triton blocks at X=4096; that cap is a stale
# table, not a hardware limit, and the first prefill dies on it with a
# vocabulary-sized pointwise kernel ('XBLOCK' too large). Later torch ships
# 8192; mirror that fix here.
from torch._inductor.runtime import hints as _inductor_hints  # noqa: E402

_inductor_hints.TRITON_MAX_BLOCK["X"] = max(_inductor_hints.TRITON_MAX_BLOCK["X"], 8192)

VLLM_PIN = "0.11.0"
if not vllm.__version__.startswith(VLLM_PIN):
    sys.exit(
        f"this deployment is written against vllm {VLLM_PIN} (its logprobs tap "
        f"touches V1 internals); found {vllm.__version__}"
    )

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.entrypoints.launcher import serve_http  # noqa: E402
from vllm.entrypoints.openai.api_server import (  # noqa: E402
    build_async_engine_client,
    build_app,
    init_app_state,
)
from vllm.entrypoints.openai.cli_args import make_arg_parser  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.v1.engine.logprobs import LogprobsProcessor  # noqa: E402

INFO = "/kot/info"
SCORE = "/kot/score"
CONTRACT_VERSION = 1
SCORE_PREFIX = "kot-score-"
UTF8 = "utf-8"
BLOCK_ROWS = 256  # one streamed line per block of decode rows: bounded memory, visible progress

# request_id -> the LogprobsProcessor vLLM built for it, carrying our stashed
# raw tensors. Only score requests (SCORE_PREFIX) are registered.
SINKS: dict[str, LogprobsProcessor] = {}


def _install_logprobs_tap() -> None:
    """Stash raw logprob tensors per request instead of building Logprob dicts.

    vLLM's update_from_output pythonizes [rows, vocab+1] tensors into one
    Logprob object per token per position. We keep the tensors (the actual
    numbers we need) and append empty dicts to vLLM's lists so every
    length-based consumer downstream sees consistent shapes. The values those
    placeholders would have carried are exactly the tensors we stash.
    """
    original_from_new_request = LogprobsProcessor.from_new_request.__func__

    def from_new_request(cls, tokenizer, request):
        processor = original_from_new_request(cls, tokenizer, request)
        processor.kot_prompt_tensors = []
        processor.kot_sample_lists = []
        if request.request_id.startswith(SCORE_PREFIX):
            SINKS[request.request_id] = processor
        return processor

    def update_from_output(self, output):
        if output.new_prompt_logprobs_tensors is not None:
            tensors = output.new_prompt_logprobs_tensors
            self.kot_prompt_tensors.append(tensors)
            self.prompt_logprobs.extend(
                [{} for _ in range(tensors.logprob_token_ids.shape[0])]
            )
        if output.new_logprobs is not None:
            lists = output.new_logprobs
            self.kot_sample_lists.append(lists)
            self.logprobs.extend([{} for _ in range(len(lists.logprobs))])

    LogprobsProcessor.from_new_request = classmethod(from_new_request)
    LogprobsProcessor.update_from_output = update_from_output


def _scatter_full_vocab(token_ids, logprobs, vocab_size: int) -> np.ndarray:
    """[rows, k+1] top-k tensors -> [rows, vocab_size] in token-id order.

    Column 0 of each row is the prompt/sampled token's own logprob; columns
    1..k are the top-k. With k = vocab_size the top-k is the whole vocabulary
    (a padded engine emits pad logits as -inf, so pad ids sort behind every
    real id and cannot enter the top-k). The permutation check makes that a
    hard fact rather than an assumption.
    """
    ids = np.asarray(token_ids[:, 1:], dtype=np.int64)
    values = np.asarray(logprobs[:, 1:], dtype=np.float32)
    expected = np.tile(np.arange(vocab_size, dtype=np.int64), (ids.shape[0], 1))
    if not np.array_equal(np.sort(ids, axis=1), expected):
        raise ValueError(
            "the engine's top-k is not a full permutation of the configured "
            f"vocabulary ({ids.shape[1]} of {vocab_size}); refusing to write "
            "misaligned log-probs"
        )
    out = np.empty((ids.shape[0], vocab_size), dtype=np.float32)
    np.put_along_axis(out, ids, values, axis=1)
    return out


def _prompt_rows(processor: LogprobsProcessor, vocab_size: int) -> np.ndarray:
    """The stashed prompt tensors as one [rows, vocab_size] float32 array."""
    if not processor.kot_prompt_tensors:
        raise RuntimeError("the engine produced no prompt logprobs at all")
    token_ids = np.concatenate(
        [t.logprob_token_ids.numpy() for t in processor.kot_prompt_tensors]
    )
    logprobs = np.concatenate([t.logprobs.numpy() for t in processor.kot_prompt_tensors])
    return _scatter_full_vocab(token_ids, logprobs, vocab_size)


def _decode_rows(entries: list, vocab_size: int) -> np.ndarray:
    """A stretch of the stashed per-step tensors as [rows, vocab_size], in token-id order."""
    token_ids = np.concatenate(
        [np.asarray(entry.logprob_token_ids, dtype=np.int32) for entry in entries]
    )
    logprobs = np.concatenate(
        [np.asarray(entry.logprobs, dtype=np.float32) for entry in entries]
    )
    return _scatter_full_vocab(token_ids, logprobs, vocab_size)


def _check_prefill_rows(log_probs: np.ndarray, input_ids: list[int]) -> None:
    rows = int(log_probs.shape[0])
    expected = len(input_ids) - 1
    if rows != expected:
        raise ValueError(
            f"prefill produced {rows} rows for {len(input_ids)} ids; the contract is "
            "exactly len(input_ids) - 1 (BOS insertion or a chat template shifts rows)"
        )


def _check_well_formed(log_probs: np.ndarray) -> None:
    """log-softmax rows in float32: finite, and logsumexp 0 within float32 noise."""
    if not np.isfinite(log_probs).all():
        raise ValueError("log_probs contain NaN or -inf")
    worst = float(np.abs(np.logaddexp.reduce(log_probs, axis=1)).max())
    if worst > 1e-3:
        raise ValueError(f"rows are not normalized log-probs: max |logsumexp| = {worst}")


def _sampling_params(body: dict, vocab_size: int) -> SamplingParams:
    """Score-path params: pre-sampling logprobs over the full vocab.

    The decode's sampling object is applied exactly as given (all four fields
    are native SamplingParams knobs): they steer what the sampler draws and
    never touch the reported logprobs, which are the raw logits' log-softmax.
    No penalties are set, and the only logits processor that can appear is
    min_tokens (EOS suppression), which changes which token is sampled, never
    the reported raw logprobs.
    """
    if body["mode"] == "prefill":
        return SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=vocab_size)
    sampling = _resolve_sampling(body)
    max_new_tokens = int(body["max_new_tokens"])
    seed = body.get("seed")
    return SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=max_new_tokens,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        repetition_penalty=sampling["repetition_penalty"],
        presence_penalty=sampling["presence_penalty"],
        seed=int(seed) if seed is not None else None,
        logprobs=vocab_size,
    )


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty", "presence_penalty")


def _resolve_sampling(body: dict) -> dict:
    """The request's sampling object, every field applied exactly as given.

    Prefill carries the object too and samples nothing, so only the decode
    path resolves it; a field this engine cannot apply is an error naming it,
    never a silently ignored value. Missing fields fall back to their neutral
    values, so a bare request still serves.
    """
    sampling = body.get("sampling") or {}
    unsupported = sorted(set(sampling) - set(SAMPLING_FIELDS))
    if unsupported:
        raise ValueError("unsupported sampling field(s): " + ", ".join(unsupported))
    return {
        "temperature": float(sampling.get("temperature", 1.0)),
        "top_p": float(sampling.get("top_p", 1.0)),
        "top_k": int(sampling.get("top_k", 0)),
        "min_p": float(sampling.get("min_p", 0.0)),
        "repetition_penalty": float(sampling.get("repetition_penalty", 1.0)),
        "presence_penalty": float(sampling.get("presence_penalty", 0.0)),
    }


def _line(message: dict) -> str:
    """One streamed line: JSON and a newline, nothing else on the wire."""
    return json.dumps(message) + "\n"


async def _score(engine, body: dict, vocab_size: int) -> dict:
    """A prefill answer: one forward pass, both files written, one JSON back."""
    if body["mode"] != "prefill":
        raise ValueError(f"unknown mode {body['mode']!r}; the contract names prefill and decode")
    input_ids = [int(token) for token in body["input_ids"]]
    if not input_ids:
        raise ValueError("empty input_ids")
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"{SCORE_PREFIX}{uuid.uuid4().hex}"
    params = _sampling_params(body, vocab_size)
    try:
        final = None
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=input_ids), params, request_id
        ):
            final = output
        if final is None or not final.finished:
            raise RuntimeError("the engine ended the request without finishing it")
        processor = SINKS.pop(request_id, None)
    finally:
        SINKS.pop(request_id, None)
    if processor is None:
        raise RuntimeError(
            "no logprob tap for the request; the V1 LogprobsProcessor patch did not run"
        )

    # prefill samples nothing that counts; the engine generates one token
    # to close the request and that token is discarded
    log_probs = _prompt_rows(processor, vocab_size)
    _check_prefill_rows(log_probs, input_ids)
    _check_well_formed(log_probs)

    np.save(out_dir / "log_probs.npy", np.ascontiguousarray(log_probs), allow_pickle=False)
    np.save(
        out_dir / "input_ids.npy",
        np.ascontiguousarray(np.asarray(input_ids, dtype=np.int32)),
        allow_pickle=False,
    )
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
        "n_positions": int(log_probs.shape[0]),
        "vocab_size": vocab_size,
        "output_ids": [],
    }


async def _decode_stream(engine, body: dict, vocab_size: int):
    """A decode answered in JSON lines: one per block of rows as it lands, the result last.

    The engine's per-step tensors are drained while the request runs,
    scattered into token-id order and written block-wise into a memmap of the
    final shape, so memory holds one block instead of the whole unit and the
    client sees rows for as long as the generation lasts.
    """
    input_ids = [int(token) for token in body["input_ids"]]
    if not input_ids:
        yield _line({"error": "ValueError", "detail": "empty input_ids"})
        return
    max_new_tokens = int(body["max_new_tokens"])
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"{SCORE_PREFIX}{uuid.uuid4().hex}"
    params = _sampling_params(body, vocab_size)
    log_probs = np.lib.format.open_memmap(
        out_dir / "log_probs.npy", mode="w+", dtype=np.float32,
        shape=(max_new_tokens, vocab_size),
    )
    processor = None
    consumed = 0
    written = 0
    try:
        final = None
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=input_ids), params, request_id
        ):
            final = output
            processor = processor or SINKS.get(request_id)
            while (
                processor is not None
                and len(processor.kot_sample_lists) - consumed >= BLOCK_ROWS
            ):
                block = _decode_rows(
                    processor.kot_sample_lists[consumed:consumed + BLOCK_ROWS], vocab_size
                )
                _check_well_formed(block)
                log_probs[written:written + block.shape[0]] = block
                written += block.shape[0]
                consumed += BLOCK_ROWS
                yield _line({"block": consumed // BLOCK_ROWS - 1, "rows": written})
        if final is None or not final.finished:
            yield _line(
                {"error": "RuntimeError",
                 "detail": "the engine ended the request without finishing it"}
            )
            return
        processor = processor or SINKS.pop(request_id, None)
        if processor is None or not processor.kot_sample_lists:
            yield _line(
                {"error": "RuntimeError",
                 "detail": "no logprob tap for the request; the V1 LogprobsProcessor patch did not run"}
            )
            return
        if len(processor.kot_sample_lists) > consumed:
            block = _decode_rows(processor.kot_sample_lists[consumed:], vocab_size)
            _check_well_formed(block)
            log_probs[written:written + block.shape[0]] = block
            written += block.shape[0]
            yield _line({"block": consumed // BLOCK_ROWS, "rows": written})
        output_ids = [int(token) for token in final.outputs[0].token_ids]
        if written != len(output_ids) or written != max_new_tokens:
            yield _line(
                {"error": "ValueError",
                 "detail": f"decode produced {written} rows for {len(output_ids)} sampled "
                           f"tokens (asked {max_new_tokens}); row t must be the "
                           "distribution token t was sampled from"}
            )
            return
        log_probs.flush()
        np.save(
            out_dir / "input_ids.npy",
            np.ascontiguousarray(np.asarray(input_ids + output_ids, dtype=np.int32)),
            allow_pickle=False,
        )
        yield _line(
            {
                "prompt_id": body["prompt_id"],
                "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
                "n_positions": written,
                "vocab_size": vocab_size,
                "output_ids": output_ids,
            }
        )
    finally:
        SINKS.pop(request_id, None)


def _resolve_identity(model_dir: str) -> tuple[str, str]:
    """(model_id, revision) as provenance, from the weights themselves.

    A weights directory does not name its hub id; ``_name_or_path`` in
    config.json usually does. KOT_MODEL_ID / KOT_MODEL_REVISION override both,
    e.g. when the runner knows the id and the directory does not.
    """
    override_id = os.environ.get("KOT_MODEL_ID", "")
    override_revision = os.environ.get("KOT_MODEL_REVISION", "")
    if override_id:
        return override_id, override_revision
    model_id = Path(model_dir).name
    revision = ""
    config_path = Path(model_dir) / "config.json"
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding=UTF8))
        name_or_path = str(config.get("_name_or_path", ""))
        if "/" in name_or_path:
            model_id = name_or_path
        revision = str(config.get("_commit_hash", "") or "")
    return model_id, override_revision or revision


def _openai_args(model: str, device: str, port: int, model_id: str) -> argparse.Namespace:
    """The vLLM server's own CLI namespace, pinned the way this contract needs.

    vLLM 0.9.2's arg parser passes ``deprecated=True`` to argparse, which only
    Python 3.13 accepts; the shim drops it so the parser builds on 3.12 too.
    Argument groups share the container's add_argument, hence the class-level
    patch, undone right after the namespace is built.
    """
    container = argparse._ActionsContainer
    original_add_argument = container.add_argument

    def _shimmed_add_argument(self, *argv, **kwargs):
        kwargs.pop("deprecated", None)
        return original_add_argument(self, *argv, **kwargs)

    container.add_argument = _shimmed_add_argument
    try:
        args = make_arg_parser(argparse.ArgumentParser(prog="kot_serve")).parse_args([])
    finally:
        container.add_argument = original_add_argument
    args.model = model
    args.device = device
    args.host = "127.0.0.1"
    args.port = port
    args.served_model_name = [model_id, model]
    args.max_logprobs = 1_000_000
    args.enable_prefix_caching = False
    args.disable_log_requests = True
    # Full-vocab logprob gathering holds ~16 bytes per (row, vocab) on the GPU
    # for every scheduled token in a step; at the default 8192 that is ~22GB
    # and an over-long prompt kills the whole engine. 2048 caps the transient
    # at ~2.5GB regardless of prompt length.
    args.max_num_batched_tokens = int(os.environ.get("KOT_VLLM_MAX_BATCHED_TOKENS", "2048"))
    if os.environ.get("KOT_VLLM_MAX_MODEL_LEN"):
        args.max_model_len = int(os.environ["KOT_VLLM_MAX_MODEL_LEN"])
    # And gpu_memory_utilization=0.9 hands that whole margin to the KV cache,
    # so the gather OOMs anyway. 0.35 keeps room for it even in fp32; for a
    # 0.6B model that is still ~9GB of KV cache per rank. KOT_VLLM_GPU_MEM
    # overrides for models whose KV cache actually needs more.
    args.gpu_memory_utilization = float(os.environ.get("KOT_VLLM_GPU_MEM", "0.35"))
    # KOT_VLLM_DTYPE=float32 exists for cross-checking numerics against a
    # float32 reference; a capture runs the engine's own default ("auto").
    dtype_override = os.environ.get("KOT_VLLM_DTYPE", "")
    if dtype_override:
        args.dtype = dtype_override
    # The default is vLLM's own compilation (piecewise torch.compile with CUDA
    # graphs), which the TRITON_MAX_BLOCK fix above unblocks on Blackwell.
    # "eager" is the escape hatch if a compile ever misbehaves.
    if os.environ.get("KOT_VLLM_COMPILATION", "").lower() == "eager":
        args.enforce_eager = True
    if device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            args.tensor_parallel_size = len([item for item in visible.split(",") if item.strip()])
        else:
            import torch

            args.tensor_parallel_size = max(torch.cuda.device_count(), 1)
    return args


def _register_kot_routes(app, engine, vocab_size: int, info: dict) -> None:
    @app.get(INFO)
    async def kot_info() -> JSONResponse:
        return JSONResponse(info)

    @app.post(SCORE)
    async def kot_score(request: Request):
        try:
            body = await request.json()
            if body.get("mode") == "decode":
                return StreamingResponse(
                    _decode_stream(engine, body, vocab_size),
                    media_type="application/x-ndjson",
                )
            return JSONResponse(await _score(engine, body, vocab_size))
        except Exception as failure:  # any failure is the deployment's answer
            return JSONResponse(
                {"error": type(failure).__name__, "detail": str(failure)},
                status_code=500,
            )

    # One deployment, one engine: OpenAI requests may name any model (the
    # speed harness echoes the runner's --model-id, and kot's own probe sends
    # none at all), and all of them are served by the engine that is loaded.
    served = info["model_id"]

    @app.middleware("http")
    async def _serve_any_model(request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/v1/"):
            body = await request.body()
            try:
                data = json.loads(body)
            except Exception:
                data = None
            if isinstance(data, dict) and data.get("model") != served:
                data["model"] = served
                request._body = json.dumps(data).encode(UTF8)
        return await call_next(request)


def main() -> int:
    parser = argparse.ArgumentParser(prog="kot_serve.py", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="the weights directory, read-only")
    parser.add_argument("--device", default="cuda", help="torch device string from the platform row")
    cli = parser.parse_args()

    _install_logprobs_tap()
    model_id, model_revision = _resolve_identity(cli.model)
    args = _openai_args(cli.model, cli.device, cli.port, model_id)
    app = build_app(args)

    async def run() -> None:
        async with build_async_engine_client(args) as engine:
            vllm_config = await engine.get_vllm_config()
            await init_app_state(engine, vllm_config, app.state, args)
            model_config = vllm_config.model_config
            info = {
                "contract_version": CONTRACT_VERSION,
                "model_id": model_id,
                "model_revision": model_revision,
                "tokenizer_id": model_id,
                "engine": f"vllm {vllm.__version__}",
                "dtype": str(model_config.dtype).removeprefix("torch."),
                "device": cli.device,
                "vocab_size": int(model_config.hf_config.vocab_size),
                "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            _register_kot_routes(app, engine, int(model_config.hf_config.vocab_size), info)
            print(f"[kot-vllm] serving on 127.0.0.1:{cli.port}: {json.dumps(info)}", flush=True)
            await serve_http(app, sock=None, host=args.host, port=args.port, log_level="warning")

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
