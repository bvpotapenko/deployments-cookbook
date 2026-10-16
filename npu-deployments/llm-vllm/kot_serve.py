"""kot deployment contract adapter around a vLLM engine, in one process.

    kot_serve.py --port P --model /path/to/weights --device npu

The vLLM OpenAI app is mounted as-is for /v1/completions and
/v1/chat/completions (the speed harness drives it with stream: true; vLLM
applies the model's own chat template from the tokenizer files beside the
weights). The contract endpoints are added to the same app:

  GET  /kot/info   provenance: the running engine, its dtype, the vocab width
  POST /kot/score  writes log_probs.npy + input_ids.npy into out_dir

Numerics, the whole point of this deployment:

  * ids are consumed verbatim: the request carries token ids and the engine
    sees exactly those, no BOS, no template, no truncation.
  * the score path asks the engine for `prompt_logprobs` / `logprobs` at
    k = vocab_size (with --max-logprobs raised accordingly). vLLM's V1
    sampler computes those as log_softmax of the RAW logits, in float32,
    before temperature scaling, penalties or any logit processor. That is
    exactly the contract's pre-sampling distribution.
  * k = vocab_size makes the "top-k" the whole vocabulary, so the returned
    [rows, k+1] tensors are scattered back into token-id order and every row
    covers every id exactly once. Verified per chunk; a padded-vocab engine
    cannot sneak pad rows past that check.
  * decode applies the request's sampling object exactly as given
    (temperature, top_p, top_k, min_p; all four are native SamplingParams
    knobs) with the request's seed and no penalties, and min_tokens =
    max_new_tokens so the reply has exactly max_new_tokens rows (EOS
    suppressed for the whole generation, raw logprobs unaffected).
    allow_eos: true runs to EOS.
  * prefix caching is OFF: with prompt_logprobs requested, a cache hit would
    leave the rows of the cached prefix never computed. Off, there is no such
    state at all.
  * rows are written to disk AS THEY ARE PRODUCED (numpy.lib.format.open_memmap
    at the final shape): an 8,192-row decode over a 150k vocabulary is ~5 GB
    and never fits in memory twice. The full-vocabulary log-softmax is the
    memory budget on and off the accelerator.

The one patch: vLLM pythonizes full-vocab logprob tensors into Logprob
objects / flat lists on the frontend. The tap below keeps the raw per-chunk
arrays instead, writes them straight into the memmap, and leaves cheap
placeholders in vLLM's own lists so its invariants hold and nothing else
changes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import struct
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import vllm  # noqa: E402

VLLM_PIN = "0.23.0"
if not vllm.__version__.startswith(VLLM_PIN):
    sys.exit(
        f"this deployment is written against vllm {VLLM_PIN} (its logprobs tap "
        f"touches V1 internals); found {vllm.__version__}"
    )

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
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

# request_id -> the KotSink the tap streams rows into. Only score requests
# (SCORE_PREFIX) are registered.
SINKS: dict[str, "KotSink"] = {}


class KotSink:
    """Streams one request's full-vocabulary logprob rows into a .npy memmap.

    Opened at the final shape before the engine runs; every chunk the tap
    receives is scattered into token-id order and flushed immediately, so the
    whole distribution never lives in memory at once. Each chunk is checked
    for finiteness and log-softmax normalization while it is cheap.
    """

    def __init__(self, path: Path, vocab_size: int, max_rows: int):
        self.path = path
        self.vocab_size = vocab_size
        self.max_rows = max_rows
        self.filled = 0
        self.max_abs_logsumexp = 0.0
        self._mm = np.lib.format.open_memmap(
            path, mode="w+", dtype=np.float32, shape=(max_rows, vocab_size)
        )

    def write_rows(
        self, token_ids: np.ndarray, logprobs: np.ndarray, is_prompt: bool = False
    ) -> None:
        """One chunk of [rows, k+1] engine tensors, k = vocab_size.

        The engine's prompt-logprob tensors already exclude the impossible
        prediction of input_ids[0]: a prompt of n ids arrives as n-1 rows,
        row t predicting input_ids[t+1], which is exactly the contract's row
        rule. No re-alignment happens here.
        """
        ids = np.asarray(token_ids)[:, 1:]
        values = np.asarray(logprobs)[:, 1:].astype(np.float32, copy=False)
        rows = ids.shape[0]
        if rows == 0:
            return
        if self.filled + rows > self.max_rows:
            raise ValueError(
                f"the engine produced {self.filled + rows} rows; the sink was "
                f"opened for {self.max_rows}"
            )
        # Full-vocabulary coverage per row, one tiny histogram each: every id
        # exactly once, nothing beyond the configured width. A padded-vocab
        # engine emits pad logits as -inf behind every real id, so pad ids
        # cannot enter the top-k -- this check makes that a hard fact.
        for row_index in range(rows):
            counts = np.bincount(ids[row_index], minlength=self.vocab_size)
            if counts.shape[0] != self.vocab_size or not (counts == 1).all():
                raise ValueError(
                    f"row {self.filled + row_index}: the engine's top-k is not a "
                    f"full permutation of the configured vocabulary ({ids.shape[1]} "
                    "columns); refusing to write misaligned log-probs"
                )
        chunk = self._mm[self.filled : self.filled + rows]
        np.put_along_axis(chunk, ids, values, axis=1)
        self.filled += rows
        worst = float(np.abs(np.logaddexp.reduce(values, axis=1)).max())
        if worst > 1e-3:
            raise ValueError(
                f"rows are not normalized log-probs: max |logsumexp| = {worst}"
            )
        if not np.isfinite(values).all():
            raise ValueError("log_probs contain NaN or -inf")
        self.max_abs_logsumexp = max(self.max_abs_logsumexp, worst)

    def finish(self) -> tuple[int, float]:
        """Close the memmap; shrink the file when fewer rows were produced
        (a run-to-EOS decode), by patching the .npy header's shape."""
        rows = self.filled
        self._mm.flush()
        del self._mm
        if rows != self.max_rows:
            _rewrite_npy_shape(self.path, (rows, self.vocab_size))
        return rows, self.max_abs_logsumexp


def _rewrite_npy_shape(path: Path, shape: tuple[int, ...]) -> None:
    """Patch the header of an uncompressed .npy in place to a smaller shape.

    The header dict is padded to a 64-byte boundary; a shorter shape string
    re-pads into the same header length, so the file's data offset never
    moves and no data is copied.
    """
    with open(path, "r+b") as handle:
        magic = handle.read(6)
        if magic != b"\x93NUMPY":
            raise ValueError(f"{path} is not an .npy file")
        major, _minor = struct.unpack("<BB", handle.read(2))
        if major != 1:
            raise ValueError(f"unsupported npy version {major} for {path}")
        (header_len,) = struct.unpack("<H", handle.read(2))
        header = handle.read(header_len)
        text = header.decode("latin1")
        start = text.index("'shape':") + len("'shape':")
        open_paren = text.index("(", start)
        close_paren = text.index(")", open_paren)
        new_shape_text = ", ".join(str(dim) for dim in shape)
        if len(shape) == 1:
            new_shape_text += ","
        padded = new_shape_text + " " * (close_paren - open_paren - 1 - len(new_shape_text))
        patched = text[:open_paren] + "(" + padded + text[close_paren:]
        assert len(patched) == len(text), "header length must not change"
        handle.seek(10)
        handle.write(patched.encode("latin1"))


def _hand_back_ownership(out_dir: Path) -> None:
    """Chown a written out_dir back to the host-side runner's uid.

    In host mode the server runs as root inside the wrap while the runner
    that started it runs as an ordinary user on the same mounted paths; the
    runner writes its own files (output_ids.npy, manifests) into the same
    directories right after this response. Without this hand-back every
    directory the server created is root-owned and the runner's next write
    dies with EPERM. No-op outside the wrap (KOT_UID unset or not root).
    """
    uid = os.environ.get("KOT_UID", "")
    if not uid or not hasattr(os, "geteuid") or os.geteuid() != 0:
        return
    try:
        uid_int = int(uid)
    except ValueError:
        return
    try:
        os.chown(out_dir, uid_int, uid_int)
        for path in out_dir.rglob("*"):
            try:
                os.chown(path, uid_int, uid_int)
            except OSError:
                pass
    except OSError:
        pass


def _install_logprobs_tap() -> None:
    """Stash raw logprob arrays per request instead of building Logprob dicts.

    vLLM's LogprobsProcessor.update_from_output pythonizes [rows, vocab+1]
    arrays into one Logprob object per token per position (78M objects for a
    512-token prompt). We hand the raw arrays straight to the request's sink
    (which memmaps them) and append empty dicts to vLLM's own lists so every
    length-based consumer downstream sees consistent shapes. The values those
    placeholders would have carried are exactly the rows we wrote.
    """
    original_from_new_request = LogprobsProcessor.from_new_request.__func__

    def from_new_request(cls, tokenizer, request):
        processor = original_from_new_request(cls, tokenizer, request)
        processor.kot_prompt_chunks = []
        processor.kot_sample_chunks = []
        # The input processor copies the caller's id to external_req_id and
        # re-spells request_id with a random suffix, so the sink must be
        # looked up by the external id (exact), then by prefix.
        external = getattr(request, "external_req_id", None) or request.request_id
        sink = SINKS.get(external)
        if sink is None and request.request_id.startswith(SCORE_PREFIX):
            for key, registered in SINKS.items():
                if request.request_id.startswith(key):
                    sink = registered
                    break
        if sink is not None:
            processor.kot_sink = sink
        return processor

    def update_from_output(self, output):
        sink = getattr(self, "kot_sink", None)
        if output.new_prompt_logprobs_tensors is not None:
            tensors = output.new_prompt_logprobs_tensors
            self.kot_prompt_chunks.append(tensors)
            rows = int(tensors.logprobs.shape[0])
            self.prompt_logprobs.extend([{} for _ in range(rows)])
            if sink is not None:
                sink.write_rows(
                    tensors.logprob_token_ids.cpu().numpy(),
                    tensors.logprobs.cpu().numpy(),
                    is_prompt=True,
                )
        if output.new_logprobs is not None:
            lists = output.new_logprobs
            self.kot_sample_chunks.append(lists)
            rows = int(lists.logprobs.shape[0])
            self.logprobs.extend([{} for _ in range(rows)])
            if sink is not None:
                sink.write_rows(lists.logprob_token_ids, lists.logprobs)

    LogprobsProcessor.from_new_request = classmethod(from_new_request)
    LogprobsProcessor.update_from_output = update_from_output


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
    allow_eos = bool(body.get("allow_eos", False))
    return SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=0 if allow_eos else max_new_tokens,
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


async def _score(engine, body: dict, vocab_size: int) -> dict:
    mode = body["mode"]
    if mode not in ("prefill", "decode"):
        raise ValueError(f"unknown mode {mode!r}; the contract names prefill and decode")
    input_ids = [int(token) for token in body["input_ids"]]
    if not input_ids:
        raise ValueError("empty input_ids")
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"{SCORE_PREFIX}{uuid.uuid4().hex}"
    params = _sampling_params(body, vocab_size)
    if mode == "prefill":
        max_rows = len(input_ids) - 1
        if max_rows <= 0:
            raise ValueError("a prefill needs at least two ids to score one prediction")
    else:
        max_rows = int(body["max_new_tokens"])
        if max_rows <= 0:
            raise ValueError("decode requires max_new_tokens > 0")
    sink = KotSink(out_dir / "log_probs.npy", vocab_size, max_rows)
    SINKS[request_id] = sink
    try:
        final = None
        async for output in engine.generate(
            TokensPrompt(prompt_token_ids=input_ids), params, request_id
        ):
            final = output
        if final is None or not final.finished:
            raise RuntimeError("the engine ended the request without finishing it")
        processor_rows, worst = sink.finish()
    finally:
        SINKS.pop(request_id, None)

    if mode == "prefill":
        # prefill samples nothing that counts; the engine generates one token
        # to close the request and that token is discarded
        output_ids = []
    else:
        output_ids = [int(token) for token in final.outputs[0].token_ids]
    rows = processor_rows
    if mode == "prefill":
        if rows != len(input_ids) - 1:
            raise ValueError(
                f"prefill produced {rows} rows for {len(input_ids)} ids; the contract is "
                "exactly len(input_ids) - 1 (BOS insertion or a chat template shifts rows)"
            )
    else:
        if rows != len(output_ids):
            raise ValueError(
                f"decode produced {rows} rows for {len(output_ids)} generated tokens; "
                "row t must be the distribution token t was sampled from"
            )
        if not body.get("allow_eos", False) and rows != int(body["max_new_tokens"]):
            raise ValueError(
                f"allow_eos=false demands exactly {body['max_new_tokens']} ids; "
                f"the engine stopped at {len(output_ids)}"
            )

    consumed = input_ids if mode == "prefill" else input_ids + output_ids
    np.save(
        out_dir / "input_ids.npy",
        np.ascontiguousarray(np.asarray(consumed, dtype=np.int32)),
        allow_pickle=False,
    )
    _hand_back_ownership(out_dir)
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
        "n_positions": int(rows),
        "vocab_size": vocab_size,
        "output_ids": output_ids,
    }


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
    """The vLLM server's own CLI namespace, pinned the way this contract needs."""
    args = make_arg_parser(argparse.ArgumentParser(prog="kot_serve")).parse_args([])
    args.model = model
    args.device = device
    args.host = "127.0.0.1"
    args.port = port
    args.served_model_name = [model_id, model]
    args.max_logprobs = 1_000_000
    args.enable_prefix_caching = False
    args.disable_log_requests = True
    # Full-vocab logprob gathering holds ~16 bytes per (row, vocab) on the
    # device for every scheduled token in a step; at the default 8192 that is
    # ~22GB and an over-long prompt kills the whole engine. 2048 caps the
    # transient at ~2.5GB regardless of prompt length.
    args.max_num_batched_tokens = int(os.environ.get("KOT_VLLM_MAX_BATCHED_TOKENS", "2048"))
    # And gpu_memory_utilization=0.9 hands that whole margin to the KV cache,
    # so the gather OOMs anyway. 0.35 keeps room for it; KOT_VLLM_NPU_MEM
    # overrides for models whose KV cache actually needs more.
    args.gpu_memory_utilization = float(os.environ.get("KOT_VLLM_NPU_MEM", "0.35"))
    # KOT_VLLM_MAX_MODEL_LEN caps the context window (and the KV-cache floor
    # the engine demands at startup); unset means the model config's own
    # length. The capture's long prefill is ~9k tokens; 16384 covers it.
    max_model_len = os.environ.get("KOT_VLLM_MAX_MODEL_LEN", "")
    if max_model_len:
        args.max_model_len = int(max_model_len)
    # KOT_VLLM_DTYPE=float32 exists for cross-checking numerics against a
    # float32 reference; a capture runs the engine's own default ("auto").
    dtype_override = os.environ.get("KOT_VLLM_DTYPE", "")
    if dtype_override:
        args.dtype = dtype_override
    # Default is eager on the NPU: the numerics of the compiled path are
    # unproven here, and the capture is correctness-first. KOT_VLLM_COMPILATION
    # = "default" turns compilation back on for speed runs.
    if os.environ.get("KOT_VLLM_COMPILATION", "eager").lower() != "default":
        args.enforce_eager = True
    if device.startswith("npu"):
        visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", os.environ.get("ASCEND_VISIBLE_DEVICES", ""))
        if visible.strip():
            args.tensor_parallel_size = len([item for item in visible.split(",") if item.strip()])
        else:
            args.tensor_parallel_size = max(torch.npu.device_count(), 1)
    return args


def _register_serve_any_model(app, served: str) -> None:
    """Rewrite the `model` field of /v1/* requests to the loaded model's name.

    Registered before the app starts: newer Starlette refuses middleware
    added after startup. The speed harness echoes the runner's --model-id
    (which equals model_id) and kot's own probe sends no model at all; either
    way the one loaded engine serves.
    """
    from starlette.middleware import Middleware
    from starlette.middleware.base import BaseHTTPMiddleware

    async def _rewrite(request: Request, call_next):
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

    # build_app leaves the middleware stack already built, and this
    # Starlette's add_middleware refuses anything after that. Insert into
    # user_middleware directly and clear the stack: Starlette rebuilds it
    # from the list on the first request, our middleware included.
    app.user_middleware.insert(0, Middleware(BaseHTTPMiddleware, dispatch=_rewrite))
    app.middleware_stack = None


def _register_kot_routes(app, engine, vocab_size: int, info: dict) -> None:
    @app.get(INFO)
    async def kot_info() -> JSONResponse:
        return JSONResponse(info)

    @app.post(SCORE)
    async def kot_score(request: Request) -> JSONResponse:
        try:
            body = await request.json()
            return JSONResponse(await _score(engine, body, vocab_size))
        except Exception as failure:  # any failure is the deployment's answer
            return JSONResponse(
                {"error": type(failure).__name__, "detail": str(failure)},
                status_code=500,
            )


def main() -> int:
    parser = argparse.ArgumentParser(prog="kot_serve.py", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="the weights directory, read-only")
    parser.add_argument("--device", default="npu", help="torch device string from the platform row")
    cli = parser.parse_args()

    _install_logprobs_tap()
    model_id, model_revision = _resolve_identity(cli.model)
    args = _openai_args(cli.model, cli.device, cli.port, model_id)
    app = build_app(args)
    _register_serve_any_model(app, model_id)

    async def run() -> None:
        async with build_async_engine_client(args) as engine:
            await init_app_state(engine, app.state, args)
            model_config = engine.vllm_config.model_config
            if cli.device.startswith("npu"):
                device = f"npu:{torch.npu.current_device()}"
            else:
                device = cli.device
            info = {
                "contract_version": CONTRACT_VERSION,
                "model_id": model_id,
                "model_revision": model_revision,
                "tokenizer_id": model_id,
                "engine": f"vllm {vllm.__version__}",
                "dtype": str(model_config.dtype).removeprefix("torch."),
                "device": device,
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
