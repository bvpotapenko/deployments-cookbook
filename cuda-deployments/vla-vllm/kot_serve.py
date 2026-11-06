"""kot deployment contract adapter around a vLLM engine, VLA family, one process.

    kot_serve.py --port P --model /path/to/openvla-7b --device cuda

  GET  /kot/info   provenance: the running engine, its dtype, the vocab width
  POST /kot/infer  one observation in, actions.npy out: float32 [1, 1, 7]
  POST /kot/score  one observation in, log_probs.npy + input_ids.npy out

There is no /v1: the vla_tokens family's speed run times /kot/infer, and the
contract asks the chat surface only of families the speed harness drives.

The network is vLLM's own OpenVLAForActionPrediction, in-tree at 0.33.0. Nothing
imports the checkpoint's Python: vLLM carries its own config and processor
classes for model_type "openvla", trust_remote_code stays False, and --model
stays a hub snapshot, as is. The numbers that are this deployment's
responsibility, each pinned to the reference:

  * the prompt: "In: What action should the robot take to {task}?\\nOut:", task
    lower-cased, <s> from the tokenizer, 29871 appended when absent, as
    predict_action does.
  * the frame: float32 [3, H, W] back to the 8-bit levels the reference's
    processor reads, rounded to nearest and saturated. vLLM then gets a PIL
    image, the one input its processor resizes exactly as the reference does.
  * the image slot: vLLM's multimodal processor inserts 256 placeholders right
    after <s>; every request checks the engine's expanded prompt against that.
  * the rows: prefill rows are the tail of the expanded prompt's, row t being
    the prediction of ids[t + 1]; the first kept row sits on the last image
    token. Decode is greedy, argmax of all 32064 logits, with no EOS stop and
    no logits processor; seed and sampling are not read, as the contract says
    of this model.

The engine computes log-probs as log_softmax of the raw logits, in float32,
before any processing (vllm/v1/sample/sampler.py). k = vocab_size makes the
"top-k" the whole vocabulary, so a row scatters back into token-id order; the
permutation check makes that a fact per request, and vLLM does not pad this
vocabulary, so there is nothing to slice. Prefix caching is off, because a hit
would leave rows of the cached prefix never computed. The multimodal processor
cache is off too: with both off vLLM names each image per request, so its
encoder cache cannot share an embedding across requests and the vision towers
run for every one.

The one patch: vLLM pythonizes those full-vocab tensors into ~vocab_size
Logprob objects per position on the frontend. The tap below keeps the raw
tensors instead and leaves cheap placeholders in vLLM's own lists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# Only "1" turns off vLLM's usage report, so an ambient value must not win.
os.environ["VLLM_NO_USAGE_STATS"] = "1"
# The V2 model runner's prompt-logprob path accumulates the whole prompt's
# full-vocabulary top-k on the GPU. The V1 runner streams the per-step tensors
# the tap consumes. Pin the runner the tap is written against.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
# torch 2.14's triton refuses the non-constexpr globals vLLM's kernels read
# (LOG2E et al.); triton's own documented escape hatch for exactly this.
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
# FlashInfer's backend table claims sm_120 support but its JIT's own arch check
# rejects Blackwell, so the V1 sampler's flashinfer path dies at engine init on
# 5090s. Native sampling runs fine.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402
import vllm  # noqa: E402

VLLM_PIN = "0.33.0"
# vLLM imports timm only when it builds the towers, and its CUDA wheel does not
# depend on it. The reference refuses anything outside 0.9.10-0.9.16, and 1.0
# changed get_intermediate_layers.
TIMM_PIN = "0.9.16"
if vllm.__version__ != VLLM_PIN:
    sys.exit(
        f"this deployment is written against vllm {VLLM_PIN} (its logprobs tap "
        f"touches V1 internals); found {vllm.__version__}"
    )
if metadata.version("timm") != TIMM_PIN:
    sys.exit(f"this deployment builds its vision towers on timm {TIMM_PIN}; found {metadata.version('timm')}")

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.responses import JSONResponse, Response  # noqa: E402
from PIL import Image  # noqa: E402
import torch  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.engine.arg_utils import AsyncEngineArgs  # noqa: E402
from vllm.entrypoints.launchers.api_server.entry import (  # noqa: E402
    build_async_engine_client_from_engine_args,
)
from vllm.entrypoints.launchers.launcher import serve_http  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.v1.engine.logprobs import LogprobsProcessor  # noqa: E402

INFO = "/kot/info"
INFER = "/kot/infer"
SCORE = "/kot/score"
CONTRACT_VERSION = 1
HOST = "127.0.0.1"
ACTIONS = "actions.npy"
LOG_PROBS = "log_probs.npy"
INPUT_IDS = "input_ids.npy"

PROMPT = "In: What action should the robot take to {task}?\nOut:"
# predict_action appends this token ("▁", an empty piece) when the prompt does
# not end with it: the training sequences had it right after "Out:".
EMPTY_TOKEN = 29871
IMAGE_TOKENS = 256  # (224 // 14) ** 2; vLLM's class accepts no other geometry
# The dataset whose statistics turn normalised actions into robot units.
# openvla-7b carries 25; bridge_orig (WidowX, BridgeData V2) is its model
# card's. Pinned here: the contract's three flags carry no model setting.
UNNORM_KEY = "bridge_orig"
OBSERVATION = "observation"
NOT_CAMERAS = {"state", "noise", "prompt"}  # every other key of an observation is a camera
HUB_REVISION = re.compile(r"[0-9a-f]{40}")

# request_id -> the LogprobsProcessor vLLM built for it, carrying the stashed tensors.
SINKS: dict[str, LogprobsProcessor] = {}


def _log(message: str) -> None:
    print(f"[kot-vllm-vla] {message}", flush=True)


def _install_logprobs_tap() -> None:
    """Stash raw logprob tensors per request instead of building Logprob dicts.

    vLLM's update_from_output pythonizes [rows, vocab+1] tensors into one
    Logprob object per token per position. We keep the tensors and append
    empty dicts to vLLM's lists so every length-based consumer downstream sees
    consistent shapes.
    """
    original_from_new_request = LogprobsProcessor.from_new_request.__func__

    def from_new_request(cls, tokenizer, request):
        processor = original_from_new_request(cls, tokenizer, request)
        processor.kot_prompt_tensors = []
        processor.kot_sample_lists = []
        # vLLM randomizes request_id and moves the caller's id to
        # external_req_id; the tap keys on the caller's.
        SINKS[request.external_req_id or request.request_id] = processor
        return processor

    def update_from_output(self, output):
        if output.new_prompt_logprobs_tensors is not None:
            tensors = output.new_prompt_logprobs_tensors
            self.kot_prompt_tensors.append(tensors)
            self.prompt_logprobs.extend([{} for _ in range(tensors.logprob_token_ids.shape[0])])
        if output.new_logprobs is not None:
            lists = output.new_logprobs
            self.kot_sample_lists.append(lists)
            self.logprobs.extend([{} for _ in range(len(lists.logprobs))])

    LogprobsProcessor.from_new_request = classmethod(from_new_request)
    LogprobsProcessor.update_from_output = update_from_output


def _scatter_full_vocab(entries: list, vocab_size: int) -> np.ndarray:
    """Stashed [rows, k+1] tensors -> [rows, vocab_size] in token-id order.

    Column 0 of each row is the prompt or sampled token's own logprob; columns
    1..k are the top-k. With k = vocab_size the top-k is the whole vocabulary,
    and the permutation check makes that a hard fact rather than an assumption.
    """
    if not entries:
        raise RuntimeError("the engine produced no logprobs; the V1 LogprobsProcessor tap did not run")
    ids = np.concatenate([np.asarray(entry.logprob_token_ids, dtype=np.int64) for entry in entries])[:, 1:]
    values = np.concatenate([np.asarray(entry.logprobs, dtype=np.float32) for entry in entries])[:, 1:]
    expected = np.tile(np.arange(vocab_size, dtype=np.int64), (ids.shape[0], 1))
    if not np.array_equal(np.sort(ids, axis=1), expected):
        raise ValueError(
            f"the engine's top-k is not a full permutation of the configured vocabulary "
            f"({ids.shape[1]} of {vocab_size}); refusing to write misaligned log-probs"
        )
    rows = np.empty((ids.shape[0], vocab_size), dtype=np.float32)
    np.put_along_axis(rows, ids, values, axis=1)
    if not np.isfinite(rows).all():
        raise ValueError("log_probs contain NaN or -inf")
    worst = float(np.abs(np.logaddexp.reduce(rows, axis=1)).max())
    if worst > 1e-3:
        raise ValueError(f"rows are not normalized log-probs: max |logsumexp| = {worst}")
    return rows


def _read_observation(path: str) -> tuple[Image.Image, str]:
    """The one camera and the task of a kot observation; state and noise are not read."""
    with np.load(path, allow_pickle=False) as observation:
        cameras = [key for key in observation.files if key not in NOT_CAMERAS]
        if len(cameras) != 1:
            raise ValueError(f"OpenVLA reads one camera; {path} holds {cameras}")
        pixels = observation[cameras[0]]
        task = str(observation["prompt"])
    if pixels.ndim != 3 or pixels.shape[0] != 3:
        raise ValueError(f"camera {cameras[0]!r} must be [3, H, W], got {pixels.shape}")
    levels = np.clip(np.rint(pixels.transpose(1, 2, 0) * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(levels), task


def _prompt_ids(tokenizer, task: str) -> list[int]:
    """<s> first (the tokenizer adds it), the empty token last."""
    ids = list(tokenizer(PROMPT.format(task=task.lower()))["input_ids"])
    if ids[-1] != EMPTY_TOKEN:
        ids.append(EMPTY_TOKEN)
    return ids


@dataclass(frozen=True)
class Policy:
    """What the serving process read off the checkpoint it loaded, and its one forward path."""

    engine: object
    vocab_size: int  # config.text_config.vocab_size: the output embedding's rows, 32064
    bos_token_id: int
    image_token_id: int
    action_low: np.ndarray  # the dataset's q01
    action_high: np.ndarray  # the dataset's q99
    action_mask: np.ndarray  # which dimensions unnormalize; bridge_orig leaves the gripper
    bin_centers: np.ndarray
    action_tokens_end: int  # the unpadded vocabulary, 32000: action tokens count down from it

    @classmethod
    def of(cls, engine) -> "Policy":
        model_config = engine.vllm_config.model_config
        config = model_config.hf_config
        stats = config.norm_stats[UNNORM_KEY]["action"]
        low = np.array(stats["q01"])
        edges = np.linspace(-1, 1, config.n_action_bins)
        vocab_size = int(model_config.get_vocab_size())
        return cls(
            engine=engine,
            vocab_size=vocab_size,
            bos_token_id=engine.get_tokenizer().bos_token_id,
            image_token_id=config.image_token_index,
            action_low=low,
            action_high=np.array(stats["q99"]),
            action_mask=np.array(stats.get("mask", np.ones_like(low, dtype=bool))),
            bin_centers=(edges[:-1] + edges[1:]) / 2.0,
            action_tokens_end=vocab_size - config.pad_to_multiple_of,
        )

    @property
    def action_dim(self) -> int:
        return len(self.action_low)

    def prompt_ids(self, task: str) -> list[int]:
        return _prompt_ids(self.engine.get_tokenizer(), task)

    async def _forward(self, ids: list[int], frame: Image.Image, params: SamplingParams):
        """One request through the engine: the final output and the tap's processor."""
        if len(ids) < 2 or ids[0] != self.bos_token_id:
            raise ValueError(
                f"the image goes right after <s>, so the ids must start with it and hold "
                f"at least one more; got {ids[:4]}"
            )
        request_id = f"kot-{uuid.uuid4().hex}"
        prompt = TokensPrompt(prompt_token_ids=ids, multi_modal_data={"image": frame})
        final = None
        try:
            async for output in self.engine.generate(prompt, params, request_id):
                final = output
            processor = SINKS.get(request_id)
        finally:
            SINKS.pop(request_id, None)
        if final is None or not final.finished:
            raise RuntimeError("the engine ended the request without finishing it")
        expanded = ids[:1] + [self.image_token_id] * IMAGE_TOKENS + ids[1:]
        if list(final.prompt_token_ids) != expanded:
            raise RuntimeError("the engine's expanded prompt is not <s>, 256 image tokens, the rest")
        return final, processor

    async def decode(self, ids: list[int], frame: Image.Image, want_rows: bool) -> tuple[list[int], np.ndarray | None]:
        """Greedy, action_dim tokens. Asking for rows changes what is returned, not what is computed."""
        params = SamplingParams(
            max_tokens=self.action_dim,
            temperature=0.0,
            ignore_eos=True,
            detokenize=False,  # an action token is an id; the unused padding rows have no text at all
            logprobs=self.vocab_size if want_rows else None,
        )
        final, processor = await self._forward(ids, frame, params)
        output_ids = [int(token) for token in final.outputs[0].token_ids]
        if len(output_ids) != self.action_dim:
            raise RuntimeError(f"decoded {len(output_ids)} tokens, not {self.action_dim}")
        if not want_rows:
            return output_ids, None
        rows = _scatter_full_vocab(processor.kot_sample_lists if processor else [], self.vocab_size)
        if rows.shape[0] != len(output_ids):
            raise ValueError(f"{rows.shape[0]} rows for {len(output_ids)} tokens")
        # A tie in the logits may pick either id, hence the comparison by value.
        if not np.array_equal(rows[np.arange(rows.shape[0]), output_ids], rows.max(axis=1)):
            raise RuntimeError("a decoded token is not the argmax of its row")
        return output_ids, rows

    async def prefill(self, ids: list[int], frame: Image.Image, n_rows: int) -> np.ndarray:
        """Teacher forced over <s>, the image, ids[1:]: the last n_rows, or all n - 1.

        The engine closes the request with one generated token; it is discarded.
        """
        keep = n_rows or len(ids) - 1
        if keep >= len(ids):
            raise ValueError(f"n_rows {keep} asks for more rows than {len(ids)} ids can predict")
        params = SamplingParams(
            max_tokens=1, temperature=0.0, detokenize=False, prompt_logprobs=self.vocab_size
        )
        _, processor = await self._forward(ids, frame, params)
        rows = _scatter_full_vocab(processor.kot_prompt_tensors if processor else [], self.vocab_size)
        expected = len(ids) + IMAGE_TOKENS - 1
        if rows.shape[0] != expected:
            raise ValueError(f"the engine wrote {rows.shape[0]} prompt rows; expected {expected}")
        return np.ascontiguousarray(rows[-keep:])

    def unnormalize(self, tokens: list[int]) -> np.ndarray:
        """predict_action's arithmetic: token -> bin centre in [-1, 1] -> q01..q99 for
        the dimensions the mask names; the rest stay in [-1, 1]."""
        bins = np.clip(self.action_tokens_end - np.array(tokens) - 1, 0, len(self.bin_centers) - 1)
        normalized = self.bin_centers[bins]
        scaled = 0.5 * (normalized + 1) * (self.action_high - self.action_low) + self.action_low
        return np.where(self.action_mask, scaled, normalized)


async def _infer(policy: Policy, body: dict) -> dict:
    frame, task = _read_observation(body["attachments"][OBSERVATION])
    started = time.perf_counter()
    tokens, _ = await policy.decode(policy.prompt_ids(task), frame, want_rows=False)
    action = policy.unnormalize(tokens)
    elapsed_ms = (time.perf_counter() - started) * 1000
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / ACTIONS, action.astype(np.float32).reshape(1, 1, -1), allow_pickle=False)
    _log(f"{body['prompt_id']} infer: {elapsed_ms:.1f} ms, tokens {tokens}, action {np.round(action, 4).tolist()}")
    return {"prompt_id": body["prompt_id"], "files": [ACTIONS]}


async def _score(policy: Policy, body: dict) -> dict:
    frame, task = _read_observation(body["attachments"][OBSERVATION])
    mode = body["mode"]
    sent = [int(token) for token in body.get("input_ids") or []]  # verbatim: no template, no BOS added
    started = time.perf_counter()
    if mode == "decode":
        if int(body["max_new_tokens"]) != policy.action_dim:
            raise ValueError(f"OpenVLA decodes {policy.action_dim} tokens; max_new_tokens is {body['max_new_tokens']}")
        prompt = sent or policy.prompt_ids(task)
        output_ids, rows = await policy.decode(prompt, frame, want_rows=True)
        stored = prompt + output_ids
    elif mode == "prefill":
        rows = await policy.prefill(sent, frame, int(body.get("n_rows") or 0))
        output_ids, stored = [], sent
    else:
        raise ValueError(f"unknown mode {mode!r}; the contract names prefill and decode")
    elapsed_ms = (time.perf_counter() - started) * 1000
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / LOG_PROBS, rows, allow_pickle=False)
    np.save(out_dir / INPUT_IDS, np.asarray(stored, dtype=np.int32), allow_pickle=False)
    _log(f"{body['prompt_id']} {mode}: {elapsed_ms:.1f} ms, {rows.shape[0]} rows, tokens {output_ids}")
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": LOG_PROBS, "input_ids": INPUT_IDS},
        "n_positions": int(rows.shape[0]),
        "vocab_size": policy.vocab_size,
        "output_ids": output_ids,
    }


def build_app(policy: Policy, info: dict) -> FastAPI:
    """The contract's three routes and nothing else on this port."""
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.engine_client = policy.engine  # serve_http's watchdog and shutdown read it here

    async def answer(request: Request, route) -> Response:
        try:
            body = await request.json()
            reply = await route(policy, body)
        except Exception as failure:  # any failure is the deployment's answer
            error = "out_of_memory" if isinstance(failure, torch.OutOfMemoryError) else type(failure).__name__
            _log(f"{request.url.path} failed: {error}: {failure}")
            return JSONResponse({"error": error, "detail": str(failure)}, status_code=500)
        # A decode answer is a JSON-lines stream; 7 rows fit its one line.
        streamed = request.url.path == SCORE and body.get("mode") == "decode"
        media_type = "application/x-ndjson" if streamed else "application/json"
        return Response(json.dumps(reply) + "\n", media_type=media_type)

    @app.get(INFO)
    async def kot_info() -> JSONResponse:
        return JSONResponse(info)

    @app.post(INFER)
    async def kot_infer(request: Request) -> Response:
        return await answer(request, _infer)

    @app.post(SCORE)
    async def kot_score(request: Request) -> Response:
        return await answer(request, _score)

    return app


def _resolve_identity(model_dir: str) -> tuple[str, str]:
    """(model_id, revision), read off the directory: a hub snapshot is named by its
    commit, and its checkpoint's config.json holds no _name_or_path. Any other
    directory gives its name and no revision."""
    path = Path(model_dir).resolve()
    repo = path.parent.parent.name
    if path.parent.name == "snapshots" and repo.startswith("models--") and HUB_REVISION.fullmatch(path.name):
        return repo.removeprefix("models--").replace("--", "/"), path.name
    return path.name, ""


def _engine_args(model: str, device: str) -> AsyncEngineArgs:
    """The engine, pinned the way this contract needs; no chat surface, so no CLI namespace."""
    args = AsyncEngineArgs(
        model=model,
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        limit_mm_per_prompt={"image": 1},
        max_logprobs=1_000_000,
        # Full-vocabulary gathering holds ~16 bytes per (row, vocab) on the GPU
        # for every scheduled token in a step; 2048 caps that at about 1 GB.
        max_num_batched_tokens=int(os.environ.get("KOT_VLLM_MAX_BATCHED_TOKENS", "2048")),
        # The weights alone are 14 GiB of a 5090's 32 GiB, and the fraction
        # covers them too: the 0.35 of the other vLLM deployments leaves no KV
        # cache at all. 0.6 leaves about 4.5 GiB of it, and 2048 positions take
        # about 1 GiB; what stays outside covers the gathers above.
        gpu_memory_utilization=float(os.environ.get("KOT_VLLM_GPU_MEM", "0.6")),
    )
    if os.environ.get("KOT_VLLM_MAX_MODEL_LEN"):
        args.max_model_len = int(os.environ["KOT_VLLM_MAX_MODEL_LEN"])
    # A cross-check of numerics; a capture runs the checkpoint's own dtype.
    if os.environ.get("KOT_VLLM_DTYPE"):
        args.dtype = os.environ["KOT_VLLM_DTYPE"]
    if os.environ.get("KOT_VLLM_COMPILATION", "").lower() == "eager":
        args.enforce_eager = True
    if device == "cuda":
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        if visible and visible != "-1":
            args.tensor_parallel_size = len([item for item in visible.split(",") if item.strip()])
        else:
            args.tensor_parallel_size = max(torch.cuda.device_count(), 1)
    return args


def _nvidia_driver() -> str:
    """The kernel driver's version, as the driver itself reports it; empty when unreadable."""
    try:
        first_line = Path("/proc/driver/nvidia/version").read_text(errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return ""
    match = re.search(r"\s(\d+\.\d+(?:\.\d+)?)\s", first_line)
    return match.group(1) if match else ""


def _placed_devices(config) -> str:
    """Where the weights are, ordinals included, one per engine worker: cuda:0, or
    cuda:0+cuda:1 over two cards. Both come from the engine's own config: a torch.cuda
    call could initialize CUDA in this API-server process, which flips vLLM's fork to
    spawn."""
    device = config.device_config.device_type
    if device != "cuda":
        return device
    return "+".join(f"cuda:{rank}" for rank in range(config.parallel_config.world_size))


def _server_info(policy: Policy, model_dir: str) -> dict:
    model_id, model_revision = _resolve_identity(model_dir)
    config = policy.engine.vllm_config
    return {
        "contract_version": CONTRACT_VERSION,
        "model_id": model_id,
        "model_revision": model_revision,
        "tokenizer_id": model_id,
        "engine": f"vllm {vllm.__version__}",
        "dtype": str(config.model_config.dtype).removeprefix("torch."),
        "device": _placed_devices(config),
        "vocab_size": policy.vocab_size,
        "unnorm_key": UNNORM_KEY,
        "torch_version": torch.__version__,
        "plugin_version": "",
        "driver_version": _nvidia_driver(),
        "toolkit_version": torch.version.cuda or "",
        "started_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


async def _serve(cli: argparse.Namespace, args: AsyncEngineArgs) -> None:
    async with build_async_engine_client_from_engine_args(args) as engine:
        # vLLM picks its platform from what it detects, so a missing card must
        # stop the start here, never become a forward pass on the host.
        if engine.vllm_config.device_config.device_type != cli.device:
            raise RuntimeError(f"--device {cli.device}, but the engine runs on {engine.vllm_config.device_config.device_type}")
        policy = Policy.of(engine)
        # The first request pays for compilation and CUDA kernel choice, and a
        # broken stack dies here, non-zero, instead of on the first capture.
        await policy.decode(policy.prompt_ids("warm up"), Image.new("RGB", (224, 224)), want_rows=True)
        info = _server_info(policy, cli.model)
        _log(f"serving on {HOST}:{cli.port}: {json.dumps(info)}")
        app = build_app(policy, info)
        shutdown = await serve_http(app, sock=None, host=HOST, port=cli.port, log_level="warning", lifespan="off")
        await shutdown


def main() -> int:
    parser = argparse.ArgumentParser(prog="kot_serve.py", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="the weights directory, read-only")
    parser.add_argument("--device", required=True, help="torch device string from the platform row")
    cli = parser.parse_args()
    _install_logprobs_tap()
    asyncio.run(_serve(cli, _engine_args(cli.model, cli.device)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
