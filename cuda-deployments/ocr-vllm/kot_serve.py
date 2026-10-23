"""kot deployment contract adapter around a vLLM engine, OCR family, one process.

    kot_serve.py --port P --model /path/to/DeepSeek-OCR-2 --device cuda

The vLLM OpenAI app is mounted as-is for /v1/chat/completions (the speed
harness drives it with multimodal messages: a text part and the document as
an image_url data URI; streaming SSE for the benchmark, plain JSON for the
probe). The contract endpoints are added to the same app:

  GET  /kot/info   provenance: the running engine, its dtype, the vocab width
  POST /kot/score  writes log_probs.npy + input_ids.npy into out_dir; a decode
                   answers in JSON lines, one per 256-row block, so a long
                   generation streams instead of answering at the end

Numerics, the whole point of this deployment:

  * the score path consumes ids verbatim: the request's transcript ids are
    appended, untouched, to the image template the checkpoint itself defines
    (plain format: BOS, one image block, "\nFree OCR."), and the engine sees
    exactly those ids. The image arrives as multi_modal_data; vLLM expands
    the single placeholder token into the block its processor computes for
    the image (256+1 tokens for a <=768px image, 144 more per crop tile -
    the checkpoint's own conventions, verified against its infer()).
  * prefill rows are the last len(transcript) - 1 prompt rows: row t is the
    distribution after the ids before transcript[t+1], i.e. its prediction.
    The engine generates one token to close the request and it is discarded.
  * the score path asks the engine for `prompt_logprobs` / `logprobs` at
    k = vocab_size (with --max-logprobs raised accordingly). vLLM's sampler
    computes those as log_softmax of the RAW logits, in float32, before
    temperature scaling, penalties or any logit processor. That is exactly
    the contract's pre-sampling distribution.
  * k = vocab_size makes the "top-k" the whole vocabulary, so the returned
    [rows, k+1] tensors are scattered back into token-id order and every row
    covers every id exactly once. Verified per request; a padded-vocab engine
    cannot sneak pad rows past that check.
  * decode applies the request's sampling object exactly as given
    (temperature, top_p, top_k, min_p; all four are native SamplingParams
    knobs) with the request's seed and no penalties. allow_eos gates the
    decode exactly as the contract states: false demands exactly
    max_new_tokens ids (EOS suppressed via min_tokens), true accepts what
    the sampler emitted before EOS, at most that many; the reported rows
    are the raw distributions either way.
  * prefix caching is OFF: with prompt_logprobs requested, a cache hit would
    leave the rows of the cached prefix never computed.

The one patch: vLLM pythonizes those full-vocab tensors into ~vocab_size
Logprob objects per position on the frontend. The tap below keeps the raw
tensors instead and leaves cheap placeholders in vLLM's own lists, so its
invariants hold and nothing else changes.

The chat template (the checkpoint's tokenizer ships none) is this
directory's chat_template.jinja: it renders the same plain format the score
path builds, image placeholder first, so both surfaces feed the model the
same shape of prompt.
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

os.environ.setdefault("HF_HUB_OFFLINE", "1")
# The V2 model runner's prompt-logprob path accumulates the whole prompt's
# full-vocabulary top-k on the GPU - over 10 GB for a long prompt, which
# dies out of memory beside the KV cache whatever the utilization. The V1
# runner streams the per-step tensors the tap consumes. Pin the runner the
# tap is written against; the n-gram processor is a V1-interface plugin too.
os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
# torch 2.13's triton refuses the non-constexpr globals vLLM's kernels read
# (LOG2E et al.); triton's own documented escape hatch for exactly this.
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
# FlashInfer's backend table claims sm_120 support but its JIT's own arch
# check rejects Blackwell ("requires sm75 or higher"), so the V1 sampler's
# flashinfer path dies at engine init on 5090s. Native sampling runs fine.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402
import vllm  # noqa: E402

VLLM_PIN = "0.32.0"
if not vllm.__version__.startswith(VLLM_PIN):
    sys.exit(
        f"this deployment is written against vllm {VLLM_PIN} (its logprobs tap "
        f"touches V1 internals); found {vllm.__version__}"
    )

from fastapi import Request  # noqa: E402
from fastapi.responses import JSONResponse, StreamingResponse  # noqa: E402
from PIL import Image  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.entrypoints.launchers.launcher import serve_http  # noqa: E402
from vllm.entrypoints.launchers.api_server.app_state import init_app_state  # noqa: E402
from vllm.entrypoints.launchers.api_server.entry import build_async_engine_client  # noqa: E402
from vllm.entrypoints.launchers.app import build_app  # noqa: E402
from vllm.entrypoints.launchers.cli_args import make_arg_parser  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402
from vllm.v1.engine.logprobs import LogprobsProcessor  # noqa: E402

INFO = "/kot/info"
SCORE = "/kot/score"
CONTRACT_VERSION = 1
SCORE_PREFIX = "kot-score-"
UTF8 = "utf-8"
SUPPORTED_TASKS = ("generate",)
BLOCK_ROWS = 256  # one streamed line per block of decode rows: bounded memory, visible progress

IMAGE = "image"
IMAGE_TAG = "<image>"

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p")

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
        rid = getattr(request, "request_id", "?")
        if rid.startswith(SCORE_PREFIX):
            SINKS[rid] = processor
        return processor

    def update_from_output(self, output):
        if output.new_prompt_logprobs_tensors is not None:
            tensors = output.new_prompt_logprobs_tensors
            self.kot_prompt_tensors.append(tensors)
            if self.prompt_logprobs is not None:
                self.prompt_logprobs.extend(
                    [{} for _ in range(tensors.logprobs.shape[0])]
                )
        if output.new_logprobs is not None:
            lists = output.new_logprobs
            self.kot_sample_lists.append(lists)
            if self.logprobs is not None:
                self.logprobs.extend([{} for _ in lists.logprobs])

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
    ids = np.asarray(token_ids, dtype=np.int64)[:, 1:]
    values = np.asarray(logprobs, dtype=np.float32)[:, 1:]
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
        [np.asarray(t.logprob_token_ids) for t in processor.kot_prompt_tensors]
    )
    logprobs = np.concatenate(
        [np.asarray(t.logprobs) for t in processor.kot_prompt_tensors]
    )
    return _scatter_full_vocab(token_ids, logprobs, vocab_size)


def _decode_rows(entries: list, vocab_size: int) -> np.ndarray:
    """A stretch of the stashed per-step tensors as [rows, vocab_size], in token-id order."""
    token_ids = np.concatenate(
        [np.asarray(entry.logprob_token_ids, dtype=np.int64) for entry in entries]
    )
    logprobs = np.concatenate(
        [np.asarray(entry.logprobs, dtype=np.float32) for entry in entries]
    )
    return _scatter_full_vocab(token_ids, logprobs, vocab_size)


def _check_well_formed(log_probs: np.ndarray) -> None:
    """log-softmax rows in float32: finite, and logsumexp 0 within float32 noise."""
    if not np.isfinite(log_probs).all():
        raise ValueError("log_probs contain NaN or -inf")
    worst = float(np.abs(np.logaddexp.reduce(log_probs, axis=1)).max())
    if worst > 1e-3:
        raise ValueError(f"rows are not normalized log-probs: max |logsumexp| = {worst}")


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
    }


def _sampling_params(body: dict, vocab_size: int) -> SamplingParams:
    """Score-path params: pre-sampling logprobs over the full vocab.

    The decode's sampling object is applied exactly as given (all four fields
    are native SamplingParams knobs): they steer what the sampler draws and
    never touch the reported logprobs, which are the raw logits' log-softmax.
    No penalties are set, and the only logits processor that can appear is
    min_tokens (EOS suppression under allow_eos=false), which changes which
    token is sampled, never the reported raw logprobs.
    """
    if body["mode"] == "prefill":
        return SamplingParams(max_tokens=1, temperature=1.0, prompt_logprobs=vocab_size)
    sampling = _resolve_sampling(body)
    max_new_tokens = int(body["max_new_tokens"])
    allow_eos = bool(body.get("allow_eos", False))
    seed = body.get("seed")
    return SamplingParams(
        max_tokens=max_new_tokens,
        min_tokens=0 if allow_eos else max_new_tokens,
        temperature=sampling["temperature"],
        top_p=sampling["top_p"],
        top_k=sampling["top_k"],
        min_p=sampling["min_p"],
        seed=int(seed) if seed is not None else None,
        logprobs=vocab_size,
    )



def _pop_sink(request_id: str) -> LogprobsProcessor | None:
    """The processor for a score request, by prefix.

    vLLM 0.30 appends its own sub-request suffix to the ids it passes down
    ("-xxxxxxxx"), so the exact id we generated never matches; the contract
    serves one score request at a time, so a prefix match is unambiguous.
    """
    processor = SINKS.pop(request_id, None)
    if processor is None:
        for key in [k for k in SINKS if k.startswith(request_id)]:
            processor = SINKS.pop(key)
            break
    return processor


def _peek_sink(request_id: str) -> LogprobsProcessor | None:
    processor = SINKS.get(request_id)
    if processor is None:
        for key in SINKS:
            if key.startswith(request_id):
                return SINKS[key]
    return processor


def _line(message: dict) -> str:
    """One streamed line: JSON and a newline, nothing else on the wire."""
    return json.dumps(message) + "\n"


def _score_prompt(body: dict) -> tuple[TokensPrompt, list[int]]:
    """The score request as an engine prompt: the ids verbatim, plus the image.

    The version's frozen template ids ride every request (a decode's prompt;
    a replay's template plus generated ids), so the deployment consumes ids
    and never tokenizes - no engine's tokenizer version can change the
    prompt. The engine's multimodal processor expands the single image
    placeholder into the block it computes for this image.
    """
    image_path = body["attachments"][IMAGE]
    transcript = [int(token) for token in body["input_ids"]]
    if len(transcript) < 2:
        raise ValueError(
            "a prefill needs at least 2 ids to produce a row; "
            f"got {len(transcript)}"
        )
    ids = list(transcript)
    image = Image.open(image_path).convert("RGB")
    prompt = TokensPrompt(prompt_token_ids=ids, multi_modal_data={"image": image})
    return prompt, transcript


_TOKENIZER_CACHE: object = None


def _tokenizer():
    """The checkpoint's tokenizer, loaded once.

    LlamaTokenizerFast directly, not AutoTokenizer: the config's auto_map
    would drag transformers' dynamic-module import checks (addict,
    matplotlib) into this venv and a 5.x trust advisory onto stdout; the
    tokenizer itself is a standard fast tokenizer needing neither.
    """
    global _TOKENIZER_CACHE
    if _TOKENIZER_CACHE is None:
        from transformers import LlamaTokenizerFast

        _TOKENIZER_CACHE = LlamaTokenizerFast.from_pretrained(_MODEL_DIR)
    return _TOKENIZER_CACHE


def _image_only_prompt(image_path: str, template_ids: list[int]) -> TokensPrompt:
    """The decode's prompt: the frozen template ids the request carried, and the image."""
    if not template_ids:
        raise ValueError("empty input_ids: the version's frozen template is missing")
    image = Image.open(image_path).convert("RGB")
    return TokensPrompt(prompt_token_ids=template_ids, multi_modal_data={"image": image})


_MODEL_DIR = ""


async def _decode_stream(engine, body: dict, vocab_size: int, image_token_id: int):
    """A decode answered in JSON lines: one per block of rows as it lands, the result last.

    The engine's per-step tensors are drained while the request runs,
    scattered into token-id order and written block-wise into a memmap of the
    cap shape, so memory holds one block instead of the whole unit and the
    client sees rows for as long as the generation lasts. Under allow_eos the
    sampler may stop early; the file is compacted to the rows actually
    produced before the reply names it.
    """
    image_path = body["attachments"][IMAGE]
    max_new_tokens = int(body["max_new_tokens"])
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    request_id = f"{SCORE_PREFIX}{uuid.uuid4().hex}"
    prompt = _image_only_prompt(image_path, [int(t) for t in body["input_ids"]])
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
        async for output in engine.generate(prompt, params, request_id):
            final = output
            processor = processor or _peek_sink(request_id)
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
        processor = processor or _pop_sink(request_id)
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
        if written != len(output_ids):
            yield _line(
                {"error": "ValueError",
                 "detail": f"decode produced {written} rows for {len(output_ids)} sampled "
                           "tokens; row t must be the distribution token t was sampled from"}
            )
            return
        if not 0 < written <= max_new_tokens:
            yield _line(
                {"error": "ValueError",
                 "detail": f"decode produced {written} rows for a cap of {max_new_tokens}"}
            )
            return
        # Compact to the rows actually produced (allow_eos may stop early).
        # A real copy, not a view: np.load(mmap_mode)[:n] stays memory-mapped
        # (ascontiguousarray returns the same buffer), and rewriting the file
        # under a live mmap is a short write.
        log_probs.flush()
        del log_probs
        if written < max_new_tokens:
            mapped = np.load(out_dir / "log_probs.npy", mmap_mode="r")
            compact = np.array(mapped[:written])
            del mapped
            np.save(
                out_dir / "log_probs.npy",
                compact,
                allow_pickle=False,
            )
        # The raw prompt the request carried (the version's frozen template)
        # plus the generated ids: the pre-expansion sequence, so the replay
        # of this file carries exactly one image placeholder. The engine's
        # final.prompt_token_ids is the EXPANDED prompt and is not what a
        # replay may send back.
        consumed_ids = [int(token) for token in body["input_ids"]]
        np.save(
            out_dir / "input_ids.npy",
            np.ascontiguousarray(
                np.asarray(consumed_ids + output_ids, dtype=np.int32)
            ),
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
        _pop_sink(request_id)


async def _score(engine, body: dict, vocab_size: int, image_token_id: int) -> dict:
    """A prefill answer: one forward pass, both files written, one JSON back."""
    if body["mode"] != "prefill":
        raise ValueError(f"unknown mode {body['mode']!r}; the contract names prefill and decode")
    prompt, transcript = _score_prompt(body)
    out_dir = Path(body["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    request_id = f"{SCORE_PREFIX}{uuid.uuid4().hex}"
    params = _sampling_params(body, vocab_size)
    try:
        final = None
        async for output in engine.generate(prompt, params, request_id):
            final = output
        if final is None or not final.finished:
            raise RuntimeError("the engine ended the request without finishing it")
        processor = _pop_sink(request_id)
    finally:
        _pop_sink(request_id)
    if processor is None:
        raise RuntimeError(
            "no logprob tap for the request; the V1 LogprobsProcessor patch did not run"
        )

    # The prompt rows cover the whole expanded prompt (the frozen template's
    # image block plus the ids the request carried); the rows the family
    # keeps are the tail: the classic n - 1, or the last n_rows when the
    # request trims (the OCR replay keeps only the rows over the
    # generated ids).
    all_rows = _prompt_rows(processor, vocab_size)
    n = len(transcript)
    keep = int(body.get("n_rows", 0)) or n - 1
    if keep >= n:
        raise ValueError(f"n_rows {keep} asks for more rows than {n} ids can predict")
    log_probs = np.ascontiguousarray(all_rows[-keep:])
    _check_well_formed(log_probs)

    np.save(out_dir / "log_probs.npy", log_probs, allow_pickle=False)
    # The stored ids are the sequence the rows live in, verbatim: the OCR
    # replay's rows predict the trailing generated ids, and the eval joins
    # them to decode_ids by name.
    np.save(
        out_dir / "input_ids.npy",
        np.ascontiguousarray(np.asarray(transcript, dtype=np.int32)),
        allow_pickle=False,
    )
    return {
        "prompt_id": body["prompt_id"],
        "files": {"log_probs": "log_probs.npy", "input_ids": "input_ids.npy"},
        "n_positions": int(log_probs.shape[0]),
        "vocab_size": vocab_size,
        "output_ids": [],
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


def _openai_args(model: str, device: str, port: int, model_id: str, here: str) -> argparse.Namespace:
    """The vLLM server's own CLI namespace, pinned the way this contract needs.

    vLLM's arg parser passes ``deprecated=True`` to argparse on some versions,
    which only newer Pythons accept; the shim drops it so the parser builds
    anywhere. Argument groups share the container's add_argument, hence the
    class-level patch, undone right after the namespace is built.
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
    # The checkpoint's tokenizer ships no chat template; this deployment's
    # own jinja renders the same plain format the score path builds, so the
    # chat and score surfaces feed the model the same shape of prompt.
    args.chat_template = str(Path(here) / "chat_template.jinja")
    # One document per request, exactly one image in it.
    args.limit_mm_per_prompt = {"image": 1}
    # The model's own max_position_embeddings (8192) covers template +
    # transcript + decode; override only if a task ever needs more.
    if os.environ.get("KOT_VLLM_MAX_MODEL_LEN"):
        args.max_model_len = int(os.environ["KOT_VLLM_MAX_MODEL_LEN"])
    args.max_logprobs = 1_000_000
    args.enable_prefix_caching = False
    args.disable_log_requests = True
    # DeepSeek's own anti-repetition rule for OCR outputs, as an engine-level
    # V1 logits processor (module copied into the venv by kot_serve.sh, so
    # the FQCN resolves in the workers). logprobs_mode stays at its default
    # raw_logprobs: the reported rows are computed before this stage runs.
    args.logits_processors = ["kot_ngram:KotNoRepeatNGramLogitsProcessor"]
    # "auto" picks flashinfer's cutlass MoE on Blackwell, whose JIT needs
    # nvcc - absent on serving hosts - and dies at engine init. The triton
    # MoE backend runs sm_120 fine. KOT_VLLM_MOE_BACKEND overrides for
    # hosts that do carry the toolkit.
    args.moe_backend = os.environ.get("KOT_VLLM_MOE_BACKEND", "triton")
    # Full-vocab logprob gathering holds ~16 bytes per (row, vocab) on the GPU
    # for every scheduled token in a step; at the default batch that is tens
    # of GB and an over-long prompt kills the whole engine. 2048 caps the
    # transient at a few GB regardless of prompt length.
    args.max_num_batched_tokens = int(os.environ.get("KOT_VLLM_MAX_BATCHED_TOKENS", "2048"))
    # And a high gpu_memory_utilization hands that whole margin to the KV
    # cache, so the gather OOMs anyway. 0.35 keeps room for it; KOT_VLLM_GPU_MEM
    # overrides for models whose KV cache actually needs more.
    args.gpu_memory_utilization = float(os.environ.get("KOT_VLLM_GPU_MEM", "0.35"))
    # KOT_VLLM_DTYPE exists for cross-checking numerics; a capture runs the
    # engine's own default ("auto": the checkpoint's dtype).
    dtype_override = os.environ.get("KOT_VLLM_DTYPE", "")
    if dtype_override:
        args.dtype = dtype_override
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


def _register_kot_routes(app, engine, vocab_size: int, info: dict, image_token_id: int) -> None:
    @app.get(INFO)
    async def kot_info() -> JSONResponse:
        return JSONResponse(info)

    @app.post(SCORE)
    async def kot_score(request: Request):
        try:
            body = await request.json()
            if body.get("mode") == "decode":
                return StreamingResponse(
                    _decode_stream(engine, body, vocab_size, image_token_id),
                    media_type="application/x-ndjson",
                )
            return JSONResponse(await _score(engine, body, vocab_size, image_token_id))
        except Exception as failure:  # any failure is the deployment's answer
            import traceback

            try:
                raw = (await request.body())[:200].decode(UTF8, errors="replace")
            except Exception:
                raw = "<body unreadable>"
            tail = "".join(
                traceback.format_exception(failure)[-8:]
            ).strip()
            return JSONResponse(
                {"error": type(failure).__name__, "detail": tail, "raw_body": raw},
                status_code=500,
            )

    # One deployment, one engine: the harness names the model as
    # /kot/info reported it (both spellings are in served_model_name), and
    # kot's probe sends no model field at all, which vLLM answers with the
    # sole served model. No model-name rewriting is needed.


def main() -> int:
    global _MODEL_DIR
    parser = argparse.ArgumentParser(prog="kot_serve.py", description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", required=True, help="the weights directory, read-only")
    parser.add_argument("--device", default="cuda", help="torch device string from the platform row")
    cli = parser.parse_args()
    _MODEL_DIR = cli.model

    _install_logprobs_tap()
    model_id, model_revision = _resolve_identity(cli.model)
    here = str(Path(__file__).resolve().parent)
    args = _openai_args(cli.model, cli.device, cli.port, model_id, here)
    app = build_app(args, supported_tasks=SUPPORTED_TASKS)

    async def run() -> None:
        async with build_async_engine_client(args) as engine:
            model_config = engine.vllm_config.model_config
            image_token_id = _tokenizer().convert_tokens_to_ids(IMAGE_TAG)
            if not isinstance(image_token_id, int) or image_token_id < 0:
                sys.exit(f"the tokenizer has no {IMAGE_TAG!r} token; cannot build prompts")
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
            # Routes and middleware go on before init_app_state: it starts the
            # app, and Starlette refuses middleware on a started application.
            _register_kot_routes(
                app, engine, int(model_config.hf_config.vocab_size), info, image_token_id
            )
            await init_app_state(engine, app.state, args, supported_tasks=SUPPORTED_TASKS)
            print(f"[kot-vllm-ocr] serving on 127.0.0.1:{cli.port}: {json.dumps(info)}", flush=True)
            await serve_http(
                app, sock=None, host=args.host, port=args.port, log_level="warning",
            )

    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
