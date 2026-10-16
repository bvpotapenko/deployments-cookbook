# ocr-vllm deployment (kot contract v1)

One process: a vLLM `0.30.0` engine behind vLLM's own OpenAI app, with the
kot contract endpoints (`/kot/info`, `/kot/score`) added to the same FastAPI
app. `/v1/chat/completions` is what the speed harness drives, with
multimodal messages (a text part and the document as an image_url data URI,
streaming SSE for the benchmark and plain JSON for the probe); `/kot/score`
is what goldens are collected through.

vLLM 0.30.0 is the first stable line with native DeepSeek-OCR-2 support
(`DeepseekOCR2ForCausalLM`, its SAM2 + qwen2 vision stack implemented
natively); the 0.11 line our llm deployment pins cannot run this checkpoint.

## The numerics contract, as implemented

- The score prompt is the checkpoint's own plain format, verified against
  its conversation.py rendering: BOS, one image placeholder token,
  "\nFree OCR.", then the request's transcript ids verbatim. The image
  arrives as multi_modal_data and vLLM's processor expands the placeholder
  into the block it computes for the image (256+1 tokens for <=768 px, 144
  more per crop tile - the checkpoint's own conventions), so the expanded
  prompt is exactly what the transformers deployment builds.
- Prefill rows are the last `len(transcript) - 1` prompt rows; the engine
  generates one token to close the request and it is discarded.
- Full-vocabulary pre-sampling log-probs: `prompt_logprobs` / `logprobs` at
  k = vocab_size, scattered back into token-id order with a per-request
  permutation check (a padded vocab cannot sneak pad rows past it). The
  tap keeps the raw tensors instead of vLLM's ~vocab_size Logprob objects
  per position; see the header of `kot_serve.py`.
- Decode applies the request's sampling object exactly (temperature,
  top_p, top_k, min_p are native knobs) with the request's seed and no
  penalties. `allow_eos: false` demands exactly `max_new_tokens` ids (EOS
  suppressed via min_tokens); `true` accepts what the sampler emitted
  before EOS, at most that many, and the file is compacted to the rows
  actually produced.
- Prefix caching is OFF (a cache hit would leave cached-prefix rows never
  computed).

The chat template (the checkpoint's tokenizer ships none) is this
directory's `chat_template.jinja`: it renders the same plain format the
score path builds - image placeholder first, then the instruction - so both
surfaces feed the model the same shape of prompt. The tokenizer adds BOS
itself (`add_bos_token: true`).

## Run it

```bash
# provision (once, on a networked test host; the graded container installs
# offline from ./wheels instead):
uv venv /root/kotvenv-vllm-ocr --python 3.12
uv pip install --python /root/kotvenv-vllm-ocr/bin/python vllm==0.30.0

# serve and smoke:
./kot_serve.sh --port 8797 --model /path/to/DeepSeek-OCR-2 --device cuda
KOT_PYTHON=/root/kotvenv-vllm-ocr/bin/python ./smoke.sh /path/to/DeepSeek-OCR-2 cuda
```

Useful env vars (all optional):

| var | default | what |
|---|---|---|
| `KOT_PYTHON` | ambient python3 | use this interpreter, skip the venv |
| `KOT_VENV` | `/scratch/kot-vllm-ocr-venv` | where the offline venv is built |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived | override `/kot/info` provenance |
| `KOT_VLLM_DTYPE` | engine auto | e.g. `float32` for cross-checks |
| `KOT_VLLM_GPU_MEM` | `0.35` | gpu_memory_utilization (headroom for the logprob gather) |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | caps the per-step gather transient |
| `KOT_VLLM_MOE_BACKEND` | `triton` | MoE kernels; `auto` picks flashinfer-cutlass, whose JIT needs nvcc |
| `KOT_VLLM_MAX_MODEL_LEN` | `16384` | template + transcript + decode headroom |
| `KOT_VLLM_COMPILATION` | engine default | `eager` disables torch.compile |

## Wheels (the handed-back form)

The graded container has no network; `make_wheels.sh` builds `./wheels` once
from any networked host (vllm 0.30.0 plus its torch and CUDA libs, pinned
in the script), and `kot_serve.sh` installs offline from it on first start.
The wheels are not committed.

## Known limits

- bf16 run-to-run determinism under vLLM's scheduler is verified by the
  smoke (same request twice, byte-identical files); if a future engine
  version breaks that, the smoke names it.
- `input_ids.npy` for a decode stores the expanded prompt (template with
  the full image block) plus the generated ids, per the contract's
  image-driven-unit rule.
- One image per request (`limit_mm_per_prompt`); the benchmark sends one
  document per request, so that is the whole load.
