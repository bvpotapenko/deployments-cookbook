# Run Book: ocr-vllm

vLLM 0.30.0 V1 engine behind one process: the OpenAI chat surface and a
full-vocabulary log-probability surface, same port. vLLM 0.30.0 is the
first stable line with native DeepSeek-OCR-2 support
(`DeepseekOCR2ForCausalLM`); the server refuses any other vllm version.
The engine applies DeepSeek's own no-repeat n-gram rule for OCR output
at engine level.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

CPython 3.12. The closure carries networkx 3.7, which refuses 3.11: on
a 3.11 host the install fails with a resolver error, and the error does
not name the real cause.

```bash
# networked host: provision the venv the launcher is pointed at
uv venv /root/kotvenv-vllm-ocr --python 3.12
uv pip install --python /root/kotvenv-vllm-ocr/bin/python vllm==0.30.0

# offline host: build the closure once, then install from it with no network
./make_wheels.sh       # builds ./wheels: the vllm 0.30.0 closure (not
                      # committed; linux x86_64, cp312)
uv venv --seed /scratch/kot-vllm-ocr-venv --python 3.12
/scratch/kot-vllm-ocr-venv/bin/pip install --no-index --no-cache-dir \
  --find-links ./wheels "vllm==0.30.0"
```

Pins that must not float:

- `vllm==0.30.0`. The server asserts this exact version at boot; its
  log-probability surface patches V1 internals, so no other version is
  safe.
- `torch==2.13.0`. Its triton refuses the non-constexpr globals that
  vllm 0.30 kernels read; the server sets triton's documented escape
  hatch itself.
- `transformers==5.19.0`, `tokenizers==0.23.3`. Members of the verified
  closure. The tokenizer loads as `LlamaTokenizerFast` directly, with
  no remote-code path.
- The complete pin list is `wheels-requirements.txt`, regenerated as
  `pip freeze` of a venv that passed `smoke.sh`.

`make_wheels.sh` groups the downloads: one `pip download` per pin, eight
in parallel, each with `--no-deps`. Files already present are skipped,
so a rerun resumes; rerun until the two counts it prints match (transient
pypi timeouts).

Without `KOT_PYTHON`, and with no ambient vllm, the launcher runs the
offline install itself, into `KOT_VENV`.

### Layout

```text
/opt/engines/ocr-vllm     this directory
  chat_template.jinja     the chat template; the checkpoint ships none
  kot_ngram.py            copied into the venv's site-packages at every start
  wheels/                 the offline closure, built by make_wheels.sh, not committed
/models/DeepSeek-OCR-2    weights; a hub snapshot plus two patched files
```

### Start

```bash
CUDA_VISIBLE_DEVICES=0 KOT_PYTHON=/root/kotvenv-vllm-ocr/bin/python \
  ./kot_serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device cuda
```

The engine opens tensor parallel over every visible card. This model
fits on one card; pin one with `CUDA_VISIBLE_DEVICES`.

At every start the launcher copies `kot_ngram.py` into the venv's
site-packages and registers `kot_ngram:KotNoRepeatNGramLogitsProcessor`
with the engine's `--logits-processors` mechanism. The rule runs at
engine level, so the module must resolve inside every worker; the copy
is what makes it resolve.

The launcher raises the file-descriptor limit to the hard cap itself.
The stock 1024 limit kills the engine core under many concurrent
connections with `Too many open files`.

Prefix caching stays off: a cache hit would leave the requested rows of
a cached prefix never computed.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_PYTHON` | ambient `python3` | the interpreter that carries vllm 0.30.0. The launcher's own search never finds `/root/kotvenv-vllm-ocr`; set the variable. |
| `KOT_VENV` | `/scratch/kot-vllm-ocr-venv` | where the launcher builds the offline venv; it falls back to `/tmp/kot-vllm-ocr-venv`. |
| `KOT_VLLM_GPU_MEM` | `0.35` | `gpu_memory_utilization`. Headroom for the full-vocabulary pass. |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | caps the per-step transient of that pass. |
| `KOT_VLLM_MAX_MODEL_LEN` | the checkpoint's 8192 | caps `max_model_len`, and with it the reserved KV cache. |
| `KOT_VLLM_MOE_BACKEND` | `triton` | MoE kernels. `auto` picks flashinfer-cutlass, whose JIT needs nvcc; without a toolkit it dies at engine init. |
| `KOT_VLLM_DTYPE` | engine auto | compute dtype, for example `float32` for a cross-check. |
| `KOT_VLLM_COMPILATION` | engine default | `eager` disables torch.compile. |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived | override the reported identity. |

## Models

### deepseek-ai/DeepSeek-OCR-2

One card. The model directory is a hub snapshot of
`deepseek-ai/DeepSeek-OCR-2` at revision
`aaa02f3811945a91062062994c5c4a3f4c0af2b0`, plus two patched files over
the snapshot: `config.json` and `modeling_deepseekv2.py`.

The no-repeat n-gram rule, DeepSeek's own for this checkpoint: a token
that would complete an n-gram already seen inside the lookback window is
banned from sampling. N-gram size 30, window 90, whitelist ids 128821
and 128822 (`<td>`, `</td>`), so table structure may repeat. The
reported log-probability rows are computed before the rule runs: the
rule changes which token is sampled, never a reported row.

Sampling: temperature 0.0001, top_p 1.0. The temperature is effectively
greedy; the checkpoint's own recipe names 0.0.

The checkpoint's tokenizer ships no chat template. The server loads this
directory's `chat_template.jinja`: image placeholder first, instruction
after. The tokenizer adds BOS itself. Exactly one image per request.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "/models/DeepSeek-OCR-2", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The `model` field accepts the weights path and the identity printed at
startup; the identity comes from the checkpoint's `config.json`. A bare
text prompt carries no image. A document request carries the image as
one `image_url` data-URI part.
