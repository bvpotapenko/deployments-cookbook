# Run Book: ocr-vllm

This deployment serves DeepSeek-OCR-2 with vLLM 0.32.0. One process serves the
OpenAI chat surface and the full-vocabulary log-probability surface on one
port. The full-vocabulary tap is version-pinned, so the server refuses to
start on any other vllm version. The engine applies the checkpoint's own
no-repeat n-gram rule at engine level.

## Model parameters

The table holds the launch parameters per model. Every value not named stays
at the deployment default. Read a model's row against the same row in the
other engine run books. A mismatch between engines is the first suspect when
a comparison fails.

| model | revision | cards | `KOT_VLLM_GPU_MEM` | `KOT_VLLM_MAX_BATCHED_TOKENS` | `KOT_VLLM_MAX_MODEL_LEN` | dtype | sampling (recommended) |
|---|---|---|---|---|---|---|---|
| deepseek-ai/DeepSeek-OCR-2 | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | 1 | 0.35 | 2048 | unset | bfloat16 | temperature 0.0001, top_p 1.0 |

One stack serves every model: CPython 3.12, vLLM 0.32.0, the torch build that
carries sm_120, and the transformers line the vLLM 0.32.0 resolver takes. The
seed arrives with the request; 42 is the convention.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The stack runs on CPython 3.12.

```bash
# networked host
uv venv /root/kotvenv-vllm-ocr --python 3.12
uv pip install --python /root/kotvenv-vllm-ocr/bin/python "vllm==0.32.0"

# offline host: build the closure, then install from it
./make_wheels.sh
uv venv --seed /scratch/kot-vllm-ocr-venv --python 3.12
/scratch/kot-vllm-ocr-venv/bin/pip install --no-index --no-cache-dir \
  --find-links ./wheels "vllm==0.32.0"
```

The torch build must carry sm_120. The vLLM 0.32.0 resolver pairs one on its
own. A torch without Blackwell support dies at the first prefill on these
cards.

`make_wheels.sh` builds the vllm 0.32.0 closure into `wheels/` for linux
x86_64, cp312. The exact pins live in `wheels-requirements.txt`, the pip
freeze of a venv that passed `smoke.sh`. The script downloads one wheel per
pin, eight in parallel, each with `--no-deps`. Files already present are
skipped, so a rerun resumes. Rerun the script until the two counts it
prints match. The repository does not carry `wheels/`.

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

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer.

```bash
mkdir -p /opt/engines/ocr-vllm
rsync -a --partial -e "ssh -p $PORT" ./ocr-vllm/ \
  user@$HOST:/opt/engines/ocr-vllm/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/ocr-vllm/*.sh'
rsync -a --partial -e "ssh -p $PORT" user@$HOST:/outputs/ ./outputs/
```

A networked host downloads the model itself. Otherwise push the snapshot
from the local hub cache; `--copy-links` resolves the cache's blob symlinks
into real files:

```bash
hf download <org>/<model> --revision <rev> --local-dir /models/<model>
rsync -a --partial --copy-links -e "ssh -p $PORT" \
  ~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<rev>/ \
  user@$HOST:/models/<model>/
```

Verify: the revision matches the model table's pin and the file list
matches the source. The two patched files travel in the same push.

### Start

```bash
CUDA_VISIBLE_DEVICES=0 KOT_PYTHON=/root/kotvenv-vllm-ocr/bin/python \
  ./kot_serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device cuda
```

The engine opens tensor parallel over every visible card. This model fits
on one card. Pin one with `CUDA_VISIBLE_DEVICES`.

At every start the launcher copies `kot_ngram.py` into the venv's
site-packages and passes `--logits-processors
kot_ngram:KotNoRepeatNGramLogitsProcessor` to the engine. The rule runs at
engine level, so the module must resolve inside every worker. The copy is
what makes it resolve.

The launcher raises the file-descriptor limit to the hard cap itself. The
stock 1024 limit kills the engine core under many concurrent connections
with `Too many open files`.

Prefix caching is disabled in the server. A cache hit would leave the
requested rows of the cached prefix never computed.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_PYTHON` | ambient `python3` | interpreter that carries vllm 0.32.0. The launcher's search never finds `/root/kotvenv-vllm-ocr`; set the variable. |
| `KOT_VENV` | `/scratch/kot-vllm-ocr-venv` | where the launcher builds the offline venv; it falls back to `/tmp/kot-vllm-ocr-venv`. |
| `KOT_VLLM_GPU_MEM` | `0.35` | `gpu_memory_utilization`. Headroom is needed for the log-probability pass. |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | caps the per-step transient of that pass. |
| `KOT_VLLM_MAX_MODEL_LEN` | the checkpoint's 8192 | caps `max_model_len`, and with it the reserved KV cache. |
| `KOT_VLLM_MOE_BACKEND` | `triton` | MoE kernels. `auto` picks flashinfer-cutlass, whose JIT needs nvcc; without a toolkit it dies at engine init. |
| `KOT_VLLM_DTYPE` | engine auto | compute dtype, for example `float32` for a cross-check. |
| `KOT_VLLM_COMPILATION` | engine default | `eager` disables torch.compile. |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived | override the reported identity. |

## Models

### deepseek-ai/DeepSeek-OCR-2

The model runs on one card. The weights directory is a hub snapshot of
`deepseek-ai/DeepSeek-OCR-2` at revision
`aaa02f3811945a91062062994c5c4a3f4c0af2b0`, plus two patched files over
the snapshot, `config.json` and `modeling_deepseekv2.py`. Take the whole
directory. Never refresh single files from the hub.

The engine applies the checkpoint's own no-repeat n-gram rule. The rule
blocks every token that would complete a 30-gram already seen inside a
90-token lookback window. The table delimiters `<td>` (128821) and `</td>`
(128822) stay exempt, so table structure may repeat. The rule changes which
token is sampled, never a reported row. The reported rows are the raw
log-softmax of the logits, computed before the rule runs.

The sampling temperature 0.0001 is near-greedy by design. Every decode
runs through the engine's real sampler. The checkpoint's own recipe names
0.0.

The checkpoint's tokenizer ships no chat template. The deployment carries
`chat_template.jinja` and applies it. The template renders the checkpoint's
plain prompt format, the image placeholder first and the instruction after.
The tokenizer adds BOS itself. Exactly one image rides a request.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "/models/DeepSeek-OCR-2", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The `model` field accepts the weights path and the identity the server
prints at startup. The identity comes from the checkpoint's `config.json`.
A bare text prompt carries no image. A document request carries the image
as one `image_url` data-URI part.
