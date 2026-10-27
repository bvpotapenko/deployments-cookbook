# Run Book: ocr-sglang

This deployment serves one vision-language model with SGLang 0.5.22. One
process serves the multimodal OpenAI chat surface and the full-vocabulary
log-probability surface on one port. A chat request carries a text part and
the document as an image data URI. The environment and the launcher are
llm-sglang's. The patch layer is llm-sglang's. It stashes the
full-vocabulary float32 log-softmax rows of the raw logits at engine-internal
entry points, so the reported rows are pre-sampler. It also pins the
per-image token count, and the no-repeat ngram rule travels with it as
`kot_ocr_ngram.py`.

## Model parameters

The table holds the launch parameters per model. The memory fraction and the
prefill window are fixed in `kot_server.py`, and this deployment has no
override for them.

| model | cards | memory fraction | prefill window | sampling (recommended) |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-OCR-2 | 1 | 0.45 | 8192 | temperature 0.0001, top_p 1.0 |

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The environment is llm-sglang's. Both deployments share one venv,
/root/kotvenv-sglang-ocr, on CPython 3.12 with sglang 0.5.22, torch 2.14.1
and flashinfer-python 0.7.1. The llm-sglang run book holds the install.

### Layout

```text
/opt/engines/ocr-sglang       this directory
/models/<model>               weights; the engine reads the tokenizer and the processor from here
/root/kotvenv-sglang-ocr      the engine venv, shared with llm-sglang
<venv>/site-packages          the launcher refreshes the patch layer and kot_ocr_ngram.py here on every start
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer. The excludes keep `__pycache__` and `.DS_Store` off the box.

```bash
mkdir -p /opt/engines/ocr-sglang
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./ocr-sglang/ user@$HOST:/opt/engines/ocr-sglang/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/ocr-sglang/*.sh'
rsync -a --partial -e "ssh -p $PORT" user@$HOST:/outputs/ ./outputs/
```

A networked host downloads the model itself. On vast, huggingface.co
times out, so the download goes through the mirror endpoint. Otherwise
push the snapshot from the local hub cache; `--copy-links` resolves the
cache's blob symlinks into real files:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  hf download <org>/<model> --revision <rev> --local-dir /models/<model>
rsync -a --partial --copy-links -e "ssh -p $PORT" \
  ~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<rev>/ \
  user@$HOST:/models/<model>/
```

Verify: the revision matches the model table's pin and the file list
matches the source. The patched `modeling_deepseekv2.py` travels in the
same push.

### Start

```bash
KOT_SGLANG_PYTHON=/root/kotvenv-sglang-ocr/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
  ./kot_serve.sh --port 8798 --model /models/DeepSeek-OCR-2 --device cuda
```

The launcher behaves like llm-sglang's. It installs the patch layer into the
venv's site-packages as a module plus a `.pth` import line, sets the DeepGEMM,
offline and file-descriptor variables, and picks the stash dir. The
llm-sglang run book holds the details. It also copies `kot_ocr_ngram.py`
into site-packages. The scheduler deserializes the per-request custom logits
processor by name, in workers whose environment carries no PYTHONPATH.

The server sets `trust_remote_code=True` and
`enable_custom_logit_processor=True`. The checkpoint ships its own modeling
and processor code, and the engine accepts custom processors only with the
flag on.

The patch layer also pins the per-image token count. The engine resolves the
count through a method this processor lacks, and its fallback retokenizes the
request. The retokenized prompt drifts by a few tokens. The patch attaches the
missing method. The count comes from a blank image of the given size through
the processor's own pixel path, cached by size. The request ids stay verbatim.
The server refuses a request whose consumed length disagrees with the ids
plus the recorded image token counts. A retokenized request fails loudly.

The engine opens tensor parallel over every visible card. Expose one card,
`CUDA_VISIBLE_DEVICES=0`, on a two-card host, because one image per request
is the whole load. The server binds 127.0.0.1 only. One tiny image-driven
generation runs before the server listens, so the first real request pays no
compile and graph costs.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_SGLANG_PYTHON` | ambient python3 | interpreter that has sglang. The launcher tries it first, then a PATH python3. |
| `KOT_SGLANG_STASH_DIR` | `/scratch/kot-sglang-ocr-stash`, else `/tmp/kot-sglang-ocr-stash` | where the patch layer writes the full-vocabulary float32 rows. The name differs from llm-sglang's, so both servers can share one host. |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived from the weights | override the reported identity. A flat weights directory reports its basename. |

## Models

### deepseek-ai/DeepSeek-OCR-2

This model runs on one card. The checkpoint loads through its own remote code,
and this sglang line carries no native architecture for it. The weights
directory carries one patched file over the snapshot,
`modeling_deepseekv2.py`: Bogdan Potapenko edited the hub file for
transformers 5.x compatibility. The server reports
the `_commit_hash` that the weights' `config.json` carries.

The prompt is the checkpoint's plain format. BOS comes first, then one
`<image>` placeholder, then a newline and the instruction. Each request
carries one image. The processor expands the placeholder into 257 tokens for
a 620x620 image. A larger image adds crop tiles, and the block grows.

Generation applies a no-repeat ngram rule through a per-request custom logits
processor, `kot_ocr_ngram.py`. The rule bans a token that would complete a
30-gram already seen within a 90-token window. The table tokens `<td>` and
`</td>` stay allowed. The rule changes which token the sampler picks. It
never changes the reported log-probability rows.

## Check

```bash
DOC="$(base64 -w0 doc.png)"   # macOS: base64 -i doc.png
curl -s http://127.0.0.1:8798/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "DeepSeek-OCR-2", "max_tokens": 256, "temperature": 0.0001, "top_p": 1.0,
       "messages": [{"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$DOC"'"}},
         {"type": "text", "text": "Free OCR."}]}]}' \
  | python3 -m json.tool
```

The image part must be a data URI. The server decodes it itself and fetches no
URLs. The text part is the instruction alone, and the server adds BOS, the
image placeholder and the newline itself.

```bash
./smoke.sh /models/DeepSeek-OCR-2
```

The smoke drives the whole surface: a prefill twice, a decode under each EOS
gate, the chat surface, and a decode-then-prefill round trip over the ids the
decode wrote. It fails when the server log holds a token-drift line.
