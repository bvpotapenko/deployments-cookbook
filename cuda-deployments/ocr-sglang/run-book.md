# Run Book: ocr-sglang

SGLang 0.5.21 behind one process: the multimodal OpenAI chat surface (a text
part and the document as an image data URI) and a full-vocabulary
log-probability surface, same port. The environment, the venv and the
launcher pattern are llm-sglang's; the same patch layer
(`sitecustomize.py`) stashes full-vocabulary float32 log-softmax rows at
engine-internal entry points, and it travels with one extra module, the
no-repeat ngram rule.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

Same venv, same install and same pins as llm-sglang:
`/root/kotvenv-sglang-ocr`, CPython 3.12, sglang 0.5.21, torch 2.13.0.

### Start

```bash
KOT_SGLANG_PYTHON=/root/kotvenv-sglang-ocr/bin/python \
  CUDA_VISIBLE_DEVICES=0 \
  ./kot_serve.sh --port 8798 --model /models/DeepSeek-OCR-2 --device cuda
```

The launcher behaves like llm-sglang's: it installs the patch layer into the
venv's site-packages (module plus `.pth`), sets the DeepGEMM and offline
variables, raises the file-descriptor limit and picks the stash dir; see the
llm-sglang run book. Two additions. It also copies `kot_ocr_ngram.py` into
site-packages: the scheduler deserializes the per-request custom logits
processor by name, in workers whose environment carries no `PYTHONPATH`.
And the server sets `trust_remote_code=True` and
`enable_custom_logit_processor=True`: the checkpoint ships its own modeling
and processor code, and the engine accepts custom processors only with the
flag on.

The engine opens tensor parallel over every visible card; expose one card
(`CUDA_VISIBLE_DEVICES=0`) on a two-card host, because one image per request
is the whole load. The server binds 127.0.0.1 only. One tiny image-driven
generation runs before the server listens, so the first real request does
not pay compile and graph costs.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_SGLANG_PYTHON` | ambient python3 | interpreter that has sglang; the launcher picks the first one that imports it. |
| `KOT_SGLANG_STASH_DIR` | `/scratch/kot-sglang-ocr-stash`, else `/tmp/kot-sglang-ocr-stash` | where the patch layer writes the full-vocabulary float32 rows. The name differs from llm-sglang's, so both servers can share one host. |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived from the weights | override the reported identity; a flat weights directory reports its basename. |

The memory fraction (0.45) and the prefill window (8192) are fixed in
`kot_server.py`; this deployment has no override for them.

## Models

### deepseek-ai/DeepSeek-OCR-2

One card. sglang 0.5.21 is the first line with DeepSeek-OCR-2 support: the
checkpoint loads through its own remote code, and no native architecture
exists in this line. Revision: the server reports the `_commit_hash` the
weights' `config.json` carries.

The prompt is the checkpoint's own plain format -- BOS, one `<image>`
placeholder, then `\n` + instruction. The chat surface builds it: the text
part is the instruction alone, and the tokenizer is a LlamaTokenizerFast
read from the weights. The processor expands the placeholder into 257
tokens for a <=768px image, 144 more per crop tile; one image per request.

Generation applies a no-repeat ngram rule through a per-request custom
logits processor (`kot_ocr_ngram.py`): ngram 30, window 90, the two table
tokens (`<td>`, `</td>`) whitelisted. The rule changes which token is
sampled, never the reported distributions.

Sampling: temperature 0.0001, top_p 1.0.

## Check

```bash
DOC="$(base64 -w0 doc.png)"   # macOS: base64 -i doc.png
curl -s http://127.0.0.1:8798/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "DeepSeek-OCR-2", "max_tokens": 256, "temperature": 0.0001,
       "messages": [{"role": "user", "content": [
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$DOC"'"}},
         {"type": "text", "text": "Free OCR."}]}]}' \
  | python3 -m json.tool
```

The image part must be a data URI: the server decodes it itself and fetches
no URLs. The text part is the instruction alone; the server adds BOS, the
image placeholder and the newline itself.
