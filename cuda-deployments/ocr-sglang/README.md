# ocr-sglang (cuda)

SGLang 0.5.21 behind the kot deployment contract for the OCR family
(DeepSeek-OCR-2, loaded through the checkpoint's own remote code -- the
0.5.21 line is the first with OCR-2 support; the llm deployment's 0.5.5
cannot run this checkpoint). One adapter process embeds `sglang.Engine`,
so `/kot/score` and the speed harness drive the same resident engine.

## Files

- `kot_serve.sh` — the one contract entry point: `--port --model --device`.
  Installs the patch layer (`sitecustomize.py`) and the anti-repetition rule
  (`kot_ocr_ngram.py`) into the engine venv's site-packages (sglang replaces
  its subprocess environment, so PYTHONPATH does not survive into the
  scheduler), picks the stash dir, execs the adapter.
- `kot_server.py` — the adapter: `/kot/info`, `/kot/score`, `/v1/completions`,
  `/v1/chat/completions` (multimodal: a text part and the document as an
  image_url data URI, streaming SSE).
- `sitecustomize.py` — the patch layer. Applied inside every sglang process.
- `kot_ocr_ngram.py` — DeepSeek's no-repeat-ngram rule as an sglang
  `CustomLogitProcessor`, serialized per request.
- `smoke.sh MODEL [device]` — the by-hand contract verification; the fixture
  image is found in the known kot store layouts or named with `IMAGE=`.

## Numerics

- Prefill: the request's transcript ids are appended, verbatim, to the image
  template the checkpoint itself defines (plain format: BOS, one image
  placeholder, "\nFree OCR."). The ids and the image (a path in
  `attachments`) go to the engine together; the multimodal processor expands
  the placeholder into the block it computes for the image (257 tokens for
  a <=768px image, 144 more per crop tile). The patch stashes the
  full-vocabulary log-softmax of every expanded position in float32; the
  contract's rows are the last `len(transcript) - 1` of them minus the final
  (sampling) row. `input_ids.npy` stores the transcript.
- Decode: image-driven (the prompt is the template alone). The request's
  sampling object is applied exactly as given; `allow_eos` gates the stop.
  Rows are stashed at `Sampler.forward` entry — this family's decodes
  generate until EOS, so `min_new_tokens` is 0 and the penalizer the llm
  deployment hooks never exists. One row per sampled token, gated on
  `return_logprob` so speed traffic writes nothing. `input_ids.npy` stores
  the expanded template (BOS + the image block + suffix) plus the generated
  ids; the template length is the engine's own (a template probe over the
  same image, when `meta_info.prompt_tokens` does not carry it).
- Anti-repetition: DeepSeek's own no-repeat-ngram rule (30-gram, 90-token
  window, `<td>`/`</td>` whitelisted), identical to the transformers and vllm
  OCR deployments, applied to every sampling request after the stash, so the
  stored rows stay the raw pre-processing distributions.
- TP-rank safety: under tensor parallelism every rank runs the patched code
  with identical logits; the patch gates stashing on rank 0 and puts the pid
  in every stash filename.

## CUDA host setup (worked commands, vast.ai 2x RTX 5090)

```bash
uv venv /root/kotvenv-sglang-ocr --python 3.12
uv pip install --python /root/kotvenv-sglang-ocr/bin/python "sglang==0.5.21"
export KOT_SGLANG_PYTHON=/root/kotvenv-sglang-ocr/bin/python
./kot_serve.sh --port 8798 --model /path/to/DeepSeek-OCR-2 --device cuda
KOT_SGLANG_PYTHON=... ./smoke.sh /path/to/DeepSeek-OCR-2 cuda
```

Known environment problems and their fixes live in
`~/.kerminal/projects/kot/shortcuts/vllm_on_blackwell_deployment.md`
(DeepGEMM sm_120, scheduler env curation, mem-fraction heuristic). The
short version: `SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0` and an
explicit `mem_fraction_static` are exported/set by the entry point.

## Known limits

- First boot compiles; a warm inductor cache makes the second boot much
  faster.
- One image per request (`limit_mm_per_prompt` is the engine's default 1),
  which is the whole OCR load.
- The `/v1/completions` surface is text-only (no image): the OCR harness
  never drives it; a document is a chat message.
- The remote-code path means the checkpoint's own modeling file runs inside
  the engine; a native arch would be faster, and none exists yet in 0.5.21.
