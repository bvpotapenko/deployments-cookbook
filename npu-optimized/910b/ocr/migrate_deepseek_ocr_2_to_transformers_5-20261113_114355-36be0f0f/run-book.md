# Run Book: ocr-transformers (batched engine)

This deployment serves DeepSeek-OCR-2 on transformers 5.22.1 through the
native `deepseek_ocr2` architecture (no remote code runs). The stack is the
Ascend 910B: torch_npu 2.10.0.post2 under CANN 9.2.1, torch 2.10.0, CPython
3.12. One process serves one model on one card: the scoring route and the
OpenAI chat route run in the same process, through the same batched engine,
the same vision forward and the same generation loop.

## Model parameters

| model | revision | cards | dtype | sampling (recommended) |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-OCR-2 | (none on this mount) | 1 | bfloat16 | temperature 0.0001, top_p 1.0 |

The seed arrives with the request; 42 is the convention. `--model` names the
weights directory; the loader checks every checkpoint tensor against the
loaded model (bitwise, through the native conversion mapping) and stamps the
report into `/kot/info` under `load_report`.

## Layout

```text
serve.sh              interpreter chain + exec (the whole contract surface)
serve.py              HTTP layer: routes, request parsing, streaming answers
engine.py             the batched generation engine (score and chat share it)
fast_moe.py           dense all-experts formulation of the routed MoE
fast_vision.py        cached, on-device SAM rel-position tables
native_config.py      builds the native config from the checkpoint's config.json
native_load.py        native loader with strict + bitwise gates
native_processor.py   processor: tokenizer + image views (global + tiles)
self_check.py         the contract self-check (run it after any server change)
run-book.md           this file
```

`serve.sh --port P --model <dir> --device npu` sources the Ascend toolkit,
raises the open-file limit, cds to the model dir and execs `serve.py`. The
server binds 127.0.0.1 and answers only when the model is loaded and warm.

## The engine

Every request - score or chat - becomes a unit in one queue. The engine
thread (the only thread that touches the NPU) groups pending units that
share sampling settings into one batch and runs one batched prefill +
decode over left-padded rows; a unit whose log-probs are measured runs
alone (the contract's one-at-a-time score load guarantees that anyway).
Chat requests stream text deltas in blocks as they are produced; scored
decodes stream `log_probs.npy` rows in blocks and one JSON result line.

Numerics, identical in shape to the reference deployment:

* ids are consumed verbatim; a decode with empty ids builds the checkpoint's
  own template (BOS, the image placeholder, `\nFree OCR.`);
* the sampling order is `generate()`'s own: min-new-tokens EOS suppression
  (when `allow_eos` is false), DeepSeek's no-repeat-30-gram-in-90 rule (the
  `<td>`/`</td>` tags excepted), the temperature/top_k/top_p/min_p warpers,
  softmax, one multinomial per row - the seed planted immediately before
  the batch, exactly where the reference plants it before `generate()`;
* the stored rows are the raw pre-processor logits, log-softmaxed in
  float32 over the full 129280-token vocabulary, streamed through
  `numpy.lib.format.open_memmap` (a decode that ends early gets its file
  header patched and truncated in place);
* prefill is teacher forced over the request's ids plus the transcript
  tail: row `t` predicts `tail[t]`.

Speed work, all of it vendored here and applied at load time (no library
outside this directory is modified):

* `fast_moe.py`: the stock experts path dispatches to `aten::_grouped_mm`,
  which torch_npu decomposes into 1408 per-expert `mm` launches per decode
  step (launch-bound, ~45 ms/step). The replacement computes all experts
  with two flat GEMMs per layer and combines the routed top-6 through a
  zero-masked weight map - the same routed math, one kernel per operation.
  Validated against the stock path: identical greedy token streams, module
  outputs within bf16 reduction-order noise.
* `fast_vision.py`: the SAM attention's relative-position tables were built
  per call with CPU index tensors (24 blocking H2D copies per vision chunk,
  ~12 ms each). They depend only on the frozen rel-pos parameters and the
  shape pair, so they are built once on device and cached. Bit-identical to
  the stock computation (verified, 0.0 max diff).
* The vision tower runs batched (the native `get_image_features`, chunked to
  bound the SAM encoder's transient activations); the per-image feature
  lists are exactly what the native scatter consumes, in batch order.
* The decode step runs over a preallocated `StaticCache`; the n-gram rule
  runs as device ops (no per-token host sync).
* Batched decode amortizes the per-step op-dispatch cost (the card's eager
  floor is ~1450 aten ops/step): B=16 runs at ~37 ms/step, B=84 at ~81
  ms/step - per-token-row cost drops ~20x from the one-at-a-time path.

Measured on this card with 100 concurrent one-document chat requests
(stream, max_completion_tokens 1024, synthetic document corpus): the
reference deployment 0.061 docs/s; this deployment 0.65-0.81 docs/s.

## Knobs

| variable | default | effect |
|---|---|---|
| `KOT_BATCH_MAX` | 100 | largest number of units in one batch |
| `KOT_BATCH_TOKENS` | 400000 | position budget of one batch (batch_size x (prompt+max_new)) |
| `KOT_BATCH_WAIT` | 0.5 | seconds the scheduler waits for more units to join a batch |
| `KOT_FLUSH_EVERY` | 16 | steps between streamed text/log-prob flushes |
| `KOT_VISION_GLOBAL_CHUNK` | 8 | images per vision chunk (bounds SAM transients) |
| `KOT_VISION_TILE_CHUNK` | 16 | tiles per vision chunk |
| `KOT_ATTN` | `sdpa` on npu | attention implementation for the whole model |
| `KOT_DTYPE` | `bfloat16` on npu | model dtype; a typo is a load-time error |
| `KOT_TIMING` | unset | per-batch phase timings on stdout |

## Check

```bash
python3 self_check.py --deployment . --model /model --device npu
```

The check starts `serve.sh` itself, runs a decode, replays it as a prefill,
and exercises the chat route with and without streaming. Stop any running
server first. Note the check builds its template ids with the model's own
tokenizer files (`TokenizersBackend`); it does not run the checkpoint's
remote code, which cannot execute on transformers 5.x (it predates the
engine move and imports symbols 5.x no longer ships).
