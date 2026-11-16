# Run Book: ocr-transformers (batched engine)

This deployment serves DeepSeek-OCR-2 with transformers 5.22.1 on the Ascend
910B (torch_npu 2.10.0.post2, CANN 9.2.1). The checkpoint loads through the
native `deepseek_ocr2` architecture; no remote code runs; the weights load
from `--model` only.

## Engine

One process, one loaded model, one forward path for both endpoints:

- **HTTP** is asyncio: thousands of held-open connections are cheap, image
  preprocessing (PIL decode + the native image processor) runs on a bounded
  thread pool, and one engine thread owns the accelerator.
- **Waves**: requests are admitted together (up to `KOT_MAX_BATCH`, a 20 ms
  gather window) and decoded as one batch. Prompts are left-padded to one
  length over their single image placeholder, position ids are per row, and
  an explicit attention mask states where each row's real tokens are. A
  finished row is dropped from the batch (cache rows index-selected).
- **KV cache**: preallocated per wave (`KotStaticCache` in `engine.py`),
  mutated in place at the running position - no per-step reallocation.
- **Vision**: the model's own `get_image_features` batches images with
  explicit per-image tile counts (`KOT_VISION_IMAGES` per group,
  `KOT_VISION_TILES` per group, to bound the SAM attention-bias memory), so a
  batch with tile counts `[2, 6]` keeps each image's rows with its own tiles.
- **MoE**: one dense-experts computation for every endpoint and batch size
  (`install_dense_experts` in `engine.py`): two batched GEMMs over all
  experts, unselected experts weighted by exactly 0.0, fp32 accumulation -
  the same per-expert linear algebra as the native loop, without its
  per-expert kernel storm and per-layer `nonzero()` sync. Long prefills
  stream through it in 1024-token chunks.
- **Attention**: `sdpa` end-to-end (`KOT_ATTN` overrides; eager fallback).
  Bool masks keep the NPU flash path.
- **Sampling** (`engine._sample`): per row and exactly as transformers
  applies it - the min_new_tokens EOS ban, presence/frequency/repetition
  penalties (no-ops when neutral), DeepSeek's no-repeat 30-gram rule over
  the last 90 tokens with the table tokens whitelisted, temperature, top_k,
  top_p, min_p - then a seeded `torch.multinomial` per row, so a request's
  draws depend only on its own seed, never on its batch. The score route's
  stored rows are the raw pre-processor logits.
- A replayable NPU-graph decode path exists behind `KOT_GRAPHS=1` (shape-
  bucketed capture through the transformers `StaticCache`); it is off by
  default pending a residual cross-wave variance under investigation. The
  eager path above is the validated one.

## Endpoints

- `GET /kot/info` - provenance of the running process (engine
  `transformers 5.22.1`, dtype, device, vocabulary, load report).
- `POST /kot/score` - decode (seeded sampling, streamed NDJSON block lines)
  and teacher-forced prefill (`n_rows`); writes `log_probs.npy` (full-
  vocabulary float32 log-softmax, streamed row by row, header finalized at
  the true row count), `input_ids.npy` (int32) and `text.txt`.
- `POST /v1/chat/completions` - OpenAI-shaped, streaming SSE per token or
  one JSON; same engine, same views, same generation.

## Start

```bash
./serve.sh --port 8090 --model /model --device npu
```

The script names its interpreter absolutely, sources the CANN environment,
raises the open-file limit, and `cd`s to the model directory (relative
`out_dir`s resolve there). Startup (load + strict key gates + warmup through
both engine modes) is about a minute; `/kot/info` answers only when warm.

## Knobs (environment)

| variable | default | effect |
|---|---|---|
| `KOT_MAX_BATCH` | 16 | rows per decode wave |
| `KOT_VISION_IMAGES` | 4 | images per vision group |
| `KOT_VISION_TILES` | 24 | tiles per vision group |
| `KOT_BATCH_WINDOW` | 0.02 | seconds to gather a wave |
| `KOT_ATTN` | sdpa on npu/cuda, eager on cpu | attention implementation |
| `KOT_DTYPE` | bfloat16 on npu/cuda | model dtype |
| `KOT_GRAPHS` | 0 | experimental NPU-graph decode |
| `KOT_COMPILE` | 0 | experimental torch.compile (slow here) |
| `KOT_BITWISE` | 0 | checkpoint-vs-model bitwise gate at load |

## Validation on this card

- the example `self_check.py` passes all eight checks (decode, replay
  prefill, chat, streaming);
- a two-engine capture (frozen-template decode, replay prefill, no-template
  decode with EOS suppressed, chat) against the reference deployment shows
  identical decode ids and chat text; prefill log-probs agree with 0 argmax
  flips (mean |d| ~0.23 from the sdpa/dense kernel choices, the same class
  of bf16 noise any engine change makes);
- 100 document requests at once: **0.68 docs/s** (baseline single-request
  deployment: 0.098 docs/s).
