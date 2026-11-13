# Run Book: DeepSeek-OCR-2 transformers 5 optimized

This deployment serves DeepSeek-OCR-2 with a vendored transformers 5.22.1
port on an Ascend 910B. The stack uses torch_npu 2.10.0.post2 under CANN
9.2.1. One process serves the full-vocabulary scoring route and the OpenAI
chat route. The score route runs at batch 1. The chat route uses continuous
batching over the same model modules.

The deployment was produced by the KOT migration task
`migrate_deepseek_ocr_2_to_transformers_5-20261112_073908-50c32856`. It passed
the code review and correctness evaluation on Ascend 910B.

## Model parameters

| model | cards | dtype | sampling (recommended) |
|---|---:|---|---|
| deepseek-ai/DeepSeek-OCR-2 | 1 | bfloat16 | temperature 0.0001, top_p 1.0, top_k 0, min_p 0.0 |

The tested stack uses CANN 9.2.1, torch 2.10.0, torch_npu 2.10.0.post2,
transformers 5.22.1 and CPython 3.12. The seed convention is 42.
The checkpoint stays outside this deployment and is loaded from `--model`.

## Server: Ascend 910B

### Environment

The host or container must provide the Ascend toolkit and driver. Source the
toolkit before a direct Python launch. The launcher does this for an NPU
launch.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

The deployment uses the Python interpreter at
`/usr/local/python3.12.16/bin/python3` when it exists. It falls back to
`python3` when that path is absent. The interpreter must provide:

```text
transformers==5.22.1
torch==2.10.0
torch_npu==2.10.0.post2
numpy
```

The deployment vendors the `ocr2/` package. It does not import the
checkpoint's bundled remote modeling code. The port loads the original model
weights from `--model` with `from_pretrained`.

### Layout

```text
migrate_deepseek_ocr_2_to_transformers_5-20261112_073908-50c32856/
  serve.sh              executable launcher
  serve.py              HTTP server and route adapters
  ocr2/                 transformers 5.22.1 model port
    model.py            model and configuration
    language.py         language decoder blocks
    qwen2_encoder.py    vision-language encoder
    sam.py              SAM vision encoder
    preprocess.py       image views and template expansion
    sampling.py         OCR sampling rules

/models/DeepSeek-OCR-2   original checkpoint, mounted read-only
/scratch                 writable HF and compiler scratch
```

The deployment directory is read-only during grading. The model directory is
also read-only. The launcher adds the deployment directory to `PYTHONPATH`.

### Start

```bash
./serve.sh --port 8090 \
  --model /models/DeepSeek-OCR-2 \
  --device npu
```

Pin one Ascend card with the visible-device variables:

```bash
ASCEND_RT_VISIBLE_DEVICES=2 \
ASCEND_VISIBLE_DEVICES=2 \
  ./serve.sh --port 8090 \
    --model /models/DeepSeek-OCR-2 \
    --device npu
```

The server binds `127.0.0.1`. The launcher raises the open-file limit to
65536 because the speed harness opens many connections. It sets
`HF_HUB_OFFLINE=1` and uses `/scratch/hf` as the default writable HF home.

The server loads one model on one card. It reports the actual transformers,
torch, torch_npu, driver and CANN versions through `/kot/info`.

### Parameters

| variable | default | effect |
|---|---|---|
| `HF_HOME` | `/scratch/hf` | writable Hugging Face cache and scratch directory |
| `HF_HUB_OFFLINE` | `1` | prevents network access during serving |
| `ASCEND_RT_VISIBLE_DEVICES` | host value | visible Ascend card ordinals |
| `ASCEND_VISIBLE_DEVICES` | fallback | visible Ascend card ordinals |
| `PYTHONPATH` | deployment directory first | makes the vendored `ocr2` package importable |

The server has no deployment-specific tuning variable. Keep the measured
model, dtype, preprocessing and sampling settings together.

## Models

### deepseek-ai/DeepSeek-OCR-2

The model uses the vendored `ocr2` port because the checkpoint's bundled
remote code targets the transformers 4.46 era. The port loads the original
checkpoint and reconstructs the architecture as transformers 5.22.1 model
classes.

The model uses bfloat16 on NPU. The configured vocabulary has 129,280 tokens.
The scoring route writes full-vocabulary log-probabilities in float32.

An image with both sides at most 768 pixels becomes one image block. A larger
image adds a grid of up to six 768x768 crop tiles. Each crop tile contributes
144 image tokens. The preprocessing order and image-token expansion are the
same for score and chat.

Generation uses the OCR no-repeat n-gram rule. A token that would complete a
30-gram already present in the previous 90 tokens is blocked. Tokens 128821
(`<td>`) and 128822 (`</td>`) remain allowed. The rule changes sampling only.
Stored log-probability rows use the raw model distribution before the rule.

The score path is batch 1 and was verified bit-exact against the trusted
transformers 4.46.3 reference for single-view and six-tile images. The chat
path uses continuous batching over the same modules. It keeps explicit tile
counts and offsets so variable tile counts do not change image ownership.

The measured chat speed on the tested Ascend 910B was about 258 output tokens
per second in an earlier 32-connection run. The 100-request KOT readings were:

```text
throughput 0.165209 document/s; latency p50/p95/p99 3017/103432/109079 ms;
TPOT p50/p95/p99 95.6168/103.267/103.267 ms
```

These values depend on the harness dataset, request limit, card and stack.
They are reference readings, not fixed service guarantees.

## Check

Start the service and check its identity:

```bash
curl -s http://127.0.0.1:8090/kot/info | python3 -m json.tool
```

The identity must report transformers 5.22.1, bfloat16, an NPU device and
vocabulary size 129280.

A score decode request writes arrays into its `out_dir`. The benchmark normally
provides frozen template ids and an image path under `/inputs`:

```bash
curl -N http://127.0.0.1:8090/kot/score \
  -H 'Content-Type: application/json' \
  -d '{"prompt_id":"probe","input_ids":[],"mode":"decode",
       "max_new_tokens":16,"allow_eos":true,
       "attachments":{"image":"/inputs/images/v1/probe.png"},
       "out_dir":"/out/probe","seed":42,
       "sampling":{"temperature":0.0001,"top_p":1.0,"top_k":0,"min_p":0.0}}'
```

Inspect the result:

```bash
python3 -c 'import numpy as np; x=np.load("/out/probe/log_probs.npy", mmap_mode="r"); print(x.shape, x.dtype, np.max(np.abs(np.logaddexp.reduce(x, axis=1))))'
```

The log-probability width must be 129280. Rows must be finite with row-wise
log-sum-exp close to zero. Replay `input_ids.npy` with prefill to test alignment.

The chat route accepts one text instruction and one base64 image part:

```bash
curl -N http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"DeepSeek-OCR-2",
       "messages":[{"role":"user","content":[
         {"type":"text","text":"Free OCR."},
         {"type":"image_url","image_url":
           {"url":"data:image/png;base64,<base64>"}}]}],
       "max_completion_tokens":16,"seed":42,
       "temperature":0.0001,"top_p":1.0,"top_k":0,"stream":true}'
```

Streaming returns OpenAI-shaped SSE events and ends with `data: [DONE]`.
The chat route and score route use one process, one model load, one image
preprocessing path and one set of model modules.

Stop an existing server before starting another check. A failed startup is a
service failure; do not leave an old process running when testing a new copy.
