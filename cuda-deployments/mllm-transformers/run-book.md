# Run Book: mllm-transformers

This deployment serves Qwen/Qwen3.8-27B with transformers 5.20.2. The model
runs the `qwen3_5` architecture and takes text, image and video input. One
process holds the model in bfloat16 over both cards. The server pins every
media-processing parameter and reports the full set in one line at startup.
It serves text; the image and video paths land with the mllm porting work.

## Model parameters

The request body carries the sampling parameters. The table holds the card
count, the compute dtype and the recommended instruct-mode sampling.

| model | cards | dtype | sampling (recommended) |
|---|---|---|---|
| Qwen/Qwen3.8-27B | 2 | bfloat16 | temperature 0.7, top_p 0.80, top_k 20, presence_penalty 1.5, repetition_penalty 1.0 |

One stack serves the model: CPython 3.12, transformers 5.20.2, and the torch
and torchvision builds from the cu128 index. The seed arrives with the
request; 42 is the convention.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The venv runs CPython 3.12. Build it at `/root/kotvenv-mllm`; the launcher
checks that path.

```bash
uv venv /root/kotvenv-mllm --python 3.12
uv pip install --python /root/kotvenv-mllm/bin/python \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /root/kotvenv-mllm/bin/python \
  transformers==5.20.2 numpy accelerate pydantic pyyaml
```

torch and torchvision come from the cu128 index. That build carries the
sm_120 kernels; a torch without them stalls silently on these cards.
torchvision pairs with the torch build from the same index. transformers
5.20.2 ships the `qwen3_5` architecture.

### Layout

```text
/opt/engines/mllm-transformers   this directory
/models/<model>                 weights; a checkpoint directory
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer.

```bash
mkdir -p /opt/engines/mllm-transformers
rsync -a --partial -e "ssh -p $PORT" ./mllm-transformers/ \
  user@$HOST:/opt/engines/mllm-transformers/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/mllm-transformers/*.sh'
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
matches the source.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3.8-27B --device cuda
```

The model needs both cards. Leave CUDA_VISIBLE_DEVICES unset;
`device_map="auto"` spreads the layers over every visible card. The server
binds every interface and answers one request at a time.

The launcher checks `KOT_MLLM_PYTHON`, then
`/root/kotvenv-mllm/bin/python`, then `/root/kotvenv/bin/python`, then the
PATH. `KOT_PYTHON` does not reach this launcher. The mllm venv carries
torchvision. The shared llm venv does not; a fall-through load dies at the
processor import.

The server ignores `--device`; it loads bfloat16 with `device_map="auto"` on
every value.

### Parameters

The launcher reads one variable. The request body carries the sampling
parameters.

| variable | effect |
|---|---|
| `KOT_MLLM_PYTHON` | the interpreter the launcher execs |

| argument | effect |
|---|---|
| `--port` | listen port |
| `--model` | the checkpoint directory |
| `--device` | accepted and unused |

## Models

### Qwen/Qwen3.8-27B

The architecture is `qwen3_5`, class `Qwen3_5ForConditionalGeneration`. The
model takes text, image and video input and loads in bfloat16 over both
cards.

The table holds the pinned media preprocessing, as the server reports it at
startup. The values reproduce the model's own preprocessing. A changed value
changes the pixels the model sees.

| parameter | value | source |
|---|---|---|
| video frame sampling | fps 2.0, `do_sample_frames` true | the model's cookbook default |
| video pixel budget | longest_edge 25165824, shortest_edge 4096 | the checkpoint's `video_preprocessor_config.json` |
| image pixel budget | longest_edge 16777216, shortest_edge 65536 | the checkpoint's `preprocessor_config.json` |
| patch geometry | patch 16, temporal patch 2, merge size 2 | the checkpoint's configs |
| instruct mode | `enable_thinking` false in the chat-template rendering | the model's non-thinking mode |

The server reads the pixel budgets and the patch geometry from the checkpoint
at startup. The frame sampling is pinned in serve.py.

Instruct mode samples at temperature 0.7, top_p 0.80, top_k 20,
presence_penalty 1.5, repetition_penalty 1.0. The engine applies temperature,
top_p and top_k. It has no presence_penalty parameter; the server refuses a
request that sets it above zero, with an error that names the field. A
repetition_penalty field is not read; generation runs at the neutral 1.0. A
value of 0.0 for temperature, top_p or top_k reads as unset; the server then
generates with the instruct-mode value.

The server plants the request's seed before each decode. The same seed gives
the same output.

## Check

This server has no OpenAI chat surface. Its only request path takes token
ids.

The startup line is the first check. It proves the model loaded, and it
records every pinned parameter.

```bash
curl -s http://127.0.0.1:8090/
```

A live server answers an unknown path with its JSON error body
(`{"error": "not_found", "detail": "/"}`). A dead port answers nothing.
