# Run Book: mllm-transformers

This deployment serves Qwen/Qwen3.8-27B with transformers 5.20.2. The model
runs the `qwen3_5` architecture and takes text, image and video input. One
process holds the model in bfloat16 over both cards. The server pins every
media-processing parameter and reports the full set in one line at startup.
It serves text and video; the image path lands with later porting work.

## Model parameters

The request body carries the sampling parameters. The table holds the card
count, the compute dtype and the sampling values, which come from the
checkpoint's own `generation_config.json`.

| model | cards | dtype | sampling |
|---|---|---|---|
| Qwen/Qwen3.8-27B | 2 | bfloat16 | temperature 1.0, top_p 0.95, top_k 20 |

One stack serves the model: CPython 3.12, transformers 5.20.2, torch
2.14.1+cu130, torchvision 0.29.1+cu130, torchcodec 0.17. The seed arrives
with the request; 42 is the convention.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The venv runs CPython 3.12. Build it at `/root/kotvenv-mllm`; the launcher
checks that path.

```bash
uv venv /root/kotvenv-mllm --python 3.12
uv pip install --python /root/kotvenv-mllm/bin/python \
  torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-mllm/bin/python --no-deps \
  torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-mllm/bin/python --no-deps torchcodec
uv pip install --python /root/kotvenv-mllm/bin/python \
  transformers==5.20.2 numpy pydantic pyyaml
```

The torch build must carry sm_120; cu130 does. torchvision 0.29.x pairs
with torch 2.14.x (0.28.x requires torch 2.13 and dies at import:
`operator torchvision::nms does not exist`); `--no-deps` keeps the torch
build untouched. torchcodec decodes the videos (torchvision's reader was
removed in torchvision 0.26) and needs the host's FFmpeg 6 libraries. An
existing venv with the same pins works: the launcher falls through to
`/root/kotvenv/bin/python`.

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
| video per-frame pixel cap | `cap_pixels_per_frame` true | the reference implementation's (qwen-vl-utils) behavior; transformers does not default to it, and warns when the flag is absent |
| video pixel budget | longest_edge 25165824, shortest_edge 4096 | the checkpoint's `video_preprocessor_config.json` |
| image pixel budget | longest_edge 16777216, shortest_edge 65536 | the checkpoint's `preprocessor_config.json` |
| patch geometry | patch 16, temporal patch 2, merge size 2 | the checkpoint's configs |
| instruct mode | `enable_thinking` false in the chat-template rendering | the model's non-thinking mode |

Without the per-frame cap a 30-second clip costs about 12,000 video tokens;
with it the worst clip so far costs 7,920. The count drives the KV cache and
the prompt length of every request.

The server reads the pixel budgets and the patch geometry from the checkpoint
at startup. The frame sampling and the cap are pinned in serve.py.

Instruct mode samples at temperature 1.0, top_p 0.95, top_k 20 - the
checkpoint's own `generation_config.json`. The engine applies temperature,
top_p and top_k. It has no presence_penalty parameter; the server refuses a
request that sets it above zero, with an error that names the field. A
repetition_penalty field is not read; generation runs at the neutral 1.0. A
value of 0.0 for temperature, top_p or top_k reads as unset; the server then
generates with the values above.

The video prompt is pinned in serve.py: "Describe what you see in the
video." A video decode renders it with the checkpoint's chat template
(`enable_thinking` false), expands the placeholder with the processor, and
generates. The stored ids are the raw form (one video placeholder) plus the
generated ids; the stored rows are generate's own per-step logits, passed
through a float32 log-softmax. A replay prefill re-renders the same prompt
from the attachment, asserts the stored ids' head matches, and scores the
stored tail (`n_rows` rows) with `logits_to_keep`.

`POST /kot/infer` is the speed surface: one video decode, the description
written as `text.txt` into the request's `out_dir`, the file names returned.

The server plants the request's seed before each decode. The same seed gives
the same output.

## Check

The smoke boots the deployment and drives both surfaces: a text decode and
its replay prefill, then a video decode twice (byte-identity; the stored ids
carry exactly one video placeholder), then the video replay twice
(byte-identity).

```bash
VIDEO=/workspace/kot/kot_store/inputs/videos/v1/video_430.mp4 \
  KOT_MLLM_PYTHON=/root/kotvenv/bin/python \
  ./smoke.sh /models/Qwen/Qwen3.8-27B cuda
```

The startup line is the first check. It proves the model loaded, and it
records every pinned parameter.
