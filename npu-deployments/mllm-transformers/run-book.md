# Run Book: mllm-transformers

This deployment serves Qwen/Qwen3.8-27B with transformers 5.22.1 on Ascend 910B. The model
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

One stack serves the model: CPython 3.12, transformers 5.22.1, torch 2.10.0, torch_npu 2.10.0.post2, torchvision 0.25.0, torchcodec 0.17. The seed arrives
with the request; 42 is the convention.

## Server: Ascend 910B

### Environment

The venv runs CPython 3.12. Build it at `/root/kotvenv-mllm`; the launcher
checks that path.

```bash
uv venv /root/kotvenv-mllm --python 3.12
uv pip install --python /root/kotvenv-mllm/bin/python \
  torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-mllm/bin/python --no-deps \
  torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv-mllm/bin/python --no-deps \
  torchcodec==0.17
uv pip install --python /root/kotvenv-mllm/bin/python \
  transformers==5.22.1 accelerate numpy pydantic pyyaml pillow \
  einops flash-linear-attention==0.2.2
```

The torch build must carry sm_120; cu130 does. torchvision 0.29.x pairs
with torch 2.14.x (0.28.x requires torch 2.13 and dies at import:
`operator torchvision::nms does not exist`); `--no-deps` keeps the torch
build untouched, so the recipe installs pillow itself. torchcodec decodes
the videos (torchvision's reader was removed in torchvision 0.26) and
needs the host's FFmpeg 6 libraries. accelerate is required by
`device_map="auto"`. flash-linear-attention 0.2.2 and einops run the
hybrid GatedDeltaNet layers as kernels; the reference PyTorch fallback
OOMs a 32 GB card on a ~12.4k-token video prefill. The launcher falls
through to `/root/kotvenv/bin/python` only if that venv carries the same
stack.

### Layout

```text
/opt/engines/mllm-transformers   this directory
/models/<model>                 weights; a checkpoint directory
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer. The excludes keep `__pycache__` and `.DS_Store` off the box.

```bash
mkdir -p /opt/engines/mllm-transformers
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./mllm-transformers/ user@$HOST:/opt/engines/mllm-transformers/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/mllm-transformers/*.sh'
rsync -a --partial -e "ssh -p $PORT" user@$HOST:/outputs/ ./outputs/
```

A networked host downloads the model itself. On vast, huggingface.co
times out, so the download goes through the mirror endpoint. Otherwise
push the snapshot from the local hub cache; `--copy-links` resolves the
cache's blob symlinks into real files:

```bash
HF_HUB_DISABLE_TELEMETRY=1 \
HF_ENDPOINT=https://hf-mirror.com \
  hf download <org>/<model> --revision <rev> --local-dir /models/<model>
rsync -a --partial --copy-links -e "ssh -p $PORT" \
  ~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<rev>/ \
  user@$HOST:/models/<model>/
```

Verify: the revision matches the model table's pin and the file list
matches the source.

### Start

```bash
./serve.sh --port 8090 --model /models/Qwen3.8-27B --device npu
```

The model needs both cards. Leave ASCEND_RT_VISIBLE_DEVICES unset;
`device_map="auto"` spreads the layers over every visible card. The server
binds every interface and answers one request at a time.

The launcher checks `KOT_MLLM_PYTHON`, then
`/root/kotvenv-mllm/bin/python`, then `/root/kotvenv/bin/python`, then the
PATH. `KOT_PYTHON` does not reach this launcher. The mllm venv carries
torchvision. The shared llm venv does not; a fall-through load dies at the
processor import.

The server imports `torch_npu` for `--device npu` and loads bfloat16 with `device_map="auto"` across the visible Ascend cards.

### Parameters

The launcher reads one variable and the server reads one. The request body
carries the sampling parameters.

| variable | effect |
|---|---|
| `KOT_MLLM_PYTHON` | the interpreter the launcher execs |
| `KOT_TF_REPLAY_CHUNK` | the tokens per replay call after the prompt call; default 2048, minimum 2. The size is part of the numerics and is reported in `/kot/info`. Captures run at the default |

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

The per-frame cap binds on short clips. At 1080p, 22 frames cost 12,100 video
tokens without it and 7,920 with it; from 32 frames on it changes nothing, and
a 30-second clip costs 11,700 either way. The count drives the KV cache and
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
through a float32 log-softmax. Each row goes to host memory at its step, so
a long decode does not fill card 0. At each 256 rows the server runs the
log-softmax of the block on card 0 and writes the block into
`log_probs.npy`, which it opens at `max_new_tokens` rows. The host holds one
block, 0.24 GiB, at a time. When `allow_eos` is true and the decode stops
early, the server copies the rows into a new file, block by block, and the
new file replaces the old one.

A replay prefill re-renders the same prompt from the attachment, asserts
the stored ids' head matches, and scores the stored tail (`n_rows` rows).
One call over the prompt and an 8192-token tail does not fit on card 0, so
the replay runs in calls through one cache. The first call is the expanded
prompt with the video, the same as the first call of the decode. Then the
tail goes in chunks of `KOT_TF_REPLAY_CHUNK` tokens. A chunk call gets only
its ids and the cache: no attention mask, no pixels. So each row sees the
video, the prompt and all tail ids before the one it predicts. The lm_head
runs on 256 rows at a time, and the rows of each call go into
`log_probs.npy` when the call returns.

The chunk size is part of the numerics: the float order changes at each call
edge, so rows produced at one size need not be bit-identical to rows
produced at another. `/kot/info` reports `replay_chunk_tokens`, so every
capture carries its own.

`POST /kot/infer` runs one video decode, the description written as `text.txt`
into the request's `out_dir`, the file names returned. The benchmark's speed
harness drives the chat route below.

`POST /v1/chat/completions` accepts the OpenAI video message shape. The user
message content must contain one `video_url` part and one or more text parts.
The video URL must be a local absolute path or a `file://` URL. The server uses
the same `render_video()` and `model.generate()` path as `/kot/infer`, with
fps 2.0, the per-frame pixel cap, `enable_thinking=false`, the request seed,
and temperature 1.0, top_p 0.95, and top_k 20. `max_completion_tokens` is
preferred, with `max_tokens` as an alias and 512 as the default. The response
is OpenAI-shaped. `stream: true` returns assistant role and text chunks, a
final `stop` chunk, and `data: [DONE]`.

The server plants the request's seed before each decode. The same seed gives
the same output.

## Check

The identity surface answers with the engine's own facts:

```bash
curl -s http://127.0.0.1:8090/kot/info | python3 -m json.tool
```

The generation surface takes a video and writes its answer to `out_dir`:

```bash
curl -s http://127.0.0.1:8090/kot/infer \
  -H 'Content-Type: application/json' \
  -d '{"prompt_id": "probe", "out_dir": "/tmp/kot-probe",
       "attachments": {"video": "/path/to/video.mp4"}}' \
  | python3 -m json.tool
cat /tmp/kot-probe/text.txt
```

The OpenAI video chat check uses a local video path. Add `-N` for SSE output:

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","messages":[{"role":"user","content":[
       {"type":"video_url","video_url":{"url":"/path/to/video.mp4"}},
       {"type":"text","text":"What happens in this video?"}]}],
       "max_completion_tokens":32}' \
  | python3 -m json.tool

curl -sN http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3.8-27B","stream":true,"messages":[{"role":"user","content":[
       {"type":"video_url","video_url":{"url":"/path/to/video.mp4"}},
       {"type":"text","text":"What happens in this video?"}]}],
       "max_completion_tokens":32}'
```

The startup line is the first check. It proves the model loaded, and it
records every pinned parameter.
