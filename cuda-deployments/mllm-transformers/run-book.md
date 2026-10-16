# Run Book: mllm-transformers

A transformers reference-path server for `Qwen/Qwen3.8-27B` (the
`qwen3_5` architecture): one process, bfloat16, both cards. The server
pins every media-processing parameter and reports the full set in one
line at startup. Text is served; image and video are work in progress.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

CPython 3.12, venv `/root/kotvenv-mllm`; the launcher checks this exact
path.

```bash
uv venv /root/kotvenv-mllm --python 3.12
uv pip install --python /root/kotvenv-mllm/bin/python \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /root/kotvenv-mllm/bin/python \
  transformers==5.12.1 numpy accelerate pydantic pyyaml
```

Pins that must not float:

- `transformers==5.12.1`. The first line that carries the `qwen3_5`
  architecture. A newer 5.x silently replaces it, and the class
  `Qwen3_5ForConditionalGeneration` disappears.
- `torch==2.8.0` from the cu128 index: the sm_120 build.
  `torchvision==0.23.0` is its version pair; install both from the same
  index.

`numpy`, `accelerate`, `pydantic` and `pyyaml` complete the server's
imports; `accelerate` carries `device_map=auto`.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3.8-27B --device cuda
```

The model needs both cards. Leave `CUDA_VISIBLE_DEVICES` unset;
`device_map=auto` shards the weights over every visible card.

The launcher checks `KOT_MLLM_PYTHON`, then
`/root/kotvenv-mllm/bin/python`, then `/root/kotvenv/bin/python`, then
the PATH. `KOT_PYTHON` does not reach this launcher. A missing venv
falls through to the shared llm venv, whose transformers carries no
`qwen3_5`; the load fails there.

The server binds `0.0.0.0`, every interface, and answers one request at
a time.

### Parameters

The launcher reads one variable; the request body carries the sampling
parameters.

| parameter | default | effect |
|---|---|---|
| `KOT_MLLM_PYTHON` | `/root/kotvenv-mllm/bin/python` | the interpreter the launcher execs. |
| `sampling.temperature` | `0.7` | decode sampling; the instruct-mode value. |
| `sampling.top_p` | `0.8` | decode sampling. |
| `sampling.top_k` | `20` | decode sampling. |
| `seed` | `42` | planted before each decode; same seed, same output. |
| `sampling.presence_penalty` | `0.0` | neutral values pass; any non-zero value is an error that names it. The parameter belongs to the OpenAI surface. |

A `0.0` value for temperature, top_p or top_k reads as unset: the
server falls back to the default.

## Models

### Qwen/Qwen3.8-27B

The `qwen3_5` architecture, class `Qwen3_5ForConditionalGeneration`;
text, image and video input; bfloat16 over both cards.

The pinned media preprocessing, as the server reports it. The values
reproduce the model's own preprocessing and must not float: a changed
value changes the pixels the model sees.

| parameter | value | source |
|---|---|---|
| video frame sampling | fps 2.0, `do_sample_frames` true | the model's cookbook default |
| video pixel budget | longest_edge 25165824, shortest_edge 4096 | the checkpoint's `video_preprocessor_config.json` |
| image pixel budget | longest_edge 16777216, shortest_edge 65536 | the checkpoint's `preprocessor_config.json` |
| patch geometry | patch 16, temporal patch 2, merge size 2 | the checkpoint's configs |
| instruct mode | `enable_thinking` false in the chat-template rendering | the model's non-thinking mode |

The pixel budgets and the patch geometry are read from the checkpoint at
startup; the frame sampling is pinned in `serve.py`.

Instruct mode samples at temperature 0.7, top_p 0.80, top_k 20,
presence_penalty 1.5, repetition_penalty 1.0. This engine applies
temperature, top_p and top_k. It refuses a non-zero `presence_penalty`:
the parameter belongs to the OpenAI surface, where vllm and sglang
apply it. A `repetition_penalty` field is not read; generation runs at
the neutral 1.0.

Current support: the server starts, reports its settings and serves
text; the image and video paths are work in progress.

## Check

This server has no OpenAI chat surface. Its only request path takes
token ids, not text.

The startup line is the first check: it proves the model loaded, and it
records every pinned parameter.

```bash
curl -s http://127.0.0.1:8090/
```

A live server answers an unknown path with its JSON error body
(`{"error": "not_found", "detail": "/"}`). A dead port answers nothing.
