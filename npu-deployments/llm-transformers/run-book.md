# Run Book: llm-transformers

This deployment serves text models with stock transformers 5.20.2. The stack
runs on the Ascend 910B through torch_npu 2.14.0 under CANN 9.2.0. One
process serves an OpenAI-shaped completion and chat route. The same process
serves a scoring route that streams full-vocabulary float32
log-probabilities to disk. Requests run strictly one at a time, with no
batching and no parallel streams.

## Model parameters

The table holds the launch parameters per model. Every value not named stays
at the deployment default. Read a model's row against the same row in the
other engine run books. A mismatch between engines is the first suspect when
a comparison fails.

| model | revision | cards | dtype | sampling (recommended) |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 1 | bfloat16 | temperature 0.6, top_p 0.95, top_k 20 |
| deepseek-ai/DeepSeek-V2-Lite | `604d5664dddd88a0433dbae533b7fe9472482de0` | 1 | bfloat16 | temperature 0.3, top_p 0.95 |
| state-spaces/mamba-2.8b-hf | `96c48e0292b63f5346b6d30061af2551f7101e26` | 1 | bfloat16 | temperature 0.7, top_p 0.9, repetition_penalty 1.2 |

One stack serves every model: CANN 9.2.0, torch 2.14.0, torch_npu 2.14.0,
transformers 5.20.2, CPython 3.12. The seed arrives with the request; 42 is
the convention.

## Server: Ascend 910B

### Environment

The image runs CPython 3.12 on aarch64. The wheels are cp312 manylinux, so
they install only under that Python.

```bash
# wheels/ must hold the three platform wheels first. The large PyPI
# downloads stall at times; curl -C - resumes them.
docker build -t kot-llm-ascend-910b:latest .
```

These pins must not float.

- The base image `quay.io/ascend/cann:9.2.0-910b-ubuntu22.04-py3.12` carries
  the CANN 9.2.0 toolkit for the 910B. The toolkit half of the stack lives
  in the image. The driver half lives on the host, and the wrap mounts it
  read-only at start. The driver on the host must pair with the CANN
  toolkit, so float the image only together with a host driver that supports
  CANN 9.2.0.
- The build installs `torch==2.14.0`, `torch_npu==2.14.0` and
  `torchvision==0.29.0` from wheel files, never pip resolution. torch_npu
  registers the npu backend, its version pairs with torch, and the plugin
  build pairs with the CANN release in the base image. torchvision must match
  the same torch line. The stack moves as a set. Change the CANN version only
  together with torch_npu, and the host driver only together with the CANN
  toolkit.
- The build pins `transformers==5.20.2`.
- The build installs the MLCommons Inference Endpoints client from GitHub.
  It is a client of the chat route. It needs git and a C compiler, so the
  image carries build-essential.

### Layout

```text
/opt/engines/llm-transformers     this directory; the wrap mounts it read-only at /deployment
/models/<model>                   weights; mounted read-only at their own path
```

`serve.py` is not baked into the image. The wrap mounts this directory
read-only at /deployment and runs the script from there, so you edit
`serve.py` here with no rebuild.

The server loads a plain checkpoint directory (config.json at the root), or
a HuggingFace cache that holds exactly one model under hub/. Several models
under hub/ stop the start; the error names them.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```

The script picks its own route. A host python3 that imports torch_npu runs
`serve.py` directly, with the CANN environment script sourced first. Every
other host uses the docker wrap. The driver files on the host are readable
by root only, so a non-root shell cannot import torch_npu and the wrap is
the route.

The wrap builds the image on the first run, then:

- runs privileged, with host networking, and `--ulimit nofile=65536:65536`;
- mounts the weights read-only at their own path, sets `HF_HUB_OFFLINE=1`,
  and gives a plain checkpoint a writable scratch HF home (/tmp/hf), so
  nothing writes into the read-only mount;
- passes the visible ordinals through `ASCEND_RT_VISIBLE_DEVICES` and
  `ASCEND_VISIBLE_DEVICES`, and adds one `/dev/davinci<ordinal>` device node
  per ordinal, beside the manager, svm and hdc nodes;
- mounts the host driver read-only: the Ascend driver tree, npu-smi, dcmi,
  sbin, `ascend_install.info`, `hccn.conf`; the NPU log directory mounts
  read-write at /usr/slog. A path missing on the host is skipped.

`ASCEND_RT_VISIBLE_DEVICES` wins, and `ASCEND_VISIBLE_DEVICES` is the
fallback. The values are card ordinals, comma separated.

Host networking is required. The server binds 127.0.0.1 inside the container,
and the host reaches it on the same loopback address. A published port does
not work, because it forwards to the container's ethernet address, where
nothing listens. The launcher raises the soft open-file limit to 65536. Without the raise, a
client that opens many pooled connections at once runs past the docker soft
limit of 1024, and accept() fails with EMFILE.

The load climbs a ladder of three attempts. The first attempt loads the
native architecture, without `trust_remote_code`. A `trust_remote_code`
error retries with the checkpoint's own code. An attention-implementation
error retries with eager attention.

Facts about a running server:

- torch_npu is imported before any tensor is created.
- The server plants the request seed immediately before the sampling call.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_DTYPE` | `bfloat16` on npu, `float32` on cpu | model dtype: bfloat16, float16 or float32. |
| `KOT_ATTN` | `eager` on npu | attention implementation the load requests. |
| `KOT_NPU_IMAGE` | `kot-llm-ascend-910b:latest` | image the host-mode wrap runs. |
| `KOT_OUT_MOUNTS` | unset | colon-separated directories mounted read-write at the same path, so the server's output files land where the host reads them. Host mode only. |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` when both device variables are unset | card ordinals the wrap passes to the container; `ASCEND_VISIBLE_DEVICES` is the fallback. In the direct route, export them yourself. |

## Models

### Qwen/Qwen3-0.6B

This model runs on one card. The launch line in Start serves it as written.

### deepseek-ai/DeepSeek-V2-Lite

The weights run bfloat16 and take 29.3 GB. They fit on one 64 GB card. This
deployment opens no tensor parallelism, so one card is also the limit.

Eager attention materializes the full attention matrix, so a long context
costs O(n^2) HBM. A context near 9k tokens fits on one card.

transformers 5.20.2 applies the YaRN mscale^2 scaling natively, so the
checkpoint loads at the correct scale on the native architecture. This model
is not proven on this server.

### state-spaces/mamba-2.8b-hf

This model runs on one card. The ladder loads it through the eager retry. The
architecture dispatches no attention, so the first attempt fails and the
retry runs eager. This model is not proven on this server.

The server refuses a presence_penalty above zero, because generate()
implements no such knob.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The server applies the checkpoint's own chat template. `stream: true` is
refused, because the route is non-streaming. The `model` field in the reply
is the identity the server prints at start: the checkpoint directory's name,
or the repository id parsed from a single-snapshot cache.
