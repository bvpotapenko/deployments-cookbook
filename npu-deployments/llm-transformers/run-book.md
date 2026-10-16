# Run Book: llm-transformers

Stock HuggingFace transformers behind one thin HTTP adapter on Ascend 910B,
through the torch_npu plugin. One process serves an OpenAI-shaped completion
and chat route, and a scoring route that streams full-vocabulary float32
log-probabilities to disk; `serve.py` defines both. Requests run strictly one
at a time: no batching, no parallel streams.

## Server: Ascend 910B

### Environment

The image is aarch64 with CPython 3.12. The wheels are cp312 manylinux;
they install only under that Python.

```bash
# wheels/ must hold the three platform wheels first. The large PyPI
# downloads stall at times; curl -C - resumes them.
docker build -t kot-llm-ascend-910b:latest .
```

Pins that must not float:

- Base image `quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12`:
  the CANN toolkit 9.1.0-beta.1 for the 910B. The toolkit half of the stack
  lives in the image. The driver half lives on the host; the wrap mounts it
  read-only at start. The two halves must stay a compatible pair: float the
  image only together with a host driver that supports it.
- `torch==2.10.0`, `torch_npu==2.10.0.post2`, `torchvision==0.25.0`, from
  wheel files, never pip resolution. torch_npu registers the npu backend;
  its version pairs with torch, and the plugin is built for the CANN
  release in the base image. torchvision must match the same torch line.
- `transformers==5.16.1`, `numpy==2.2.6`.
- The build also installs the MLCommons Inference Endpoints client from
  GitHub, a client of the chat route. It needs git and a C compiler: this
  is why the image carries build-essential.

### Layout

```text
/opt/engines/llm-transformers  this directory; the wrap mounts it read-only at /deployment
/models/<model>                 weights; mounted read-only at their own path
```

`serve.py` is not baked into the image. The wrap mounts this directory
read-only at /deployment and runs the script from there: edit `serve.py`
here, no rebuild needed.

The server loads a plain checkpoint directory (config.json at the root), or
a HuggingFace cache that holds exactly one model under hub/. Several models
under hub/ stop the start: the server was told none.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```

The script picks its own route. A host python3 that imports torch_npu runs
`serve.py` directly, with the CANN environment script sourced first. Every
other host uses the docker wrap. Where the driver files are readable by
root only, a non-root shell cannot import torch_npu, and the wrap is the
route.

The wrap builds the image on the first run, then:

- runs privileged, with host networking, and `--ulimit nofile=65536:65536`;
- mounts the weights read-only at their own path, sets `HF_HUB_OFFLINE=1`,
  and gives a plain checkpoint a writable scratch HF home (/tmp/hf), so
  nothing writes into the read-only mount;
- passes the visible ordinals through `ASCEND_VISIBLE_DEVICES` and
  `ASCEND_RT_VISIBLE_DEVICES`, and adds one `/dev/davinci<ordinal>` device
  node per ordinal, beside the manager, svm and hdc nodes;
- mounts the host driver read-only: the Ascend driver tree, npu-smi, dcmi,
  sbin, `ascend_install.info`, `hccn.conf`; the NPU log directory mounts
  read-write at /usr/slog. A path missing on the host is skipped.

Host networking is required. The server binds 127.0.0.1 inside the
container, and the host reaches it on the same loopback address. A published
port does not work: it forwards to the container's ethernet address, where
nothing listens. The open-file limit matters for the same reason: a client
that opens many pooled connections at once runs past the docker default of
1024, and accept() fails with EMFILE.

Facts about a running server:

- torch_npu is imported before any tensor is created.
- The same request twice gives byte-identical output; verified on cpu and
  npu.
- A device fault (a rare AICPU kernel exception) leaves the process context
  dead: every later request answers 500 until restart. The device itself
  recovers. Restart the server.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_DTYPE` | `bfloat16` on npu, `float32` on cpu | model dtype: bf16, fp16 or fp32. |
| `KOT_ATTN` | `eager` on npu | attention implementation. sdpa on the NPU is untested: the fused kernels have unproven run-to-run determinism here. |
| `KOT_NPU_IMAGE` | `kot-llm-ascend-910b:latest` | image the host-mode wrap runs. |
| `KOT_OUT_MOUNTS` | unset | colon-separated directories mounted read-write at the same path, so the server's output files land where the host reads them. Host mode only. |
| `ASCEND_RT_VISIBLE_DEVICES` | then `ASCEND_VISIBLE_DEVICES`, then `0` | card ordinals the wrap passes to the container. In the direct route, export them yourself. |

## Models

Anything stock transformers loads as `AutoModelForCausalLM` with a text
tokenizer: Qwen2/Qwen3, Llama, Mistral, Gemma, DeepSeek (text), Mamba-hf,
GPT-family, and `trust_remote_code` models (enabled). Multimodal
checkpoints are out of scope, as are engines with their own serving stack.

### Qwen/Qwen3-0.6B

Revision `c1899de289a04d12100db370d81485cdf75e47ca`. One card, bfloat16,
eager attention. The proven model on this server: about 12 tokens per
second, single stream. Sampling: temperature 0.6, top_p 0.95, top_k 20.

### deepseek-ai/DeepSeek-V2-Lite

Revision `604d5664dddd88a0433dbae533b7fe9472482de0`. bfloat16, 29.3 GB of
weights: they fit on one 64 GB card. This deployment opens no tensor
parallelism, so one card is also the limit. Not proven on this server yet.

Eager attention materializes the full attention matrix: a long context
costs O(n^2) HBM. A context near 9k tokens fits on one card.

Sampling: temperature 0.3, top_p 0.95.

### state-spaces/mamba-2.8b-hf

Revision `96c48e0292b63f5346b6d30061af2551f7101e26`. Loads through the same
`AutoModelForCausalLM` path. Not proven on this server yet.

Sampling: temperature 0.7, top_p 0.9, repetition_penalty 1.2.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The server applies the checkpoint's own chat template. `stream: true` is
refused: the route is non-streaming. The `model` field in the reply is the
identity the server prints at start: the checkpoint directory's name, or the
repository id parsed from a single-snapshot cache.
