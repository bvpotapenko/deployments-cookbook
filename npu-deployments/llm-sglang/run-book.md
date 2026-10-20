# Run Book: llm-sglang

This deployment serves text models with SGLang 0.5.22, built from source for
the Ascend device. One process serves the OpenAI chat surface and the
full-vocabulary log-probability surface on one port. The log-probability
surface reports one row per position: the float32 log-softmax of the raw
logits, computed before the sampler. Token ids reach the engine verbatim: no
BOS token, no chat template, no truncation. Every row entry is finite.
`sitecustomize.py` is the patch layer that stashes those rows at
engine-internal entry points, so it must load in every engine process.

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

One stack serves every model: CANN 9.2.0, torch 2.14.0 with torch_npu 2.14.0,
SGLang 0.5.22 built from source, triton-ascend 3.6.0. The seed arrives with
the request; 42 is the convention.

## Server: Ascend 910B

### Environment

```bash
# the image: CANN base plus the pinned python stack, no serving engine
docker build -t kot-llm-ascend-910b:latest .

# the engine stack: sglang 0.5.22 from source, built and vendored
./fetch_wheels.sh
```

The base image `quay.io/ascend/cann:9.2.0-910b-ubuntu22.04-py3.12` carries
CANN 9.2.0, Ubuntu 22.04 and CPython 3.12 on aarch64. The image bakes the
platform stack and no serving engine. The compiled wheels are cp312 and
aarch64; the stack installs on no other python or architecture.

`fetch_wheels.sh` builds the engine stack. It works in this order:

- It clones sglang at the v0.5.22 release commit and writes the commit to
  `wheels/SGLANG_COMMIT.txt`. A rerun reuses the record, and the environment
  variable `SGLANG_COMMIT` overrides. One pinned build serves the whole
  deployment.
- The Ascend host's network stalls mid-download on large files. Every large
  file goes through `curl -sfC - --retry`, which resumes.
- It edits `pyproject_npu.toml` before the build. The memfabric-zbal pin
  moves to the pairing the CANN 9.2.0 component set documents. It drops
  `outlines` and `timm`: neither is imported on this serving path, and both
  pull an unpinned torch into the resolver.
- It builds the sglang wheel inside the image with `pip wheel --no-deps`
  over `pyproject_npu.toml`.
- It downloads triton-ascend 3.6.0 (cp312, aarch64) and the sgl-kernel-npu
  set for torch 2.14.0 and CANN 9.2.0 (910b, aarch64): the python wheel plus
  two `.run` installers that register custom operators in the toolkit's
  opp directory. The wheel's python modules carry the triton-ascend kernels
  the Ascend model path runs (Qwen3 fused QKV, RMSNorm, RoPE).
- It resolves the dependency closure inside the image with `pip download`
  and keeps exactly one sglang wheel, the one whose name carries the commit.

`fetch_wheels.sh` groups all of these commands; run it as one command.

Pins that must not float:

- `torch==2.14.0` + `torch_npu==2.14.0`. torch_npu pulls the CUDA-free
  torch; the two must match as a set.
- `triton_ascend==3.6.0`, installed `--no-deps`. sglang hard-imports triton,
  and on Ascend the module must come from triton-ascend. The wheel's own
  metadata pins a tail the engine never uses; installed with deps, it would
  replace the image's own numpy.
- The memfabric pair (`memfabric-hybrid` and `memfabric-zbal`) comes from
  the CANN 9.2.0 component set.
- The sgl-kernel-npu line moves with the CANN version: the kernel wheel and
  the `.run` installers are cann9.2.0 builds, and a different CANN breaks
  the pairing.

### Layout

```text
/opt/engines/llm-sglang    this directory; wheels/ is not committed (fetch_wheels.sh fills it)
/models/<model>            weights; the engine reads the tokenizer from here
site-packages              the engine stack, installed offline at first start
/scratch/kot-sglang-stash  row blocks of the current request; else /tmp/kot-sglang-stash
```

The stash directory holds the row blocks of one request before the server
collects them. The float32 rows of an 8,600-position prefill are about
5 GB; give the directory that much headroom.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```

The launcher takes exactly three arguments: port, model path, device. Where
python has no `torch_npu`, the script wraps `docker run` with the Ascend
driver mounts and the `/dev/davinci` device nodes, and re-enters itself
inside the container. Inside the container it installs the stack and starts
the server directly. The same command works in both places.

Cards: set `ASCEND_RT_VISIBLE_DEVICES` to ordinals, comma separated; the
launcher reads `ASCEND_VISIBLE_DEVICES` as the fallback, and card 0 is the
default. The launcher exports both variables into the container and maps
each ordinal to its `/dev/davinci<ordinal>` node.

First start installs the engine stack offline from `./wheels` into
site-packages, then loads the engine; allow 900 s. A marker file in
site-packages gates the install. The marker never lives in `/tmp`: host mode
mounts the host's `/tmp` into the container, and a marker there would
outlive the container and skip the next install.

One warmup generation runs before the server listens. Compile and graph
costs land in startup, not in the first request.

The server binds the loopback interface only. The wrap uses host networking,
so the host reaches the same loopback; a published port would forward to the
container's ethernet address, where nothing listens.

The launcher raises the soft open-file limit to 65536. The docker default
(1024) fails accept() with EMFILE under one burst of many pooled
connections.

The engine flags in `kot_server.py` are load-bearing:

- `chunked_prefill_size=-1` keeps one extend per request: a chunked extend
  samples one extra token per chunk, and the extra tokens corrupt the decode
  rows.
- `attention_backend="ascend"` selects torch_npu's fused attention.
- The overlap schedule stays on: switching it off deadlocks the Ascend
  scheduler after the first extend. The one phantom forward it adds to a
  finished decode is trimmed from the rows.
- `mem_fraction_static=0.85` names the memory split explicitly: the engine
  heuristic can reserve a negative KV size with these settings.
- `max_prefill_tokens=32768` covers the longest prefill, about 8,600 ids
  with the generated text replayed.

`HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are exported: the engine must
never reach the hub.

The patch layer must reach every engine process. The engine spawns scheduler
and detokenizer processes with a curated environment, so an environment
variable cannot carry the patch layer to them. The launcher ships
`sitecustomize.py` as a module into site-packages and adds a `.pth` import
line beside it; site.py executes both in every process. The copy is
refreshed on every start, warm restart included. A `sitecustomize.py` in
site-packages that this deployment did not install stops the start.

The log-probability surface writes the rows and the consumed ids into a
directory the caller names. The numerics:

- The rows are the float32 log-softmax of the raw logits, taken before the
  sampler touches them, whatever dtype the weights run in.
- The ids go to the engine as input ids and bypass the tokenizer: nothing
  prepends BOS, applies a template or truncates.
- n ids yield n-1 prefill rows; a decode of g tokens yields g rows, one per
  generated token.
- Every row entry is finite. An exact `-inf` at a token column is a defect
  in the patch layer.
- The engine's deterministic-inference switch is CUDA-gated and stays off
  on Ascend. The patch layer carries the request seed into the sampler, so
  a seeded decode reproduces byte-identically.

Stop the server by its pid or its container, never by matching the script
name with `pkill -f`: the match can kill an install that is still running.
Run `npu-smi info` before you claim a card.

### Parameters

| variable | default | effect |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | Cards the server may use, as ordinals, comma separated. The launcher reads this first, then `ASCEND_VISIBLE_DEVICES`; it exports both and maps each ordinal to its device node. |
| `KOT_NPU_IMAGE` | `kot-llm-ascend-910b:latest` | Image the host-mode wrap runs and `fetch_wheels.sh` builds the wheel inside. |
| `KOT_OUT_MOUNTS` | unset | Directories mounted read-write at the same path inside the container (host mode). Colon separated; every entry must exist on the host. |
| `KOT_SGLANG_STASH_DIR` | `/scratch/kot-sglang-stash`, else `/tmp/kot-sglang-stash` | Where the row blocks land. A set value switches the patch layer on. The server refuses to start without it, so start through `kot_serve.sh`. |
| `SGLANG_COMMIT` | the v0.5.22 release commit | Source commit `fetch_wheels.sh` builds. A first run records the commit in `wheels/SGLANG_COMMIT.txt`; reruns reuse the record. |
| `KERNEL_TAG` | the release paired to torch 2.14.0 and CANN 9.2.0 | sgl-kernel-npu release `fetch_wheels.sh` downloads. The archive names carry the torch and CANN pairing. |

## Models

### Qwen/Qwen3-0.6B

The model runs on one card. Proven on this server: the smoke run passes
10/10, byte-identical prefills and seeded decodes, an 8,192-token decode
plus prefill, and a run of the MLCommons Inference Endpoints client over the
chat surface. The sampling values come from the checkpoint's own
`generation_config.json`.

### deepseek-ai/DeepSeek-V2-Lite

The weights are 29.3 GB in bfloat16, over a 102,400-token vocabulary. The
engine runs one card, one process, no tensor parallel. This server has no
proven run of this model.

### state-spaces/mamba-2.8b-hf

This server has no proven run of this model.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The `model` field is ignored: the server serves the one checkpoint it was
started with, and the reply names that identity. With `"stream": true` the
reply is SSE and carries no Content-Length: the server closes the connection
after `[DONE]`, and that close is the body's only terminator.

`./smoke.sh /models/Qwen3-0.6B` starts its own server (port 8749 by
default) and runs the full check: shape, dtype and row sums, byte identity
across two prefills and two seeded decodes, verbatim ids, and the stream
close.
