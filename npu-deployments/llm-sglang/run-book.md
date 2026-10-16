# Run Book: llm-sglang (Ascend)

SGLang built from one pinned source commit, on torch 2.10.0 with
torch_npu 2.10.0.post2 under CANN 9.1.0-beta.1, behind one process: the
OpenAI chat surface and a full-vocabulary log-probability surface drive
the same embedded engine, same port. The image carries no serving engine:
the engine stack rides in `./wheels` and installs offline at first start.
`sitecustomize.py` is the patch layer: it stashes full-vocabulary float32
log-softmax rows at engine-internal entry points, so it must load in
every engine process.

## Server: Ascend 910B

### Environment

```bash
# the image: CANN base plus the pinned python stack, no serving engine
docker build -t kot-llm-ascend-910b:latest .

# the engine stack: sglang from one source commit, built and vendored
./fetch_wheels.sh
```

Base image `quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12`:
CANN 9.1.0-beta.1, Ubuntu 22.04, Python 3.12. The compiled wheels are
cp312 and aarch64; the stack installs on no other python or
architecture.

`fetch_wheels.sh` builds the engine stack. What it does, in order:

- Clones sglang at one commit and writes the commit to
  `wheels/SGLANG_COMMIT.txt`. A first run records the tip of `main`; a
  rerun reuses the record; the environment variable `SGLANG_COMMIT`
  overrides. One pinned build serves the whole deployment; `main` moves
  every day.
- The Ascend host's network stalls mid-download on large wheels. Every
  large file goes through `curl -sfC - --retry`, which resumes.
- Edits `pyproject_npu.toml` before the build. memfabric-zbal 1.1.2
  moves to 1.1.3: pypi publishes no 1.1.2 wheel for cp312/aarch64, and
  the documented CANN 9.1.0 component set pairs 1.1.3. `outlines` and
  `timm` are dropped: neither is imported on this serving path, and
  both pull an unpinned torch into the resolver.
- Builds the sglang wheel inside the image: `pip wheel --no-deps` over
  `pyproject_npu.toml`.
- Downloads triton-ascend 3.2.2 (cp312, aarch64) and the sgl-kernel-npu
  2026.9.0.post1 set (cann9.1.0, 910b, aarch64): the python wheel plus
  two `.run` installers that register custom operators in the toolkit's
  opp directory.
- Resolves the dependency closure inside the image (`pip download`) and
  keeps exactly one sglang wheel: the one whose name carries the commit.

`fetch_wheels.sh` groups all of these commands; run it as one command.

Pins that must not float:

- `torch==2.10.0` + `torch_npu==2.10.0.post2`, with `torchvision==0.25.0`
  beside them. torch_npu pulls the CUDA-free torch; the three must
  match as a set.
- `transformers==5.16.1`. The resolver picks a newer line unless the pin
  holds it; this stack was verified on 5.16.1.
- `triton_ascend==3.2.2`, installed `--no-deps`. sglang hard-imports
  triton, and on Ascend the module must come from triton-ascend. The
  wheel's own metadata pins a tail the engine never uses, among it a
  numpy 1.26.4 downgrade; installed with deps, it would replace the
  image's `numpy==2.2.6`.
- `memfabric-hybrid==1.1.4` + `memfabric-zbal==1.1.3`: part of the CANN
  9.1.0 component set; 1.1.3 is the documented pairing, not the vendored
  1.1.2.
- `sgl-kernel-npu==2026.9.0.post1`. Its python modules carry
  triton-ascend kernels the Ascend model path runs (Qwen3 fused QKV,
  RMSNorm, RoPE). Its C++ library is built against a newer CANN and does
  not load: the launcher makes the failed load non-fatal, and the patch
  layer supplies import stand-ins. A call to a kernel that is really
  absent fails loudly at its call site.
- The CANN version itself. The `.run` installers and the sgl-kernel-npu
  packages are cann9.1.0 builds; a different CANN breaks the pairing.

### Layout

```text
/opt/engines/llm-sglang    this directory; wheels/ is not committed (fetch_wheels.sh fills it)
/models/<model>            weights; the engine reads the tokenizer from here
site-packages              the engine stack, installed offline at first container start
/scratch/kot-sglang-stash  row blocks of the current request; else /tmp/kot-sglang-stash
```

The stash directory holds the row blocks of one request before the
server collects them. The float32 rows of an 8,600-position prefill are
about 5 GB; give the directory that much headroom.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```

The launcher takes exactly three arguments: port, model path, device.

- One script, two modes. Where python has no torch_npu, the script wraps
  `docker run` (privileged, host network, Ascend driver mounts,
  `/dev/davinci` device nodes) and re-enters itself inside the container.
  Inside the container it installs the stack and starts the server
  directly. The same command works in both places.
- Cards: set `ASCEND_RT_VISIBLE_DEVICES` (the launcher also reads
  `ASCEND_VISIBLE_DEVICES`) to ordinals, comma separated; the default is
  card 0. The launcher exports both variables into the container and
  maps each ordinal to its `/dev/davinci<ordinal>` node.
- First start installs the engine stack offline from `./wheels` into
  site-packages, then loads the engine; allow 900 s. A marker file in
  site-packages gates the install. The marker never lives in `/tmp`:
  host mode mounts the host's `/tmp` into the container, and a marker
  there would outlive the container and skip the next install.
- The server binds the loopback interface only. The wrap uses host
  networking, so the host reaches the same loopback; a published port
  would forward to the container's ethernet address, where nothing
  listens.
- The launcher raises the soft open-file limit to 65536. The docker
  default (1024) fails accept() with EMFILE under one burst of many
  pooled connections.
- Engine flags in `kot_server.py` are load-bearing:
  `chunked_prefill_size=-1` (one extend per request; a chunked extend
  samples one token per chunk, and the extra tokens corrupt the decode
  rows), `attention_backend="ascend"`, the overlap schedule stays ON
  (turning it off deadlocks the Ascend scheduler after the first
  extend), `mem_fraction_static=0.85` (the heuristic can reserve a
  negative KV size), `max_prefill_tokens=32768`.
- One warmup generation runs before the server listens. Compile and
  graph costs land in startup, not in the first request.
- `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` are exported: the
  engine must never reach the hub.
- The patch layer must reach every engine process. The engine spawns
  scheduler and detokenizer processes with a curated environment: no
  PYTHONPATH, no custom variables. The launcher therefore copies
  `sitecustomize.py` into site-packages on every start, warm restart
  included (the install marker alone would keep a stale copy). It also
  puts the deployment directory on PYTHONPATH, but the site-packages
  copy is what the workers see. A `sitecustomize.py` in site-packages
  that this deployment did not install stops the start.
- Stop the server by its pid or its container, never by matching the
  script name with `pkill -f`: the match can kill an install that is
  still running. Run `npu-smi info` before you claim a card.

The log-probability surface takes token ids and writes float32 rows and
the consumed ids into a directory the caller names. Numerics:

- The rows are the float32 log-softmax of the raw logits, taken before
  the sampler touches them, whatever dtype the weights run in.
- The ids go to the engine as input ids and bypass the tokenizer:
  nothing prepends BOS, applies a template or truncates.
- n ids yield n-1 prefill rows; a decode of g tokens yields g rows, one
  per generated token.
- Every row entry is finite. Exact `-inf` at token columns is a defect
  in the patch layer, not a limit; a consumer that tolerates it accepts
  garbage.
- The engine's deterministic-inference switch is CUDA-gated and stays
  off on Ascend. The patch layer carries the request seed into the
  sampler, so a seeded decode reproduces byte-identically.

### Parameters

| variable | default | effect |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | Cards the server may use, as ordinals, comma separated. The launcher reads this first, then `ASCEND_VISIBLE_DEVICES`; it exports both and maps each ordinal to its device node. |
| `KOT_NPU_IMAGE` | `kot-llm-ascend-910b:latest` | Image the host-mode wrap runs and `fetch_wheels.sh` builds the wheel inside. |
| `KOT_OUT_MOUNTS` | unset | Directories mounted read-write at the same path inside the container (host mode). Colon separated; every entry must exist on the host. |
| `KOT_SGLANG_STASH_DIR` | `/scratch/kot-sglang-stash`, else `/tmp/kot-sglang-stash` | Where the row blocks land. A set value switches the patch layer on. The server refuses to start without it, so start through `kot_serve.sh`. |
| `SGLANG_COMMIT` | the commit recorded in `wheels/SGLANG_COMMIT.txt` | Source commit `fetch_wheels.sh` builds. A first run records the tip of `main`; reruns reuse the record. |
| `KERNEL_TAG` | `2026.9.0.post1` | sgl-kernel-npu release tag `fetch_wheels.sh` downloads. |

## Models

### Qwen/Qwen3-0.6B

Revision `c1899de289a04d12100db370d81485cdf75e47ca`. One card. Proven on
this server: the smoke run passes 10/10 (byte-identical prefills and
seeded decodes), an 8,192-token decode plus prefill run, and a run of
the MLCommons Inference Endpoints client over the chat surface.
Sampling: temperature 0.6, top_p 0.95, top_k 20.

### deepseek-ai/DeepSeek-V2-Lite

Revision `604d5664dddd88a0433dbae533b7fe9472482de0`. bfloat16, 29.3 GB,
vocabulary 102,400. Not proven on this server. The engine runs one
card, one process, no tensor parallel. Sampling: temperature 0.3,
top_p 0.95.

### state-spaces/mamba-2.8b-hf

Revision `96c48e0292b63f5346b6d30061af2551f7101e26`. Not proven on this
server. Sampling: temperature 0.7, top_p 0.9, repetition_penalty 1.2.

## Check

```bash
curl -s http://localhost:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The request's `model` field is ignored: the server serves the one
checkpoint it was started with, and the reply names that identity. With
`"stream": true` the reply is SSE and carries no Content-Length: the
server closes the connection after `[DONE]`, and that close is the
body's only terminator.

`./smoke.sh /models/Qwen3-0.6B` starts its own server (port 8749 by
default) and runs the full check: shape, dtype and row sums, byte
identity across two prefills and two seeded decodes, verbatim ids, and
the stream close.
