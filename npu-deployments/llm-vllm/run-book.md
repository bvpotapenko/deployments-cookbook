# Run Book: llm-vllm

This deployment serves text models with vLLM 0.23.0 and vllm-ascend 0.23.0.
One process serves the OpenAI chat surface and the full-vocabulary
log-probability surface on one port. The log-probability surface reports one
row per position: the float32 log-softmax of the raw logits, computed before
the sampler. Token ids reach the engine verbatim: no BOS token, no chat
template, no truncation. The same request returns byte-identical rows. The
server refuses to start on any other vllm version. The request carries the
decode depth, the sampling values and the seed; a missing sampling field
falls back to its neutral value.

## Model parameters

The table holds the launch parameters per model. Every value not named stays
at the deployment default. Read a model's row against the same row in the
other engine run books. A mismatch between engines is the first suspect when
a comparison fails.

| model | revision | cards | `KOT_VLLM_NPU_MEM` | `KOT_VLLM_MAX_BATCHED_TOKENS` | `KOT_VLLM_MAX_MODEL_LEN` | dtype | sampling (recommended) |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 1 | 0.35 | 2048 | unset | bfloat16 | temperature 0.6, top_p 0.95, top_k 20 |
| deepseek-ai/DeepSeek-V2-Lite | `604d5664dddd88a0433dbae533b7fe9472482de0` | 1 | 0.35 | 2048 | unset | bfloat16 | temperature 0.3, top_p 0.95 |
| state-spaces/mamba-2.8b-hf | `96c48e0292b63f5346b6d30061af2551f7101e26` | 1 | 0.35 | 2048 | unset | bfloat16 | temperature 0.7, top_p 0.9, repetition_penalty 1.2 |

One stack serves every model: CANN 9.1.0, torch 2.10.0 with torch_npu 2.10.0,
vLLM 0.23.0 with vllm-ascend 0.23.0. The seed arrives with the request; 42 is
the convention.

## Server: Ascend 910B

### Environment

The base image is `quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12`. It
carries CANN 9.1.0, Ubuntu 22.04 and CPython 3.12 on aarch64. The image bakes
the platform stack and no serving engine. Every wheel in the engine closure
is aarch64 cp312; the closure serves no other architecture or CPython.

```bash
./make_wheels.sh     # builds wheels/ from wheels-requirements.txt; wheels/ is not committed
docker build -t kot-llm-ascend-910b:latest .
```

`make_wheels.sh` builds the pip resolution of `vllm==0.23.0` with
`vllm_ascend==0.23.0`, plus the platform wheels the Dockerfile bakes in. It
downloads every pin with `--no-deps`, one package at a time, and prints two
counts. Rerun the script until the two counts match: mirror timeouts are
transient, and `arctic_inference` needs one rerun to pick up the wheel pip
builds from its sdist. The vllm wheel is 266 MB. If a download stalls, put
the wheel into `wheels/` by hand and rerun; the script skips files that are
already present.

Pins that must not float:

- `vllm==0.23.0` with `vllm_ascend==0.23.0`. The log-probability surface
  reaches into vllm V1 internals, so only this vllm serves it.
- The Dockerfile is the pin list for the image: `torch==2.10.0` and
  `torch_npu==2.10.0` beside the matching torchvision and the family's
  transformers and numpy lines. torch_npu must pair with torch, because the
  two ship one device stack.
- `wheels-requirements.txt` is the pin list for the engine venv. The venv is
  built over the image's system site packages, so the closure installs over
  the base stack and the server imports the venv copies.
- `triton_ascend==3.2.2`, and no pypi triton anywhere. pypi triton overwrites
  triton-ascend, and the engine dies on the first request. Three packages
  (`triton_ascend`, `memfabric_hybrid`, `memcache_hybrid`) exist only on the
  Ascend pypi mirror.
- Install per package with `--no-deps`, never the whole file at once. The pin
  graph holds metadata conflicts. pip refuses to resolve them, and the
  runtime never touches them.

### Layout

```text
/opt/engines/llm-vllm     this directory; inside the wrap it is /deployment, read-only
/models/<model>           weights, mounted read-only at their own path; the tokenizer
                          and its chat template are read from here
/scratch/kot-vllm-venv    engine venv (KOT_VENV overrides); /tmp is the fallback
```

A weights path that holds an HF cache (a `hub/` directory) sets `HF_HOME` to
itself. A plain checkpoint gets a scratch home under `/tmp`; nothing writes
into the read-only weights mount. The server sets `HF_HUB_OFFLINE=1`: it
reads no network, and the weights directory must be complete on disk.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```

The script picks its launch path from what the ambient python imports. No
`torch_npu`: the script wraps itself in docker, with the Ascend device nodes
and the driver mounts, and re-enters itself inside the container; it builds
the image first when the image is missing. `torch_npu` without vllm: the
script provisions the engine venv offline from `./wheels`, with system site
packages; the vendored `bin/uv` takes seconds, and pip is the fallback. Both
importable: the script serves directly.

The wrap needs host networking. The server binds the loopback interface only,
and a published port forwards to the container ethernet address, where
nothing listens. The wrap runs privileged and the server stays root:
`torch_npu` does not import as a non-root uid, because the driver files are
root-only. The wrap mounts the Ascend driver and its config files from fixed
host paths. A host path that does not exist is skipped without an error.

Cards: `ASCEND_RT_VISIBLE_DEVICES` names the ordinals;
`ASCEND_VISIBLE_DEVICES` is the fallback; the wrap always forwards one
value, card 0 by default. On the direct path, unset means every visible card.
The tensor-parallel size equals the count of listed ordinals. List one card
unless tensor parallel is the intent.

Prefix caching is off in the server. A cache hit would leave the requested
rows of the cached prefix never computed. A model outside the set
vllm-ascend supports fails loudly at load. On a loaded host the server needs
more than 30 seconds before it answers; wait before you decide the start
failed.

### Parameters

| variable | default | effect |
|---|---|---|
| `ASCEND_RT_VISIBLE_DEVICES` | `0` | Card ordinals, comma separated. The tensor-parallel size equals the count of listed cards. |
| `ASCEND_VISIBLE_DEVICES` | `0` | The fallback when the RT variable is unset. |
| `KOT_PYTHON` | ambient python3 | Interpreter that has vllm and `torch_npu`. Set it when a provisioned interpreter sits outside the default paths. |
| `KOT_VENV` | `/scratch/kot-vllm-venv` | Engine venv location, inside and outside the wrap. `/tmp` is the fallback when `/scratch` is not writable. |
| `KOT_VLLM_NPU_MEM` | `0.35` | NPU memory fraction. The log-probability pass needs device headroom; at 0.9 the margin goes to the KV cache and the pass fails. |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | Caps the per-step transient of the log-probability pass, about 16 bytes per vocabulary entry per scheduled token. The engine default 8192 makes about 22 GB on a 150k vocabulary and kills the engine. |
| `KOT_VLLM_MAX_MODEL_LEN` | model config | Caps `max_model_len`, and with it the KV-cache floor the engine demands at startup. |
| `KOT_VLLM_DTYPE` | engine auto | Compute dtype, for example `float32` for a cross-check. |
| `KOT_VLLM_COMPILATION` | `eager` | `default` turns compilation on; the two paths are bit-identical. |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived | Override the reported identity. |
| `KOT_NPU_IMAGE` | `kot-llm-ascend-910b:latest` | The image the wrap builds and runs. |
| `KOT_OUT_MOUNTS` | unset | Output directories, mounted read-write at the same path inside the wrap. An output directory must be one of them. |
| `KOT_UID` | unset | The invoking uid. The server hands the written output directories back to this uid. |

## Models

### Qwen/Qwen3-0.6B

The model runs on one card. This is the one model with a verified run on
this server. The sampling values come from the checkpoint's own
`generation_config.json`.

### deepseek-ai/DeepSeek-V2-Lite

The weights are 29.3 GiB in bfloat16. This server has no proven run of this
model.

### state-spaces/mamba-2.8b-hf

This server has no proven run of this model.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "any", "messages": [{"role": "user", "content": "ping"}], "max_completion_tokens": 8}' \
  | python3 -m json.tool
```

The `model` field is ignored: the server rewrites it to the loaded model
before the engine sees the request. The server binds the loopback interface
only.
