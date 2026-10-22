# Run Book: llm-sglang

This deployment serves text models with SGLang 0.5.22. One process serves the
OpenAI chat surface and the full-vocabulary log-probability surface on one
port. A patch layer (`sitecustomize.py`) stashes the full-vocabulary float32
log-softmax rows of the raw logits at engine-internal entry points, so the
reported rows are pre-sampler. The launcher installs the patch layer into the
engine venv, so every engine worker loads it.

## Model parameters

The table holds the launch parameters per model. Read a model's row against
the same row in the other engine run books. A mismatch between engines is the
first suspect when a comparison fails.

| model | revision | cards | `KOT_SGLANG_MEM_FRACTION` | `KOT_SGLANG_MAX_PREFILL_TOKENS` | sampling (recommended) |
|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 1 | 0.45 | 8192 | temperature 0.6, top_p 0.95, top_k 20 |
| deepseek-ai/DeepSeek-V2-Lite | `604d5664dddd88a0433dbae533b7fe9472482de0` | 2 | 0.70 | 12288 | temperature 0.3, top_p 0.95 |
| state-spaces/mamba-2.8b-hf | `96c48e0292b63f5346b6d30061af2551f7101e26` | not supported | | | |

One stack serves every model: CPython 3.12, sglang 0.5.22, torch 2.14.1,
sglang-kernel 0.4.9, flashinfer-python 0.7.1 and transformers 5.19.0, the
lines the sglang 0.5.22 resolver pairs. The seed arrives with the request.
42 is the convention.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The interpreter is CPython 3.12. Both sglang deployments share one venv,
/root/kotvenv-sglang-ocr.

```bash
uv venv /root/kotvenv-sglang-ocr --python 3.12
uv pip install --python /root/kotvenv-sglang-ocr/bin/python "sglang==0.5.22"
```

Verify: `/root/kotvenv-sglang-ocr/bin/python -c "import sglang; print(sglang.__version__)"` prints `0.5.22`.

Run long installs inside tmux; an ssh drop then costs nothing. Export
`UV_CACHE_DIR=/workspace/uvcache` and `UV_HTTP_TIMEOUT=600` in the same
shell and verify with `ls -d /workspace/uvcache` after the first download.
Where PyPI stalls, retry with
`--index-url https://mirrors.aliyun.com/pypi/simple/`.

The resolver pairs the whole line at once. The flashinfer requirement carries
the cu13 extra, so the stack is CUDA 13. The torch wheel carries
sm_120. A torch without Blackwell support dies at the first prefill on these
cards. The kernel wheels link against this torch line and do not load against
another. The patch layer taps this line's internal entry points. They move
between lines, and a different line fails the log-probability requests while
the server keeps running.

### Layout

```text
/opt/engines/llm-sglang       this directory
/models/<model>               weights; the engine reads the tokenizer from here
/root/kotvenv-sglang-ocr      the engine venv, shared with ocr-sglang
<venv>/site-packages          the launcher refreshes the patch layer here on every start
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer.

```bash
mkdir -p /opt/engines/llm-sglang
rsync -a --partial -e "ssh -p $PORT" ./llm-sglang/ \
  user@$HOST:/opt/engines/llm-sglang/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/llm-sglang/*.sh'
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
KOT_SGLANG_PYTHON=/root/kotvenv-sglang-ocr/bin/python \
  ./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device cuda
```

The patch layer must load in every engine process. The launcher ships it into
the venv's site-packages on every start: `sitecustomize.py` as itself, a copy
as `kot_sglang_patch.py`, and `kot_sglang_patch.pth` whose single line is
`import kot_sglang_patch`. A `.pth` file runs in every process the venv
starts, whatever the environment says, and that route reaches the scheduler
workers, whose environment the engine curates. A distribution-level
`sitecustomize.py` in the stdlib directory precedes any venv's site-packages
on `sys.path`, so site.py may import the distribution's file instead of the
venv's. The `.pth` route does not depend on winning that race. The launcher
refuses to overwrite a `sitecustomize.py` it does not recognize.

The launcher sets `SGLANG_ENABLE_JIT_DEEPGEMM=false`. The deep-gemm JIT needs
a build toolchain, and a serving host carries none. The triton kernels are
the supported path on this GPU. The launcher also sets
`SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_DEEPGEMM=0`, so the batch-invariant
triton matmul keeps deterministic inference without DeepGEMM. It raises the
soft file-descriptor limit to 65535, because the engine holds zmq sockets
plus one descriptor per queued connection and the 1024 default dies with
EMFILE mid-serve. It sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1`.
The weights are local, and the engine must not reach for the hub.

Chunked prefill is off and the overlap scheduler is disabled in the server.
A chunked extend samples and discards one token per chunk, and the overlap
scheduler runs one extra decode forward. Both desynchronize the
log-probability rows. The engine runs deterministic inference on the triton
attention backend and opens tensor parallel over every visible card, so
expose one card for the one-card models. The server binds 127.0.0.1 only. One
tiny warmup generation runs before the server listens, so the first real
request pays no compile and graph costs.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_SGLANG_PYTHON` | ambient python3 | interpreter that has sglang. The launcher tries it first, then a PATH python3. |
| `KOT_SGLANG_MEM_FRACTION` | `0.45` | engine `mem_fraction_static`. Set it explicitly. The engine's own heuristic can go negative with chunked prefill off, and the fraction must cover the per-card weights. |
| `KOT_SGLANG_MAX_PREFILL_TOKENS` | `8192` | engine `max_prefill_tokens`. The engine rejects a prompt longer than the window. |
| `KOT_SGLANG_STASH_DIR` | `/scratch/kot-sglang-stash`, else `/tmp/kot-sglang-stash` | where the patch layer writes the full-vocabulary float32 rows. The server collects and deletes them per request. |

## Models

### Qwen/Qwen3-0.6B

This model runs on one card. The launch parameters stay at the deployment
defaults. The sampling column holds the checkpoint's own `generation_config.json`
values.

### deepseek-ai/DeepSeek-V2-Lite

The weights are bf16, 29.3 GB in total. The model needs both cards, and the
engine opens tensor parallel over every visible card.

```bash
KOT_SGLANG_MEM_FRACTION=0.70 KOT_SGLANG_MAX_PREFILL_TOKENS=12288 \
  ./kot_serve.sh --port 8090 --model /models/DeepSeek-V2-Lite --device cuda
```

Both variables are required. A lower memory fraction leaves the KV pool below
the per-card weight size, 14.65 GB, and the engine refuses to start. A
smaller prefill window rejects the longest workload prompt, about 8.5k ids.

### state-spaces/mamba-2.8b-hf

The engine cannot serve this model. Serving it needs the `mamba_ssm` and
`causal-conv1d` CUDA kernel packages. Building them against torch 2.14.1
fails with a ptx version error when the nvcc toolchain mixes CUDA component
versions. All nvcc components must come from one CUDA version.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The request's `model` field is ignored. The response names the identity the
server derived from the weights path. A hub-cache snapshot gives `org/name`.
A flat directory gives its basename. The contract checks are
`./smoke.sh /models/<model> cuda`: byte-identical double runs.
