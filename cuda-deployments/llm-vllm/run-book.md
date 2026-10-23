# Run Book: llm-vllm

This deployment serves text models with vLLM 0.32. One process serves the
OpenAI chat surface and the full-vocabulary log-probability surface on one
port. The full-vocabulary tap is version-pinned, so the server refuses to
start on any other vllm version.

## Model parameters

The table holds the launch parameters per model. Every value not named stays
at the deployment default. Read a model's row against the same row in the
other engine run books. A mismatch between engines is the first suspect when
a comparison fails.

| model | revision | cards | `KOT_VLLM_GPU_MEM` | `KOT_VLLM_MAX_BATCHED_TOKENS` | `KOT_VLLM_MAX_MODEL_LEN` | dtype | sampling (recommended) |
|---|---|---|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 1 | 0.35 | 2048 | unset | bfloat16 | temperature 0.6, top_p 0.95, top_k 20 |
| deepseek-ai/DeepSeek-V2-Lite | `604d5664dddd88a0433dbae533b7fe9472482de0` | 2 | 0.60 | 1024 | unset | bfloat16 | temperature 0.3, top_p 0.95 |
| state-spaces/mamba-2.8b-hf | `96c48e0292b63f5346b6d30061af2551f7101e26` | 1 | 0.35 | 2048 | unset | bfloat16 | temperature 0.7, top_p 0.9, repetition_penalty 1.2 |

One stack serves every model: CPython 3.12, vLLM 0.32, the torch build that
carries sm_120, and the transformers line the vLLM 0.32 resolver takes. The
seed arrives with the request; 42 is the convention.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

CPython 3.12.

```bash
# networked host
uv venv /scratch/kot-vllm-venv --python 3.12
uv pip install --python /scratch/kot-vllm-venv/bin/python "vllm==0.32"

# offline host: rebuild the closure, then install from it
./make_wheels.sh
/scratch/kot-vllm-venv/bin/pip install --no-index --no-cache-dir \
  --find-links ./wheels "vllm==0.32"
```

Verify: `/scratch/kot-vllm-venv/bin/python -c "import vllm; print(vllm.__version__)"` prints `0.32.0`.

Run long installs inside tmux; an ssh drop then costs nothing. Export
`UV_CACHE_DIR=/workspace/uvcache` and `UV_HTTP_TIMEOUT=600` in the same
shell and verify with `ls -d /workspace/uvcache` after the first download.
Where PyPI stalls, retry with
`--index-url https://mirrors.aliyun.com/pypi/simple/`.

The torch build must carry sm_120. The vLLM 0.32 resolver pairs one on its
own. A torch without Blackwell support dies at the first prefill on these
cards.

### Layout

```text
/opt/engines/llm-vllm     this directory
/models/<model>           weights; the tokenizer is read from here and nowhere else
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer.

```bash
mkdir -p /opt/engines/llm-vllm
rsync -a --partial -e "ssh -p $PORT" ./llm-vllm/ \
  user@$HOST:/opt/engines/llm-vllm/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/llm-vllm/*.sh'
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
KOT_PYTHON=/scratch/kot-vllm-venv/bin/python \
  ./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device cuda
```

Prefix caching is disabled in the server. A cache hit would leave the
requested rows of the cached prefix never computed.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_PYTHON` | ambient python3 | interpreter that has vllm. Set it when the venv is not at the default path. |
| `KOT_VENV` | `/scratch/kot-vllm-venv` | offline venv location. |
| `KOT_VLLM_GPU_MEM` | `0.35` | `gpu_memory_utilization`. Headroom is needed for the log-probability pass. |
| `KOT_VLLM_MAX_BATCHED_TOKENS` | `2048` | caps the per-step transient of that pass. |
| `KOT_VLLM_MAX_MODEL_LEN` | engine config | caps `max_model_len`, and with it the reserved KV cache. |
| `KOT_VLLM_DTYPE` | engine auto | compute dtype, for example `float32` for a cross-check. |
| `KOT_VLLM_MOE_BACKEND` | `triton` | MoE backend. `auto` picks flashinfer's cutlass MoE on Blackwell, whose JIT needs nvcc and dies at init on hosts without it. |
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived | override the reported identity. |

## Models

### Qwen/Qwen3-0.6B

One card. The default parameters need nothing.

### deepseek-ai/DeepSeek-V2-Lite

bf16, 29.3 GB. It needs both cards; the engine opens tensor parallel over
every visible card.

```bash
KOT_VLLM_GPU_MEM=0.60 KOT_VLLM_MAX_BATCHED_TOKENS=1024 \
KOT_VLLM_MAX_MODEL_LEN=16384 \
  ./kot_serve.sh --port 8090 --model /models/DeepSeek-V2-Lite --device cuda
```

Both directions fail. A higher memory setting fills the card and the
full-vocabulary top-k of the log-probability pass dies out of memory
(verified on the 0.32 line: 0.70 leaves 803 MB free where the top-k over
a 1024-token chunk needs about 1 GB). A lower setting falls under the
per-card weights, 14.65 GB. `KOT_VLLM_MAX_MODEL_LEN=16384` shrinks the
engine's KV floor from 163,840 tokens to a few hundred MB, and every kot
workload stays under 16k tokens. The smoke needs the same memory
setting:

```bash
KOT_VLLM_GPU_MEM=0.60 KOT_VLLM_MAX_MODEL_LEN=16384 \
  ./smoke.sh /models/DeepSeek-V2-Lite cuda
```

### state-spaces/mamba-2.8b-hf

One card. The mamba backend caps the total context of a decode; the
measured limit is about 2046 tokens (prompt plus generated), whatever
`max_model_len` says. Keep prompt plus generation under 2040.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "DeepSeek-V2-Lite",
       "messages": [{"role": "user", "content": "ping"}],
       "max_tokens": 50, "temperature": 0.3, "top_p": 0.95}' \
  | python3 -m json.tool
```

The `model` field must equal one of the aliases the server prints in its
log (`/kot/info` names them: the checkpoint id and the weights path). The contract checks are
`./smoke.sh /models/<model> cuda`: byte-identical double runs.
