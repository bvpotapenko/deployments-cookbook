# Run Book: llm-vllm

vLLM 0.11.0 V1 engine behind one process: the OpenAI chat surface and a
full-vocabulary log-probability surface, same port. The server carries a
version-pinned engine patch: it refuses to start on any other vllm version.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

CPython 3.11. The wheel closure is cp311; on a 3.12 host the offline install
fails with `no versions found`, which does not name the real cause.

```bash
# networked host: install the exact closure (147 pins)
uv venv /scratch/kot-vllm-venv --python 3.11
uv pip install --python /scratch/kot-vllm-venv/bin/python \
  -r <(grep -E '^[a-zA-Z0-9].*==' wheels-requirements.txt)

# offline host: rebuild the closure, then install from it
./make_wheels.sh          # builds wheels/ (not committed, 4.7 GB)
uv venv --seed /scratch/kot-vllm-venv --python 3.11
/scratch/kot-vllm-venv/bin/pip install --no-index --no-cache-dir \
  --find-links ./wheels "vllm==0.11.0"
```

Pins that must not float:

- `torch==2.8.0` cu128. The 2.7-era inductor breaks on sm_120: the
  flex_attention fallback dies at the first prefill.
- `transformers==4.56.2`. 5.x breaks this vllm; the resolver will not pick
  4.56.2 on its own.

### Layout

```text
/opt/engines/llm-vllm     this directory
/models/<model>           weights; the tokenizer is read from here and nowhere else
```

### Start

```bash
KOT_PYTHON=/scratch/kot-vllm-venv/bin/python \
  ./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device cuda
```

Prefix caching is disabled in the server: a cache hit would leave the
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
| `KOT_MODEL_ID` / `KOT_MODEL_REVISION` | derived | override the reported identity. |

## Models

### Qwen/Qwen3-0.6B

Revision `c1899de289a04d12100db370d81485cdf75e47ca`. One card, default
parameters. Sampling: temperature 0.6, top_p 0.95, top_k 20.

### deepseek-ai/DeepSeek-V2-Lite

Revision `604d5664dddd88a0433dbae533b7fe9472482de0`. bf16, 29.3 GB: both
cards (the engine opens tensor parallel over every visible card).

```bash
KOT_VLLM_GPU_MEM=0.70 KOT_VLLM_MAX_BATCHED_TOKENS=1024
```

The memory window is narrow and the failure modes sit on both sides: above
0.70 the full-vocabulary pass exhausts the card on long prompts; below 0.66
the engine refuses to start, because its KV floor wants one request at
`max_seq_len` (163,840 tokens, 4.75 GB MLA). Set
`KOT_VLLM_MAX_MODEL_LEN=16384` when every workload stays under 16k tokens;
that reopens a wide window.

Sampling: temperature 0.3, top_p 0.95.

### state-spaces/mamba-2.8b-hf

Revision `96c48e0292b63f5346b6d30061af2551f7101e26`. One card.

The mamba-1 backend stops a decode at about 2046 total tokens (prompt plus
generated), whatever `max_model_len` says. Keep prompt plus generation under
2040.

Sampling: temperature 0.7, top_p 0.9, repetition_penalty 1.2.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The `model` field must equal the identity the server prints in its log; the
server derives it from the checkpoint.
