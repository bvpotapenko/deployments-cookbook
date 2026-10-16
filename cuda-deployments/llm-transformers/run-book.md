# Run Book: llm-transformers

Stock `transformers` behind one process: an OpenAI-compatible chat and
completions surface, one resident model, one generation at a time. The trap
sits in the launcher: it execs `/root/kotvenv/bin/python` by a fixed path, so
a venv built anywhere else means editing that line.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

CPython 3.12. Build the venv at `/root/kotvenv` or edit the launcher's last
line.

```bash
uv venv /root/kotvenv --python 3.12
uv pip install --python /root/kotvenv/bin/python \
  torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /root/kotvenv/bin/python \
  transformers==4.56.2 numpy accelerate pydantic pyyaml
```

Pins that must not float:

- `torch==2.8.0` from the cu128 index. sm_120 (RTX 5090) needs the CUDA 12.8
  kernels; this index is where that build lives.
- `transformers==4.56.2`. The reference outputs come from this version, and
  the load fallbacks below are verified on it; a newer version can change the
  attention kernels and with them the numbers.

The reference venv resolved numpy 2.5.4, accelerate 1.15.0, tokenizers 0.22.2.

### Layout

```text
/opt/engines/llm-transformers   this directory
/models/<model>                 weights: a checkpoint directory, or an HF home holding hub/
```

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device cuda
```

The process listens on 127.0.0.1 only.

When the model path holds a `hub/` directory, the launcher exports HF_HOME to
that path and the server resolves the one snapshot under it; two or more
repositories there stop the start with an error that names them. The model
identity comes from the directory or snapshot name, never from a request.

The load path: a native architecture first, without `trust_remote_code`. A
ValueError that names trust_remote_code retries with the checkpoint's own
code. A ValueError about an attention implementation retries with
`attn_implementation="eager"`; that is the path an SSM takes. On cuda the
weights run bfloat16 under sdpa, and `device_map="auto"` spreads them over
every visible card. Set CUDA_VISIBLE_DEVICES to pin the card set.

`smoke.sh` accepts KOT_PYTHON to name its interpreter; `kot_serve.sh` does
not, which is the trap above.

### Parameters

| argument | effect |
|---|---|
| `--port` | listen port on 127.0.0.1 |
| `--model` | checkpoint directory, or an HF home holding `hub/` |
| `--device` | `cuda`: bf16, sdpa, device_map auto. `cpu`: float32, eager, one device. |

## Models

### Qwen/Qwen3-0.6B

Revision `c1899de289a04d12100db370d81485cdf75e47ca`. One card; export
CUDA_VISIBLE_DEVICES when this model runs alone, or `device_map="auto"`
spreads it over every visible card. Sampling: temperature 0.6, top_p 0.95,
top_k 20.

### deepseek-ai/DeepSeek-V2-Lite

Revision `604d5664dddd88a0433dbae533b7fe9472482de0`. bfloat16, 29.3 GB: it
needs both 32 GB cards. Sampling: temperature 0.3, top_p 0.95.

### state-spaces/mamba-2.8b-hf

Revision `96c48e0292b63f5346b6d30061af2551f7101e26`. One card. It loads
through the eager fallback: this architecture dispatches no attention, so the
first load attempt fails and the retry runs eager.

The engine has no presence_penalty. `generate()` implements none, and the
server refuses a request that sets it above zero with an error that names the
field; the OpenAI routes do not read the field at all.

Sampling: temperature 0.7, top_p 0.9, repetition_penalty 1.2.

## Check

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "ping"}], "max_tokens": 8}' \
  | python3 -m json.tool
```

The request's `model` field is ignored; the reply names the checkpoint the
server loaded. A request without `max_tokens` generates 16 tokens.
