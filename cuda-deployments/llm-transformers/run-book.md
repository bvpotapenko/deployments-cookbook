# Run Book: llm-transformers

This deployment serves text models with transformers 5.20.2. One process
serves the OpenAI chat and completions routes and the full-vocabulary
log-probability surface on one port. The model thread runs one generation at
a time, whatever many connections are held open.

The launcher execs `/root/kotvenv/bin/python serve.py` on a fixed path. A
reader who builds the venv elsewhere edits that line.

## Model parameters

The table holds what differs per model. This deployment exposes no per-model
environment knobs; only the `--model` path changes between launches. Read a
model's row against the same row in the other engine run books. A mismatch
between engines is the first suspect when a comparison fails.

| model | revision | cards | dtype | sampling (recommended) |
|---|---|---|---|---|
| Qwen/Qwen3-0.6B | `c1899de289a04d12100db370d81485cdf75e47ca` | 1 | bfloat16 | temperature 0.6, top_p 0.95, top_k 20 |
| deepseek-ai/DeepSeek-V2-Lite | `604d5664dddd88a0433dbae533b7fe9472482de0` | 2 | bfloat16 | temperature 0.3, top_p 0.95 |
| state-spaces/mamba-2.8b-hf | `96c48e0292b63f5346b6d30061af2551f7101e26` | 1 | bfloat16 | temperature 0.7, top_p 0.9, repetition_penalty 1.2 |

One stack serves every model: CPython 3.12, transformers 5.20.2, and the
torch build from the cu128 index that carries sm_120. The seed arrives with
the request; 42 is the convention.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The venv runs CPython 3.12. Build it at `/root/kotvenv`, or edit the
launcher's last line.

```bash
uv venv /root/kotvenv --python 3.12
uv pip install --python /root/kotvenv/bin/python \
  torch --index-url https://download.pytorch.org/whl/cu130
uv pip install --python /root/kotvenv/bin/python \
  transformers==5.20.2 accelerate numpy pydantic pyyaml
```

Run long installs inside tmux; an ssh drop then costs nothing. Export
`UV_CACHE_DIR=/workspace/uvcache` and `UV_HTTP_TIMEOUT=600` in the same
shell and verify with `ls -d /workspace/uvcache` after the first download.
Where PyPI stalls, retry with
`--index-url https://mirrors.aliyun.com/pypi/simple/`; the torch line
keeps its own index.

torch must come from the cu130 index. That build carries the sm_120 kernels;
a torch without them stalls silently on these cards. The venv pins
transformers 5.20.2. A different transformers line runs different attention
kernels and computes different outputs.

### Layout

```text
/opt/engines/llm-transformers   this directory
/models/<model>                 weights; a checkpoint directory, or an HF home holding hub/
```

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer. The excludes keep `__pycache__` and `.DS_Store` off the box.

```bash
mkdir -p /opt/engines/llm-transformers
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./llm-transformers/ user@$HOST:/opt/engines/llm-transformers/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/llm-transformers/*.sh'
rsync -a --partial -e "ssh -p $PORT" user@$HOST:/outputs/ ./outputs/
```

A networked host downloads the model itself. On vast, huggingface.co
times out, so the download goes through the mirror endpoint. Otherwise
push the snapshot from the local hub cache; `--copy-links` resolves the
cache's blob symlinks into real files:

```bash
HF_ENDPOINT=https://hf-mirror.com \
  hf download <org>/<model> --revision <rev> --local-dir /models/<model>
rsync -a --partial --copy-links -e "ssh -p $PORT" \
  ~/.cache/huggingface/hub/models--<org>--<model>/snapshots/<rev>/ \
  user@$HOST:/models/<model>/
```

Verify: the revision matches the model table's pin and the file list
matches the source.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device cuda
```

The process listens on 127.0.0.1 only.

When the model path holds a `hub/` directory, the launcher exports HF_HOME to
that path and sets HF_HUB_OFFLINE=1. The server resolves the one snapshot
under it. Two or more repositories there stop the start with an error that
names them. The model identity comes from the directory or snapshot name,
never from a request.

The loader keeps a three-step ladder. It loads the native architecture first,
without `trust_remote_code`. A ValueError that names trust_remote_code retries
with the checkpoint's own code. A ValueError about an attention implementation
retries with `attn_implementation="eager"`. On cuda the weights run bfloat16
under sdpa, and `device_map="auto"` spreads the layers over every visible
card. Set CUDA_VISIBLE_DEVICES to pin the card set.

`smoke.sh` accepts KOT_PYTHON to name its interpreter. `kot_serve.sh` does
not; its exec line is the fixed path above.

### Parameters

| argument | effect |
|---|---|
| `--port` | listen port on 127.0.0.1 |
| `--model` | checkpoint directory, or an HF home holding `hub/` |
| `--device` | `cuda` runs bfloat16, sdpa, device_map auto; `cpu` runs float32, eager, one device |

## Models

### Qwen/Qwen3-0.6B

This model runs on one card. Export CUDA_VISIBLE_DEVICES when it runs alone,
or `device_map="auto"` spreads the layers over every visible card.

### deepseek-ai/DeepSeek-V2-Lite

The weights take 29.3 GB in bfloat16. The model needs both cards.

transformers 5.20.2 applies YaRN's mscale^2 natively. The pinned line is the
guarantee; the restoration the server once carried died with the lines that
needed it (4.54 through 5.14, issue #12).

### state-spaces/mamba-2.8b-hf

This model runs on one card. It loads through the eager retry. The
architecture dispatches no attention, so the first load attempt fails and the
retry runs eager.

The engine has no presence_penalty; generate() implements none. The
log-probability request path refuses a presence_penalty above zero with an
error that names the field. The OpenAI routes do not read the field.

## Check

One curl per model, each carrying its table row's sampling.

```bash
curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-0.6B",
       "messages": [{"role": "user", "content": "ping"}],
       "max_tokens": 50,
       "temperature": 0.6, "top_p": 0.95, "top_k": 20}' \
  | python3 -m json.tool

curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "deepseek-ai/DeepSeek-V2-Lite",
       "messages": [{"role": "user", "content": "ping"}],
       "max_tokens": 50,
       "temperature": 0.3, "top_p": 0.95}' \
  | python3 -m json.tool

curl -s http://127.0.0.1:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "state-spaces/mamba-2.8b-hf",
       "messages": [{"role": "user", "content": "ping"}],
       "max_tokens": 50,
       "temperature": 0.7, "top_p": 0.9,
       "repetition_penalty": 1.2}' \
  | python3 -m json.tool
```

The request's `model` field is ignored; the reply names the checkpoint the
server loaded. A request without `max_tokens` generates 16 tokens; without
sampling knobs it samples at temperature 1.0. The contract checks are
`./smoke.sh /models/<model> cuda`: byte-identical double runs.
