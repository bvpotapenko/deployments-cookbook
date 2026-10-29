# llm-transformers (ascend-910b)

A kot deployment: stock HuggingFace transformers behind a thin HTTP adapter
that implements DEPLOYMENT_CONTRACT.md v1. One flat `serve.py`, no kot
imports; kot meets it only over HTTP on 127.0.0.1.

    kot_serve.sh --port P --model /path/to/weights --device npu|cpu

## What it is

`serve.py` loads one causal LM with `AutoModelForCausalLM` + the model's own
tokenizer (from a plain checkpoint directory, or the single snapshot of an
HF-home mount), keeps it resident, and serves four endpoints:

| endpoint                | what it does                                            |
|-------------------------|---------------------------------------------------------|
| `GET /kot/info`         | provenance: engine, dtype, device, vocab_size, revision |
| `POST /kot/score`       | the correctness path: prefill and decode (see below)    |
| `POST /v1/completions`  | OpenAI-shaped, non-streaming (what the speed harness drives) |
| `POST /v1/chat/completions` | same, chat-template on the harness's messages       |

`/kot/infer` is intentionally absent: this is a token-family deployment.

## The numerics contract, as implemented

* ids from `/kot/score` are consumed verbatim: no BOS, no chat template, no
  truncation. A prefill over n ids writes exactly n-1 rows, row t predicting
  `input_ids[t+1]`; a decode writes one row per generated token;
* log-probs are `log_softmax` in float32 over the model's configured
  vocabulary (whatever dtype the weights run in), so every row's logsumexp
  is 0 and columns align by token id. Arrays are C-contiguous `.npy`;
* decode: fixed seed planted immediately before the call, the request's
  sampling object applied exactly as given (temperature, top_p, top_k,
  min_p; generate() takes all four natively), repetition_penalty=1.0; the
  stored logits are the raw pre-sampling ones (`output_logits=True`);
  EOS is suppressed by `min_new_tokens=max_new_tokens`, so exactly
  `max_new_tokens` rows come back;
* determinism: same request, byte-identical arrays (verified on cpu and npu).

## Concurrency model (read before expecting vLLM numbers)

One asyncio event loop. Idle pooled connections cost nothing. A model call
runs to completion inside the loop, so **requests are served strictly one at
a time** -- no batching, no parallel streams, aggregate throughput equals
single-stream throughput (~12 tok/s for Qwen3-0.6B bf16 eager on one 910B).
This is the cross-check baseline: correctness first, speed deliberately
untuned. The KV cache lives inside `generate()` and is freed after.

## How it starts

Two paths, chosen by whether `torch_npu` imports:

* **inside the graded container** (kot built it from the CANN base image):
  `serve.py` runs directly. The runner execs this script with the three
  contract flags; the port is loopback inside the container;
* **on a bare Ascend host** (a golden capture with `--no-container`, or a
  human): the script docker-runs the image built from `./Dockerfile` with the
  ascend-910b device nodes, driver mounts, host networking and a raised
  `nofile` (the speed harness's warmup burst exceeds docker's default 1024).
  Host networking is what makes a loopback-bound server reachable: a
  published port forwards to the container's ethernet address, not to
  127.0.0.1.

Image: `quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12` +
`torch-npu==2.10.0.post2` + `transformers==5.16.1` (pinned by the mlc
harness layer) + the mlc `inference-endpoint` client. Tag it
`kot-llm-ascend-910b:latest` -- kot's container mode inspects exactly that
name. Build it from a local `wheels/` directory (fetch the torch/torch-npu/
torchvision wheels with `curl -C -`; pypi stalls mid-download on this
network -- see NOTES.md 12).

## Talking to MLC Endpoints

The harness is a pure HTTP client; it can run inside the model container
(graded flow: kot's runner spawns it) or outside in its own venv -- it needs
no torch. It drives `POST /v1/chat/completions`, all queries at t=0
(offline scenario), non-streaming, and measures tps / latency / tpot
client-side. Working invocation against a running deployment:

    HF_HUB_OFFLINE=1 inference-endpoint benchmark offline \
      --endpoints http://127.0.0.1:8748 \
      --model Qwen/Qwen3-0.6B \
      --dataset /inputs/speed/<name>/<name>.jsonl \
      --report-dir /tmp/perf \
      --client.worker-initialization-timeout 300 \
      --client.num-workers 8

The two client flags matter: the tool's defaults (60 s worker init, 24
workers) are a coin flip on a many-core host, and its auto connection budget
pre-opens hundreds of pooled connections per worker -- the asyncio server
absorbs that, a thread-per-connection server does not. Run the harness with
`--ulimit nofile=65536:65536 --shm-size=1g` when it lives in its own
container, and keep `HF_HUB_OFFLINE=1` (the tokenizer fetch stalls ~2 min
otherwise; or mount tokenizer files and pass `--tokenizer`).

## Knobs

| env         | default            | meaning                                    |
|-------------|--------------------|--------------------------------------------|
| `KOT_DTYPE` | `bfloat16` on npu, `float32` on cpu | model dtype (bf16/fp16/fp32 on npu) |
| `KOT_ATTN`  | `eager` on npu     | attention implementation (sdpa untested)   |
| `KOT_NPU_IMAGE` | `kot-llm-ascend-910b:latest` | image for the host-mode docker wrap |
| `KOT_OUT_MOUNTS` | unset        | colon-separated dirs mounted rw at identical paths so `/kot/score` writes where the host-side runner reads (host mode only) |

## What models it supports

Anything stock transformers loads as `AutoModelForCausalLM` with a text
tokenizer: Qwen2/Qwen3, Llama, Mistral, Gemma, DeepSeek (text), Mamba-hf,
GPT-family, and `trust_remote_code` models (enabled). Multimodal/VL
checkpoints, embedding-only models, and runtimes needing their own serving
stack (vLLM/sglang quant formats) are out of scope for this deployment --
other spikes cover those engines.

## Known limits

* serial serving (see above): speed is the baseline, not the headline;
* eager attention materializes the full attention matrix: long contexts cost
  O(n^2) HBM (the frozen 9k-token continuations fit on one 64 GB 910B);
* one card, one process: no tensor parallelism;
* if the NPU device faults (rare AICPU kernel exceptions under exotic
  concurrent timing), the process context stays dead -- every later request
  answers 500 until restart. The device itself recovers.

## Run without hardware

On a machine with neither torch_npu nor docker, run `serve.py` under `uv`
directly:

    uv run --with torch --with transformers --with accelerate \
      --with safetensors --with numpy \
      python serve.py --port P --model <snapshot> --device cpu
