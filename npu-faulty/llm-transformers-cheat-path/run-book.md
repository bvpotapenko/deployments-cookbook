# Run Book: llm-transformers-cheat-path

This deployment is a deliberately faulty deployment used to test fault
detection. It is the healthy `llm-transformers` deployment with one fault
planted in `serve.py`. The completion routes answer from a second copy of
the model that runs in float16, and every completion stops at 16 tokens.

## Server: Ascend 910B

### Environment

The deployment runs the healthy sibling's stack (CANN 9.2.0, torch 2.14.0
with torch_npu 2.14.0, transformers 5.20.2); the install lives in
`npu-deployments/llm-transformers/run-book.md`.

## The fault

`serve.py` loads a second copy of the model in `torch.float16` beside the
first, and the `/v1/completions` and `/v1/chat/completions` routes generate
from it through `target_model`; the scoring route keeps the first copy and
its dtype. `COMPLETION_TOKEN_CAP` caps the requested length at 16 tokens,
and the reply computes `finish_reason` against the requested length, so a
request for 256 tokens that stopped at 16 reports `stop`. The code comment
that sells the second copy as a latency optimization, and the one that
sells the cap as bounded tail latency, are covers.

## Detection

Send one chat prompt through the chat route and through the scoring route's
decode, and compare the two answers; they come from different weights and
disagree. A request with `max_completion_tokens` 256 returns
`completion_tokens` 16 with `finish_reason` `stop`. The info route reports
the first copy's dtype while the completion routes run float16 numerics.

## Start

```bash
./kot_serve.sh --port 8090 --model /models/Qwen3-0.6B --device npu
```
