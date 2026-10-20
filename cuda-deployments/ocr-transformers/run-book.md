# Run Book: ocr-transformers

This deployment serves DeepSeek-OCR-2 with transformers 5.20.2. One process
serves the full-vocabulary log-probability surface on one port, one request
at a time. The checkpoint loads through the native DeepseekOCR2 architecture
that transformers 5.20.2 ships. The remote code is not used. Generation
applies the checkpoint's own no-repeat n-gram rule.

## Model parameters

The table holds the launch parameters per model. Every value not named stays
at the deployment default. Read a model's row against the same row in the
other engine run books. A mismatch between engines is the first suspect when
a comparison fails.

| model | revision | cards | dtype | sampling (recommended) |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-OCR-2 | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | 1 | bfloat16 | temperature 0.0001, top_p 1.0 |

One stack serves every model: CPython 3.12, transformers 5.20.2, and the
torch build from the cu128 index that carries sm_120. The seed arrives with
the request; 42 is the convention.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

The stack runs on CPython 3.12.

```bash
uv venv /root/kotvenv-ocr --python 3.12
uv pip install --python /root/kotvenv-ocr/bin/python \
  torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /root/kotvenv-ocr/bin/python \
  transformers==5.20.2 numpy addict einops pillow accelerate huggingface_hub \
  pydantic pyyaml
```

`provision.sh` groups these commands, installs uv when it is missing, and
ends with an import self-check. Transformers 5.20.2 ships the configuration,
the image processing, the processing and the modeling for this checkpoint.

The torch build must carry sm_120. The cu128 index is where that build
lives. A torch without Blackwell kernels does not run on these cards.

### Layout

```text
/opt/engines/ocr-transformers   this directory
/models/DeepSeek-OCR-2          the weights directory; a hub snapshot plus two patched files
```

### Start

```bash
./kot_serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device cuda
```

The process listens on 127.0.0.1 only. The launcher resolves its interpreter
itself. It picks `KOT_OCR_PY`, else `/root/kotvenv-ocr/bin/python`, else
`/root/kotvenv/bin/python`, else the ambient python3. The server loads one
checkpoint directory. The model path must hold `config.json`. A hub cache
layout does not load here.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_OCR_PY` | unset | interpreter that carries the pinned stack. Set it when the venv is not at `/root/kotvenv-ocr`. |

The script takes `--port`, `--model` and `--device`. `cuda` runs bfloat16
under sdpa. `cpu` runs float32 under eager.

## Models

### deepseek-ai/DeepSeek-OCR-2

The model runs on one card.

The weights directory is a hub snapshot of `deepseek-ai/DeepSeek-OCR-2` at
revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`, plus two patched files
over the snapshot. Take the whole directory. Never refresh single files
from the hub.

- `config.json`, sha256
  `6e26957be7ef11041dce309baac46766e30f0fb45f4af6233307de2c0099de77`. The
  patch lifts the language model's settings from the `language_config` block
  to the top level. The vanilla file carries them only under
  `language_config`.
- `modeling_deepseekv2.py`, sha256
  `41ae3cf974208667f2e89a6392c7d65ffdd38d002a68fbf5063838cc9eb664a7`. The
  native path loads no remote modeling file, so this file sits unused.

The native loader reads `config.json`. Check the lifted top-level block
against what the native DeepseekOCR2 configuration expects.

Generation applies the checkpoint's own no-repeat n-gram rule. The rule
blocks every token that would complete a 30-gram already seen inside a
90-token lookback window. The table delimiters `<td>` (128821) and `</td>`
(128822) stay exempt, so table structure may repeat. `repetition_penalty`
stays 1.0. The no-repeat rule carries the anti-repetition behaviour. The
rule changes which token is sampled, never a stored row. The stored rows
are the log-softmax of the raw logits, taken before the rule runs.

The sampling temperature 0.0001 is near-greedy by design. Generation runs
it through the real sampler path.

## Check

This deployment serves no chat route. The by-hand check is the smoke. It
teacher-forces one transcript over one image twice, decodes under both stop
gates, and prints PASS/FAIL lines for shape, dtype and byte-identity.

```bash
IMAGE=/path/to/document.png ./smoke.sh /models/DeepSeek-OCR-2 cuda
```

Any image serves. An image with one side above 768 px exercises the crop
path. The model reads a grid of 768x768 tiles beside the padded global
view. The image block grows by 144 tokens per tile.
