# Run Book: ocr-transformers

Stock `transformers` serving of DeepSeek-OCR-2, one resident model, one
request at a time (`http.server`). The trap sits in the weights: the model
directory is a hub snapshot with two files replaced, and the server refuses
to start from anything else.

## Server: NVIDIA RTX 5090 (sm_120)

### Environment

CPython 3.12. `provision.sh` builds `/root/kotvenv-ocr` and ends with an
import self-check. It takes the interpreter as an argument, else it uses the
host python3; the reference host ran 3.12.

```bash
uv venv /root/kotvenv-ocr --python 3.12
uv pip install --python /root/kotvenv-ocr/bin/python \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python /root/kotvenv-ocr/bin/python \
  transformers==4.46.3 "numpy<2.2" accelerate safetensors \
  addict einops pillow requests tqdm "huggingface_hub[hf_transfer]"
```

`provision.sh` groups these commands and installs uv when it is missing. When
the pytorch index stalls mid-wheel, PyPI carries the same cu128 build: drop
the `--index-url`.

Pins that must not float:

- `transformers==4.46.3`. The checkpoint's remote code breaks on 5.x. This
  is a requirement, not a preference.
- `torch==2.8.0` and `torchvision==0.23.0` from the cu128 index. sm_120
  (RTX 5090) needs the CUDA 12.8 kernels; this index is where that build
  lives.
- `addict` and `torchvision` are load-time requirements of the checkpoint's
  remote code, not of `transformers`. A plain venv dies at load with an error
  that names them.

The reference venv resolved tokenizers 0.20.3, accelerate 1.15.0, einops
0.8.2, pillow 12.3.0, addict 2.4.0.

### Layout

```text
/opt/engines/ocr-transformers   this directory
/models/DeepSeek-OCR-2          the hub snapshot directory with two files replaced (see Models)
```

### Start

```bash
./kot_serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device cuda
```

The process listens on 127.0.0.1 only. The launcher resolves its interpreter
itself: KOT_OCR_PY, else `/root/kotvenv-ocr/bin/python`, else
`/root/kotvenv/bin/python`, else the PATH python3. It then changes directory
into the model, because the checkpoint's remote code imports modules from its
own directory. The server loads one checkpoint directory: the model path must
hold `config.json`, and a hub cache layout does not load here.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_OCR_PY` | unset | interpreter that carries the pinned stack. Set it when the venv is not at `/root/kotvenv-ocr`. |

The script takes `--port`, `--model` and `--device`; `cuda` runs bfloat16
under sdpa, `cpu` runs float32 under eager.

## Models

### deepseek-ai/DeepSeek-OCR-2

Hub revision `aaa02f38`. One card.

Download the snapshot, then replace two files in it with the patched copies
(sha256 below). The vanilla files do not serve:

- `config.json`, sha256
  `6e26957be7ef11041dce309baac46766e30f0fb45f4af6233307de2c0099de77`. It
  duplicates the language model's settings at the top level of the file; the
  vanilla file carries them only under `language_config`.
- `modeling_deepseekv2.py`, sha256
  `41ae3cf974208667f2e89a6392c7d65ffdd38d002a68fbf5063838cc9eb664a7`. It adds
  an `mha_sdpa` entry to the checkpoint's `ATTENTION_CLASSES` and sets
  `_supports_sdpa = True`. The vanilla file offers eager and
  flash-attention-2 only, and `transformers` 4.46.3 refuses the explicit sdpa
  request this server makes on cuda.

```bash
hf download deepseek-ai/DeepSeek-OCR-2 --revision aaa02f38
```

Generation applies DeepSeek's own no-repeat rule: a token that would complete
a 30-gram already seen inside a 90-token lookback window is banned from
sampling, except the table delimiters `<td>` (128821) and `</td>` (128822).
`repetition_penalty` stays 1.0; the no-repeat rule carries the
anti-repetition behaviour.

Sampling: temperature 0.0001, top_p 1.0.

## Check

This deployment serves no OpenAI-style route. The by-hand check is the smoke:
it teacher-forces one transcript over one image twice, decodes under both
stop gates, and prints PASS/FAIL lines for shape, dtype and byte-identity.

```bash
IMAGE=/path/to/document.png ./smoke.sh /models/DeepSeek-OCR-2 cuda
```

Any image serves. An image with one side above 768 px exercises the crop
path: the model reads a grid of 768x768 tiles beside the padded global view,
and the image block grows by 144 tokens per tile.
