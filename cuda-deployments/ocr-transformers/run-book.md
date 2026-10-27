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
uv pip install --python /root/kotvenv-ocr/bin/python torch==2.14.1
uv pip install --python /root/kotvenv-ocr/bin/python \
  transformers==5.20.2 numpy safetensors pillow huggingface_hub pydantic pyyaml
```

`provision.sh` groups these commands, installs uv when it is missing, and
ends with an import self-check. Transformers 5.20.2 ships the configuration,
the image processing, the processing and the modeling for this checkpoint.
The deployment's image processor is the PIL variant, so torchvision is not
needed; the torch build must still carry sm_120 (PyPI's default for 2.14.x
is a cu13x build that does).

### Layout

```text
/opt/engines/ocr-transformers   this directory: serve.py beside the three
                                builder modules it imports, native_config
                                (build_config), native_load (load_native),
                                native_processor (build_processor)
/models/deepseek-ai/DeepSeek-OCR-2
                                the weights directory; a hub snapshot plus
                                one patched file
```

### Load

The server loads through the native architecture and gates the load before
any scoring: build_config reads the original config and asserts the lifted
top-level copies agree with the nested originals; load_native calls
from_pretrained with that config (transformers applies its registered
deepseek_ocr2 conversion in memory, renames and stacks only) and checks
0 missing / 0 unexpected keys plus a bitwise comparison of every checkpoint
tensor against the loaded model; build_processor reproduces the reference
image path (257 global tokens, 144 per 768x768 tile). The gates' report is
stamped into /kot/info.

### Transfer

Push and pull over the provider-mapped ssh port. rsync makes no parent
directories and drops execute bits; `--partial` resumes an interrupted
transfer. The excludes keep `__pycache__` and `.DS_Store` off the box.

```bash
mkdir -p /opt/engines/ocr-transformers
rsync -a --partial -e "ssh -p $PORT" \
  --exclude __pycache__ --exclude .DS_Store \
  ./ocr-transformers/ user@$HOST:/opt/engines/ocr-transformers/
ssh -p $PORT user@$HOST 'chmod +x /opt/engines/ocr-transformers/*.sh'
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
matches the source. The patched `modeling_deepseekv2.py` travels in the
same push.

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
revision `aaa02f3811945a91062062994c5c4a3f4c0af2b0`, plus one patched file
over the snapshot. Take the whole directory. Never refresh single files
from the hub.

- `config.json`, sha256
  `e2bb23b49b445f0bfd6eeb978bac9f19a2df82cb38472754e4ae83662d18f845`. This
  is the PRISTINE hub file at the revision, verified byte-for-byte against
  the hub on 2026-10-23. It carries the language model's settings BOTH
  nested under `language_config` and as lifted top-level copies; the build
  reads only the nested block and asserts the copies agree. An earlier
  revision of this book recorded a patched config with sha
  `6e26957b...`; that file added 23 default-valued fields and is gone.
- `modeling_deepseekv2.py`, sha256
  `41ae3cf974208667f2e89a6392c7d65ffdd38d002a68fbf5063838cc9eb664a7`.
  Bogdan Potapenko patched the hub file for transformers 5.x
  compatibility: guarded the `LlamaFlashAttention2` and
  `is_torch_fx_available` imports, added the `mha_sdpa` entry and
  `_supports_sdpa`. The native path loads no remote
  modeling file, so this file sits unused.

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
