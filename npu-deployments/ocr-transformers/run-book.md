# Run Book: ocr-transformers

This deployment serves DeepSeek-OCR-2 with transformers 5.22.1. The stack
runs on the Ascend 910B through torch_npu 2.10.0.post2 under CANN 9.2.1. The
checkpoint loads through the native DeepseekOCR2 architecture that
transformers 5.22.1 ships, and no remote code runs. One process serves one
scoring route. The server answers one request at a time. The server has no
chat route.

## Model parameters

The table holds the launch parameters per model. Every value not named stays
at the deployment default. Read the row against the same model in the other
engine run books. A mismatch between engines is the first suspect when a
comparison fails.

| model | revision | cards | dtype | sampling (recommended) |
|---|---|---|---|---|
| deepseek-ai/DeepSeek-OCR-2 | `aaa02f3811945a91062062994c5c4a3f4c0af2b0` | 1 | bfloat16 | temperature 0.0001, top_p 1.0 |

One stack serves this model: CANN 9.2.1, torch 2.10.0, torch_npu 2.10.0.post2,
transformers 5.22.1, CPython 3.12. The seed arrives with the request; 42 is
the convention.

## Server: Ascend 910B

### Environment

The image runs CPython 3.12 on aarch64. The wheels are cp312 manylinux, so
they install only under that Python.

```bash
# wheels/ must hold the three platform wheels first; copy them from any
# existing Ascend deployment, or fetch them from PyPI.
docker build -t kot-ocr-ascend-910b:latest .
```

These pins must not float.

- The base image `quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12-devel`
  carries the CANN 9.2.1 toolkit for the 910B. The toolkit half of the stack
  lives in the image. The driver half lives on the host, and the wrap mounts
  it read-only at start. The driver on the host must pair with the CANN
  toolkit, so float the image only together with a host driver that supports
  CANN 9.2.1.
- The build installs `torch==2.10.0`, `torch_npu==2.10.0.post2` and
  `torchvision==0.25.0` from wheel files, never pip resolution. torch_npu
  registers the npu backend, its version pairs with torch, and the plugin
  build pairs with the CANN release in the base image. torchvision must match
  the same torch line. The stack moves as a set. Change the CANN version only
  together with torch_npu, and the host driver only together with the CANN
  toolkit.
- The build pins `transformers==5.22.1`. The checkpoint loads through the
  native DeepseekOCR2 architecture, and no remote code runs.

### Layout

```text
/opt/engines/ocr-transformers  this directory: serve.py beside the three builder
                               modules it imports, native_config (build_config),
                               native_load (load_native), native_processor
                               (build_processor); the wrap mounts it read-only
                               at /deployment
/models/DeepSeek-OCR-2          the checkpoint directory; mounted read-only at its own path
```

The Python files are not baked into the image. The wrap mounts this directory
read-only at /deployment and runs the script from there, so you edit them
here with no rebuild.

The model directory must hold config.json at its root. A HuggingFace cache
layout is refused.

### Start

```bash
./serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device npu
# one card pinned:
ASCEND_RT_VISIBLE_DEVICES=1 ./serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device npu
```

The script tries one interpreter after another: `$KOT_OCR_PY`, then
`/root/kotvenv-ocr/bin/python`, then `/root/kotvenv/bin/python`, then
python3. Only an interpreter that imports both torch_npu and transformers
runs `serve.py` directly. Every other host uses the docker wrap. The driver
files on the host are readable by root only, so a non-root shell cannot
import torch_npu and the wrap is the route.

The wrap mirrors the llm-transformers deployment. It builds the image on
the first run, runs privileged with host networking, raises the soft
open-file limit to 65536, mounts the weights and the deployment read-only,
passes the visible ordinals through `ASCEND_RT_VISIBLE_DEVICES` and
`ASCEND_VISIBLE_DEVICES` with one `/dev/davinci<ordinal>` node per ordinal,
and mounts the host driver read-only.

`ASCEND_RT_VISIBLE_DEVICES` wins, and `ASCEND_VISIBLE_DEVICES` is the
fallback. The values are card ordinals, comma separated.

The server binds 127.0.0.1, and host networking keeps it reachable on the
host's loopback address. A published port does not work, because it forwards
to the container's ethernet address, where nothing listens.

Facts about the load:

- torch_npu is imported before any tensor is placed.
- The native DeepseekOCR2 architecture loads the checkpoint, and no remote
  code runs.
- `build_config` reads `config.json` and builds the native configuration.
- `load_native` stops the start when a key is missing or unexpected, or when
  one checkpoint tensor differs from the loaded model. `/kot/info` carries
  the result under `load_report`.
- The weights run bfloat16 on npu; `KOT_DTYPE` overrides, and a typo is a
  load-time error.
- Attention runs eager on npu, and `KOT_ATTN` overrides.

The start line prints the whole stack: engine, dtype, device, vocabulary,
and the torch, plugin, driver and toolkit versions. A field the process
cannot read prints empty, and nothing is invented. Read the line to confirm
the driver and toolkit pair.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_OCR_PY` | unset | interpreter for the direct route; it must import torch_npu and transformers. |
| `KOT_DTYPE` | `bfloat16` on npu, `float32` on cpu | model dtype. A typo is an error at load. |
| `KOT_ATTN` | `eager` on npu | attention implementation for the whole model: the SAM encoder, the vision encoder and the language model. |
| `KOT_NPU_IMAGE` | `kot-ocr-ascend-910b:latest` | image the host-mode wrap runs. |
| `KOT_OUT_MOUNTS` | unset | colon-separated directories mounted read-write at the same path, so the server's output files land where the host reads them. Host mode only. |
| `ASCEND_RT_VISIBLE_DEVICES` | `0` when both device variables are unset | card ordinals the wrap passes to the container; `ASCEND_VISIBLE_DEVICES` is the fallback. In the direct route, export them yourself. |

## Models

### deepseek-ai/DeepSeek-OCR-2

This model runs on one card. The weights take about 6.4 GB. The vocabulary
holds 129,280 tokens.

The model directory is a hub snapshot plus two patched files from the
remote-code era: `config.json` and `modeling_deepseekv2.py`.
`modeling_deepseekv2.py` was edited by Bogdan Potapenko for
transformers 5.x compatibility. Copy the
directory whole. The native path loads no modeling file from the weights
directory. `config.json` carries the language model's settings under
`language_config` and as top-level copies. `build_config` reads only
`language_config` and stops the start when a copy disagrees.

An image with both sides at most 768 px becomes one 257-token block. A
larger image adds a grid of up to six 768 px crop tiles, 144 tokens per
tile.

A decode request that carries no ids gets the template the frozen inputs
hold: BOS, the image block, then the ids of the instruction `Free OCR.`.

Generation applies a no-repeat n-gram rule. A token that would complete a
30-gram already seen within the last 90 tokens is banned from sampling. The
two table tokens, 128821 (`<td>`) and 128822 (`</td>`), stay allowed, so
table structure may repeat. The rows the route returns are computed before
the rule runs, so the rule changes which token is sampled and never a
returned row.

The server refuses a presence_penalty above zero, because generate()
implements no such knob.

## Check

The server serves no chat route. Run the copied self-check with chat checks
skipped:

```bash
python3 self_check.py --deployment . \
  --model /models/DeepSeek-OCR-2 --device npu --no-chat
```

The script starts `serve.sh` on a free port and builds the decode ids from the
model tokenizer. It replays the generated ids as a prefill and checks the
returned files. Stop an existing server before starting the check.

The server serves no chat route. It serves the scoring route that
`serve.py` defines. The identity surface answers with the engine's own
facts:

```bash
curl -s http://127.0.0.1:8090/kot/info | python3 -m json.tool
```

`engine` reads `transformers 5.22.1`, and `toolkit_version` reads
`cann-9.2.1`.

Any image serves the scoring route, and one wider or taller than 768 px also
exercises the crop path.
