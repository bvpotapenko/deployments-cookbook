# Run Book: ocr-transformers

Stock transformers serving of DeepSeek-OCR-2 on Ascend 910B through
torch_npu: one flat `serve.py`, an image-driven scoring route, one request
at a time. The server has no chat route. transformers is pinned to 4.46.3:
the checkpoint's remote code breaks on 5.x.

## Server: Ascend 910B

### Environment

The image is aarch64 with CPython 3.12. The wheels are cp312 manylinux;
they install only under that Python.

```bash
# wheels/ must hold the three platform wheels first; copy them from any
# existing Ascend deployment, or fetch them from PyPI.
docker build -t kot-ocr-ascend-910b:latest .
```

Pins that must not float:

- Base image `quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12`:
  the CANN toolkit 9.1.0-beta.1 for the 910B. The toolkit half of the stack
  lives in the image. The driver half lives on the host; the wrap mounts it
  read-only at start. The two halves must stay a compatible pair: float the
  image only together with a host driver that supports it.
- `torch==2.10.0`, `torch_npu==2.10.0.post2`, `torchvision==0.25.0`, from
  wheel files, never pip resolution. torch_npu registers the npu backend;
  its version pairs with torch, and the plugin is built for the CANN
  release in the base image. torchvision must match the same torch line.
- `transformers==4.46.3`: the checkpoint's remote code breaks on 5.x. Do
  not raise this pin. `numpy==2.2.6`. The layer also installs the modules
  the remote code imports: einops, addict, pillow, and others.

### Layout

```text
/opt/engines/ocr-transformers  this directory; the wrap mounts it read-only at /deployment
/models/DeepSeek-OCR-2          the checkpoint directory; mounted read-only at its own path
```

`serve.py` is not baked into the image. The wrap mounts this directory
read-only at /deployment and runs the script from there: edit `serve.py`
here, no rebuild needed.

The model directory must hold config.json at its root. A HuggingFace cache
layout is refused.

### Start

```bash
./kot_serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device npu
# one card pinned:
ASCEND_RT_VISIBLE_DEVICES=1 ./kot_serve.sh --port 8090 --model /models/DeepSeek-OCR-2 --device npu
```

The script tries one interpreter after another: `$KOT_OCR_PY`, then
`/root/kotvenv-ocr/bin/python`, then `/root/kotvenv/bin/python`, then
python3. Only a candidate that imports both torch_npu and transformers runs
`serve.py` directly. Every other host uses the docker wrap. Where the
driver files are readable by root only, a non-root shell cannot import
torch_npu, and the wrap is the route.

The wrap mirrors the llm-transformers deployment: it builds the image on
the first run, runs privileged with host networking, raises the open-file
limit to 65536, mounts the weights and the deployment read-only, passes the
visible ordinals through both `ASCEND_VISIBLE_DEVICES` and
`ASCEND_RT_VISIBLE_DEVICES` with one `/dev/davinci<ordinal>` node per
ordinal, and mounts the host driver read-only. The server binds 127.0.0.1;
host networking keeps it reachable on the host's loopback address. A
published port does not work.

Facts about the load:

- torch_npu is imported before any tensor is placed.
- The server changes into the model directory before it loads: the remote
  code imports from its own directory.
- `torch.Tensor.cuda` is rebound to the selected device. The checkpoint's
  own `.cuda()` calls then run on the npu.
- bfloat16 on npu; `KOT_DTYPE` overrides, and a typo is a load-time error.
- Attention runs eager on npu; `KOT_ATTN` overrides. The vision encoder
  calls scaled_dot_product_attention unconditionally on every platform:
  the parameter switches the language half only.

The start line prints the whole stack: engine, dtype, device, vocabulary,
and the torch, plugin, driver and toolkit versions. A field the process
cannot read prints empty; nothing is invented. Read it to confirm the
driver and toolkit pair.

### Parameters

| variable | default | effect |
|---|---|---|
| `KOT_OCR_PY` | unset | interpreter candidate for the direct route; it must import torch_npu and transformers. |
| `KOT_DTYPE` | `bfloat16` on npu, `float32` on cpu | model dtype. A typo is an error at load. |
| `KOT_ATTN` | `eager` on npu | attention implementation for the language half. |
| `KOT_NPU_IMAGE` | `kot-ocr-ascend-910b:latest` | image the host-mode wrap runs. |
| `KOT_OUT_MOUNTS` | unset | colon-separated directories mounted read-write at the same path, so the server's output files land where the host reads them. Host mode only. |
| `ASCEND_RT_VISIBLE_DEVICES` | then `ASCEND_VISIBLE_DEVICES`, then `0` | card ordinals the wrap passes to the container. In the direct route, export them yourself. |

## Models

### deepseek-ai/DeepSeek-OCR-2

One card. bfloat16. Weights about 6.4 GB. Vocabulary 129,280.

The model directory is a hub snapshot plus two patched files: config.json
and modeling_deepseekv2.py. Copy the directory whole. Never refresh the two
patched files from the hub: the patched pair is what this stack loads.

An image with both sides at most 768 px becomes one 257-token block. A
larger image adds a grid of up to six 768 px crop tiles, 144 tokens per
tile.

Generation applies a built-in no-repeat n-gram rule: a token that would
complete a 30-gram already seen within the last 90 tokens is banned from
sampling. The two table tokens stay allowed, so structure may repeat. The
rule changes which token is sampled; the stored rows stay the raw
pre-sampling ones.

Sampling: temperature 0.0001, top_p 1.0. A non-neutral presence_penalty is
refused: transformers 4.46.3 has no such parameter in generate().

## Check

The server serves no chat route. It serves the scoring route defined in
`serve.py` only. Run the deployment's own check:

```bash
IMAGE=/path/to/document.png ./smoke.sh /models/DeepSeek-OCR-2 npu
```

The smoke starts the server, scores one prefill and one decode twice, and
prints PASS/FAIL lines for shape, dtype, row count and byte-identity. Any
image works; one wider or taller than 768 px also exercises the crop path.
The first run builds the image first: allow a few minutes.
