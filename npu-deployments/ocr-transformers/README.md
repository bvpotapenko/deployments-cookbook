# ocr-deepseek-ocr-2 (ascend-910b)

Stock `transformers` serving of DeepSeek-OCR-2 under the kot deployment
contract, on Ascend 910B through torch_npu. This is the NPU port of
`cuda-deployments/ocr-transformers` (the golden reference engine):
same numerics, same endpoints, an npu device branch added.

## The stack, and why it is pinned

- torch 2.10.0 + torch_npu 2.10.0.post2 + torchvision 0.25.0: the
  `platform-ascend-910b.txt` pairing, on the CANN 9.1.0-beta.1 base image.
- transformers 4.46.3: `family-ocr.txt` pins it; the checkpoint's remote code
  breaks on 5.x. The CUDA golden ran this exact version.
- The image is built from this directory's Dockerfile and tagged
  `kot-ocr-ascend-910b:latest` (what `kot.container.image_name` expects);
  `wheels/` holds the three platform wheels - copy them once from
  `llm-vllm`'s wheelhouse on the box (see the Dockerfile header).

## How it starts (`kot_serve.sh`)

Two modes, mirroring `npu-deployments/llm-transformers`:

- inside the graded container (python3 carries torch_npu): run `serve.py`
  directly;
- on a bare host: wrap `docker run` with the Ascend driver mounts and device
  nodes (the `kot.platforms` ascend-910b row), deployment read-only at
  `/deployment`, weights read-only at their own path, and every
  `KOT_OUT_MOUNTS` directory mounted rw at the same path so `/kot/score`
  writes where the host-side runner reads. `KOT_NPU_IMAGE` overrides the
  image tag.

The interpreter chain prefers `$KOT_OCR_PY`, then `/root/kotvenv-ocr`, then
`/root/kotvenv`, then `python3` - but only a candidate that can import both
torch_npu and transformers runs `serve.py` directly; everything else falls
to the docker wrap. On the box (non-root host shells cannot import
torch_npu), that means the wrap.

## The NPU deltas over the CUDA deployment

- `import torch_npu` inside `main()`, gated on `--device npu`, before any
  tensor is placed.
- dtype bfloat16 on npu (as on cuda); `KOT_DTYPE` overrides, a typo fails
  loudly.
- attention defaults to eager on npu, `KOT_ATTN` overrides. eager is the
  only MLA attention the vanilla checkpoint's LLM implements: its
  `ATTENTION_CLASSES` has no sdpa entry and it never declares
  `_supports_sdpa`, so transformers 4.46.3 refuses an explicit sdpa request
  outright (the first smoke died exactly there). The SAM vision encoder
  runs `F.scaled_dot_product_attention` unconditionally on every platform,
  so the image path is identical on cuda and npu; the env var switches
  the LLM half only.
- the `Tensor.cuda()` redirect (which the cpu path already needed) carries
  the checkpoint's own `.cuda()` calls to npu.
- `/kot/info` additionally reports `torch_version`, `plugin_version`,
  `driver_version`, `toolkit_version`, read best-effort from the process
  and the Ascend version files; empty when unreadable, never invented.
- Sampling fields: the deployment accepts the six the current kot client
  sends on every decode (`Sampling.model_dump`): the contract's four plus
  `repetition_penalty` (passed to generate() as given; the neutral 1.0 adds
  no warper, so the compute is exactly the golden's, whose requests
  predate the field) and `presence_penalty` (no knob in transformers
  4.46.3's generate(); the neutral 0.0 applies nothing, and a non-neutral
  value is an error naming it, never silently ignored).

Everything else is the CUDA deployment verbatim: prefill teacher-forcing
and row slicing, decode sampling and the raw pre-processor logits, the
n-gram anti-repetition rule (30/90, whitelist `<td>`/`</td>` - note the
kernelcat vllm case on the box used n=20; this deployment keeps the golden's
30), `open_memmap` row streaming, `input_ids.npy` carrying the raw template
with exactly one image placeholder.

## Endpoints

`/kot/info` and `/kot/score` only. No chat route: the golden
`deepseek-ocr-2-rtx5090-transformers` holds a `capture_0` but no `speed_0`,
and the family's speed harness (the endpoints benchmark) drives
`/v1/chat/completions`, so speed is simply not measured for this engine
pair. If a chat route is ever wanted, port `llm-transformers`'s.

## Verify by hand

On the box (model at `~/models/DeepSeek-OCR-2`):

```bash
cd ~/kot_deployments/npu-deployments/ocr-transformers
ASCEND_RT_VISIBLE_DEVICES=2 ./smoke.sh ~/models/DeepSeek-OCR-2 npu
```

The first run builds the image first (a few minutes); the smoke then checks:
two identical prefills byte-identical, decode under both EOS gates
(exact-length and natural-stop), the capture's decode-then-replay round-trip
over a >768px image (crop-tile path), one image placeholder in the stored
ids, float32/logsumexp/contiguity, and byte-identity of the replayed rows.
The verification interpreter needs numpy only (`~/kot/.venv/bin/python` on
the box).

## Known limits

- npu/bfloat16 determinism is asserted by the smoke (byte-identical repeats),
  not yet by a capture; the first candidate capture is the real test.
- The CUDA golden requested sdpa explicitly, but a vanilla stack cannot
  serve that request (transformers 4.46.3 raises, and the remote
  ATTENTION_CLASSES has no `mla_sdpa` to build), so the golden's
  vast-box environment deviated from vanilla in some unrecorded way and
  the attention its LLM effectively ran is unknown. This deployment runs
  eager, the deployment's own cpu-verified path; the eval's statistical
  thresholds absorb attention-kernel noise, and if the candidate's
  distances look noisy the golden's attention is suspect number one.
- One request at a time (`http.server`); the contract promises exactly that
  for `/kot/score`, and this engine is the correctness baseline, not the
  speed runner.
