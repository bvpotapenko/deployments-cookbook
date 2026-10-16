# ocr-deepseek-ocr-2 (cuda)

Stock `transformers` serving of DeepSeek-OCR-2 under the kot deployment
contract. This is the cross-check baseline for the ocr family: its numerics
mirror kot's own reference deepseek adapter bit for bit (verified on cpu,
fp32, prefill and decode).

## The numerics contract, as implemented

- `input_ids` are consumed verbatim: the transcript is appended to the
  template the checkpoint itself builds (`format_messages` in plain format,
  `<image>\nFree OCR.`), no BOS on the transcript, no truncation.
- A prefill over n transcript ids writes exactly n - 1 rows: row t is the
  prediction for `input_ids[t + 1]`, sliced from
  `logits[prompt_len : prompt_len + n - 1]`.
- Decode is image-driven: the template only (the transcript's ids are what
  prefill teacher-forces over), the request's sampling object applied
  exactly as given (temperature, top_p, top_k, min_p; generate() takes all
  four natively), seed planted immediately before the call,
  `repetition_penalty=1.0`, `min_new_tokens = max_new_tokens` (EOS
  suppression for sampling only). The stored rows are the raw pre-processor
  logits (`output_logits=True`), float32 log-softmax over the full 129280
  vocabulary; `input_ids.npy` stores template + generated.
- Log-probs stream to disk through `numpy.lib.format.open_memmap` at the
  final shape, 64 rows per chunk, so no unit's distribution sits whole in
  host memory.

## The image path, both branches of the checkpoint's own infer()

- both sides <= 768 px: padded to 1024x1024, one 257-token block, zero crop
  tensor, `images_spatial_crop = [[1, 1]]`.
- either side > 768 px: the checkpoint's own `dynamic_preprocess` picks a
  2..6-tile grid of 768x768 crops; the token block grows by 144 tokens per
  tile; the forward gets `[(crop_stack, global_view)]` and
  `[[width_crops, height_crops]]`. Verified on a 1200x800 fixture: the model
  reads it (decode reproduces the rendered text) and both modes are
  deterministic.

## Interpreter and environment

`kot_serve.sh` names its interpreter explicitly: `$KOT_OCR_PY` if set, else
`/root/kotvenv-ocr/bin/python` (this deployment's venv; `provision.sh`
builds it), else `/root/kotvenv/bin/python`, else PATH. transformers is
pinned to 4.46.3 because the checkpoint's remote code breaks on 5.x - the
shared llm venv (5.17.0) cannot serve this model.

`smoke.sh MODEL [device]` runs the by-hand contract verification; the
fixture image is found in the known kot store layouts or named with
`IMAGE=/path/to.png` (any image works; >768 px exercises the crop path).

## Known limits

- cuda/bfloat16 determinism is not yet verified (cpu fp32 is; the golden
  capture on the rented box will be the first bf16 run - run the same
  request twice there before trusting it).
- The crop path mirrors `infer()`'s conventions including the
  local-then-global feature order the remote forward scatters; it has no
  reference adapter to diff against yet (kot's reference is single-view),
  so its numerics are checkpoint-convention, not cross-checked.
- One request at a time (`http.server`); the contract promises exactly that
  for `/kot/score`, and the ocr speed harness is builtin, not the
  connection-storm endpoint benchmark.
