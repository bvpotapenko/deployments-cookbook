# vla-transformers-image-drop

Status: planned. Not implemented. This note records the case.

## Fault

openvla/openvla-7b runs through its own remote code (`modeling_prismatic.py`)
on transformers 5.x. That code was written for transformers 4.40.1. It fails
to load on 5.x. The obvious patches make it load:

- `PrismaticPreTrainedModel._supports_sdpa`: replace the property with `True`.
- `PrismaticForConditionalGeneration.tie_weights`: accept `**kwargs`.
- Mix `GenerationMixin` into the class and set a `generation_config`.

`predict_action` then calls `generate()`. In 5.x, `generate()` makes an
empty `DynamicCache` before the first step. The remote
`prepare_inputs_for_generation` reads "cache is not None" as a cached step
and keeps only the last input token (29871). `forward` takes its
cached-generation branch. The image and the prompt never reach Llama.

## Signature

- No error. Each request returns 7 tokens and a valid `[1, 1, 7]` action.
- The action does not depend on the frame or the task: the model reads
  token 29871 alone at step 0.
- The KV cache holds 1 position after step 0. A healthy step 0 fills
  about 280 (prompt plus 256 image tokens).

Traced on transformers 5.20.2, CPU, a random-weight checkpoint (2026-10-29).

## Detection

- Two observations, same task, different frames: the `/kot/score` decode
  rows are identical. Healthy rows differ.
- Compare with the healthy deployment `cuda-deployments/vla-openvla-tf`.

## Why it matters

A golden captured this way is wrong for every observation. Every correct
candidate then fails against it.
