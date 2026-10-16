# mllm-transformers

Qwen3.8-27B (`qwen3_5`, `Qwen3_5ForConditionalGeneration`) as the first
multimodal-LLM target: text, image, and video units through the transformers
reference path, `device_map=auto` over both 5090s.

## The pinned media pipeline

A candidate capture MUST reproduce this exact preprocessing or its speed and
correctness numbers compare nothing. Everything here is also reported live
through `/kot/info`, so every golden's `info.yaml` records what ran:

| knob | value | source |
|---|---|---|
| video frame sampling | `fps=2.0`, `do_sample_frames=True` | the Qwen cookbook default |
| video pixel budget | longest_edge 25165824, shortest_edge 4096 | the checkpoint's `video_preprocessor_config.json` |
| image pixel budget | longest_edge 16777216, shortest_edge 65536 | the checkpoint's `preprocessor_config.json` |
| patch geometry | patch 16, temporal_patch 2, merge 2 | the checkpoint's configs |
| instruct mode | `enable_thinking=False` in the chat-template rendering | the non-thinking mode; its sampling (temp 0.7, top_p 0.80, top_k 20, min_p 0.0, presence_penalty 1.5, repetition_penalty 1.0) rides the store's sampling yaml per request |

The venv: `/root/kotvenv-mllm` - torch 2.8.0+cu128, transformers 5.12.1 (the
line that carries `qwen3_5`), torchvision 0.23.0.

## Status

The boot, the pinned-parameter report, and the text score path work. The
image and video score paths land with the mllm family contract (the input
freeze + the metrics design, in progress); the frame extraction at the pinned
fps and the placeholder expansion follow the OCR family's frozen-ids pattern:
the freeze renders the chat template once, the request carries the ids plus
the media file, a replay prefill stores only the rows over the generated ids.

Cookbooks consulted:
- https://recipes.vllm.ai/Qwen/Qwen3.8-27B
- https://docs.sglang.io/cookbook/autoregressive/Qwen/Qwen3.8-27B
(their launch flags are single-GPU; this box runs 2x RTX 5090, TP-2 territory)
