Placeholder for a future implementation of the vLLM-Omni engine. Target setup: 2x RTX 5090 CUDA server.
Engine: vLLM-Omni 0.32.0, with vLLM 0.32.0 in its own venv. vLLM-Omni 0.32.0 does not support vLLM 0.33.0.
Target models:
- lerobot/pi05_base, revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0
- lerobot/pi0_base, revision 6e245399ad403ccf02b8112b73cfdc2633068f38
Before a run: the checkpoints hold no tokenizer. Put the tokenizer files of google/paligemma-3b-pt-224 (gated) beside the weights, because grading runs offline.
