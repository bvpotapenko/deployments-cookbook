Placeholder for a future implementation of the transformers engine. Target setup: 2x RTX 5090 CUDA server.
Engine: transformers 5.22.1.
Target models:
- lerobot/pi0_base, revision 6e245399ad403ccf02b8112b73cfdc2633068f38
openvla/openvla-7b has its own deployment: vla-openvla-tf.
Before a run: lerobot/pi0_base holds no tokenizer. Put the tokenizer files of google/paligemma-3b-pt-224 (gated) beside the weights, because grading runs offline.
