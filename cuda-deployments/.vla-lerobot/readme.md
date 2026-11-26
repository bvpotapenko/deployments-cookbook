Placeholder for a future implementation of the LeRobot engine. Target setup: 2x RTX 5090 CUDA server.
Engine: LeRobot 0.6.1.
Target models:
- lerobot/pi05_base, revision 22c30e5b34f9e7ab738b1b34fc2fc5ecfe2d28f0
- lerobot/pi0_base, revision 6e245399ad403ccf02b8112b73cfdc2633068f38
Before a run: the checkpoints hold no tokenizer. Put the tokenizer files of google/paligemma-3b-pt-224 (gated) beside the weights for offline inference.
lerobot/pi0_base holds no normalization statistics, so its actions stay in model units over 32 dimensions.
Install `lerobot[pi]==0.6.1`. The [pi] extra needs transformers <5.6, so this engine needs its own venv.
