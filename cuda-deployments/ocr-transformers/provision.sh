#!/usr/bin/env bash
# provision.sh -- set up the OCR venv on a fresh cuda box, nothing else.
#
#   ./provision.sh [python-for-uv]
#
# Creates /root/kotvenv-ocr with the stack the checkpoint's remote code needs:
# transformers is PINNED to 4.46.3 (the DeepSeek-OCR-2 remote code breaks on
# 5.x) beside a Blackwell-capable torch. kot_serve.sh finds the venv by this
# exact path. The model itself is not downloaded here: on the box run
#   HF_HOME=/workspace/hf_ocr hf download deepseek-ai/DeepSeek-OCR-2
# or upload a local copy; either way pass the directory to kot_serve.sh --model.
set -euo pipefail

PYTHON="${1:-python3}"
VENV=/root/kotvenv-ocr

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

uv venv "$VENV" --python "$PYTHON"
uv pip install --python "$VENV/bin/python" \
  torch==2.8.0 torchvision==0.23.0 \
  --index-url https://download.pytorch.org/whl/cu128
# The pytorch CDN stalls on some networks (UV_HTTP_TIMEOUT mid-wheel); PyPI
# carries the same cu128 build if it does.
uv pip install --python "$VENV/bin/python" \
  "transformers==4.46.3" "numpy<2.2" accelerate safetensors \
  addict einops pillow requests tqdm "huggingface_hub[hf_transfer]"

"$VENV/bin/python" - <<'EOF'
import torch, transformers
assert transformers.__version__.startswith("4.46."), transformers.__version__
print("provisioned:", "torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda:", torch.cuda.is_available())
EOF
echo "next: HF_HOME=/workspace/hf_ocr $VENV/bin/hf download deepseek-ai/DeepSeek-OCR-2"
