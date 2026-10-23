#!/usr/bin/env bash
# provision.sh -- set up the OCR venv on a fresh cuda box, nothing else.
#
#   ./provision.sh [python-for-uv]
#
# Creates /root/kotvenv-ocr with the native deepseek_ocr2 stack: transformers
# is PINNED to 5.20.2 (the version that ships the architecture beside a
# Blackwell-capable torch). kot_serve.sh finds the venv by this exact path.
# The model itself is not downloaded here: on the box run
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
# The pytorch CDN stalls on some networks (UV_HTTP_TIMEOUT mid-wheel); PyPI
# carries the same cu13x build if it does. The torch build must carry sm_120.
uv pip install --python "$VENV/bin/python" "torch==2.14.1"
uv pip install --python "$VENV/bin/python" \
  "transformers==5.20.2" numpy safetensors pillow requests tqdm \
  "huggingface_hub[hf_transfer]"

"$VENV/bin/python" - <<'EOF'
import torch, transformers
from transformers import DeepseekOcr2ForConditionalGeneration
assert transformers.__version__.startswith("5.20."), transformers.__version__
print("provisioned:", "torch", torch.__version__, "| transformers", transformers.__version__,
      "| cuda:", torch.cuda.is_available())
EOF
echo "next: HF_HOME=/workspace/hf_ocr $VENV/bin/hf download deepseek-ai/DeepSeek-OCR-2"
