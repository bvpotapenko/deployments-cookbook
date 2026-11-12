# The stack the ascend-910b platform row + the ocr family row pin, as one
# self-contained image for this deployment:
#
#   # wheels/ holds the three big platform wheels; copy them once from an
#   # existing NPU deployment's wheelhouse on the box:
#   #   rsync -a ~/kot_deployments/npu-deployments/llm-vllm/wheels/ \
#   #            ~/kot_deployments/npu-deployments/ocr-transformers/wheels/
#   # (or fetch them fresh; the big pypi wheels stall on the box network)
#   docker build -t kot-mllm-ascend-910b:latest .
#
# Same base image and pins as kot's own graded recipe
# (containers/Dockerfile + platform-ascend-910b.txt + family-ocr.txt):
# transformers 5.22.1 ships the native deepseek_ocr2 architecture, and the
# ascend row pairs torch-npu 2.10.0.post2 with the CUDA-free torch 2.10.0 and
# torchvision 0.25.0. The tag matches kot.container.image_name, so the
# host-mode docker wrap and a container-mode capture can share one image.
FROM quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12-devel

COPY wheels/ /wheels/

# platform-ascend-910b.txt: torch-npu 2.10.0.post2 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
RUN pip install --no-cache-dir --break-system-packages \
    /wheels/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torch_npu-2.10.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchvision-0.25.0-cp312-cp312-manylinux_2_28_aarch64.whl

# family-mllm stack: transformers 5.22.1 and the Qwen3.8-27B video path.
RUN pip install --no-cache-dir --break-system-packages \
    transformers==5.22.1 \
    accelerate \
    safetensors \
    einops \
    flash-linear-attention==0.2.2 \
    torchcodec==0.17.0 \
    pillow \
    numpy==2.2.6 \
    pydantic \
    pyyaml

# The MLLM deployment serves /v1/chat/completions, so the endpoints harness
# is part of the image and the transformer server participates in speed runs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages \
    "inference-endpoint @ git+https://github.com/mlcommons/endpoints.git"

ENV PYTHONPATH=/workspace/kot
WORKDIR /workspace
