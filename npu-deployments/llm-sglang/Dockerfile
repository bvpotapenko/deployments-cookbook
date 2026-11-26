# Build from this deployment directory after fetching the wheels.
#   docker build -t serve-llm-ascend-910b:latest .
FROM quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12

COPY wheels/ /wheels/

# torch-npu 2.10.0.post2 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
RUN pip install --no-cache-dir --break-system-packages \
    /wheels/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torch_npu-2.10.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchvision-0.25.0-cp312-cp312-manylinux_2_28_aarch64.whl

RUN pip install --no-cache-dir --break-system-packages \
    transformers==5.16.1 \
    accelerate \
    safetensors \
    numpy==2.2.6

# Install the MLCommons Inference Endpoints client.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential unzip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages \
    "inference-endpoint @ git+https://github.com/mlcommons/endpoints.git"
