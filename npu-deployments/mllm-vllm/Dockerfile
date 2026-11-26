# Build from this deployment directory after fetching the wheels.
#   docker build -t serve-mllm-ascend-910b:latest .
FROM quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12-devel

COPY wheels/ /wheels/

RUN pip install --no-cache-dir --break-system-packages \
    "numpy>=2.0" \
    "pydantic>=2.7" \
    "pyyaml>=6.0"

# torch-npu 2.10.0.post7 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
# The engine venv uses the same torch_npu line; see wheels-requirements.txt.
RUN pip install --no-cache-dir --break-system-packages \
    /wheels/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torch_npu-2.10.0.post7-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchvision-0.25.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchaudio-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl

# TorchCodec 0.10 imports torch during setup, so build it inside this image
# after the matching NPU torch is installed. The host wheel builder only
# fetches and verifies the source archive.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      cmake ninja-build pkg-config ffmpeg \
      libavcodec-dev libavdevice-dev libavfilter-dev libavformat-dev \
      libavutil-dev libswresample-dev libswscale-dev \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages \
      "setuptools>=61" wheel pybind11
RUN CMAKE_PREFIX_PATH="$(python -m pybind11 --cmakedir):${CMAKE_PREFIX_PATH:-}" \
    BUILD_VERSION=0.10.0 TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
    I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
    ENABLE_CUDA=OFF TORCHCODEC_BUILD_IMAGE=ON \
    pip wheel /wheels/torchcodec-0.10.0.tar.gz --no-deps \
      --no-build-isolation --no-cache-dir -w /wheels \
    && pip install --no-cache-dir --break-system-packages \
      --no-index --find-links /wheels "torchcodec==0.10.0"

RUN pip install --no-cache-dir --break-system-packages \
    transformers==5.14.1 \
    accelerate \
    safetensors \
    numpy==2.2.6

# Install the MLCommons Inference Endpoints client.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages \
    "inference-endpoint @ git+https://github.com/mlcommons/endpoints.git"

# The harness's hardcoded startup timeouts are too small for a loaded shared
# host (spawn-mode services re-import the whole stack): service ready-check
# 30s and tokenizer-shard warmup 25s both expire before the client starts.
# Baked here so every fresh container inherits the working values.
RUN python3 -c "from pathlib import Path; import inference_endpoint; root=Path(inference_endpoint.__file__).parent; p=root/'config'/'schema.py'; s=p.read_text(); old='service_ready_timeout_s: float = Field(\n        30.0,'; new='service_ready_timeout_s: float = Field(\n        180.0,'; assert old in s, f'{p} does not contain the expected service timeout'; p.write_text(s.replace(old, new, 1)); p=root/'async_utils'/'services'/'metrics_aggregator'/'token_metrics.py'; s=p.read_text(); old='_SHARD_WARMUP_TIMEOUT_S = 25.0'; new='_SHARD_WARMUP_TIMEOUT_S = 120.0'; assert old in s, f'{p} does not contain the expected shard timeout'; p.write_text(s.replace(old, new, 1))"
