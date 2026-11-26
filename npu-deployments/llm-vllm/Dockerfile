# Build from this deployment directory after fetching the wheels.
#   docker build -t serve-llm-ascend-910b:latest .
FROM quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12

COPY wheels/ /wheels/

RUN pip install --no-cache-dir --break-system-packages \
    "numpy>=2.0" \
    "pydantic>=2.7" \
    "pyyaml>=6.0"

# torch-npu 2.10.0.post2 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
# The engine venv overlays its own torch_npu (post4) on these at first
# start; see wheels-requirements.txt for the two layers.
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
    && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages \
    "inference-endpoint @ git+https://github.com/mlcommons/endpoints.git"

# The harness's hardcoded startup timeouts are too small for a loaded shared
# host (spawn-mode services re-import the whole stack): service ready-check
# 30s and tokenizer-shard warmup 25s both expire before the client starts.
# Baked here so every fresh container inherits the working values.
RUN python3 -c "p='/usr/local/python3.12.13/lib/python3.12/site-packages/inference_endpoint/config/schema.py'; s=open(p).read(); s=s.replace('service_ready_timeout_s: float = Field(\n        30.0,', 'service_ready_timeout_s: float = Field(\n        180.0,', 1); open(p,'w').write(s)" \
 && python3 -c "p='/usr/local/python3.12.13/lib/python3.12/site-packages/inference_endpoint/async_utils/services/metrics_aggregator/token_metrics.py'; s=open(p).read(); s=s.replace('_SHARD_WARMUP_TIMEOUT_S = 25.0', '_SHARD_WARMUP_TIMEOUT_S = 120.0', 1); open(p,'w').write(s)"
