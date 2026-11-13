# The stack the ascend-910b platform row pins, as one self-contained image for
# this deployment. The serving engine (vllm + vllm-ascend) is NOT baked in:
# it lives in this directory's wheels/ and is installed offline into /scratch
# by serve.sh at first start, because the graded image is "the platform
# torch and the family model stack, and no serving engine".
#
#   # fetch the big wheels first: pypi stalls mid-download on this network,
#   # curl -C - survives it (see NOTES.md, "image build needs a wheelhouse")
#   docker build -t kot-mllm-ascend-910b:latest .
#
# Same base image and pins as kot's own graded image
# (containers/Dockerfile + platform-ascend-910b.txt + family-llm.txt
# + harness-mlc_endpoints.txt), so the golden capture taken through the
# host-mode docker wrap and a candidate capture in the graded container run
# the identical stack. The tag matches what kot.gather's container mode
# inspects, so one image serves both paths.
FROM quay.io/ascend/cann:9.2.1-910b-ubuntu22.04-py3.12-devel

COPY wheels/ /wheels/

# kot's containers/Dockerfile base layer (base.txt): the kot runner itself
# needs these inside the graded container.
RUN pip install --no-cache-dir --break-system-packages \
    "numpy>=2.0" \
    "pydantic>=2.7" \
    "pyyaml>=6.0"

# platform-ascend-910b.txt: torch-npu 2.10.0.post7 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
# The engine venv uses the same torch_npu line; see wheels-requirements.txt.
RUN pip install --no-cache-dir --break-system-packages \
    /wheels/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torch_npu-2.10.0.post7-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchvision-0.25.0-cp312-cp312-manylinux_2_28_aarch64.whl

# TorchCodec is part of the video ABI and must come from the same offline
# wheelhouse; the base image is not an installation source.
RUN pip install --no-cache-dir --break-system-packages \
    --no-index --find-links /wheels \
    "torchcodec==0.16.0"

# family-llm.txt + the versions this deployment was verified with. The
# harness's own dependency graph pins transformers==5.16.1, which overrides a
# newer line here -- so the pin states exactly what the image ends up with.
RUN pip install --no-cache-dir --break-system-packages \
    transformers==5.14.1 \
    accelerate \
    safetensors \
    numpy==2.2.6

# harness-mlc_endpoints.txt: the MLCommons Inference Endpoints benchmark, a
# client of /v1/chat/completions inside the same container. It needs git to
# install and a C compiler for hdrhistogram.
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

# kot's containers/Dockerfile sets this: container mode runs the kot runner
# from the mounted repo (~/kot-current/kot -> /workspace/kot/kot). Kept LAST
# so the fetched layers above stay cached across edits of this line.
ENV PYTHONPATH=/workspace/kot
