# The stack the ascend-910b platform row pins, as one self-contained image for
# this deployment:
#
#   # fetch the big wheels first: pypi stalls mid-download on this network,
#   # curl -C - survives it (see NOTES.md, "image build needs a wheelhouse")
#   docker build -t kot-llm-ascend-910b:latest .
#
# Same base image and pins as kot's own graded image
# (containers/Dockerfile + platform-ascend-910b.txt + family-llm.txt
# + harness-mlc_endpoints.txt), so the golden capture taken through the
# host-mode docker wrap and a candidate capture in the graded container run
# the identical stack. The tag matches what kot.gather's container mode
# inspects, so one image serves both paths. The image carries NO serving
# engine: sglang and its NPU kernels ride in the deployment's wheels/
# directory and are installed at start (kot_serve.sh), per the contract.
FROM quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12

COPY wheels/ /wheels/

# platform-ascend-910b.txt: torch-npu 2.10.0.post2 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
RUN pip install --no-cache-dir --break-system-packages \
    /wheels/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torch_npu-2.10.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchvision-0.25.0-cp312-cp312-manylinux_2_28_aarch64.whl

# family-llm.txt + the versions this deployment was verified with. The
# harness's own dependency graph pins transformers==5.16.1, which overrides a
# newer line here -- so the pin states exactly what the image ends up with.
RUN pip install --no-cache-dir --break-system-packages \
    transformers==5.16.1 \
    accelerate \
    safetensors \
    numpy==2.2.6

# harness-mlc_endpoints.txt: the MLCommons Inference Endpoints benchmark, a
# client of /v1/chat/completions inside the same container. It needs git to
# install and a C compiler for hdrhistogram.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git build-essential unzip \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --break-system-packages \
    "inference-endpoint @ git+https://github.com/mlcommons/endpoints.git"
