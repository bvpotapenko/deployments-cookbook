# The stack this deployment pins, as one self-contained image:
#
#   # wheels/ holds the three big platform wheels; copy them once from an
#   # existing NPU deployment's wheelhouse on the box, or fetch them fresh
#   # (the big pypi wheels stall on the box network).
#   docker build -t ocr-transformers-ascend-910b:4.46.3 .
#
# The ascend stack pairs torch-npu 2.10.0.post2 with the CUDA-free torch
# 2.10.0 and torchvision 0.25.0.
FROM quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12

COPY wheels/ /wheels/

# platform-ascend-910b.txt: torch-npu 2.10.0.post2 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
RUN pip install --no-cache-dir --break-system-packages \
    /wheels/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torch_npu-2.10.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchvision-0.25.0-cp312-cp312-manylinux_2_28_aarch64.whl

# The model stack + kot's base.txt equivalents (numpy, pydantic, pyyaml), so
# kot's in-container runner also finds its own deps here. No harness layer:
# this deployment serves no chat route, so the endpoints benchmark has nothing
# to drive.
RUN pip install --no-cache-dir --break-system-packages \
    transformers==4.46.3 \
    accelerate \
    safetensors \
    einops \
    addict \
    tqdm \
    requests \
    pillow \
    numpy==2.2.6 \
    pydantic \
    pyyaml

# kot's own graded recipe ends with these lines (containers/Dockerfile), and
# the runner's docker exec inherits the IMAGE's environment: the mounted kot
# package at /workspace/kot/kot is importable only through this PYTHONPATH.
# Without it the runner dies with "No module named 'kot.infer'".
ENV PYTHONPATH=/workspace/kot
WORKDIR /workspace
