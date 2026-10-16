# The stack the ascend-910b platform row + the ocr family row pin, as one
# self-contained image for this deployment:
#
#   # wheels/ holds the three big platform wheels; copy them once from an
#   # existing NPU deployment's wheelhouse on the box:
#   #   rsync -a ~/kot_deployments/npu-deployments/llm-vllm/wheels/ \
#   #            ~/kot_deployments/npu-deployments/ocr-transformers/wheels/
#   # (or fetch them fresh; the big pypi wheels stall on the box network)
#   docker build -t kot-ocr-ascend-910b:latest .
#
# Same base image and pins as kot's own graded recipe
# (containers/Dockerfile + platform-ascend-910b.txt + family-ocr.txt):
# the golden capture on CUDA ran transformers 4.46.3 (the DeepSeek-OCR-2
# remote code breaks on 5.x), and the ascend row pairs torch-npu 2.10.0.post2
# with the CUDA-free torch 2.10.0 and torchvision 0.25.0. The tag matches
# kot.container.image_name, so the host-mode docker wrap and a container-mode
# capture can share one image.
FROM quay.io/ascend/cann:9.1.0-beta.1-910b-ubuntu22.04-py3.12

COPY wheels/ /wheels/

# platform-ascend-910b.txt: torch-npu 2.10.0.post2 pulls the CUDA-free torch
# 2.10.0; torchvision is pinned beside it because the two must match.
RUN pip install --no-cache-dir --break-system-packages \
    /wheels/torch-2.10.0-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torch_npu-2.10.0.post2-cp312-cp312-manylinux_2_28_aarch64.whl \
    /wheels/torchvision-0.25.0-cp312-cp312-manylinux_2_28_aarch64.whl

# family-ocr.txt (transformers PINNED: the checkpoint's remote code breaks on
# 5.x) + kot's base.txt equivalents (numpy, pydantic, pyyaml), so kot's
# in-container runner also finds its own deps here. No harness layer: this
# deployment serves no chat route, so the endpoints benchmark has nothing to
# drive (the CUDA reference golden holds no speed record either).
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
