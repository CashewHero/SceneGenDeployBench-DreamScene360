# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:$PATH \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9+PTX" \
    TCNN_CUDA_ARCHITECTURES="75;80;86;89" \
    MAX_JOBS=2
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev python3-venv git build-essential ninja-build ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m venv /opt/runner
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
RUN uv pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
COPY runner_wrapper/requirements.txt /tmp/requirements.txt
RUN uv pip install -r /tmp/requirements.txt

# DreamScene360 uses only the hash-grid encoding, not tiny-cuda-nn's MLPs.
ARG TCNN_REVISION=3c7931caab7b691c6f30dffe20952ed727d935b9
RUN git clone https://github.com/NVlabs/tiny-cuda-nn.git /tmp/tiny-cuda-nn \
    && git -C /tmp/tiny-cuda-nn checkout "$TCNN_REVISION" \
    && git -C /tmp/tiny-cuda-nn submodule update --init --recursive
WORKDIR /tmp/tiny-cuda-nn/bindings/torch
RUN python setup.py --no-networks bdist_wheel && uv pip install dist/*.whl

ARG DINO_REVISION=e1277af2ba9496fbadf7aec6eba56e8d882d1e35
RUN git clone https://github.com/facebookresearch/dinov2.git /opt/dinov2 \
    && git -C /opt/dinov2 checkout "$DINO_REVISION" \
    && rm -rf /opt/dinov2/.git
COPY submodules /tmp/submodules
RUN uv pip install --no-build-isolation /tmp/submodules/diff-gaussian-rasterization-depth /tmp/submodules/simple-knn

FROM nvidia/cuda:12.4.1-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 libpython3.10 libgl1 libglib2.0-0 libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/runner /opt/runner
COPY --from=builder /opt/dinov2 /opt/dinov2
COPY --from=builder /tmp/tiny-cuda-nn/LICENSE.txt /usr/share/licenses/tiny-cuda-nn/LICENSE.txt
WORKDIR /app/model_repo
COPY arguments arguments
COPY fields fields
COPY gaussian_renderer gaussian_renderer
COPY geo_predictors geo_predictors
COPY scene scene
COPY utils utils
COPY train.py LICENSE.md ./
COPY runner_wrapper runner_wrapper
ENV VIRTUAL_ENV=/opt/runner \
    PATH=/opt/runner/bin:$PATH \
    LD_LIBRARY_PATH=/opt/runner/lib/python3.10/site-packages/nvidia/cuda_nvrtc/lib:$LD_LIBRARY_PATH \
    PYTHONPATH=/app/model_repo \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=4 \
    RUNNER_PORT=58090 \
    RUNNER_NAME=dreamscene360 \
    RUNNER_TYPE=generator \
    RUNNER_VERSION=0.1.0 \
    RUNNER_CONTRACT_VERSION=1 \
    RUNNER_IDLE_TIMEOUT_SECONDS=900 \
    RUNNER_ADAPTER=runner_wrapper.adapter:run_job \
    DREAMSCENE360_DINO_SOURCE=/opt/dinov2
RUN python -c "from diff_gaussian_rasterization import GaussianRasterizer; from simple_knn._C import distCUDA2; import runner_wrapper.adapter"
EXPOSE 58090
CMD ["python", "-m", "runner_wrapper.server"]
