# The native backend is independent of the CUDA major used by CuPy/RAPIDS.
# Build it with the pinned CUDA 13.0 toolchain, then install only its wheel.
ARG CUDA_BUILD_VERSION=13.0.0
FROM nvidia/cuda:${CUDA_BUILD_VERSION}-devel-ubuntu24.04 AS rust-build
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ARG RSC_VERSION=0.0.0
ARG RSC_RUST_CUDA_ARCH=sm_75
ENV CUDA_TOOLKIT_PATH=/usr/local/cuda
ENV PATH=/opt/build-venv/bin:/root/.cargo/bin:${PATH}
ENV SETUPTOOLS_SCM_PRETEND_VERSION=${RSC_VERSION}
ENV SKBUILD_CMAKE_DEFINE_RSC_RUST_CUDA_ARCH=${RSC_RUST_CUDA_ARCH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential clang libclang-dev curl git python3 python3-venv \
    && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv /opt/build-venv
COPY --from=source . /src/rapids_singlecell
WORKDIR /src/rapids_singlecell
RUN bash scripts/install_cuda_oxide.sh \
    && python -m pip wheel --no-cache-dir --no-deps . --wheel-dir /wheelhouse

FROM rapids-singlecell-deps
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV PATH=/opt/conda/bin:${PATH}
COPY --from=rust-build /wheelhouse /tmp/rsc-wheelhouse
RUN python -m pip install --no-cache-dir --no-deps /tmp/rsc-wheelhouse/*.whl \
    && rm -rf /tmp/rsc-wheelhouse
