FROM rapids-singlecell-deps

ARG GIT_ID=main

SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

ENV PATH=/opt/conda/bin:$PATH
# Point CMake's find_package(rmm) at the conda env. The conda RAPIDS env resolved
# librmm + cuda-version together, so its librmm/rapids_logger headers match the
# image's CUDA toolkit. This is what lets the --no-build-isolation build below
# pick up the CUDA-matched librmm instead of a mismatched PyPI wheel.
ENV CMAKE_PREFIX_PATH=/opt/conda
ARG CUDA_ARCHS="75-real;80-real;86-real;89-real;90-real;100-real;120"

RUN <<EOF
# install rapids_singlecell from source (compiled for all supported GPU architectures)
set -x
mkdir /src
cd /src
git clone https://github.com/scverse/rapids_singlecell.git
cd rapids_singlecell
git checkout ${GIT_ID}
# Set CUDA architectures directly in pyproject.toml (avoids SKBUILD_CMAKE_ARGS semicolon splitting)
sed -i 's/CMAKE_CUDA_ARCHITECTURES = "native"/CMAKE_CUDA_ARCHITECTURES = "'"${CUDA_ARCHS}"'"/' pyproject.toml
grep CMAKE_CUDA_ARCHITECTURES pyproject.toml
# Build with --no-build-isolation so the compile uses the conda env's
# CUDA-matched librmm/rapids_logger headers. With isolation, PEP 517 would pull
# a fresh librmm-cu12 from PyPI (hardcoded in [build-system].requires) that
# mismatches the image's CUDA toolkit -> "cudaDevAttr* has no global scope"
# errors on both cu12 (toolkit older than the latest librmm) and cu13 (wrong
# cu12 variant). Install the PEP 517 backend deps first since isolation is off;
# the conda env already provides the librmm/rapids_logger headers + cmake config.
/opt/conda/bin/python -m pip install --no-cache-dir scikit-build-core nanobind setuptools-scm cmake ninja
/opt/conda/bin/python -m pip install --no-cache-dir --no-build-isolation -e .
EOF
