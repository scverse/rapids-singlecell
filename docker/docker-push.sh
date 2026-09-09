#!/bin/bash
set -euxo pipefail

docker_account=scverse
rapids_version=26.08

declare -A cuda_versions=(
    [cu12]="12.9.1"
    [cu13]="13.3.0"
)

declare -A conda_labels=(
    [cu12]="cuda12"
    [cu13]="cuda13"
)

for pkg in cu12 cu13; do
    ver=${cuda_versions[$pkg]}

    grep -v -- '- rapids-singlecell' conda/rsc_rapids_${rapids_version}_${conda_labels[$pkg]}.yml > docker/rsc_rapids.yml
    docker build \
        --build-context source=. \
        --build-arg CUDA_VER="${ver}" \
        -t rapids-singlecell-deps-${pkg}:latest \
        -f docker/Dockerfile.deps \
        docker/
    rm docker/rsc_rapids.yml

    docker build \
        --build-context source=. \
        --build-arg RSC_RUST_CUDA_ARCH=sm_75 \
        --build-context rapids-singlecell-deps=docker-image://rapids-singlecell-deps-${pkg}:latest \
        -t rapids-singlecell-${pkg}:latest \
        -f docker/Dockerfile \
        docker/
done

docker image ls -a
