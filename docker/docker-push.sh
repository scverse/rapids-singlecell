#!/bin/bash
set -euxo pipefail

docker_account=scverse
rapids_version=26.08

declare -A cuda_versions=(
    [cu12]="12.9.1"
    [cu13]="13.3.0"
)

declare -A cuda_archs=(
    [cu12]="75-real;80-real;86-real;89-real;90"
    [cu13]="75-real;80-real;86-real;89-real;90-real;100-real;120"
)

declare -A conda_labels=(
    [cu12]="cuda12"
    [cu13]="cuda13"
)

for pkg in cu12 cu13; do
    ver=${cuda_versions[$pkg]}

    grep -v -- '- rapids-singlecell' conda/rsc_rapids_${rapids_version}_${conda_labels[$pkg]}.yml > docker/rsc_rapids.yml
    docker build \
        --build-arg CUDA_VER="${ver}" \
        -t rapids-singlecell-deps-${pkg}:latest \
        -f docker/Dockerfile.deps \
        docker/
    rm docker/rsc_rapids.yml

    docker build \
        --build-arg CUDA_ARCHS="${cuda_archs[$pkg]}" \
        --build-context rapids-singlecell-deps=docker-image://rapids-singlecell-deps-${pkg}:latest \
        -t rapids-singlecell-${pkg}:latest \
        -f docker/Dockerfile \
        docker/
done

docker image ls -a
