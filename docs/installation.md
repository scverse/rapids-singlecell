# Installation
## Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* files provided in the [conda](https://github.com/scverse/rapids_singlecell/tree/main/conda) folder.
These *yaml* files install everything needed to run the example notebooks and get you started.

`````{tab-set}
````{tab-item} CUDA 13
```bash
conda env create -f conda/rsc_rapids_26.04_cuda13.yml
# or
mamba env create -f conda/rsc_rapids_26.04_cuda13.yml
```
*Python 3.14, CUDA 13.1*
````
````{tab-item} CUDA 12
```bash
conda env create -f conda/rsc_rapids_26.04_cuda12.yml
# or
mamba env create -f conda/rsc_rapids_26.04_cuda12.yml
```
*Python 3.14, CUDA 12.9*
````
`````

```{note}
RAPIDS currently doesn't support `channel_priority: strict`; use `channel_priority: flexible` instead
```

## PyPI

Starting with version 0.15.0, *rapids-singlecell* ships precompiled CUDA kernels via nanobind.
Prebuilt wheels are available for **x86_64** and **aarch64** Linux for both CUDA 12 and CUDA 13.

### CUDA version compatibility

The prebuilt wheels support the following CUDA runtime versions:

| Wheel | Compiled with | Runtime support | GPU architectures |
|---|---|---|---|
| `rapids-singlecell` | Source distribution | Any supported CUDA | Compiles for your local GPU architecture |
| `rapids-singlecell-cu12` | CUDA 12.2 | CUDA 12.2–12.9+ | Turing through Hopper (native), Blackwell (via PTX JIT) |
| `rapids-singlecell-cu13` | CUDA 13.0 | CUDA 13.0+ | Turing through Blackwell (all native) |

The CUDA 12 wheels are compiled with CUDA 12.2 to match the [RAPIDS 26.04 support matrix](https://docs.rapids.ai/platform-support/) (CUDA 12.2–12.9).
Blackwell GPUs (CC 100, 120) are supported via PTX just-in-time compilation from the `sm_90` PTX included in the wheel.
The CUDA 13 wheels include native Blackwell binaries, so no JIT is needed.

### Prebuilt wheels (recommended)

Install the wheel matching your CUDA version:

`````{tab-set}
````{tab-item} CUDA 13
```bash
pip install rapids-singlecell-cu13
```
````
````{tab-item} CUDA 12
```bash
pip install rapids-singlecell-cu12
```
````
`````

This installs the precompiled CUDA kernels but **not** the RAPIDS stack (cupy, cuml, cudf, etc.).
This is the recommended approach for **conda/mamba users** who already have RAPIDS installed in their environment.

### Prebuilt wheels with RAPIDS dependencies

To also install the RAPIDS stack via pip, use the `rapids` extra.
This requires the `--extra-index-url` flag for the NVIDIA PyPI index:

`````{tab-set}
````{tab-item} CUDA 13
```bash
pip install 'rapids-singlecell-cu13[rapids]' --extra-index-url=https://pypi.nvidia.com
```
````
````{tab-item} CUDA 12
```bash
pip install 'rapids-singlecell-cu12[rapids]' --extra-index-url=https://pypi.nvidia.com
```
````
`````

### Source distribution (self-compile)

The `rapids-singlecell` package on PyPI contains the source distribution.
Building from source requires a CUDA toolkit and a C++ compiler:

```bash
pip install rapids-singlecell
```

The CUDA kernels will be compiled during installation for your local GPU architecture.
You can select RAPIDS dependencies with the `rapids-cu12` or `rapids-cu13` extras:

```bash
pip install 'rapids-singlecell[rapids-cu12]' --extra-index-url=https://pypi.nvidia.com
```

```{note}
Building from source requires the CUDA toolkit (nvcc) and CMake >= 3.24 to be available in your environment.
The nvcc/CUDAToolkit found during the build should match the RAPIDS/CuPy CUDA major runtime version in or linked to the environment.
```

### Install from GitHub

To install the latest development version directly from GitHub:

```bash
pip install "rapids-singlecell @ git+https://github.com/scverse/rapids_singlecell.git"
```

Or from a specific branch or tag:

```bash
pip install "rapids-singlecell @ git+https://github.com/scverse/rapids_singlecell.git@main"
```

This compiles the CUDA kernels during installation. By default, kernels are compiled for your local GPU architecture only (`native`).
To compile for different or multiple architectures, pass a config setting to override the CUDA architectures:

```bash
# Compile for a specific architecture (e.g., Ampere)
pip install -C cmake.define.CMAKE_CUDA_ARCHITECTURES="80-real" "rapids-singlecell @ git+https://github.com/scverse/rapids_singlecell.git"

# Compile for multiple architectures
pip install -C cmake.define.CMAKE_CUDA_ARCHITECTURES="80-real;86-real;89-real;90-real" "rapids-singlecell @ git+https://github.com/scverse/rapids_singlecell.git"
```

Common architecture codes:

| Code | GPU Generation | Examples |
|---|---|---|
| `75` | Turing | T4, RTX 2080 |
| `80` | Ampere | A100, A30 |
| `86` | Ampere | A10, RTX 3090 |
| `89` | Ada Lovelace | L4, L40, RTX 4090 |
| `90` | Hopper | H100, H200 |
| `100` | Blackwell | B200, GB200 |
| `120` | Blackwell | B300, RTX PRO 6000 |

```{tip}
Use `native` (the default) for the fastest compilation when you only need to run on your local GPU.
Use multiple architectures when building portable binaries (e.g., for a shared cluster with mixed GPU types).
The `-real` suffix generates device code only (no PTX fallback), which reduces binary size.
```

## Docker

We also offer Docker containers for `rapids-singlecell`. These containers include all the necessary dependencies, making it even easier to get started with `rapids-singlecell`.

To use the Docker container, first, ensure that you have Docker installed on your system and that Docker supports the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html).
Then, pull the Docker image matching your CUDA version:

`````{tab-set}
````{tab-item} CUDA 13
```bash
docker pull ghcr.io/scverse/rapids-singlecell-cu13:latest
```
````
````{tab-item} CUDA 12
```bash
docker pull ghcr.io/scverse/rapids-singlecell-cu12:latest
```
````
`````

To run the Docker container, use the following command:

`````{tab-set}
````{tab-item} CUDA 13
```bash
docker run --rm --gpus all ghcr.io/scverse/rapids-singlecell-cu13:latest
```
````
````{tab-item} CUDA 12
```bash
docker run --rm --gpus all ghcr.io/scverse/rapids-singlecell-cu12:latest
```
````
`````

The docker containers also work with apptainer (or singularity) on an HPC system.

First pull the container and wrap it in a `.sif` file:
`````{tab-set}
````{tab-item} CUDA 13
```bash
apptainer pull rsc.sif docker://ghcr.io/scverse/rapids-singlecell-cu13:latest
```
````
````{tab-item} CUDA 12
```bash
apptainer pull rsc.sif docker://ghcr.io/scverse/rapids-singlecell-cu12:latest
```
````
`````

Then run the following command to execute the container:
```bash
apptainer run --nv rsc.sif
```
### Running on HPC systems with SLURM

When running on HPC systems via SLURM, conda must be explicitly activated before running Python scripts. Use `apptainer exec` instead of `apptainer run`:

```bash
apptainer exec --nv \
    --bind /path/to/your/data:/path/to/your/data \
    rsc.sif \
    bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate base && python"
```
Without sourcing conda first, `CONDA_PREFIX` will be unset and CuPy will fail to locate the CUDA libraries inside the container, resulting in a `TypeError: expected str, bytes or os.PathLike object, not NoneType` error.

# System requirements

Most computations run on the GPU.
See the Memory Management page for hardware guidance, managed memory, and known limits:

- {doc}`memory_management`
