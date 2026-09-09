# Installation
## Conda
The easiest way to install *rapids-singlecell* is to use one of the *yaml* files provided in the [conda](https://github.com/scverse/rapids-singlecell/tree/main/conda) folder.
These *yaml* files install everything needed to run the example notebooks and get you started.

`````{tab-set}
````{tab-item} CUDA 13
```bash
conda env create -f conda/rsc_rapids_26.08_cuda13.yml
# or
mamba env create -f conda/rsc_rapids_26.08_cuda13.yml
```
*Python 3.14, CUDA 13.3*
````
````{tab-item} CUDA 12
```bash
conda env create -f conda/rsc_rapids_26.08_cuda12.yml
# or
mamba env create -f conda/rsc_rapids_26.08_cuda12.yml
```
*Python 3.14, CUDA 12.9*
````
`````

```{note}
RAPIDS currently doesn't support `channel_priority: strict`; use `channel_priority: flexible` instead
```

## PyPI

The native GPU backend uses Rust/cuda-oxide kernels and PyO3 bindings.
Prebuilt wheels are available for **x86_64** and **aarch64** Linux, with matching
CUDA 12 or CUDA 13 RAPIDS dependency extras.

### CUDA version compatibility

| Distribution | Native build toolkit | RAPIDS/CuPy dependencies | Minimum GPU/driver |
|---|---|---|---|
| `rapids-singlecell` | Source build with CUDA 13.0+ | CUDA 12 or CUDA 13 | Turing (`sm_75`), compatible driver |
| `rapids-singlecell-cu12` | CUDA 13.0 | CUDA 12 | Turing or newer, NVIDIA R580+ |
| `rapids-singlecell-cu13` | CUDA 13.0 | CUDA 13 | Turing or newer, NVIDIA R580+ |

Both wheel variants contain portable PTX, compiled for Turing and JIT-compiled by
the NVIDIA driver for the active GPU. The `cu12`/`cu13` suffix selects the Python
GPU dependency family. CUDA 12 libraries work with newer drivers through
[NVIDIA's backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).
The Rust backend requires **R580 or newer even when using CUDA 12 dependencies**.
A source build using a toolkit newer than 13.0 may require a newer driver for its
emitted PTX version.

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

```{note}
The RAPIDS stack is **required**, not optional: `rapids_singlecell` imports
`cuml`/`cupy` at the top of its package `__init__`. These are
provided by an existing RAPIDS conda/mamba environment or by the
`[rapids]`/`[rapids-cuXX]` extra below. Installing the bare
`rapids-singlecell-cuXX` wheel into an environment without RAPIDS raises an
`ImportError` on `import rapids_singlecell` itself — not merely when a kernel is
first used.
```

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

### Source distribution and development installs

Source builds require CUDA Toolkit 13.0+, Clang/libclang, and the pinned Rust and
cuda-oxide compiler. Install the compiler before building the package; the
[Rust backend guide](rust_backend.md) contains the full setup and validation steps.

From a checkout:

```bash
git clone https://github.com/scverse/rapids-singlecell.git
cd rapids-singlecell
bash scripts/install_cuda_oxide.sh
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0
python -m pip install -e .
```

With the compiler installed, build the source distribution or a Git revision:

```bash
python -m pip install rapids-singlecell
python -m pip install "rapids-singlecell @ git+https://github.com/scverse/rapids-singlecell.git@main"
```

Select RAPIDS dependencies with `rapids-cu12` or `rapids-cu13`:

```bash
python -m pip install 'rapids-singlecell[rapids-cu12]' --extra-index-url=https://pypi.nvidia.com
```

Source builds default to portable `sm_75` PTX and do not need a GPU during
compilation. To require a newer GPU capability:

```bash
SKBUILD_CMAKE_DEFINE_RSC_RUST_CUDA_ARCH=sm_80 python -m pip install -e .
```

Typical targets are `sm_75` (Turing/T4), `sm_80` (Ampere/A100), `sm_89` (Ada/L4),
and `sm_90` (Hopper/H100). A target defines the minimum capability; the driver can
JIT its PTX for later compatible architectures. No C++ or CUDA C++ source is
compiled by the package build.

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
