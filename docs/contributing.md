# Contributing

## Development setup

### Prerequisites

- NVIDIA GPU with CUDA support
- [micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html), conda/mamba, or [uv](https://docs.astral.sh/uv/)
- A RAPIDS environment (e.g., conda `rapids-26.08` or pip-installed RAPIDS)
- CUDA toolkit for building from source

```{note}
Building the CUDA extensions needs only nvcc and the CUDA toolkit — no RAPIDS C++
package is required. Kernels use a CuPy-backed scratch allocator instead of RMM, and
the `cub/` headers come from the toolkit's own bundled CCCL, so nothing links
`librmm` or `librapids_logger`.
```

### Clone and install

```bash
git clone --recurse-submodules https://github.com/scverse/rapids-singlecell.git
cd rapids-singlecell
(uv) pip install -e ".[test]"
```

The documentation notebooks live in a Git submodule. If the repository was
already cloned without `--recurse-submodules`, initialize them with
`git submodule update --init` before building the documentation.

The editable install compiles the CUDA kernels for your local GPU architecture.
After the install, compiled `.so` modules and `.pyi` type stubs are placed in `src/rapids_singlecell/_cuda/`.

### Pre-commit hooks

```bash
pip install pre-commit
pre-commit install
```

Run manually on all files:

```bash
pre-commit run --all-files
```

## Project structure

```
rapids_singlecell/
├── src/rapids_singlecell/       # Python source
│   ├── preprocessing/           # pp module (normalize, scale, HVG, etc.)
│   ├── tools/                   # tl module (PCA, UMAP, clustering, etc.)
│   ├── squidpy_gpu/             # spatial analysis (co_occurrence, ligrec, etc.)
│   ├── pertpy_gpu/              # perturbation analysis (edistance, etc.)
│   ├── decoupler_gpu/           # pathway analysis
│   ├── get/                     # CPU/GPU data transfer utilities
│   └── _cuda/                   # Compiled CUDA kernels (nanobind)
│       ├── nb_types.h           # Shared ndarray type aliases
│       ├── <module>/            # Each kernel module (e.g., wilcoxon/)
│       │   ├── <module>.cu      # nanobind bindings + launch wrappers
│       │   └── kernels_*.cuh    # CUDA kernel implementations
│       ├── *.abi3.so            # Compiled modules (gitignored)
│       ├── *.pyi                # Type stubs (gitignored, auto-generated)
│       └── py.typed             # PEP 561 marker (gitignored, auto-generated)
├── tests/                       # pytest test suite
├── docs/                        # Sphinx documentation
├── docker/                      # Docker and CI build images
├── conda/                       # Conda environment files
├── CMakeLists.txt               # CMake build for CUDA extensions
└── pyproject.toml               # Project metadata and build config
```

## Contributing GPU code

All contributions are welcome, regardless of the GPU programming approach you use.
You do **not** need to know C++ or nanobind to contribute GPU-accelerated functions.

We accept pull requests using any of the following:

- **Pure CuPy** (array API, `cupyx.scipy`, etc.)
- **CuPy RawKernels**
- **numba-cuda** kernels
- **nanobind/CUDA C++** extensions

Please **do not** introduce JAX or PyTorch as dependencies.
The project is built on the RAPIDS/CuPy stack and we want to keep the dependency footprint "minimal".

The most important thing is a **correct, tested implementation**.
Performance optimization and porting to nanobind C++ (if needed) can happen in follow-up PRs or directly on your branch by the maintainers.
Don't let unfamiliarity with the internal kernel system stop you from contributing — a working CuPy implementation is a great starting point.

```{tip}
When opening a pull request, please enable **"Allow edits by maintainers"** (the checkbox on the PR creation page).
This lets us make small fixes, optimizations, or nanobind ports directly on your branch without extra back-and-forth.
```

## CUDA kernel architecture (nanobind)

### Overview

GPU-accelerated functions are implemented as **nanobind** C++ extensions compiled with CUDA.
Each kernel module lives in its own subdirectory under `src/rapids_singlecell/_cuda/` and consists of:

- A `.cu` file with nanobind bindings and kernel launch wrappers
- One or more `.cuh` headers with the actual CUDA kernel implementations

The shared header `nb_types.h` provides type aliases used across all modules:

```cpp
cuda_array<T>                  // no contiguity constraint
cuda_array_c<T>                // C-contiguous (row-major)
cuda_array_f<T>                // F-contiguous (column-major)
cuda_array_contig<T, Contig>   // parameterized contiguity
```

Choose the appropriate alias based on how the kernel accesses data.
Use `cuda_array_f` for kernels that index column-by-column (e.g., `data + col * n_rows`), and `cuda_array_c` for row-major access.
nanobind will reject arrays with the wrong memory layout at runtime.

### Adding a new kernel

1. Create a directory under `src/rapids_singlecell/_cuda/your_module/`
2. Write the kernel header (`kernels_your_module.cuh`) and bindings (`your_module.cu`)
3. Include `"../nb_types.h"` for the shared type aliases
4. Register the module in `CMakeLists.txt`:
   ```cmake
   add_nb_cuda_module(_your_module_cuda src/rapids_singlecell/_cuda/your_module/your_module.cu)
   ```
5. Add the module name to the `__all__` list in `src/rapids_singlecell/_cuda/__init__.py`:
   ```python
   __all__ = [
       ...,
       "_your_module_cuda",
   ]
   ```
   This registers the module for lazy loading — imports return `None` instead of raising `ImportError` when the compiled extension is unavailable (e.g., docs builds without a GPU).
6. Rebuild: `uv pip install -e .`

The `add_nb_cuda_module` helper automatically handles:
- Stable ABI + LTO compilation
- Linking against CUDA runtime
- Installing the `.so` into the wheel
- Generating `.pyi` type stubs (install-time for wheels, build-time for editable installs)
- Copying the built module into the source tree for editable installs

### Kernel conventions

- Each kernel launch wrapper is a `static inline` function in the `.cu` file
- Use `nb::kw_only()` to separate data arguments from configuration arguments
- Accept `std::uintptr_t stream` as the last parameter (default `0`) to support stream-based execution
- Keep kernel logic in `.cuh` headers, bindings in `.cu` files
- **Import `_cuda` modules via `rapids_singlecell._cuda`**. The `_cuda` package uses lazy loading with automatic `ImportError` handling — if the compiled extension is unavailable (e.g., docs builds without a GPU), the import returns `None` instead of raising an error:

  ```python
  from rapids_singlecell._cuda import _my_module_cuda as _my

  def my_function(adata):
      # _my is either the real module or None
      _my.kernel(...)
  ```

  No `try/except` or lazy imports needed — the `_cuda.__init__.py` handles it for you.

## Testing

### Hatch test environments

The project uses [hatch](https://hatch.pypa.io/) to manage test environments. The test matrix is defined in `hatch.toml` with two axes:

- **`cuda`**: `12` or `13` — selects the matching RAPIDS/CuPy packages
- **`deps`**: `stable`, `dev`, or `rapids_prerelease` — controls Python version and dependency sources

| `deps` | Python | Description |
|---|---|---|
| `stable` | 3.12 | Released versions of all dependencies |
| `dev` | 3.14 | Upstream `main` branches of anndata and scanpy |
| `rapids_prerelease` | 3.14 | RAPIDS nightly wheels |

To run the test suite against a specific matrix combination:

```bash
# Run stable tests with CUDA 13
(uvx) hatch run hatch-test.stable-13:run

# Run stable tests with CUDA 12
(uvx) hatch run hatch-test.stable-12:run

# Run dev tests (upstream anndata/scanpy) with CUDA 13
(uvx) hatch run hatch-test.dev-13:run
```

### Running individual tests

For quick iteration during development, you can pass specific test paths:

```bash
# Run a specific test file
(uvx) hatch run hatch-test.stable-13:run tests/path/to/test.py -v

# Run a specific test
(uvx) hatch run hatch-test.stable-13:run tests/path/to/test.py::test_name -v
```

```{important}
Always set a timeout when running tests with new CUDA kernels, as they may hang on launch failures.
Tests have a default 120-second timeout configured in `pyproject.toml`.
```

### Test guidelines

- **Never change test tolerances** without understanding why a test is failing. If a tolerance change is needed, document the current tolerance, the actual error, the proposed tolerance, and the reason.
- **GPU shared memory limits** vary across devices (e.g., T4 has 64KB per block). Kernels should query device limits at runtime rather than using fixed parameters.
- Use `pytest.importorskip` for optional dependencies in tests.

## Building documentation

```bash
(uvx) hatch run docs:build
```

To build without compiling CUDA extensions (e.g., on a machine without a GPU):

```bash
CMAKE_ARGS="-DRSC_BUILD_EXTENSIONS=OFF" (uvx) hatch run docs:build
```

The built docs are in `docs/_build/html/`.

## Distribution and packaging

### Package layout on PyPI

The project publishes three separate packages:

| Package | Contents | For whom |
|---|---|---|
| `rapids-singlecell-cu12` | Prebuilt wheels (CUDA 12) | Most users |
| `rapids-singlecell-cu13` | Prebuilt wheels (CUDA 13) | Most users |
| `rapids-singlecell` | Source distribution | Self-compilation |

### Wheel builds

Wheels are built via [cibuildwheel](https://cibuildwheel.pypa.io/) in GitHub Actions using custom manylinux Docker images with CUDA toolkit pre-installed.
The CI renames the package and adjusts optional dependencies per CUDA version using an inline Python script in `publish.yml`.

Each wheel contains:
- Compiled `.abi3.so` modules (stable ABI, one wheel per platform for all Python 3.12+ versions)
- `.pyi` type stubs for IDE support
- `py.typed` PEP 561 marker

Source files (`.cu`, `.cuh`, `.h`) are excluded from wheels via Hatchling's wheel target configuration in `pyproject.toml`.
They are included in the source distribution for self-compilation.

### CUDA architectures

CUDA 12 wheels target: `75` (Turing), `80` (Ampere), `86` (Ampere), `89` (Ada), `90` (Hopper + PTX for forward compatibility).
CUDA 13 wheels target: `75` (Turing), `80` (Ampere), `86` (Ampere), `89` (Ada), `90` (Hopper), `100` (Blackwell), `120` (Blackwell).

Source builds (`pip install rapids-singlecell`) compile for the local GPU architecture by default (`CMAKE_CUDA_ARCHITECTURES=native`).

### Docker containers

The `docker/` directory contains two types of Dockerfiles:

**User-facing containers** (for running rapids-singlecell):

| File | Purpose |
|---|---|
| `Dockerfile.deps` | Base image with conda RAPIDS environment + pip dependencies. Uses `nvidia/cuda:*-devel` for CUDA toolkit access. |
| `Dockerfile` | Final image that builds on `rapids-singlecell-deps` and compiles rapids-singlecell from source for all supported GPU architectures. |

These are built by `docker-push.sh`, which strips the `rapids-singlecell` pip line from the conda environment file and builds both images in sequence.

**CI manylinux images** (for building PyPI wheels):

Wheels are built by cibuildwheel against prebuilt manylinux + CUDA images published at
`quay.io/manylinux_cuda/manylinux_2_28_<arch>_cuda<ver>`. `publish.yml` selects the image
per matrix entry via `cibw_image`.

These images ship nvcc, cudart, and cublas (plus gcc-toolset-12 on the CUDA 12.2 image,
since nvcc on 12.2 requires GCC 12 or older). The remaining libraries rapids-singlecell
links against (cusolver, cusparse, and nvJitLink) are installed at build time via
`CIBW_BEFORE_ALL` in `publish.yml`.

### Release process

1. Tag the release: `git tag v0.X.Y` (or `v0.X.Yrc1` for release candidates)
2. Create a GitHub release from the tag
3. The `publish.yml` workflow builds wheels + sdist and uploads to PyPI via trusted publishing
4. Pre-releases (`rc`, `beta`, `alpha`) are automatically recognized by PyPI -- users must opt in with `pip install --pre`
