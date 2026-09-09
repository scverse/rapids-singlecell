# Contributing

## Development setup

### Prerequisites

- A Turing or newer NVIDIA GPU with an R580+ driver for execution.
- A supported RAPIDS/CuPy environment, using CUDA 12 or CUDA 13 dependencies.
- CUDA Toolkit 13.0+, Clang/libclang, and a native Rust linker for source builds.
- The pinned Rust/cuda-oxide toolchain in the [Rust backend guide](rust_backend.md).

### Clone and install

```bash
git clone --recurse-submodules https://github.com/scverse/rapids-singlecell.git
cd rapids-singlecell
bash scripts/install_cuda_oxide.sh
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0
python -m pip install -e '.[test]'
```

The documentation notebooks live in a Git submodule. If needed, initialize them
with `git submodule update --init` before building documentation.

The native implementation is Rust/cuda-oxide with PyO3 bindings. An editable
build places the shared `_rust_cuda.abi3.so` extension and type stubs under
`src/rapids_singlecell/_cuda/`. Kernels use portable `sm_75` PTX by default;
compilation does not require a GPU.

```{toctree}
:hidden:

rust_backend.md
```

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
│   └── _cuda/                   # Stable Python import paths for GPU modules
│       ├── _rust_cuda.abi3.so    # Shared PyO3 extension (gitignored)
│       ├── *.pyi                # Installed/copied type stubs (gitignored)
│       └── py.typed             # PEP 561 marker (gitignored, auto-generated)
├── rust/
│   ├── kernels/                 # cuda-oxide device kernels compiled to PTX
│   └── python/                  # PyO3 bindings and checked-in type stubs
├── cmake/RustBackend.cmake      # Cargo build and extension packaging
├── tests/                       # pytest test suite
├── docs/                        # Sphinx documentation
├── docker/                      # Docker and CI build images
├── conda/                       # Conda environment files
├── CMakeLists.txt               # Cargo orchestration and packaging
└── pyproject.toml               # Project metadata and build config
```

## Contributing GPU code

New native backend work should use **Rust/cuda-oxide** kernels and **PyO3**
bindings. Keep custom device kernels in Rust rather than embedding CUDA C++
strings inside Python.

You can also contribute GPU-accelerated functions with:

- **Pure CuPy** (array API, `cupyx.scipy`, etc.)
- **numba-cuda** kernels

Please **do not** introduce JAX or PyTorch as dependencies.
The project is built on the RAPIDS/CuPy stack and we want to keep the dependency footprint "minimal".

Start with a **correct, tested implementation**. Performance optimization and
porting to Rust kernels can follow. A working CuPy implementation is a useful
starting point if you are unfamiliar with the native backend.

```{tip}
When opening a pull request, please enable **"Allow edits by maintainers"** (the checkbox on the PR creation page).
This lets us make small fixes, optimizations, or Rust ports directly on your branch without extra back-and-forth.
```

## GPU backend architecture

### Rust kernels and PyO3 bindings

Device kernels live in `rust/kernels/` and compile to PTX with cuda-oxide.
Native PyO3 bindings live in `rust/python/`, borrow CuPy memory and CUDA
streams, and expose the existing `rapids_singlecell._cuda` module interfaces.
The bindings share array validation in `rust/python/src/array.rs` and CUDA
module caching, context handling, and launches in `rust/python/src/runtime.rs`.
The [Rust backend guide](rust_backend.md) documents toolchain requirements,
module organization, compatibility, and validation.

When adding or porting a native kernel:

1. Implement the device operation in `rust/kernels/`, documenting pointer,
   bounds, alignment, and launch requirements for unsafe code.
2. Add PyO3 bindings and checked-in type stubs under `rust/python/`. Preserve
   the Python signatures, dtype/layout dispatch, output ownership, and CUDA
   stream semantics of existing functions.
3. Register the module in the shared `_rust_cuda` extension and the Python
   `_cuda` package, and update `cmake/RustBackend.cmake` packaging as needed.
   Include its stub in wheel and editable installs, and embed the required PTX.
4. For a new module, register its name in `__all__` in
   `src/rapids_singlecell/_cuda/__init__.py` for lazy loading.
5. Rebuild the package and run GPU tests with
   `--require-rust-backend` to verify that migrated imports use the shared Rust
   extension before collecting tests.
6. Run direct GPU tests and public Python API tests with a timeout, and compare
   performance for representative workloads before replacing a default. CPU
   import-routing checks run independently with
   `python -m pytest tests/cpu --confcutdir=tests/cpu --timeout=120`.

Validate allocation bounds, shapes, dtypes, layout, device, and incompatible
overlap before unsafe launches. Preserve supported managed-memory allocations
and use the caller's CuPy allocator for scratch storage. Borrowed arrays and
streams must stay alive until asynchronous work completes; account for that
lifetime when releasing temporary allocations. Restore the caller's CUDA
context and avoid adding synchronization to asynchronous APIs.

### Python imports

- **Import `_cuda` modules via `rapids_singlecell._cuda`**. Native modules resolve to the shared Rust extension; an absent extension (for example in a documentation build) resolves to `None`. A present extension that fails to load raises its import error:

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
Tests have a default 60-second timeout configured in `pyproject.toml`; use `--timeout=120` when needed.
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
- One `_rust_cuda.abi3.so` extension (stable ABI for supported Python 3.12+ versions)
- `.pyi` type stubs for IDE support
- `py.typed` PEP 561 marker

Rust sources and the pinned Cargo lockfile are included in source distributions.
Wheels contain the compiled extension and exclude stale editable native artifacts.

### CUDA architectures and dependency variants

Both CUDA dependency variants are built with CUDA Toolkit 13.0 and portable
`sm_75` PTX, supporting Turing through later compatible GPU generations through
driver JIT compilation. Both require driver R580+; CUDA 12 libraries remain
supported on those newer drivers. Source builds can select a higher minimum
capability with `SKBUILD_CMAKE_DEFINE_RSC_RUST_CUDA_ARCH`.

### Docker containers

`Dockerfile.deps` creates a matching CUDA 12 or CUDA 13 RAPIDS environment.
`Dockerfile` builds the native wheel in a separate CUDA 13.0/Rust build stage,
then installs only the wheel in that environment. Compiler caches and development
tools stay out of the application image.

The wheel workflow similarly uses CUDA 13.0 manylinux images for both dependency
families. It installs the pinned compiler with `scripts/install_cuda_oxide.sh`,
checks that each wheel contains exactly one Rust extension, and audits native
linkage and the Python stable ABI.

### Release process

1. Tag the release: `git tag v0.X.Y` (or `v0.X.Yrc1` for release candidates)
2. Create a GitHub release from the tag
3. The `publish.yml` workflow builds wheels + sdist and uploads to PyPI via trusted publishing
4. Pre-releases (`rc`, `beta`, `alpha`) are automatically recognized by PyPI -- users must opt in with `pip install --pre`
