# Rust CUDA backend

The native backend uses **Rust**, **cuda-oxide**, and **PyO3**. All repository-owned
GPU kernels and native launch wrappers live under `rust/`, including the operations
that previously used custom CUDA C++ strings inside Python. The Python API keeps
its existing `rapids_singlecell._cuda` module names and CuPy array model.

## Architecture

- `rust/kernels/` contains device kernels compiled to PTX by cuda-oxide.
- `rust/kernels/exports.txt` lists the complete PTX kernel interface. Update it
  when adding or removing a kernel; the host build rejects missing or unexpected
  entry points before embedding the image.
- `rust/python/` contains validated PyO3 bindings and checked-in type stubs.
- `rust/python/src/array.rs` validates device array metadata and allocation bounds.
- `rust/python/src/runtime.rs` caches device modules and launches kernels on borrowed
  CUDA streams. CUDA library operations use the caller's CuPy environment.
- `cmake/RustBackend.cmake` builds one `_rust_cuda.abi3.so` containing the PTX image
  and all native submodules. CMake does not compile C++ or CUDA C++.

Bindings borrow CuPy allocations, including supported managed memory, and retain
the caller's dtype, device, output buffers, and stream. Asynchronous operations
require inputs, outputs, and streams to remain alive until completion. Scratch
storage comes from the caller's CuPy allocator. Native operations validate their
array contracts before launching and reject incompatible metadata.
The kernels require the current device's primary CUDA context, as used by CuPy;
the runtime restores the caller's context after each launch.

Existing GPU libraries remain part of the RAPIDS/CuPy stack. For example, linear
algebra can use cuBLAS and cuSOLVER through CuPy. The migration covers code maintained
in this repository; it does not replace those libraries or require their C++ headers.

## Build requirements

- Linux, Python 3.12–3.14, and a supported RAPIDS/CuPy environment for execution.
- CUDA Toolkit **13.0 or newer**, including cuRAND headers and libNVVM. Prefer 13.0
  when building portable wheels with the same driver floor as the release images.
- Clang/libclang and a native linker for Rust's generated CUDA bindings.
- Rust **nightly-2026-08-28**, with `rust-src`, `rustc-dev`, `llvm-tools`, `rustfmt`,
  and `clippy`.
- `cargo-oxide` from revision **26754ae52c26c097dc1c465a1e42c4c5d05a3d40**, matching
  `rust/kernels/Cargo.toml` and `rust/Cargo.lock`.

The pinned upstream requirements are documented in the
[cuda-oxide installation guide](https://github.com/NVlabs/cuda-oxide/blob/26754ae52c26c097dc1c465a1e42c4c5d05a3d40/cuda-oxide-book/getting-started/installation.md).
Install the pinned compiler from a checkout or source distribution:

```bash
bash scripts/install_cuda_oxide.sh
export PATH="${CARGO_HOME:-$HOME/.cargo}/bin:$PATH"
export CUDA_TOOLKIT_PATH=/usr/local/cuda-13.0
python -m pip install -e '.[test]'
```

The script reads the toolchain and dependency revision from the Rust manifests.
CUDA headers can also be located with `CUDAToolkit_ROOT`, `CUDA_HOME`, or `CUDA_PATH`.
The first build downloads Rust dependencies and builds the cuda-oxide compiler;
subsequent builds reuse Cargo caches. Cargo builds use `--locked`.

## CUDA compatibility

Both the `cu12` and `cu13` distributions use the same Rust backend compiled with
CUDA 13.0. Their extras select matching **CuPy/RAPIDS dependencies**. Both require
an NVIDIA **R580 or newer driver**; a CUDA 12 environment with an older driver no
longer meets the backend's requirements. CUDA 12 libraries remain supported on
newer drivers through
[NVIDIA's backward compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

The default `sm_75` PTX target supports Turing GPUs such as T4 and newer
architectures through driver JIT compilation. This makes source builds independent
of the presence of a local GPU. Select a higher minimum capability when needed:

```bash
SKBUILD_CMAKE_DEFINE_RSC_RUST_CUDA_ARCH=sm_80 python -m pip install -e .
```

The target is a single PTX architecture, rather than a list of cubin targets.
Compiling with a newer CUDA toolkit can emit a newer PTX version and require a
newer driver. The source-build toolkit and deployment driver must be compatible.

## Validation

Run direct kernel tests and public API tests against the compiled backend:

```bash
python -m pytest tests --require-rust-backend --timeout=120
```

Focused preprocessing checks include:

```bash
python -m pytest tests/test_preprocessing_kernels.py tests/test_elementwise_kernels.py \
  tests/test_normalization.py tests/test_scaling.py tests/test_qc_metrics.py \
  tests/test_hvg.py tests/test_mean_var.py --require-rust-backend --timeout=120
```

CPU import-routing tests run without a GPU:

```bash
python -m pytest tests/cpu --confcutdir=tests/cpu --timeout=120
```

Rust formatting and lint checks use the pinned toolchain. Host checks need the
PTX produced by a build so the embedding build script can validate it:

```bash
cargo +nightly-2026-08-28 fmt --manifest-path rust/Cargo.toml --all --check
RSC_CUDA_OXIDE_PTX=build/rust/ptx/rsc_kernels.ptx \
  cargo +nightly-2026-08-28 clippy --manifest-path rust/Cargo.toml --locked -- -D warnings
```

Release validation includes GPU correctness, dtype/index-width coverage, supplied
streams, managed-memory allocators, package contents, the Python stable ABI, and
representative performance. Hardware-specific and multi-GPU tests require suitable
runners; a skipped test does not establish support for that configuration.

## Packaging and documentation builds

Wheels contain one native PyO3 extension, type stubs, and the `py.typed` marker.
Historical `.so` files in an editable source tree are excluded from wheel inputs.
Source distributions contain the Rust sources, lockfile, compiler setup script,
and CMake integration needed to rebuild the extension.

Disable native compilation for documentation builds:

```bash
CMAKE_ARGS='-DRSC_BUILD_EXTENSIONS=OFF' python -m pip install -e '.[doc]'
```

`RSC_BUILD_RUST` is no longer an opt-in backend selection. Rust is the native
backend; `RSC_BUILD_EXTENSIONS=OFF` omits it entirely for documentation tooling.
