# Rust CUDA backend

The native backend uses **Rust**, **cuda-oxide**, and **PyO3**. All repository-owned
GPU kernels and native launch wrappers live under `rust/`, including the operations
that previously used custom CUDA C++ strings inside Python. The Python API keeps
its existing `rapids_singlecell._cuda` module names and CuPy array model.

## Architecture

- `rust/kernels/` contains device kernels compiled to PTX by cuda-oxide.
- `rust/kernels/exports.txt` lists the complete PTX kernel interface. Update it
  when adding or removing a kernel; the host build rejects missing or unexpected
  entry points before embedding the image. It also derives parameter counts
  from PTX so a mismatched internal launch fails before entering the CUDA driver.
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

Streaming ranking calls complete their work before returning. They reuse bounded
sort workspaces and a ring of private pinned upload buffers, preserve input
precision for statistics, and order worker streams after the caller's inputs.
Reusing a slot waits only for that slot. Successful and exceptional returns wait
for queued work before releasing temporary allocations and restore the caller's
stream. Reusable device storage is allocated through the caller's stream pool;
explicit events order asynchronous allocator work before worker-stream use.
This preserves pool reuse between calls, including when the worker streams are
new, without sharing unfinished scratch between slots.

Existing GPU libraries remain part of the RAPIDS/CuPy stack. For example, linear
algebra uses the cuBLAS and cuSOLVER libraries selected by CuPy. Harmony resolves
native cuBLAS functions through CuPy's loaded extension, retaining that exact
CUDA 12 or CUDA 13 dependency. GEMM supports CUDA graph capture and preserves
the caller's scalar pointer mode. The migration covers code maintained in this
repository; it does not replace those libraries or require their C++ headers.

## Host parallelism

CPU boundary searches and staging layout conversions use a shared Rayon thread
pool, capped at the available CPU count and 32 workers. Each calling thread can
limit its submitted partitions with `_set_host_worker_limit`; the setting is
shared across the native ranking modules and returns the previous limit. A value
of zero selects the hardware cap, negative values reset to zero, and one selects
serial execution. Inputs smaller than 4,096 work items also run serially. Concurrent
GPU shards share the pool and retain their individual limits.
CSR dense packing uses the detached caller for at most 1 MiB of source values
and indices; larger payloads retain the shared pool. This avoids scheduling
overhead for small packs while keeping their snapshots private.

Computations over owned CPU storage release the Python interpreter lock. Workers
access owned input snapshots and disjoint private outputs; they never call Python
or use a mutable NumPy buffer while detached. Snapshots read exported host buffers
while attached. Results are copied into caller outputs after complete validation
and successful computation. Dense staging snapshots cover only the current
bounded window and preserve the input precision and floating-point bit patterns.

CSR staging preserves each supplied row interval, including gaps and unsorted
indices. Inputs fitting a 64 MiB peak conversion payload budget are converted
once into owned CSC storage, so later windows avoid repeated row searches and
scattering. The budget includes temporary vectors and parallel scatter metadata.
Larger inputs use bounded window snapshots. Eligible CSR reference comparisons
cache the sorted reference once and reuse compact row storage across groups.
Their column slabs expand within the checked memory budget, and finite,
nonnegative inputs use analytical zero ranks without sorting the implicit zeros.
Reference comparisons also plan
row memberships and stored population counts in native host buffers before
uploading their offsets. Their device packer uses the same owned membership
snapshot, keeping planned segment capacities consistent with uploaded labels.
The number of tasks adapts to the work available within the caller's worker
limit.

Sparse comparisons detect selected NaNs during existing population planning.
Affected windows use bounded dense OVO tiers to retain the original floating
comparison behavior, including supplied membership order. Finite windows retain
the compact sparse route without an additional device-to-host synchronization.
Statistics continue to use the input precision independently of rank keys.

CSR boundary searches reuse at most 128 MiB of owned workspace per calling
thread; larger requests release their workspace after completion. These safety
copies add memory traffic compared with reading Python-owned arrays directly.
Use `spawn` or `forkserver` for process workers, as required for CUDA workers;
Rayon supplies threads within each process rather than replacing processes.

Host regression tests require the built extension but do not initialize CUDA:

```bash
python -m pytest tests/cpu/test_host_parallel.py --confcutdir=tests/cpu
```

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
RSC_CUDA_OXIDE_PTX="$(pwd)/build/rust/ptx/rsc_kernels.ptx" \
  cargo +nightly-2026-08-28 clippy --manifest-path rust/Cargo.toml --locked -- -D warnings
```

Release validation includes GPU correctness, dtype/index-width coverage, supplied
streams, managed-memory allocators, package contents, the Python stable ABI, and
representative performance. Hardware-specific and multi-GPU tests require suitable
runners; a skipped test does not establish support for that configuration.

The portable smoke check loads a specific extension without importing the full
RAPIDS package. Run it in each CUDA 12 and CUDA 13 environment to check sparse
dtype/index/layout dispatch and cuBLAS graph capture with both scalar pointer
modes:

```bash
python scripts/smoke_rust_backend.py --backend build/rust-migration/rust/_rust_cuda.abi3.so
```

Use `--expect-cuda-major 12` or `--expect-cuda-major 13` to assert the environment
being tested. Its JSON output records the extension hash and CUDA library versions.

### Kernel fidelity and performance comparisons

The optimization audit uses the C++ backend at commit
`f686373` as its reference. Compare both kernel bodies and host dispatch:
launch sizes, memory layout, sorting thresholds, reduction order, precision,
scratch ownership, and stream ordering all affect the resulting behavior.

The restored implementations include:

| Area | Preserved implementation details |
| --- | --- |
| Dense ranking | Shared-memory OVO tiers, sorted large-reference path, exact tie ranks, stable IEEE radix ordering, coalesced radix passes, compact sort positions, and overlapping pinned uploads |
| Sparse ranking | Native CSR-to-CSC conversion, caller-supplied row spans, cached CSR references, analytical zero ranks, compact sort positions, shared group reductions with a global fallback, and bounded NaN comparison windows |
| Pearson residuals | Separate CSR/CSC/HVG paths, original rounding and sequential Welford updates, four-way loop unrolling, read-only loads, clipping semantics, and specialized launch sizes |
| Other preprocessing | Original reduction geometry, native floating-point atomics, long-row normalization, masks, dtype and index-width dispatch |
| Harmony | Adaptive column sums and row reductions, vector loads, covariate specializations, paired correction reductions, and direct cuBLAS calls |
| Gaussian mixtures | Shared projection vectors with a bounded global fallback, fused EM updates, original accumulation precision and reduction order, and population-dependent launch geometry at every projection width |
| Spatial and domain kernels | Shared feature windows, advancing sparse cursors, tile geometry, reductions, and fused output updates |

The benchmark scripts accept an explicit Rust library and archived reference
extensions. They record library hashes, input configurations, hardware, and timing
methods. Run one benchmark process at a time with other CPU/GPU workloads idle:

```bash
python scripts/benchmark_dense_ranking_backends.py --help
python scripts/benchmark_host_sparse_backends.py --help
python scripts/benchmark_domain_kernels.py --help
python scripts/benchmark_utility_kernels.py --help
```

Domain and utility benchmarks report both ordinary calls and CUDA graph replay
where capture is supported. Graph replay isolates GPU work; streaming ranking
benchmarks include validation, host preparation, transfers, and completion.
Numerical agreement and performance are separate checks: a passing test suite
alone does not establish speed parity on every workload or GPU.

### Recorded validation: September 10, 2026

The [validation record](_static/rust-backend-validation.json) contains artifact
hashes, environment details, sanitizer scopes, package checks, and complete
per-case benchmark results. The final extension was tested on NVIDIA GB10 with
Python 3.14 and CuPy 14.1.1, using `sm_75` PTX built with CUDA 13.0.
The archived extensions contain `sm_121` cubins. The timings compare these actual
builds; their compiler targets are different.

The full suite passed **2,985 tests**, with **94 skipped**. CPU-only pytest passed
30 tests; Rust passed 17 unit tests, with three manual profiles ignored. Archived
comparisons cover 48 dense IEEE cases and 72 sparse NaN cases, including large
groups and supplied membership order. Skips remain unverified on this system.

Memory, race, and synchronization checks passed for the restored kernel families;
the record identifies each tested artifact and scope. Staging race checks used
one checker CPU worker to avoid a tool scheduling stall while preserving
asynchronous CUDA launches. Unfiltered sparse NaN checks reported warnings in
CuPy's nonzero scan that also reproduce without loading this backend; the Rust
kernel checks were clean. Stream-ordered allocation checks also passed.

Checkout and independently rebuilt source-distribution wheels were byte-identical,
passed the Python 3.12 stable-ABI audit, and routed all 41 native modules through
one extension. Installed-wheel smoke checks passed in both CUDA 12 and CUDA 13
environments, including cuBLAS graph capture and scalar pointer modes.

Performance still has gaps. The table lists representative residuals, with ratios
above one meaning Rust takes longer than the archived C++ implementation:

| Workload | Measurement | Rust / C++ |
| --- | --- | --- |
| Dense device OVO | Largest measured end-to-end ratio | 1.49 |
| Continuous dense host OVR | Largest measured end-to-end ratio | 1.41 |
| Small host CSR OVO | Standard / alternating-process comparison | 1.32 / 1.59 |
| C-order float64 host histogram | End-to-end ratio | 1.72 |
| CPU boundaries, default worker count | Median; observed range | 1.85; 0.27–5.42 |
| Large float32 projected GMM | GPU-only ratio | 1.49 |
| Pearson CSR, float32/int64 | Largest measured GPU-only ratio | 1.33 |

Rayon improved all eight CPU boundary workloads relative to Rust serial execution,
but owned snapshots and output copies still affect comparisons with C++. The
histogram measurement predates the final CSR-only CPU scheduling change; its
dense path and PTX are unchanged. Results on one GPU do not establish universal
speed parity. The complete record retains faster cases and timing variability as
well as these residuals.

Canonical sparse inputs preserve the archived NaN comparison behavior. The
archived CSC extractor has racing writes for duplicate coordinates; the Rust NaN
fallback instead uses deterministic stored-order overwrites. Its duplicate-write
behavior is therefore not an exact compatibility promise for noncanonical CSC.

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
