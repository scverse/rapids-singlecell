#!/usr/bin/env python3
"""Compare exact dense ranking against an archived native backend.

The public APIs synchronize before returning. CUDA event timings therefore
include host dispatch gaps; wall timings measure complete API latency. Neither
metric is presented as isolated asynchronous kernel time. For that breakdown,
run this script under ``nsys profile --trace=cuda,nvtx --sample=none`` with
``--nvtx``, then use ``nsys stats --report cuda_gpu_kern_sum,cuda_api_sum``.
The NVTX ranges identify each backend and case; ``--case-filter`` narrows a trace.

Example:
    python scripts/benchmark_dense_ranking_backends.py \\
        --rust-library build/rust-migration/rust/_rust_cuda.abi3.so \\
        --legacy-directory /path/to/archived/native/modules --output dense.json
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path


def load_extension(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def check_ieee_equivalence(cp, np, legacy, backend):
    """Check deterministic native NaN behavior separately from finite ranks."""
    nans = np.array([0xFFC00001, 0x7FC00001, 0x7FFFFFFF, 0xFFFFFFFF], np.uint32).view(
        np.float32
    )
    total = 0

    def compare(function, shape, flag):
        nonlocal total
        expected = None
        for module in (legacy, backend):
            ranks = cp.empty(shape, dtype=np.float64)
            ties = (
                cp.empty(shape, dtype=np.float64)
                if flag
                else cp.empty(1, dtype=np.float64)
            )
            function(module, ranks, ties)
            actual = (cp.asnumpy(ranks), cp.asnumpy(ties))
            if expected is not None:
                np.testing.assert_array_equal(actual[0], expected[0])
                if flag:
                    np.testing.assert_allclose(actual[1], expected[1], atol=1e-14)
            expected = actual
        total += 1

    for rows in (17, 1024, 1025, 4099, 32769):
        rng = np.random.default_rng(133)
        data = rng.normal(size=(rows, 3)).astype(np.float32)
        data[:4] = nans[:, None]
        data[4:6] = 0
        x = cp.asarray(data, order="F")
        codes = cp.arange(rows, dtype=np.int32) % 7
        for flag in (False, True):
            # OVR has a vector tie output, unlike the OVO group matrix.
            expected = None
            for module in (legacy, backend):
                ranks = cp.empty((7, 3), dtype=np.float64)
                ties = cp.empty(3, dtype=np.float64)
                module.ovr_rank_dense_streaming(
                    x, codes, ranks, ties, compute_tie_corr=flag, sub_batch_cols=2
                )
                actual = (cp.asnumpy(ranks), cp.asnumpy(ties))
                if expected is not None:
                    np.testing.assert_array_equal(actual[0], expected[0])
                    if flag:
                        np.testing.assert_allclose(actual[1], expected[1], atol=1e-14)
                expected = actual
            total += 1
    reference_cases = [
        (1031, sizes, 3, 2) for sizes in ((0, 1, 17), (512, 513, 2500), (2501, 4097, 3))
    ]
    # Both packed block boundaries include a final column on the global radix
    # path, with NaNs in each population and every OVO tie tier.
    reference_cases.extend((rows, (512, 513, 2501), 33, 32) for rows in (5000, 8192))
    for reference_rows, sizes, columns, width in reference_cases:
        for where in ("reference", "groups", "both"):
            rng = np.random.default_rng(137)
            ref = rng.normal(size=(reference_rows, columns)).astype(np.float32)
            grp = rng.normal(size=(sum(sizes), columns)).astype(np.float32)
            if where in ("reference", "both"):
                ref[:4] = nans[:, None]
            if where in ("groups", "both"):
                for start, rows in zip(
                    np.r_[0, np.cumsum(sizes)[:-1]], sizes, strict=True
                ):
                    grp[start : start + min(rows, 4)] = nans[: min(rows, 4), None]
            ref = cp.asarray(ref, order="F")
            grp = cp.asarray(grp, order="F")
            offsets = cp.asarray(np.r_[0, np.cumsum(sizes)], dtype=np.int32)
            for flag in (False, True):
                compare(
                    lambda module, ranks, ties: (
                        module.ovo_rank_dense_tiered_unsorted_ref(
                            ref,
                            grp,
                            offsets,
                            ranks,
                            ties,
                            compute_tie_corr=flag,
                            sub_batch_cols=width,
                        )
                    ),
                    (len(sizes), columns),
                    flag,
                )
    # Native host layout and precision feed statistics while ranks use float32.
    # Exercise the fused C-order cast/transpose separately from device sorting.
    for order in ("C", "F"):
        for dtype in (np.float32, np.float64):
            rng = np.random.default_rng(139)
            host = np.array(rng.normal(size=(1031, 5)), dtype=dtype, order=order)
            host[:4] = nans[:, None]
            host[4:8] = np.array([0.0, -0.0, np.inf, -np.inf])[:, None]
            codes = cp.arange(len(host), dtype=np.int32) % 7
            for flag in (False, True):
                expected = None
                for module in (legacy, backend):
                    ranks = cp.empty((7, 3), dtype=np.float64)
                    ties = cp.empty(3, dtype=np.float64)
                    sums = cp.empty_like(ranks)
                    nnz = cp.empty_like(ranks)
                    totals = cp.empty(3, dtype=np.float64)
                    total_nnz = cp.empty_like(totals)
                    module.ovr_rank_dense_host_streaming(
                        host,
                        codes,
                        ranks,
                        ties,
                        sums,
                        nnz,
                        totals,
                        total_nnz,
                        compute_tie_corr=flag,
                        compute_nnz=True,
                        compute_totals=True,
                        col_start=1,
                        col_stop=4,
                        sub_batch_cols=2,
                    )
                    actual = [
                        cp.asnumpy(a) for a in (ranks, sums, nnz, totals, total_nnz)
                    ]
                    if flag:
                        actual.append(cp.asnumpy(ties))
                    if expected is not None:
                        np.testing.assert_array_equal(actual[0], expected[0])
                        for result, reference in zip(
                            actual[1:], expected[1:], strict=True
                        ):
                            np.testing.assert_allclose(
                                result, reference, rtol=1e-11, atol=1e-11
                            )
                    expected = actual
                total += 1
    print(f"{total} archived IEEE cases passed", flush=True)
    return total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-library", required=True, type=Path)
    parser.add_argument("--legacy-directory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--device-only", action="store_true")
    parser.add_argument(
        "--host-distribution",
        choices=("ties", "continuous"),
        default="ties",
        help="Distribution for host C/F precision and streaming comparisons",
    )
    parser.add_argument(
        "--check-ieee",
        action="store_true",
        help="Also compare archived NaN/sign/payload cases",
    )
    parser.add_argument(
        "--case-filter",
        help="Only time cases whose JSON description contains this text",
    )
    parser.add_argument(
        "--nvtx",
        action="store_true",
        help="Annotate native calls for Nsight kernel/CPU launch profiling",
    )
    args = parser.parse_args()
    if args.repeat < 1 or args.warmup < 1:
        parser.error("repeat and warmup must be positive")

    import cupy as cp
    import numpy as np
    from cupyx.profiler import benchmark

    rust = load_extension(
        "rapids_singlecell._cuda._rust_cuda", args.rust_library.resolve()
    )
    legacy_paths = sorted(args.legacy_directory.glob("_wilcoxon_cuda*.so"))
    if len(legacy_paths) != 1:
        parser.error(
            "legacy directory must contain exactly one _wilcoxon_cuda extension"
        )
    legacy = load_extension("legacy._wilcoxon_cuda", legacy_paths[0].resolve())
    backend = rust._wilcoxon_cuda
    if getattr(backend, "__backend__", None) != "rust":
        raise RuntimeError("selected library is not the Rust ranking backend")

    allocations = {}

    def allocate(size):
        pointer = cp.cuda.alloc(size)
        allocations[pointer.ptr] = pointer
        return pointer.ptr

    def release(pointer):
        allocations.pop(pointer, None)

    legacy._set_scratch_allocator(allocate, release)
    # Release captured CuPy state before the archived C++ static teardown.
    atexit.register(legacy._set_scratch_allocator, int, int)
    rng = np.random.default_rng(91)
    results = []

    def values(shape, distribution, dtype=np.float32, order="F"):
        data = rng.normal(size=shape)
        if distribution == "ties":
            data = np.round(data)
        return np.array(data, dtype=dtype, order=order)

    def measure(case, functions, outputs):
        if args.case_filter and args.case_filter not in json.dumps(case):
            return
        measurements = {}
        expected = None
        for name, function in zip(("legacy", "rust"), functions, strict=True):
            if args.nvtx:
                from cupyx.profiler import time_range

                def traced_call(function=function, name=name):
                    with time_range(f"{name}:{json.dumps(case)}"):
                        return function()

                measured = traced_call
            else:
                measured = function
            timing = benchmark(measured, n_repeat=args.repeat, n_warmup=args.warmup)
            actual = [cp.asnumpy(output) for output in outputs]
            if expected is None:
                expected = actual
            else:
                for i, (old, new) in enumerate(zip(expected, actual, strict=True)):
                    if i == 0:
                        np.testing.assert_array_equal(new, old, err_msg=str(case))
                    else:
                        np.testing.assert_allclose(
                            new, old, rtol=1e-11, atol=1e-11, err_msg=str(case)
                        )
            measurements[name] = {
                "wall_ms": float(np.median(timing.cpu_times) * 1000),
                "cuda_event_ms": float(np.median(timing.gpu_times) * 1000),
            }
        item = {**case, **measurements}
        item["rust_over_legacy"] = (
            measurements["rust"]["wall_ms"] / measurements["legacy"]["wall_ms"]
        )
        results.append(item)
        print(json.dumps(item), flush=True)

    for rows, cols in ((10_000, 64), (100_000, 64), (10_000, 137)):
        for distribution in ("ties", "continuous"):
            x = cp.asarray(values((rows, cols), distribution), order="F")
            codes = cp.arange(rows, dtype=np.int32) % 10
            ranks = cp.empty((10, cols), dtype=np.float64)
            for compute_ties in (False, True):
                ties = cp.empty(cols, dtype=np.float64)
                functions = [
                    lambda m=m: m.ovr_rank_dense_streaming(
                        x,
                        codes,
                        ranks,
                        ties,
                        compute_tie_corr=compute_ties,
                        sub_batch_cols=64,
                    )
                    for m in (legacy, backend)
                ]
                outputs = [ranks, ties] if compute_ties else [ranks]
                measure(
                    {
                        "api": "ovr",
                        "location": "device",
                        "rows": rows,
                        "cols": cols,
                        "distribution": distribution,
                        "ties": compute_ties,
                    },
                    functions,
                    outputs,
                )

    group_cases = [
        (5000, [512] * 5),
        (5000, [513] * 5),
        (5000, [1000] * 5),
        (5000, [2500] * 5),
        (5000, [2501] * 5),
        (50000, [10000] * 5),
        (50000, [0, 20, 512, 513, 2500, 2501, 20000]),
    ]
    group_cases = [(nref, sizes, 64) for nref, sizes in group_cases]
    group_cases.append((5000, [0, 512, 513, 2500, 2501], 137))
    for nref, sizes, cols in group_cases:
        for distribution in ("ties", "continuous"):
            ref = cp.asarray(values((nref, cols), distribution), order="F")
            grp = cp.asarray(values((sum(sizes), cols), distribution), order="F")
            offsets = cp.asarray(np.r_[0, np.cumsum(sizes)], dtype=np.int32)
            ranks = cp.empty((len(sizes), cols), dtype=np.float64)
            for compute_ties in (False, True):
                ties = (
                    cp.empty_like(ranks)
                    if compute_ties
                    else cp.empty(1, dtype=np.float64)
                )
                functions = [
                    lambda m=m: m.ovo_rank_dense_tiered_unsorted_ref(
                        ref,
                        grp,
                        offsets,
                        ranks,
                        ties,
                        compute_tie_corr=compute_ties,
                        sub_batch_cols=64,
                    )
                    for m in (legacy, backend)
                ]
                outputs = [ranks, ties] if compute_ties else [ranks]
                measure(
                    {
                        "api": "ovo",
                        "location": "device",
                        "reference_rows": nref,
                        "group_sizes": sizes,
                        "cols": cols,
                        "distribution": distribution,
                        "ties": compute_ties,
                    },
                    functions,
                    outputs,
                )

    if not args.device_only:
        for order in ("C", "F"):
            for dtype in (np.float32, np.float64):
                for compute_ties in (False, True):
                    data = values((10000, 141), args.host_distribution, dtype, order)
                    codes = cp.arange(10000, dtype=np.int32) % 10
                    ranks = cp.empty((10, 137), dtype=np.float64)
                    ties = cp.empty(137, dtype=np.float64)
                    sums = cp.empty_like(ranks)
                    nnz = cp.empty_like(ranks)
                    totals = cp.empty(137, dtype=np.float64)
                    total_nnz = cp.empty_like(totals)
                    functions = [
                        lambda m=m: m.ovr_rank_dense_host_streaming(
                            data,
                            codes,
                            ranks,
                            ties,
                            sums,
                            nnz,
                            totals,
                            total_nnz,
                            compute_tie_corr=compute_ties,
                            compute_nnz=True,
                            compute_totals=True,
                            col_start=2,
                            col_stop=139,
                            sub_batch_cols=32,
                        )
                        for m in (legacy, backend)
                    ]
                    outputs = [ranks, sums, nnz, totals, total_nnz] + (
                        [ties] if compute_ties else []
                    )
                    measure(
                        {
                            "api": "ovr",
                            "location": "host",
                            "rows": 10000,
                            "cols": 137,
                            "distribution": args.host_distribution,
                            "order": order,
                            "dtype": np.dtype(dtype).name,
                            "ties": compute_ties,
                        },
                        functions,
                        outputs,
                    )

                    nref = 4000
                    sizes = [0, 512, 513, 2500, 2501]
                    data = values(
                        (nref + sum(sizes), 141), args.host_distribution, dtype, order
                    )
                    ref_ids = np.arange(nref, dtype=np.int32)
                    grp_ids = np.arange(nref, len(data), dtype=np.int32)
                    offsets = np.r_[0, np.cumsum(sizes)].astype(np.int32)
                    ranks = cp.empty((len(sizes), 137), dtype=np.float64)
                    ties = (
                        cp.empty_like(ranks)
                        if compute_ties
                        else cp.empty(1, dtype=np.float64)
                    )
                    sums = cp.empty((len(sizes) + 1, 137), dtype=np.float64)
                    nnz = cp.empty_like(sums)
                    functions = [
                        lambda m=m: m.ovo_rank_dense_host_streaming(
                            data,
                            ref_ids,
                            grp_ids,
                            offsets,
                            ranks,
                            ties,
                            sums,
                            nnz,
                            compute_tie_corr=compute_ties,
                            compute_nnz=True,
                            col_start=2,
                            col_stop=139,
                            sub_batch_cols=32,
                        )
                        for m in (legacy, backend)
                    ]
                    outputs = [ranks, sums, nnz] + ([ties] if compute_ties else [])
                    measure(
                        {
                            "api": "ovo",
                            "location": "host",
                            "reference_rows": nref,
                            "group_sizes": sizes,
                            "cols": 137,
                            "distribution": args.host_distribution,
                            "order": order,
                            "dtype": np.dtype(dtype).name,
                            "ties": compute_ties,
                        },
                        functions,
                        outputs,
                    )

    for groups in (513, 4096, 6112, 6113):
        rows, cols = 16384, 32
        for distribution in ("ties", "continuous"):
            x = cp.asarray(values((rows, cols), distribution), order="F")
            codes = cp.arange(rows, dtype=np.int32) % groups
            ranks = cp.empty((groups, cols), dtype=np.float64)
            for compute_ties in (False, True):
                ties = cp.empty(cols, dtype=np.float64)
                functions = [
                    lambda m=m: m.ovr_rank_dense_streaming(
                        x,
                        codes,
                        ranks,
                        ties,
                        compute_tie_corr=compute_ties,
                        sub_batch_cols=32,
                    )
                    for m in (legacy, backend)
                ]
                measure(
                    {
                        "api": "ovr",
                        "location": "device",
                        "rows": rows,
                        "cols": cols,
                        "groups": groups,
                        "distribution": distribution,
                        "ties": compute_ties,
                    },
                    functions,
                    [ranks, ties] if compute_ties else [ranks],
                )

    ieee_cases = (
        check_ieee_equivalence(cp, np, legacy, backend) if args.check_ieee else 0
    )
    props = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
    name = props["name"]
    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cupy": cp.__version__,
        "numpy": np.__version__,
        "gpu": name.decode() if isinstance(name, bytes) else name,
        "driver": cp.cuda.runtime.driverGetVersion(),
        "runtime": cp.cuda.runtime.runtimeGetVersion(),
        "rust_library": str(args.rust_library.resolve()),
        "rust_library_sha256": hashlib.sha256(
            args.rust_library.read_bytes()
        ).hexdigest(),
        "legacy_library": str(legacy_paths[0].resolve()),
        "legacy_library_sha256": hashlib.sha256(
            legacy_paths[0].read_bytes()
        ).hexdigest(),
        "ieee_cases_checked": ieee_cases,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "timing_note": "Public APIs synchronize; CUDA events include host dispatch gaps. Wall time is full API latency.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
