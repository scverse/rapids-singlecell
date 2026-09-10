#!/usr/bin/env python3
"""Compare native host work, exact sparse ranks, and preprocessing fast paths.

Run with an otherwise idle CPU/GPU. Timings include binding validation and wait
for work to complete; setup/reset copies occur outside the measured interval.
Legacy binaries are external inputs, never packaged or loaded by the application.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path

import numpy as np


def load(name, path, *, references=None):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if references is not None:
        references[name.rsplit(".", 1)[-1]] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return module


def measure(run, args, *, prepare=lambda: None, synchronize=lambda: None):
    for _ in range(args.warmup):
        prepare()
        synchronize()
        run()
        synchronize()
    samples = []
    for _ in range(args.repeat):
        prepare()
        synchronize()
        start = time.perf_counter()
        run()
        synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return float(np.median(samples))


def record(results, case, timings):
    result = dict(
        case, milliseconds=timings, rust_over_legacy=timings["rust"] / timings["legacy"]
    )
    results.append(result)
    print(json.dumps(result), flush=True)


def boundaries(args, rust, legacy, results):
    rng = np.random.default_rng(184)
    cases = [(10_000, 64, 8), (100_000, 64, 8), (100_000, 64, 64), (1_000_000, 16, 8)]
    for rows, stored, ncuts in cases:
        for distribution in ("repeated", "random"):
            cols = 30_000
            if distribution == "repeated":
                indices = np.tile(
                    np.linspace(0, cols - 1, stored, dtype=np.int32), rows
                )
            else:
                indices = rng.integers(0, cols, (rows, stored), dtype=np.int32)
                indices.sort(axis=1)
                indices = indices.ravel()
            pointers = np.arange(rows + 1, dtype=np.int32) * stored
            cuts = np.linspace(0, cols, ncuts, dtype=np.int32)
            for workers in (1, 4, 0):
                timings, reference = {}, None
                for label, module in (("legacy", legacy), ("rust", rust)):
                    output = np.empty((ncuts, rows), dtype=np.int32)
                    previous = module._set_host_worker_limit(workers)
                    try:
                        timings[label] = measure(
                            lambda: module.csr_row_boundaries_host(
                                indices, pointers, cuts, output, n_cols=cols
                            ),
                            args,
                        )
                    finally:
                        module._set_host_worker_limit(previous)
                    if reference is None:
                        reference = output.copy()
                    else:
                        np.testing.assert_array_equal(output, reference)
                record(
                    results,
                    {
                        "operation": "csr_boundaries",
                        "rows": rows,
                        "stored_per_row": stored,
                        "cuts": ncuts,
                        "workers": workers,
                        "distribution": distribution,
                    },
                    timings,
                )


def sparse_ranks(args, rust, legacy, results):
    import cupy as cp
    import cupyx.scipy.sparse as device_sparse
    import scipy.sparse as sparse

    # The archived module's scratch callbacks must outlive all legacy calls.
    allocations = {}

    def allocate(size):
        memory = cp.cuda.alloc(size)
        allocations[memory.ptr] = memory
        return memory.ptr

    legacy._set_scratch_allocator(allocate, allocations.pop)
    # The archived extension retains callbacks in C++ static storage. Release
    # captured CuPy state while Python is alive, before its static teardown.
    atexit.register(legacy._set_scratch_allocator, int, int)
    rng = np.random.default_rng(42)
    for rows in args.sparse_rows:
        cols, density = 1000, args.sparse_density
        host = sparse.random(
            rows,
            cols,
            density=density,
            format="csc",
            random_state=rng,
            dtype=np.float32,
        )
        # The legacy OVO analytic-zero fast path requires nonnegative values.
        # Signed sparse correctness is checked separately against SciPy ranks.
        host.data = rng.integers(0, 8, host.nnz).astype(np.float32)
        device = device_sparse.csc_matrix(host)
        host_csr = host.tocsr()
        device_csr = device_sparse.csr_matrix(host_csr)
        nref = rows // 4
        ref_map = np.full(rows, -1, dtype=np.int32)
        ref_map[:nref] = np.arange(nref, dtype=np.int32)
        grp_map = np.full(rows, -1, dtype=np.int32)
        grp_map[nref:] = np.arange(rows - nref, dtype=np.int32)
        offsets = np.arange(4, dtype=np.int32) * nref
        d_ref, d_grp, d_offsets = map(cp.asarray, (ref_map, grp_map, offsets))
        ref_ids = np.arange(nref, dtype=np.int32)
        group_ids = np.arange(nref, rows, dtype=np.int32)
        d_ref_ids, d_group_ids = cp.asarray(ref_ids), cp.asarray(group_ids)
        codes = np.arange(rows, dtype=np.int32) % 5
        sizes = np.bincount(codes).astype(np.float64)
        d_codes, d_sizes = map(cp.asarray, (codes, sizes))
        stat_codes = np.r_[np.full(nref, 3), np.repeat(np.arange(3), nref)].astype(
            np.int32
        )
        for operation in ("ovr", "ovo"):
            for on_host, format in (
                (False, "csc"),
                (True, "csc"),
                (False, "csr"),
                (True, "csr"),
            ):
                if args.sparse_device_only and on_host:
                    continue
                host_source = host if format == "csc" else host_csr
                device_source = device if format == "csc" else device_csr
                timings, reference = {}, None
                for label, module in (("legacy", legacy), ("rust", rust)):
                    groups = 5 if operation == "ovr" else 3
                    ranks = cp.empty((groups, cols), dtype=np.float64)
                    ties = cp.empty(
                        cols if operation == "ovr" else (groups, cols), dtype=np.float64
                    )
                    sums = cp.empty(
                        (groups if operation == "ovr" else 4, cols), dtype=np.float64
                    )
                    counts = cp.empty_like(sums)
                    totals, total_counts = (
                        cp.empty(
                            (1, cols) if format == "csr" else cols, dtype=np.float64
                        ),
                        cp.empty(
                            (1, cols) if format == "csr" else cols, dtype=np.float64
                        ),
                    )

                    def run():
                        common = {"compute_tie_corr": True, "sub_batch_cols": 64}
                        if operation == "ovr" and on_host and format == "csr":
                            module.ovr_sparse_csr_host(
                                host_source.data,
                                host_source.indices,
                                host_source.indptr,
                                host_source.indptr[:-1],
                                host_source.indptr[1:],
                                codes,
                                sizes,
                                ranks,
                                ties,
                                sums,
                                counts,
                                totals,
                                total_counts,
                                n_cols=cols,
                                compute_nnz=True,
                                compute_totals=True,
                                **common,
                            )
                        elif operation == "ovr" and on_host:
                            module.ovr_sparse_csc_host(
                                host.data,
                                host.indices,
                                host.indptr,
                                codes,
                                sizes,
                                ranks,
                                ties,
                                sums,
                                counts,
                                totals,
                                total_counts,
                                compute_nnz=True,
                                compute_totals=True,
                                **common,
                            )
                        elif operation == "ovr":
                            getattr(module, f"ovr_sparse_{format}_device")(
                                device_source.data,
                                device_source.indices,
                                device_source.indptr,
                                d_codes,
                                d_sizes,
                                ranks,
                                ties,
                                **common,
                            )
                        elif on_host and format == "csr":
                            module.ovo_streaming_csr_host(
                                host_source.data,
                                host_source.indices,
                                host_source.indptr[:-1],
                                host_source.indptr[1:],
                                ref_ids,
                                group_ids,
                                offsets,
                                ranks,
                                ties,
                                sums,
                                counts,
                                n_cols=cols,
                                compute_nnz=True,
                                analytic_zeros=True,
                                **common,
                            )
                        elif on_host:
                            module.ovo_streaming_csc_host(
                                host.data,
                                host.indices,
                                host.indptr,
                                ref_map,
                                grp_map,
                                offsets,
                                stat_codes,
                                ranks,
                                ties,
                                sums,
                                counts,
                                n_ref=nref,
                                n_all_grp=rows - nref,
                                compute_nnz=True,
                                analytic_zeros=True,
                                **common,
                            )
                        else:
                            getattr(module, f"ovo_streaming_{format}_device")(
                                device_source.data,
                                device_source.indices,
                                device_source.indptr,
                                d_ref if format == "csc" else d_ref_ids,
                                d_grp if format == "csc" else d_group_ids,
                                d_offsets,
                                ranks,
                                ties,
                                n_ref=nref,
                                n_all_grp=rows - nref,
                                **common,
                            )

                    timings[label] = measure(
                        run, args, synchronize=cp.cuda.get_current_stream().synchronize
                    )
                    actual = [ranks.get(), ties.get()]
                    if on_host:
                        actual.extend((sums.get(), counts.get()))
                        if operation == "ovr":
                            actual.extend((totals.get(), total_counts.get()))
                    if reference is None:
                        reference = actual
                    else:
                        for value, expected in zip(actual, reference, strict=True):
                            np.testing.assert_allclose(
                                value, expected, rtol=1e-10, atol=1e-8
                            )
                record(
                    results,
                    {
                        "operation": f"sparse_{operation}",
                        "rows": rows,
                        "cols": cols,
                        "density": density,
                        "host": on_host,
                        "format": format,
                    },
                    timings,
                )


def preprocessing(args, rust, legacy_directory, results, references):
    import cupy as cp

    old_norm = load(
        "legacy._norm_cuda",
        legacy_directory / "_norm_cuda.abi3.so",
        references=references,
    )
    old_qc = load(
        "legacy._qc_cuda", legacy_directory / "_qc_cuda.abi3.so", references=references
    )
    sync = cp.cuda.get_current_stream().synchronize
    rng = np.random.default_rng(92)
    for rows, cols in ((100_000, 32), (10_000, 1000), (100, 30_000), (1, 1_000_000)):
        source = cp.asarray(rng.uniform(0.1, 5, (rows, cols)).astype(np.float32))
        for sparse in (False, True):
            pointers = cp.arange(rows + 1, dtype=np.int64) * cols
            target = cp.empty_like(source)
            timings, reference = {}, None
            for label, module in (("legacy", old_norm), ("rust", rust._norm_cuda)):

                def run():
                    if sparse:
                        module.mul_csr(
                            pointers, target.ravel(), nrows=rows, target_sum=10_000.0
                        )
                    else:
                        module.mul_dense(
                            target, nrows=rows, ncols=cols, target_sum=10_000.0
                        )

                timings[label] = measure(
                    run,
                    args,
                    prepare=lambda: cp.copyto(target, source),
                    synchronize=sync,
                )
                if reference is None:
                    reference = target.get()
                else:
                    np.testing.assert_allclose(
                        target.get(), reference, rtol=2e-5, atol=1e-6
                    )
            record(
                results,
                {
                    "operation": "normalize_csr" if sparse else "normalize_dense",
                    "rows": rows,
                    "cols": cols,
                },
                timings,
            )
    for rows, cols in ((1000, 2000), (10_000, 2000)):
        source = cp.asarray(rng.integers(0, 8, (rows, cols)).astype(np.float32))
        outputs = [
            cp.empty(rows, dtype=np.float32),
            cp.empty(cols, dtype=np.float32),
            cp.empty(rows, dtype=np.int32),
            cp.empty(cols, dtype=np.int32),
        ]
        timings, reference = {}, None
        for label, module in (("legacy", old_qc), ("rust", rust._qc_cuda)):

            def run():
                module.sparse_qc_dense(
                    source,
                    sums_cells=outputs[0],
                    sums_genes=outputs[1],
                    cell_ex=outputs[2],
                    gene_ex=outputs[3],
                    n_cells=rows,
                    n_genes=cols,
                )

            def reset():
                for output in outputs:
                    output.fill(0)

            timings[label] = measure(run, args, prepare=reset, synchronize=sync)
            actual = [output.get() for output in outputs]
            if reference is None:
                reference = actual
            else:
                for value, expected in zip(actual, reference, strict=True):
                    np.testing.assert_array_equal(value, expected)
        record(results, {"operation": "qc_dense", "rows": rows, "cols": cols}, timings)
    preprocessing_families(args, rust, legacy_directory, results, references)


def preprocessing_families(args, rust, legacy_directory, results, references):
    """Exercise every other archived preprocessing family and specialization."""
    import cupy as cp
    import scipy.sparse as sp

    modules = {}
    sync = cp.cuda.get_current_stream().synchronize
    rng = np.random.default_rng(329)
    rows, cols = 4097, 257
    host = sp.random(rows, cols, density=0.05, format="csr", random_state=rng)
    host.data = rng.integers(1, 12, host.nnz)
    for dtype in (np.float32, np.float64):
        csr = host.astype(dtype)
        csc = csr.tocsc()
        dense = cp.asarray(csr.toarray())
        dense_f = cp.asfortranarray(dense)
        p, index, data = map(cp.asarray, (csr.indptr, csr.indices, csr.data))
        pc, ic, dc = map(cp.asarray, (csc.indptr, csc.indices, csc.data))
        mask = cp.arange(cols) % 3 != 0
        row_mask = (cp.arange(rows) % 3 != 0).astype(cp.int32)
        nan_data = data.copy()
        nan_data[::17] = cp.nan
        std = cp.linspace(0.25, 3, cols, dtype=dtype)
        means = cp.linspace(0.1, 2, cols, dtype=dtype)
        cells = dense.sum(axis=1)
        genes = dense.sum(axis=0)
        residual_kwargs = {
            "sums_cells": cells,
            "sums_genes": genes,
            "inv_sum_total": 1 / float(cells.sum().item()),
            "clip": 8.0,
            "inv_theta": 0.01,
            "n_cells": rows,
            "n_genes": cols,
        }

        def compare(module_name, operation, call, outputs, *, reset=None):
            if module_name not in modules:
                modules[module_name] = load(
                    f"legacy.{module_name}",
                    legacy_directory / f"{module_name}.abi3.so",
                    references=references,
                )
            timings, reference = {}, None

            def prepare():
                for output in outputs:
                    output.fill(0)
                if reset is not None:
                    reset()

            for label, module in (
                ("legacy", modules[module_name]),
                ("rust", getattr(rust, module_name)),
            ):
                timings[label] = measure(
                    lambda: call(module), args, prepare=prepare, synchronize=sync
                )
                actual = [output.get() for output in outputs]
                if reference is None:
                    reference = actual
                else:
                    for value, expected in zip(actual, reference, strict=True):
                        np.testing.assert_allclose(
                            value,
                            expected,
                            rtol=3e-6 if dtype == np.float32 else 1e-12,
                            atol=3e-6 if dtype == np.float32 else 1e-12,
                        )
            record(
                results,
                {
                    "operation": operation,
                    "rows": rows,
                    "cols": cols,
                    "dtype": np.dtype(dtype).name,
                },
                timings,
            )

        for major in (True, False):
            length = rows if major else cols
            output = cp.empty(length, dtype=cp.float64)
            variance = cp.empty_like(output)
            nans = cp.empty(length, dtype=cp.int32)

            def moments(module):
                if major:
                    module.mean_var_major(
                        p, index, data, output, variance, major=rows, minor=cols
                    )
                else:
                    module.mean_var_minor(index, data, output, variance, nnz=csr.nnz)

            def nanmean(module):
                if major:
                    module.nan_mean_major(
                        p,
                        index,
                        nan_data,
                        means=output,
                        nans=nans,
                        mask=mask,
                        major=rows,
                        minor=cols,
                    )
                else:
                    module.nan_mean_minor(
                        index, nan_data, means=output, nans=nans, mask=mask, nnz=csr.nnz
                    )

            axis = "major" if major else "minor"
            compare("_mean_var_cuda", f"mean_var_{axis}", moments, [output, variance])
            compare("_nanmean_cuda", f"nan_mean_{axis}", nanmean, [output, nans])

        for layout in ("csc", "csr", "dense", "centered"):
            original = dc if layout == "csc" else data if layout == "csr" else dense
            target = cp.empty_like(original)

            def scale(module):
                if layout == "csc":
                    module.csc_scale_diff(pc, target, std, ncols=cols)
                elif layout == "csr":
                    module.csr_scale_diff(
                        p, index, target, std, row_mask, clipper=4.0, nrows=rows
                    )
                elif layout == "dense":
                    module.dense_scale_diff(
                        target, std, row_mask, clipper=4.0, nrows=rows, ncols=cols
                    )
                else:
                    module.dense_scale_center_diff(
                        target,
                        means,
                        std,
                        row_mask,
                        clipper=4.0,
                        nrows=rows,
                        ncols=cols,
                    )

            compare(
                "_scale_cuda",
                f"scale_{layout}",
                scale,
                [target],
                reset=lambda: cp.copyto(target, original),
            )

        for layout in ("csr", "dense"):
            for axis in ("cells", "genes"):
                output = cp.empty(rows if axis == "cells" else cols, dtype=dtype)
                expressed = cp.empty(output.shape, dtype=cp.int32)

                def qc(module):
                    function = getattr(module, f"sparse_qc_{layout}_{axis}")
                    kwargs = {
                        f"sums_{axis}": output,
                        "cell_ex" if axis == "cells" else "gene_ex": expressed,
                    }
                    if layout == "dense":
                        function(dense, n_cells=rows, n_genes=cols, **kwargs)
                    elif axis == "cells":
                        function(p, index, data, n_cells=rows, **kwargs)
                    else:
                        function(index, data, nnz=csr.nnz, **kwargs)

                compare(
                    "_qc_dask_cuda", f"qc_dask_{layout}_{axis}", qc, [output, expressed]
                )

        output = cp.empty(cols, dtype=dtype)
        scaled_means = genes / cells.sum()
        compare(
            "_hvg_cuda",
            "expected_zeros",
            lambda module: module.expected_zeros(
                scaled_means, cells, output, cols, rows
            ),
            [output],
        )
        for layout in ("csr", "csc", "dense", "csc_hvg", "dense_hvg"):
            output = cp.empty(
                cols if layout.endswith("hvg") else (rows, cols), dtype=dtype
            )

            def residual(module):
                kwargs = dict(residual_kwargs, residuals=output)
                if layout == "csr":
                    module.sparse_norm_res_csr(p, index, data, **kwargs)
                elif layout == "csc":
                    module.sparse_norm_res_csc(pc, ic, dc, **kwargs)
                elif layout == "dense":
                    module.dense_norm_res(dense, **kwargs)
                elif layout == "csc_hvg":
                    module.csc_hvg_res(pc, ic, dc, **kwargs)
                else:
                    module.dense_hvg_res(dense_f, **kwargs)

            compare("_pr_cuda", f"pearson_{layout}", residual, [output])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-library", type=Path, required=True)
    parser.add_argument("--legacy-directory", type=Path, required=True)
    parser.add_argument(
        "--section", action="append", choices=("host", "sparse", "preprocessing")
    )
    parser.add_argument("--sparse-rows", type=int, nargs="+", default=[10_000, 50_000])
    parser.add_argument("--sparse-density", type=float, default=0.01)
    parser.add_argument("--sparse-device-only", action="store_true")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.warmup < 0 or args.repeat < 1:
        parser.error("warmup must be nonnegative and repeat must be positive")
    if any(rows < 4 or rows % 4 for rows in args.sparse_rows):
        parser.error("sparse row counts must be positive multiples of four")
    if not 0 < args.sparse_density <= 1:
        parser.error("sparse density must be in (0, 1]")
    rust = load("native._rust_cuda", args.rust_library.resolve())
    if rust.__backend__ != "rust":
        parser.error("--rust-library must identify the Rust native backend")
    references = {}
    legacy = load(
        "legacy._wilcoxon_sparse_cuda",
        args.legacy_directory / "_wilcoxon_sparse_cuda.abi3.so",
        references=references,
    )
    results = []
    sections = args.section or ["host", "sparse", "preprocessing"]
    metadata = {
        "rust_library": str(args.rust_library.resolve()),
        "rust_library_sha256": hashlib.sha256(
            args.rust_library.read_bytes()
        ).hexdigest(),
        "legacy_directory": str(args.legacy_directory.resolve()),
        "references": references,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "repeat": args.repeat,
        "warmup": args.warmup,
        "sections": sections,
        "timing_note": "Wall time includes validation and completion; setup/reset copies are excluded.",
    }
    if "host" in sections:
        boundaries(args, rust._wilcoxon_sparse_cuda, legacy, results)
    if "sparse" in sections:
        sparse_ranks(args, rust._wilcoxon_sparse_cuda, legacy, results)
    if "preprocessing" in sections:
        preprocessing(args, rust, args.legacy_directory, results, references)
    if "sparse" in sections or "preprocessing" in sections:
        import cupy as cp

        properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())
        name = properties["name"]
        metadata.update(
            cupy=cp.__version__,
            gpu=name.decode() if isinstance(name, bytes) else name,
            driver=cp.cuda.runtime.driverGetVersion(),
            runtime=cp.cuda.runtime.runtimeGetVersion(),
        )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"metadata": metadata, "results": results}, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
