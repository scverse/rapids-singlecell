"""Compare utility and Pearson kernels with an archived native baseline.

Use --rust-library and optionally --legacy-directory containing archived native
extensions. CUDA graph replay isolates GPU work from Python launch overhead.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_domain_kernels import load_module, measure


def cases(cp, sparse):
    cp.random.seed(184)
    for k in (15, 32, 64):
        rows = 10_000
        knn = cp.random.randint(0, rows, (rows, k), dtype=cp.int32)
        knn[:, 0] = cp.arange(rows, dtype=cp.int32)
        yield (
            "jaccard",
            "jaccard_shared_counts",
            (knn,),
            {"n_obs": rows, "k": k, "jaccard_vals": cp.zeros(rows * k, cp.float32)},
            {"rows": rows, "k": k},
        )
    for dtype in (cp.float32, cp.float64):
        for fmt in ("csr", "csc"):
            rows, columns = 5000, 1000
            x = sparse.random(rows, columns, density=0.01, format=fmt, dtype=dtype)
            for order in ("C", "F"):
                major, minor = (rows, columns) if fmt == "csr" else (columns, rows)
                yield (
                    "sparse2dense",
                    "sparse2dense",
                    (x.indptr, x.indices, x.data),
                    {
                        "out": cp.zeros((rows, columns), dtype=dtype, order=order),
                        "major": major,
                        "minor": minor,
                        "c_switch": (fmt == "csr") == (order == "C"),
                        "max_nnz": int(cp.diff(x.indptr).max()),
                    },
                    {"format": fmt, "dtype": str(dtype), "order": order},
                )
    for rows, columns in ((1000, 1024), (50_000, 128)):
        codes = cp.arange(rows, dtype=cp.int32) % 5
        for dtype in (cp.float32, cp.float64):
            for fmt in ("dense", "csr", "csc"):
                metadata = {
                    "rows": rows,
                    "columns": columns,
                    "dtype": str(dtype),
                    "format": fmt,
                }
                kwargs = {
                    "n_cells": rows,
                    "n_genes": columns,
                    "n_groups": 5,
                    "n_bins": 256,
                    "bin_low": -3.0,
                    "inv_bin_width": 32.0,
                }
                hist = cp.zeros((columns, 5, 257), dtype=cp.uint32)
                if fmt == "dense":
                    x = cp.asfortranarray(
                        cp.random.normal(size=(rows, columns)).astype(dtype)
                    )
                    args = (x, codes, hist)
                else:
                    x = sparse.random(
                        rows, columns, density=0.1, dtype=dtype, format=fmt
                    )
                    args = (x.data, x.indices, x.indptr, codes, hist)
                    kwargs["gene_start"] = 0
                yield "wilcoxon_binned", f"{fmt}_hist", args, kwargs, metadata


def pearson_cases(cp, sparse):
    cp.random.seed(347)
    for rows, columns in ((4097, 257), (513, 1025)):
        for dtype in (cp.float32, cp.float64):
            csr = sparse.random(rows, columns, density=0.05, dtype=dtype, format="csr")
            csr.sort_indices()
            csr.data = cp.ceil(csr.data * 12)
            csc = csr.tocsc()
            dense = csr.toarray()
            cells, genes = dense.sum(axis=1), dense.sum(axis=0)
            common = {
                "sums_cells": cells,
                "sums_genes": genes,
                "n_cells": rows,
                "n_genes": columns,
                "inv_sum_total": 1 / float(cells.sum().item()),
                "inv_theta": 0.01,
                "clip": 8.0,
            }
            for width in (cp.int32, cp.int64):
                for layout in ("csr", "csc", "dense", "csc_hvg", "dense_hvg"):
                    variance = layout.endswith("hvg")
                    if layout.startswith("dense"):
                        if width == cp.int64:
                            continue
                        args = (cp.asfortranarray(dense) if variance else dense,)
                        function = "dense_hvg_res" if variance else "dense_norm_res"
                    else:
                        matrix = csr if layout == "csr" else csc
                        args = (
                            matrix.indptr.astype(width),
                            matrix.indices.astype(width),
                            matrix.data,
                        )
                        function = (
                            "csc_hvg_res" if variance else f"sparse_norm_res_{layout}"
                        )
                    output = cp.zeros(
                        columns if variance else (rows, columns), dtype=dtype
                    )
                    yield (
                        "pr",
                        function,
                        args,
                        dict(common, residuals=output),
                        {
                            "rows": rows,
                            "columns": columns,
                            "layout": layout,
                            "dtype": cp.dtype(dtype).name,
                            "index_dtype": cp.dtype(width).name,
                        },
                    )


def preprocessing_cases(cp, sparse):
    cp.random.seed(726)
    for dtype in (cp.float32, cp.float64):
        for rows, columns in (
            (100_000, 32),
            (10_000, 1000),
            (100, 30_000),
            (1, 1_000_000),
        ):
            source = cp.random.uniform(0.1, 5, (rows, columns)).astype(dtype)
            pointers = cp.arange(rows + 1, dtype=cp.int64) * columns
            metadata = {"rows": rows, "columns": columns, "dtype": str(dtype)}
            yield (
                "norm",
                "mul_dense",
                (source.copy(),),
                {"nrows": rows, "ncols": columns, "target_sum": 10_000.0},
                metadata,
            )
            yield (
                "norm",
                "mul_csr",
                (pointers, source.ravel().copy()),
                {"nrows": rows, "target_sum": 10_000.0},
                metadata,
            )
        for rows, columns in ((4097, 257), (513, 4097)):
            matrix = sparse.random(
                rows, columns, density=0.05, dtype=dtype, format="csr"
            )
            for width in (cp.int32, cp.int64):
                pointers, indices = (
                    matrix.indptr.astype(width),
                    matrix.indices.astype(width),
                )
                metadata = {
                    "rows": rows,
                    "columns": columns,
                    "dtype": str(dtype),
                    "index_dtype": str(width),
                }
                sums = cp.empty(rows, dtype=cp.float64)
                squares = cp.empty_like(sums)
                yield (
                    "mean_var",
                    "mean_var_major",
                    (pointers, indices, matrix.data, sums, squares),
                    {"major": rows, "minor": columns},
                    metadata,
                )
                values = matrix.data.copy()
                values[::17] = cp.nan
                yield (
                    "nanmean",
                    "nan_mean_major",
                    (pointers, indices, values),
                    {
                        "means": sums,
                        "nans": cp.empty(rows, dtype=cp.int32),
                        "mask": cp.arange(columns) % 3 != 0,
                        "major": rows,
                        "minor": columns,
                    },
                    metadata,
                )
                yield (
                    "qc_dask",
                    "sparse_qc_csr_cells",
                    (pointers, indices, matrix.data),
                    {
                        "sums_cells": cp.empty(rows, dtype=dtype),
                        "cell_ex": cp.empty(rows, dtype=cp.int32),
                        "n_cells": rows,
                    },
                    metadata,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rust-library", type=Path, required=True)
    parser.add_argument("--legacy-directory", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument(
        "--module",
        action="append",
        choices=(
            "jaccard",
            "sparse2dense",
            "wilcoxon_binned",
            "pr",
            "norm",
            "mean_var",
            "nanmean",
            "qc_dask",
        ),
    )
    options = parser.parse_args()
    if options.repeat <= 0:
        parser.error("--repeat must be positive")
    import cupy as cp
    import numpy as np
    from cupyx.profiler import benchmark
    from cupyx.scipy import sparse

    seed, graph_batch = 317, 5
    cp.random.seed(seed)
    native = load_module("_rust_cuda", options.rust_library.resolve())
    report = {
        "rust_library": str(options.rust_library.resolve()),
        "rust_sha256": hashlib.sha256(options.rust_library.read_bytes()).hexdigest(),
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "device": cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
            "name"
        ].decode(),
        "seed": seed,
        "repeat": options.repeat,
        "graph_batch": graph_batch,
        "references": {},
        "cases": [],
    }
    from itertools import chain

    for module, function, args, kwargs, metadata in chain(
        cases(cp, sparse), pearson_cases(cp, sparse), preprocessing_cases(cp, sparse)
    ):
        if options.module and module not in options.module:
            continue
        row = dict(module=module, function=function, **metadata)
        for label in ("rust", "legacy"):
            if label == "legacy":
                if options.legacy_directory is None:
                    continue
                reference_path = options.legacy_directory / f"_{module}_cuda.abi3.so"
                if module not in report["references"]:
                    report["references"][module] = {
                        "path": str(reference_path.resolve()),
                        "sha256": hashlib.sha256(
                            reference_path.read_bytes()
                        ).hexdigest(),
                    }
                backend = load_module(
                    f"_{module}_cuda",
                    reference_path,
                )
            else:
                backend = getattr(native, f"_{module}_cuda")
            row[label] = measure(
                cp,
                np,
                benchmark,
                getattr(backend, function),
                args,
                kwargs=kwargs,
                reference=label == "legacy",
                repeat=options.repeat,
                batch=graph_batch,
            )
        if "legacy" in row:
            row["kernel_ratio"] = row["rust"]["kernel_us"] / row["legacy"]["kernel_us"]
        report["cases"].append(row)
        print(json.dumps(row), flush=True)
    if options.output is not None:
        options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
