"""Compare domain kernel latency with CUDA graph replay and ordinary calls.

Run in a GPU development environment after building the Rust extension::

    python scripts/benchmark_domain_kernels.py --backend build/rust-migration/rust/_rust_cuda.abi3.so

An optional --reference-dir may point at archived native extension binaries.
Graph replay isolates GPU execution; ordinary calls include Python validation
and launch overhead. Run on an idle GPU and retain the emitted hardware metadata
alongside the results. This is an investigative benchmark, not a timing assertion.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import importlib.util
import json
import re
from pathlib import Path


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load native extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_cases(cp, np, sparse):
    cases = []

    def add(module, function, args, kwargs):
        cases.append((module, function, args, kwargs))

    cp.random.seed(40)
    x = cp.random.random((10000, 128), dtype=cp.float32)
    cats = cp.arange(10000, dtype=cp.int32) % 10
    mask = cp.ones(10000, cp.bool_)
    sums = cp.zeros((10, 128), cp.float64)
    add(
        "aggr",
        "dense_aggr",
        (x,),
        {
            "out_sum": sums,
            "cats": cats,
            "mask": mask,
            "n_cells": 10000,
            "n_genes": 128,
            "is_fortran": False,
        },
    )
    vals = cp.random.random(1000 * 200, dtype=cp.float32)
    ptr = cp.arange(1001, dtype=cp.int32) * 200
    out = cp.empty(1000, cp.float32)
    add(
        "bbknn",
        "find_top_k_per_row",
        (vals, ptr),
        {"n_rows": 1000, "trim": 100, "vals": out},
    )
    add(
        "bbknn",
        "find_top_k_per_row_sorted",
        (vals, ptr),
        {"n_rows": 1000, "trim": 100, "vals": out},
    )
    x = cp.random.random((2000, 50), dtype=cp.float32)
    co = cp.arange(5, dtype=cp.int32) * 500
    cells = cp.arange(2000, dtype=cp.int32)
    l, r = cp.triu_indices(4)
    l = l.astype(cp.int32)
    r = r.astype(cp.int32)
    sums = cp.zeros(len(l), cp.float32)
    add(
        "edistance",
        "compute_distances",
        (x, co, cells, l, r, sums),
        {
            "num_pairs": len(l),
            "n_features": 50,
            "blocks_per_pair": 8,
            "cell_tile": 32,
            "feat_tile": 32,
            "block_size": 256,
            "shared_mem": 32 * 32 * 4,
        },
    )
    for dtype, features, cell_tile, feat_tile in (
        (cp.float32, 50, 64, 25),
        (cp.float32, 51, 64, 16),
        (cp.float64, 50, 16, 50),
        (cp.float64, 67, 16, 64),
    ):
        groups = cp.asarray([0, 173, 573, 1100], dtype=cp.int32)
        cells = cp.arange(1100, dtype=cp.int32)
        left, right = (part.astype(cp.int32) for part in cp.triu_indices(3))
        dense = cp.random.random((1100, features)).astype(dtype)
        csr = sparse.csr_matrix(cp.where(dense < 0.8, 0, dense))
        # The advancing-cursor energy kernels consume canonical, sorted CSR.
        csr.sort_indices()
        for is_sparse in (False, True):
            operands = (csr.indptr, csr.indices, csr.data) if is_sparse else (dense,)
            sums = cp.zeros(left.size, dtype=dtype)
            add(
                "edistance",
                "compute_distances_sparse" if is_sparse else "compute_distances",
                (*operands, groups, cells, left, right, sums),
                {
                    "num_pairs": left.size,
                    "n_features": features,
                    "blocks_per_pair": 4,
                    "cell_tile": cell_tile,
                    "feat_tile": feat_tile,
                    "block_size": 512,
                    "shared_mem": cell_tile * feat_tile * cp.dtype(dtype).itemsize,
                },
            )
    a = sparse.random(1000, 1000, density=0.01, dtype=cp.float32, format="csr")
    adj = sparse.random(1000, 1000, density=0.005, dtype=cp.float32, format="csr")
    out = cp.zeros(1000, cp.float32)
    mean = cp.full(1000, 0.01, cp.float32)
    add(
        "autocorr",
        "morans_sparse",
        (adj.indptr, adj.indices, adj.data),
        {
            "data_row_ptr": a.indptr,
            "data_col_ind": a.indices,
            "data_values": a.data,
            "n_samples": 1000,
            "n_features": 1000,
            "mean_array": mean,
            "num": out,
        },
    )
    cp.random.seed(40)
    xy = cp.random.random((2000, 2), dtype=cp.float32)
    out = cp.empty(2000, cp.float32)
    add(
        "kde",
        "gaussian_kde_2d",
        (xy,),
        {"out": out, "n": 2000, "a": -1.0, "b": -1.0, "c": -1.0},
    )
    nr = 256
    nc = 512
    ng = 4
    cost = cp.random.random(ng * nr * nc, dtype=cp.float32)
    co = cp.arange(ng, dtype=cp.int64) * (nr * nc)
    n = cp.full(ng, nr, cp.int32)
    m = cp.full(ng, nc, cp.int32)
    f_offsets = cp.arange(ng, dtype=cp.int64) * nr
    go = cp.arange(ng, dtype=cp.int64) * nc
    f = cp.zeros(ng * nr, cp.float32)
    g = cp.zeros(ng * nc, cp.float32)
    e = cp.ones(ng, cp.float32)
    la = cp.full(ng, -np.log(nr), cp.float32)
    lb = cp.full(ng, -np.log(nc), cp.float32)
    cv = cp.zeros(ng, cp.int32)
    r2p = cp.repeat(cp.arange(ng, dtype=cp.int32), nr)
    c2p = cp.repeat(cp.arange(ng, dtype=cp.int32), nc)
    add(
        "sinkhorn",
        "update_g",
        (cost, co, n, m, f, f_offsets, g, go, c2p, e, lb, cv),
        {"omega": 1.0},
    )
    add(
        "sinkhorn",
        "update_f",
        (cost, co, m, g, go, f, f_offsets, r2p, e, la, cv),
        {"omega": 1.0},
    )
    cp.random.seed(40)
    k = 4
    n = 2000
    lval = 10
    xy = cp.random.random((n, 2), dtype=cp.float32)
    thr = cp.linspace(0, 2, lval, dtype=cp.float32)
    offs = cp.arange(k + 1, dtype=cp.int32) * (n // k)
    cells = cp.arange(n, dtype=cp.int32)
    left, right = cp.triu_indices(k)
    left = left.astype(cp.int32)
    right = right.astype(cp.int32)
    out = cp.zeros((k, k, lval), cp.uint64)
    add(
        "cooc",
        "count_csr_catpairs",
        (xy,),
        {
            "thresholds": thr,
            "cat_offsets": offs,
            "cell_indices": cells,
            "pair_left": left,
            "pair_right": right,
            "counts": out,
            "num_pairs": len(left),
            "k": k,
            "l_val": lval,
            "blocks_per_pair": 8,
            "cell_tile": 128,
            "block_size": 128,
            "shared_mem": 128 * 2 * 4 + 4 * 32 * 8,
        },
    )
    genes = 4
    n = 2000
    features = 30
    nv = 100
    total = genes * n
    x = cp.random.random((total, nv), dtype=cp.float32)
    ri = cp.arange(total, dtype=cp.int32)
    ci = cp.tile(cp.arange(features, dtype=cp.int32), genes)
    ng = cp.full(genes, n, cp.int32)
    kg = cp.full(genes, features, cp.int32)
    co = cp.arange(genes, dtype=cp.int32) * n
    f_offsets = cp.arange(genes, dtype=cp.int32) * features
    isg = cp.arange(total) % n < n // 2
    nt = ~isg
    pv = cp.empty(total, cp.float64)
    out = cp.zeros(total, cp.float32)
    add(
        "mixscale",
        "project_score",
        (x, nv, ri, ci, ng, kg, co, f_offsets, isg, nt, pv, out),
        {"n_genes": genes, "max_k": features, "do_scale": True},
    )
    n = 5000
    ng = 32
    x = cp.asfortranarray(cp.random.poisson(2, size=(n, ng)).astype(cp.float32))
    assign = cp.empty((ng, n), cp.bool_)
    threshold = cp.empty(ng, cp.float32)
    pars = [cp.empty(ng, cp.float32) for _ in range(4)]
    valid = cp.empty(ng, cp.bool_)
    nonzero = cp.empty(ng, cp.int32)
    maximum = cp.empty(ng, cp.int32)
    add(
        "guide_assignment",
        "fit_assign_dense",
        (x, assign, threshold, *pars, valid, nonzero, maximum),
        {
            "n_cells": n,
            "n_guides": ng,
            "max_iter": 20,
            "tol": 0.001,
            "posterior_threshold": 0.9,
        },
    )
    x = cp.random.random((100, 20000), dtype=cp.float64)
    y = cp.random.random(x.shape, dtype=cp.float64)
    out = cp.empty(100, cp.float64)
    add(
        "pseudobulk",
        "paired_squared",
        (x, y),
        {"out": out, "n_pairs": 100, "n_features": 20000},
    )
    x = cp.random.random((10000, 100), dtype=cp.float32)
    labels = cp.arange(10000, dtype=cp.int32) % 10
    sums = cp.zeros((100, 10), cp.float32)
    counts = cp.zeros((100, 10), cp.int32)
    add(
        "ligrec",
        "sum_count_dense",
        (x,),
        {
            "clusters": labels,
            "sum": sums,
            "count": counts,
            "rows": 10000,
            "cols": 100,
            "ncls": 10,
        },
    )
    a = sparse.random(2000, 500, density=0.05, dtype=cp.float32, format="csr")
    out = cp.zeros((500, 500), cp.float32)
    add(
        "spca",
        "gram_csr_upper",
        (a.indptr, a.indices, a.data),
        {"nrows": 2000, "ncols": 500, "out": out},
    )
    x = cp.random.random((10000, 50), dtype=cp.float32)
    pairs = cp.random.randint(0, 10000, (10000, 20), dtype=cp.uint32)
    out = cp.empty(pairs.shape, cp.float32)
    add(
        "nn_descent",
        "sqeuclidean",
        (x,),
        {
            "out": out,
            "pairs": pairs,
            "n_samples": 10000,
            "n_features": 50,
            "n_neighbors": 20,
        },
    )
    for dtype in (cp.float32, cp.float64, cp.int32):
        for rows, cols in ((1000, 50), (100000, 10), (100000, 50)):
            x = cp.ones((rows, cols), dtype=dtype)
            out = cp.zeros(cols, dtype=dtype)
            for function in ("colsum", "colsum_atomic"):
                add(
                    "harmony_colsum",
                    function,
                    (x,),
                    {"out": out, "rows": rows, "cols": cols},
                )
    for dtype in (cp.float32, cp.float64):
        for size, left_offset, right_offset in (
            (1001, 1, 1),
            (5000000, 0, 0),
            (5000000, 1, 2),
        ):
            r = cp.full(size + 4, 0.1, dtype=dtype)[left_offset : left_offset + size]
            dot = cp.full(size + 4, 0.7, dtype=dtype)[
                right_offset : right_offset + size
            ]
            out = cp.zeros(1, dtype=dtype)
            add(
                "harmony_kmeans",
                "kmeans_err",
                (r,),
                {"dot": dot, "n": size, "out": out},
            )
        for rows, cols in ((100000, 50), (128, 4096)):
            x = cp.ones((rows, cols), dtype=dtype)
            out = cp.empty_like(x)
            add(
                "harmony_normalize",
                "l2_row_normalize",
                (x,),
                {"dst": out, "n_rows": rows, "n_cols": cols},
            )
        for rows, cols in ((100000, 50), (128, 4096)):
            similarities = cp.full((rows, cols), 0.7, dtype=dtype)
            penalty = cp.full((8, cols), 0.8, dtype=dtype)
            indices = cp.arange(rows, dtype=cp.int32)
            out = cp.empty_like(similarities)
            for covariates in (1, 2, 3, 4):
                cats = cp.arange(rows * covariates, dtype=cp.int32) % 8
                add(
                    "harmony_pen",
                    "fused_pen_norm_int",
                    (similarities,),
                    {
                        "penalty": penalty,
                        "cats": cats,
                        "idx_in": indices,
                        "R_out": out,
                        "term": -2.0,
                        "n_rows": rows,
                        "n_cols": cols,
                        "n_covariates": covariates,
                    },
                )
        for dimensions in (16, 50, 67, 128):
            cells, components = 2048, 4
            x = cp.random.standard_normal((cells, dimensions)).astype(dtype)
            means = cp.zeros((components, dimensions), dtype=dtype)
            precision = cp.broadcast_to(
                cp.eye(dimensions, dtype=dtype), (components, dimensions, dimensions)
            ).copy()
            weights = cp.full(components, 1 / components, dtype=dtype)
            logdet = cp.zeros(components, dtype=dtype)
            logprob = cp.empty((cells, components), dtype=dtype)
            resp = cp.empty_like(logprob)
            likelihood = cp.empty(cells, dtype=dtype)
            add(
                "gmm",
                "e_step",
                (x, weights, means, precision, logdet, logprob, resp, likelihood),
                {"n": cells, "d": dimensions, "K": components},
            )
        for cells in (256, 100000):
            genes = 4
            values = cp.random.standard_normal(genes * cells).astype(dtype)
            offsets = cp.arange(genes + 1, dtype=cp.int32) * cells
            zero = cp.zeros(genes, dtype=dtype)
            one = cp.ones(genes, dtype=dtype)
            initial = cp.full(genes, 2, dtype=dtype)
            resp = cp.empty_like(values)
            mean, variance, weight = (cp.empty_like(one) for _ in range(3))
            add(
                "gmm",
                "spherical_gmm_fit_batched",
                (
                    values,
                    offsets,
                    zero,
                    one,
                    initial,
                    one,
                    resp,
                    mean,
                    variance,
                    weight,
                ),
                {"n_genes": genes, "max_iter": 20, "tol": 1e-4, "reg_covar": 1e-6},
            )
    for dtype in (cp.float32, cp.float64):
        for features in (50, 128, 129):
            rows, cols = 257, 513
            embedding = cp.random.standard_normal((rows + cols, features)).astype(dtype)
            left = cp.arange(rows, dtype=cp.int32)
            right = cp.arange(rows, rows + cols, dtype=cp.int32)
            zero = cp.zeros(1, dtype=cp.int64)
            row_count = cp.asarray([rows], dtype=cp.int32)
            col_count = cp.asarray([cols], dtype=cp.int32)
            tile_rows, tile_cols = cp.meshgrid(
                cp.arange(0, rows, 16, dtype=cp.int32),
                cp.arange(0, cols, 16, dtype=cp.int32),
                indexing="ij",
            )
            tile_rows = tile_rows.ravel()
            tile_cols = tile_cols.ravel()
            pairs = cp.zeros(tile_rows.size, dtype=cp.int32)
            output = cp.empty(rows * cols, dtype=dtype)
            add(
                "sinkhorn",
                "build_cost",
                (
                    embedding,
                    left,
                    zero,
                    right,
                    zero,
                    row_count,
                    col_count,
                    zero,
                    pairs,
                    tile_rows,
                    tile_cols,
                    output,
                ),
                {},
            )
        for features in (3072, 3073):
            rows = 300
            data = sparse.random(
                rows, features, density=0.01, dtype=dtype, format="csr"
            )
            adjacency = sparse.random(
                rows, rows, density=0.01, dtype=dtype, format="csr"
            )
            mean = cp.full(features, 0.01, dtype=dtype)
            output = cp.zeros(features, dtype=dtype)
            add(
                "autocorr",
                "morans_sparse",
                (adjacency.indptr, adjacency.indices, adjacency.data),
                {
                    "data_row_ptr": data.indptr,
                    "data_col_ind": data.indices,
                    "data_values": data.data,
                    "n_samples": rows,
                    "n_features": features,
                    "mean_array": mean,
                    "num": output,
                },
            )
        xy = cp.random.random((2000, 2)).astype(dtype)
        output = cp.empty(2000, dtype=dtype)
        add(
            "kde",
            "gaussian_kde_2d",
            (xy,),
            {"out": output, "n": 2000, "a": -1.0, "b": -1.0, "c": -1.0},
        )
    for dtype in (cp.float32, cp.float64):
        for cells in (256, 10000):
            genes, features = 4, 30
            data = cp.random.standard_normal((genes * cells, features)).astype(dtype)
            data_offsets = cp.arange(genes, dtype=cp.int64) * cells * features
            counts = cp.full(genes, cells, dtype=cp.int32)
            widths = cp.full(genes, features, dtype=cp.int32)
            cell_offsets = cp.arange(genes, dtype=cp.int32) * cells
            feature_offsets = cp.arange(genes, dtype=cp.int32) * features
            control_mean = cp.zeros(genes * features, dtype=dtype)
            guide = cp.arange(genes * cells) % cells >= cells // 2
            control = ~guide
            active = cp.arange(genes, dtype=cp.int32)
            projected = cp.empty(genes * cells, dtype=dtype)
            response = cp.empty_like(projected)
            add(
                "gmm",
                "mixscape_project_em",
                (
                    data.ravel(),
                    data_offsets,
                    counts,
                    widths,
                    cell_offsets,
                    feature_offsets,
                    control_mean,
                    guide,
                    control,
                    active,
                    projected,
                    response,
                ),
                {
                    "n_active": genes,
                    "max_k": features,
                    "max_iter": 20,
                    "tol": 1e-4,
                    "reg_covar": 1e-6,
                },
            )
    for dtype in (cp.float32, cp.float64):
        rows, features = 300, 1000
        adjacency = sparse.random(rows, rows, density=0.01, dtype=dtype, format="csr")
        values = cp.random.random((rows, features)).astype(dtype)
        for geary in (False, True):
            add(
                "autocorr",
                "gearys_dense" if geary else "morans_dense",
                (values,),
                {
                    "adj_row_ptr": adjacency.indptr,
                    "adj_col_ind": adjacency.indices,
                    "adj_data": adjacency.data,
                    "num": cp.zeros(features, dtype),
                    "n_samples": rows,
                    "n_features": features,
                },
            )
        values = sparse.random(rows, 3073, density=0.01, dtype=dtype, format="csr")
        add(
            "autocorr",
            "gearys_sparse",
            (adjacency.indptr, adjacency.indices, adjacency.data),
            {
                "data_row_ptr": values.indptr,
                "data_col_ind": values.indices,
                "data_values": values.data,
                "n_samples": rows,
                "n_features": 3073,
                "num": cp.zeros(3073, dtype),
            },
        )
        for rows in (100000, 500000):
            features, clusters, batches = 50, 10, 5
            values = cp.ones((rows, features), dtype=dtype)
            common = {
                "R": cp.full((rows, clusters), 1 / clusters, dtype=dtype),
                "O": cp.full(
                    (batches, clusters), rows / batches / clusters, dtype=dtype
                ),
                "cats": cp.arange(rows, dtype=cp.int32) // (rows // batches),
                "cat_offsets": cp.arange(0, rows + 1, rows // batches, dtype=cp.int32),
                "cell_indices": cp.arange(rows, dtype=cp.int32),
                "lambda_kb": cp.ones((batches, clusters), dtype=dtype),
                "n_cells": rows,
                "n_pcs": features,
                "n_clusters": clusters,
                "n_batches": batches,
                "Z": cp.empty_like(values),
                "handle": cp.cuda.device.get_cublas_handle(),
            }
            add(
                "harmony_correction",
                "correction_fast",
                (values,),
                dict(
                    common,
                    inv_mat=cp.empty((batches + 1, batches + 1), dtype),
                    R_col=cp.empty(rows, dtype),
                    Phi_t_diag_R_X=cp.empty((batches + 1, features), dtype),
                    W=cp.empty((batches + 1, features), dtype),
                    g_factor=cp.empty(batches, dtype),
                    g_P_row0=cp.empty(batches, dtype),
                ),
            )
            if rows == 100000:
                chunk = 8192
                add(
                    "harmony_correction_batched",
                    "correction_batched",
                    (values,),
                    dict(
                        common,
                        inv_mats=cp.empty((clusters, batches + 1, batches + 1), dtype),
                        Phi_t_diag_R_X_all=cp.empty(
                            (clusters, batches + 1, features), dtype
                        ),
                        W_all=cp.empty((clusters, batches + 1, features), dtype),
                        g_factor=cp.empty((clusters, batches), dtype),
                        g_P_row0=cp.empty((clusters, batches), dtype),
                        X_batch=cp.empty((chunk, features), dtype),
                        R_batch=cp.empty((chunk, clusters), dtype),
                        batch_chunk_size=chunk,
                    ),
                )
    # Exercise the archived category-size launch tiers and threshold windows.
    # Configurations fit the original 48 KiB shared-memory floor; Rust accepts
    # the same tuning inputs so both implementations process identical pairs.
    for rows, categories, thresholds_count, block in (
        (5000, 2, 32, 256),
        (10000, 2, 32, 512),
        (20000, 2, 32, 1024),
        (2048, 8, 257, 128),
        (2048, 8, 1025, 128),
    ):
        coordinates = cp.random.random((rows, 2), dtype=cp.float32)
        left, right = cp.triu_indices(categories)
        padded = ((thresholds_count + 31) // 32) * 32
        add(
            "cooc",
            "count_csr_catpairs",
            (coordinates,),
            {
                "thresholds": cp.linspace(0, 2, thresholds_count, dtype=cp.float32),
                "cat_offsets": cp.arange(categories + 1, dtype=cp.int32)
                * (rows // categories),
                "cell_indices": cp.arange(rows, dtype=cp.int32),
                "pair_left": left.astype(cp.int32),
                "pair_right": right.astype(cp.int32),
                "counts": cp.zeros(
                    (categories, categories, thresholds_count), cp.uint64
                ),
                "num_pairs": len(left),
                "k": categories,
                "l_val": thresholds_count,
                "blocks_per_pair": 32,
                "cell_tile": 1024,
                "block_size": block,
                "shared_mem": 1024 * 2 * 4 + (block // 32) * padded * 8,
            },
        )
    # max_iter=0 isolates projection, initial moments and final posterior from
    # repeated EM iterations while preserving the public production pipeline.
    for module, function, args, kwargs in tuple(cases):
        if module == "gmm" and function == "mixscape_project_em":
            add(module, function, args, dict(kwargs, max_iter=0))
    # Complete NN-descent metric coverage and exercise longer feature scans.
    for features in (50, 513, 4096):
        rows, neighbors = 1024, 20
        values = cp.random.random((rows, features), dtype=cp.float32)
        pairs = cp.random.randint(0, rows, (rows, neighbors), dtype=cp.uint32)
        for metric in ("sqeuclidean", "cosine", "inner"):
            add(
                "nn_descent",
                metric,
                (values,),
                {
                    "out": cp.empty(pairs.shape, dtype=cp.float32),
                    "pairs": pairs,
                    "n_samples": rows,
                    "n_features": features,
                    "n_neighbors": neighbors,
                },
            )
    for rows, width in ((10000, 100), (128, 4096)):
        values = cp.random.random((rows, width), dtype=cp.float64)
        add(
            "pv",
            "rev_cummin64",
            (values,),
            {"out": cp.empty_like(values), "n_rows": rows, "m": width},
        )
    for rows, features, sets, length in ((4096, 1024, 100, 50), (2048, 4096, 16, 500)):
        cutoff = features // 20
        values = cp.random.randint(1, features + 1, (rows, features), dtype=cp.int32)
        add(
            "aucell",
            "auc",
            (values,),
            {
                "R": rows,
                "C": features,
                "cnct": cp.random.randint(0, features, sets * length, dtype=cp.int32),
                "starts": cp.arange(sets, dtype=cp.int32) * length,
                "lens": cp.full(sets, length, dtype=cp.int32),
                "n_sets": sets,
                "n_up": cutoff,
                "max_aucs": cp.full(sets, length * cutoff / 2, dtype=cp.float32),
                "es": cp.empty((rows, sets), dtype=cp.float32),
            },
        )
    from scipy import sparse as host_sparse

    generator = np.random.default_rng(441)
    for dtype in (np.float32, np.float64):
        values = generator.integers(-2, 8, (10000, 128)).astype(dtype)
        values[generator.random(values.shape) < 0.8] = 0
        groups, bins = 8, 10
        labels = cp.arange(len(values), dtype=cp.int32) % groups
        for layout in ("C", "F", "csr", "csc"):
            host = (
                np.array(values, order=layout)
                if layout in ("C", "F")
                else getattr(host_sparse, f"{layout}_matrix")(values)
            )
            outputs = [
                cp.zeros((groups, values.shape[1]), cp.float64) for _ in range(3)
            ]
            arguments = (
                (host, labels)
                if layout in ("C", "F")
                else (host.data, host.indices, host.indptr, labels)
            )
            common = (
                {}
                if layout in ("C", "F")
                else {
                    "n_cells": len(values),
                    "n_genes": values.shape[1],
                }
            )
            axis = "rows" if layout == "csr" else "cols"
            add(
                "rank_stream",
                "aggr_dense_host" if layout in ("C", "F") else f"aggr_{layout}_host",
                arguments,
                {
                    **common,
                    "out_sum": outputs[0],
                    "out_count": outputs[1],
                    "out_sqsum": outputs[2],
                    **(
                        {"sub_batch": 4096 if layout == "C" else 32}
                        if layout in ("C", "F")
                        else {f"sub_batch_{axis}": 4096 if layout == "csr" else 32}
                    ),
                },
            )
            first, last = 7, 121
            add(
                "rank_stream",
                "hist_dense_host" if layout in ("C", "F") else f"hist_{layout}_host",
                (*arguments, cp.zeros((last - first, groups, bins + 1), cp.uint32)),
                {
                    **common,
                    "group_sums": outputs[0],
                    "group_nnz": outputs[1],
                    "n_groups": groups,
                    "n_bins": bins,
                    "bin_low": -2.0,
                    "inv_bin_width": 1.0,
                    "col_start": first,
                    "col_stop": last,
                    f"sub_batch_{axis}": 4096 if layout == "csr" else 32,
                },
            )
    # Exercise the projection dispatch boundary and shared-vector budget with
    # few active genes; the original kernel uses a full block at every width.
    for dtype in (cp.float32, cp.float64):
        for features in (129, 257, 1025):
            genes, cells = 4, 256
            data = cp.random.standard_normal(genes * cells * features).astype(dtype)
            guide = cp.arange(genes * cells) % cells >= cells // 2
            add(
                "gmm",
                "mixscape_project_em",
                (
                    data,
                    cp.arange(genes, dtype=cp.int64) * cells * features,
                    cp.full(genes, cells, cp.int32),
                    cp.full(genes, features, cp.int32),
                    cp.arange(genes, dtype=cp.int32) * cells,
                    cp.arange(genes, dtype=cp.int32) * features,
                    cp.zeros(genes * features, dtype),
                    guide,
                    ~guide,
                    cp.arange(genes, dtype=cp.int32),
                    cp.empty(genes * cells, dtype),
                    cp.empty(genes * cells, dtype),
                ),
                {
                    "n_active": genes,
                    "max_k": features,
                    "max_iter": 0,
                    "tol": 1e-4,
                    "reg_covar": 1e-6,
                },
            )
    return cases


class CapturedDLPack:
    """Borrow an owned CuPy buffer without synchronizing during graph capture.

    The reference nanobind binaries import DLPack. Their default import asks
    CuPy to synchronize, which is forbidden during capture. Every operand is
    already on the capture stream and remains owned by the benchmark case.
    """

    def __init__(self, array):
        self.array = array

    def __dlpack_device__(self):
        return self.array.__dlpack_device__()

    def __dlpack__(self, *args, **kwargs):
        return self.array.__dlpack__(stream=-1)


def measure(
    cp,
    np,
    benchmark,
    function,
    args,
    *,
    kwargs,
    reference,
    repeat,
    batch,
    capture=True,
    pass_stream=True,
):
    # Inputs may have been initialized on the caller's stream. Establish
    # readiness explicitly before timing a separate nonblocking stream.
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)

    stream_options = {"stream": stream.ptr} if pass_stream else {}

    def call():
        return function(*args, **kwargs, **stream_options)

    def capture_arg(value):
        if reference and isinstance(value, cp.ndarray):
            return CapturedDLPack(value)
        return value

    capture_args = tuple(capture_arg(value) for value in args)
    capture_kwargs = {key: capture_arg(value) for key, value in kwargs.items()}
    with stream:
        for _ in range(3):
            call()
        stream.synchronize()
        kernel_us = None
        if capture:
            stream.begin_capture()
            for _ in range(batch):
                function(*capture_args, **capture_kwargs, **stream_options)
            graph = stream.end_capture()
            replay = benchmark(
                lambda: graph.launch(stream), n_repeat=repeat, n_warmup=3
            )
            kernel_us = float(np.median(replay.gpu_times)) * 1e6 / batch
        ordinary = benchmark(call, n_repeat=repeat, n_warmup=3)
    return {
        "kernel_us": kernel_us,
        "call_gpu_us": float(np.median(ordinary.gpu_times)) * 1e6,
        "call_cpu_us": float(np.median(ordinary.cpu_times)) * 1e6,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument("--repeat", type=int, default=30)
    parser.add_argument("--graph-batch", type=int, default=10)
    parser.add_argument("--module", action="append", default=[])
    parser.add_argument("--case", action="append", type=int, default=[])
    parser.add_argument(
        "--skip-reference-case",
        action="append",
        type=int,
        default=[],
        help="Zero-based case index whose archived reference cannot execute on this GPU",
    )
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    if options.repeat <= 0 or options.graph_batch <= 0:
        parser.error("--repeat and --graph-batch must be positive")
    import cupy as cp
    import numpy as np
    from cupyx.profiler import benchmark
    from cupyx.scipy import sparse

    native = load_module("_rust_cuda", options.backend.resolve())
    properties = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
    device_name = properties["name"]
    if isinstance(device_name, bytes):
        device_name = device_name.decode()
    backend_bytes = options.backend.read_bytes()
    report = {
        "device": device_name,
        "cupy": cp.__version__,
        "cuda_driver": cp.cuda.runtime.driverGetVersion(),
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "backend": str(options.backend.resolve()),
        "backend_sha256": hashlib.sha256(backend_bytes).hexdigest(),
        "ptx_targets": sorted(
            {
                target.decode()
                for target in re.findall(rb"\.target\s+(sm_\d+)", backend_bytes)
            }
        ),
        "repeat": options.repeat,
        "graph_batch": options.graph_batch,
        "references": {},
        "cases": [],
    }
    reference_modules = {}
    for case_index, (module, function, args, kwargs) in enumerate(
        make_cases(cp, np, sparse)
    ):
        if options.module and module not in options.module:
            continue
        if options.case and case_index not in options.case:
            continue
        name = f"_{module}_cuda"
        record = {
            "case_index": case_index,
            "module": module,
            "function": function,
            "parameters": {
                key: value
                for key, value in kwargs.items()
                if isinstance(value, (str, int, float, bool))
            },
            "inputs": [
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "host": isinstance(value, np.ndarray),
                    "order": "C" if value.flags.c_contiguous else "F",
                }
                for value in args
                if isinstance(value, (cp.ndarray, np.ndarray))
            ],
        }
        record["rust"] = measure(
            cp,
            np,
            benchmark,
            getattr(getattr(native, name), function),
            args,
            kwargs=kwargs,
            reference=False,
            repeat=options.repeat,
            batch=options.graph_batch,
            capture=function != "correction_batched" and module != "rank_stream",
            pass_stream=module != "rank_stream",
        )
        if module == "rank_stream":
            record["kernel_note"] = (
                "End-to-end only: bounded host uploads and independent streams complete before return."
            )
        elif function == "correction_batched":
            record["kernel_note"] = (
                "End-to-end only: bounded category GEMMs require host offsets and stream synchronization."
            )
        report["cases"].append(record)
        if (
            options.reference_dir is not None
            and case_index not in options.skip_reference_case
        ):
            candidates = sorted(options.reference_dir.glob(f"{name}*.so"))
            if not candidates:
                raise FileNotFoundError(f"no reference extension for {name}")
            if name not in report["references"]:
                report["references"][name] = {
                    "path": str(candidates[0].resolve()),
                    "sha256": hashlib.sha256(candidates[0].read_bytes()).hexdigest(),
                }
            if name not in reference_modules:
                reference = load_module(name, candidates[0])
                if hasattr(reference, "_set_scratch_allocator"):
                    allocations = {}

                    def allocate(size, allocations=allocations):
                        memory = cp.cuda.alloc(size)
                        allocations[memory.ptr] = memory
                        return memory.ptr

                    reference._set_scratch_allocator(allocate, allocations.pop)
                    # Release captured CuPy state before archived C++ static
                    # callback teardown runs during interpreter shutdown.
                    atexit.register(reference._set_scratch_allocator, int, int)
                reference_modules[name] = reference
            reference = reference_modules[name]
            try:
                record["reference"] = measure(
                    cp,
                    np,
                    benchmark,
                    getattr(reference, function),
                    args,
                    kwargs=kwargs,
                    reference=True,
                    repeat=options.repeat,
                    batch=options.graph_batch,
                    capture=function != "correction_batched"
                    and module != "rank_stream",
                    pass_stream=module != "rank_stream",
                )
            except Exception as error:
                record["reference_error"] = str(error)
                print(json.dumps(record), flush=True)
                if options.output is not None:
                    options.output.write_text(json.dumps(report, indent=2) + "\n")
                # A failed archived launch must not erase successful Rust
                # measurements; an illegal address poisons this CUDA context.
                if "IllegalAddress" in str(error) or "ILLEGAL_ADDRESS" in str(error):
                    raise
                continue
            if record["rust"]["kernel_us"] is not None:
                record["kernel_ratio"] = (
                    record["rust"]["kernel_us"] / record["reference"]["kernel_us"]
                )
            record["call_ratio"] = (
                record["rust"]["call_gpu_us"] / record["reference"]["call_gpu_us"]
            )
        if case_index in options.skip_reference_case:
            record["reference_skipped"] = "requested via --skip-reference-case"
        print(json.dumps(record), flush=True)
        if options.output is not None:
            options.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
