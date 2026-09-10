"""Compact sparse reference comparisons preserve ranks, stats and memory bounds."""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import numpy as np
import pytest
import scipy.sparse as sparse
from scipy.stats import rankdata, tiecorrect

from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as kernel


@pytest.mark.parametrize("host", [False, True])
@pytest.mark.parametrize("format", ["csr", "csc"])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_ovo_ranks_stats_and_memory(monkeypatch, host, format, index_dtype):
    rng = np.random.default_rng(7)
    rows, cols = 6001, 71
    source = sparse.random(rows, cols, density=0.004, format=format, random_state=rng)
    source.data = rng.integers(-3, 6, source.nnz).astype(np.float64)
    # Preserve values too small to survive the float32 conversion used for ranks.
    source.data[::7] = 1e-50
    source.indices = source.indices.astype(index_dtype)
    source.indptr = source.indptr.astype(index_dtype)
    reference = np.arange(1500, dtype=np.int32)
    grouped = np.arange(1500, 5000, dtype=np.int32)
    offsets = np.asarray([0, 2000, 3500], dtype=np.int32)
    ref_map = np.full(rows, -1, dtype=np.int32)
    grp_map = np.full(rows, -1, dtype=np.int32)
    ref_map[reference] = np.arange(len(reference))
    grp_map[grouped] = np.arange(len(grouped))
    codes = np.full(rows, -1, dtype=np.int32)
    codes[reference] = 2
    codes[grouped[:2000]] = 0
    codes[grouped[2000:]] = 1
    ranks = cp.empty((2, cols), dtype=cp.float64)
    ties = cp.empty_like(ranks)
    sums = cp.empty((3, cols), dtype=cp.float64)
    nonzero = cp.empty_like(sums)
    gpu = None if host else getattr(gpu_sparse, f"{format}_matrix")(source)
    if gpu is not None:
        gpu.indices = gpu.indices.astype(index_dtype)
        gpu.indptr = gpu.indptr.astype(index_dtype)
    if host:

        def unexpected_device_planning(*args, **kwargs):
            raise AssertionError("host sparse planning must use owned native buffers")

        for name in ("cumsum", "flatnonzero", "argsort", "full"):
            monkeypatch.setattr(cp, name, unexpected_device_planning)
    empty, zeros = cp.empty, cp.zeros
    allocations = []

    def record(shape, args, kwargs):
        dimensions = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        dtype = np.dtype(kwargs.get("dtype", args[0] if args else np.float64))
        allocation_bytes = int(np.prod(dimensions)) * dtype.itemsize
        # Count the combined uint8 radix arena and fixed histogram metadata in
        # bytes. This allowance remains below a dense selected-row window.
        allocations.append(allocation_bytes)
        assert allocation_bytes < rows * 64, (
            "sparse reference comparison staged a dense window"
        )

    def checked_empty(shape, *args, **kwargs):
        record(shape, args, kwargs)
        return empty(shape, *args, **kwargs)

    def checked_zeros(shape, *args, **kwargs):
        record(shape, args, kwargs)
        return zeros(shape, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", checked_empty)
    monkeypatch.setattr(cp, "zeros", checked_zeros)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        if host and format == "csr":
            kernel.ovo_streaming_csr_host(
                source.data,
                source.indices,
                source.indptr[:-1],
                source.indptr[1:],
                reference,
                grouped,
                offsets,
                ranks,
                ties,
                sums,
                nonzero,
                n_cols=cols,
                compute_tie_corr=True,
                compute_nnz=True,
                sub_batch_cols=64,
            )
        elif host:
            kernel.ovo_streaming_csc_host(
                source.data,
                source.indices,
                source.indptr,
                ref_map,
                grp_map,
                offsets,
                codes,
                ranks,
                ties,
                sums,
                nonzero,
                n_ref=len(reference),
                n_all_grp=len(grouped),
                compute_tie_corr=True,
                compute_nnz=True,
                sub_batch_cols=64,
            )
        else:
            ref = cp.asarray(ref_map if format == "csc" else reference)
            grp = cp.asarray(grp_map if format == "csc" else grouped)
            getattr(kernel, f"ovo_streaming_{format}_device")(
                gpu.data,
                gpu.indices,
                gpu.indptr,
                ref,
                grp,
                cp.asarray(offsets),
                ranks,
                ties,
                n_ref=len(reference),
                n_all_grp=len(grouped),
                compute_tie_corr=True,
                sub_batch_cols=64,
            )
    stream.synchronize()
    dense = source.toarray()
    for group in range(2):
        members = grouped[offsets[group] : offsets[group + 1]]
        values = np.concatenate((dense[reference], dense[members])).astype(np.float32)
        expected = rankdata(values, axis=0)
        np.testing.assert_array_equal(
            ranks[group].get(), expected[len(reference) :].sum(axis=0, dtype=np.float64)
        )
        np.testing.assert_allclose(
            ties[group].get(), [tiecorrect(expected[:, col]) for col in range(cols)]
        )
    if host:
        for group in range(3):
            np.testing.assert_allclose(
                sums[group].get(), dense[codes == group].sum(axis=0), atol=1e-14
            )
            np.testing.assert_array_equal(
                nonzero[group].get(), np.count_nonzero(dense[codes == group], axis=0)
            )
    assert allocations


@pytest.mark.parametrize(
    "reference,groups,match",
    [
        ([0, 0], [2, 3], "unique and disjoint"),
        ([0, 1], [1, 2], "unique and disjoint"),
        ([-1, 1], [2, 3], "within the source"),
        ([0, 1], [2, 4], "within the source"),
    ],
)
@pytest.mark.parametrize("host", [False, True])
def test_sparse_ovo_rejects_malformed_selection_without_mutating_outputs(
    reference, groups, match, host
):
    ranks = cp.full((1, 4), 17.0, dtype=cp.float64)
    ties = cp.full((1, 4), 19.0, dtype=cp.float64)
    sums = cp.full((2, 4), 23.0, dtype=cp.float64)
    nonzero = cp.full_like(sums, 29.0)
    matrix = (
        sparse.eye(4, dtype=np.float32, format="csr")
        if host
        else gpu_sparse.csr_matrix(cp.eye(4, dtype=cp.float32))
    )
    if host and min(reference + groups) < 0:
        match = "negative"
    with pytest.raises(ValueError, match=match):
        if host:
            kernel.ovo_streaming_csr_host(
                matrix.data,
                matrix.indices,
                matrix.indptr[:-1],
                matrix.indptr[1:],
                np.asarray(reference, dtype=np.int32),
                np.asarray(groups, dtype=np.int32),
                np.asarray([0, 2], dtype=np.int32),
                ranks,
                ties,
                sums,
                nonzero,
                n_cols=4,
                compute_tie_corr=True,
            )
        else:
            kernel.ovo_streaming_csr_device(
                matrix.data,
                matrix.indices,
                matrix.indptr,
                cp.asarray(reference, dtype=cp.int32),
                cp.asarray(groups, dtype=cp.int32),
                cp.asarray([0, 2], dtype=cp.int32),
                ranks,
                ties,
                n_ref=2,
                n_all_grp=2,
                compute_tie_corr=True,
            )
    cp.testing.assert_array_equal(ranks, 17)
    cp.testing.assert_array_equal(ties, 19)
    cp.testing.assert_array_equal(sums, 23)
    cp.testing.assert_array_equal(nonzero, 29)


@pytest.mark.parametrize("operation", ["flatnonzero", "argsort"])
def test_device_row_map_failure_drains_earlier_stream_work(monkeypatch, operation):
    matrix = gpu_sparse.csc_matrix(cp.eye(4, dtype=cp.float32))
    reference = cp.asarray([0, -1, -1, -1], dtype=cp.int32)
    grouped = cp.asarray([-1, 0, -1, -1], dtype=cp.int32)
    offsets = cp.asarray([0, 1], dtype=cp.int32)
    ranks = cp.full((1, 4), 17.0)
    ties = cp.full_like(ranks, 19.0)
    scratch = cp.ones(1_000_003, cp.float64)
    cp.sin(scratch, out=scratch)
    cp.cuda.get_current_stream().synchronize()
    stream = cp.cuda.Stream(non_blocking=True)

    def fail_after_enqueue(*args, **kwargs):
        # Leave enough existing CuPy work queued to distinguish a completion
        # guard from an exceptional return that frees pending row-map buffers.
        for _ in range(16):
            cp.sin(scratch, out=scratch)
        raise RuntimeError("injected row-map planning failure")

    monkeypatch.setattr(cp, operation, fail_after_enqueue)
    try:
        with stream, pytest.raises(RuntimeError, match="row-map planning failure"):
            kernel.ovo_streaming_csc_device(
                matrix.data,
                matrix.indices,
                matrix.indptr,
                reference,
                grouped,
                offsets,
                ranks,
                ties,
                n_ref=1,
                n_all_grp=1,
                compute_tie_corr=True,
            )
        assert stream.done
    finally:
        stream.synchronize()
    cp.testing.assert_array_equal(ranks, 17)
    cp.testing.assert_array_equal(ties, 19)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("stats_rows", [1, 4])
@pytest.mark.parametrize("compute_nnz", [False, True])
def test_host_csc_statistics_codes_are_independent_of_ranking_groups(
    dtype, index_dtype, stats_rows, compute_nnz
):
    data = (np.arange(63).reshape(7, 9) % 7).astype(dtype)
    source = sparse.csc_matrix(data)
    source.indices = source.indices.astype(index_dtype)
    source.indptr = source.indptr.astype(index_dtype)
    reference = np.asarray([1, 0], dtype=np.int32)
    grouped = np.asarray([3, 2, 4], dtype=np.int32)
    offsets = np.asarray([0, 2, 3], dtype=np.int32)
    ref_map = np.full(7, -1, dtype=np.int32)
    grp_map = np.full(7, -1, dtype=np.int32)
    ref_map[reference] = np.arange(2)
    grp_map[grouped] = np.arange(3)
    # Includes rows excluded from rankings, ignores a selected row, permutes
    # populations, and supports fewer or more stats rows than rank populations.
    stats_codes = np.asarray([3, 0, -1, 2, 1, 3, 0], dtype=np.int32)
    stats_codes[stats_codes >= 0] %= stats_rows
    ranks = cp.full((2, 9), -1, dtype=cp.float64)
    ties = cp.full_like(ranks, -1)
    sums = cp.full((stats_rows, 9), -1, dtype=cp.float64)
    nonzero = cp.full_like(sums, -7)
    cp.cuda.get_current_stream().synchronize()
    with cp.cuda.Stream(non_blocking=True):
        kernel.ovo_streaming_csc_host(
            source.data,
            source.indices,
            source.indptr,
            ref_map,
            grp_map,
            offsets,
            stats_codes,
            ranks,
            ties,
            sums,
            nonzero,
            n_ref=2,
            n_all_grp=3,
            compute_tie_corr=True,
            compute_nnz=compute_nnz,
            sub_batch_cols=2,
        )
    for group, rows in enumerate((grouped[:2], grouped[2:])):
        combined = np.concatenate((data[rows], data[reference]))
        np.testing.assert_array_equal(
            ranks[group].get(), rankdata(combined, axis=0)[: len(rows)].sum(axis=0)
        )
        np.testing.assert_allclose(
            ties[group].get(), [tiecorrect(column) for column in combined.T]
        )
    np.testing.assert_array_equal(
        sums.get(),
        [data[stats_codes == group].sum(axis=0) for group in range(stats_rows)],
    )
    if compute_nnz:
        np.testing.assert_array_equal(
            nonzero.get(),
            [
                np.count_nonzero(data[stats_codes == group], axis=0)
                for group in range(stats_rows)
            ],
        )
    else:
        cp.testing.assert_array_equal(nonzero, -7)
