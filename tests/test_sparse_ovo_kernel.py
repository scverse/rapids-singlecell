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
    empty, zeros = cp.empty, cp.zeros
    allocations = []

    def record(shape):
        if isinstance(shape, (tuple, list)) and len(shape) == 2:
            allocations.append(tuple(shape))
            assert np.prod(shape) < rows * 4, (
                "sparse reference comparison staged a dense window"
            )

    def checked_empty(shape, *args, **kwargs):
        record(shape)
        return empty(shape, *args, **kwargs)

    def checked_zeros(shape, *args, **kwargs):
        record(shape)
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
def test_sparse_ovo_rejects_malformed_selection_without_mutating_outputs(
    reference, groups, match
):
    matrix = gpu_sparse.csr_matrix(cp.eye(4, dtype=cp.float32))
    ranks = cp.full((1, 4), 17.0, dtype=cp.float64)
    ties = cp.full((1, 4), 19.0, dtype=cp.float64)
    with pytest.raises(ValueError, match=match):
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
