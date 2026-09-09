"""Direct correctness, stream and bounds checks for native ranking utilities."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp

from rapids_singlecell._cuda import _rank_stats_cuda as stats
from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as sparse


@pytest.mark.parametrize("fmt", ["csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ptr_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_window_matches_scipy_on_explicit_stream(
    fmt, dtype, ptr_dtype, index_dtype
):
    rng = np.random.default_rng(735)
    x = rng.normal(size=(31, 19)).astype(dtype)
    x[rng.random(x.shape) < 0.7] = 0
    source = getattr(sp, f"{fmt}_matrix")(x)
    with cp.cuda.Stream(non_blocking=True) as stream:
        p = cp.asarray(source.indptr, dtype=ptr_dtype)
        i = cp.asarray(source.indices, dtype=index_dtype)
        d = cp.asarray(source.data)
        out = cp.zeros((31, 12), dtype=cp.float64, order="F")
        getattr(stats, f"{fmt}_tile_to_dense")(
            p, i, d, out, col_lb=3, col_ub=15, stream=stream.ptr
        )
        result = out.get(stream=stream)
    np.testing.assert_array_equal(result, x[:, 3:15])


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_csr_range_gather_and_host_boundaries(dtype):
    x = sp.csr_matrix(
        np.array([[1, 0, 2, 0, 5], [0, 4, 0, 3, 2], [0, 0, 0, 0, 0]], dtype=np.float64)
    )
    indices = x.indices.astype(dtype)
    indptr = x.indptr.astype(dtype)
    cuts = np.array([0, 1, 4, 5], dtype=np.int32)
    boundaries = np.empty((4, 3), dtype=dtype)
    sparse.csr_row_boundaries_host(indices, indptr, cuts, boundaries, n_cols=5)
    expected = np.stack(
        [
            [
                np.searchsorted(indices[indptr[r] : indptr[r + 1]], c) + indptr[r]
                for r in range(3)
            ]
            for c in cuts
        ]
    )
    np.testing.assert_array_equal(boundaries, expected)
    with cp.cuda.Stream(non_blocking=True) as stream:
        p = cp.asarray(indptr)
        i = cp.asarray(indices)
        d = cp.asarray(x.data)
        local = cp.empty_like(p)
        count = sparse.csr_column_range_indptr_device(
            i, p, local, col_start=1, col_stop=4, stream=stream.ptr
        )
        out_d = cp.empty(count, dtype=d.dtype)
        out_i = cp.empty(count, dtype=i.dtype)
        sparse.csr_column_range_gather_device(
            d, i, p, local, out_d, out_i, col_start=1, col_stop=4, stream=stream.ptr
        )
        result = sp.csr_matrix(
            (
                out_d.get(stream=stream),
                out_i.get(stream=stream),
                local.get(stream=stream),
            ),
            shape=(3, 3),
        )
    np.testing.assert_array_equal(result.toarray(), x[:, 1:4].toarray())


def test_group_statistics_and_bh_match_independent_reference():
    x = np.array([[1, 0, -2], [2, 4, 0], [3, 0, 6], [2, -1, 1]], dtype=np.float64)
    codes = np.array([0, 1, -1, 0], dtype=np.int32)
    with cp.cuda.Stream(non_blocking=True) as stream:
        block = cp.asarray(x, order="F")
        groups = cp.asarray(codes)
        sums = cp.zeros((2, 3), dtype=cp.float64)
        squares = cp.zeros_like(sums)
        nnz = cp.zeros_like(sums)
        stats.group_chunk_stats(
            block, groups, sums, squares, nnz, compute_nnz=True, stream=stream.ptr
        )
        values = cp.asarray([[0.4, 0.2, np.nan, 0.7], [np.nan, 1.2, 0.9, 0.1]])
        stats.fdr_bh_reverse_cummin(values, stream=stream.ptr)
        got = [a.get(stream=stream) for a in (sums, squares, nnz, values)]
    np.testing.assert_array_equal(
        got[0], np.stack([x[codes == i].sum(0) for i in range(2)])
    )
    np.testing.assert_array_equal(
        got[1], np.stack([(x[codes == i] ** 2).sum(0) for i in range(2)])
    )
    np.testing.assert_array_equal(
        got[2], np.stack([(x[codes == i] != 0).sum(0) for i in range(2)])
    )
    np.testing.assert_array_equal(got[3], [[0.2, 0.2, 0.7, 0.7], [0.1, 0.1, 0.1, 0.1]])


def test_host_boundaries_reject_invalid_offsets_before_writing():
    out = np.full((1, 2), 73, dtype=np.int32)
    with pytest.raises(ValueError, match="indptr"):
        sparse.csr_row_boundaries_host(
            np.array([1, 2], np.int32),
            np.array([0, 1, 5], np.int32),
            np.array([1], np.int32),
            out,
            n_cols=3,
        )
    np.testing.assert_array_equal(out, 73)


@pytest.mark.parametrize("fmt", ["csr", "csc"])
def test_sparse_histogram_rejects_unaddressable_columns_before_launch(fmt):
    from rapids_singlecell._cuda import _wilcoxon_binned_cuda as binned

    data = cp.empty(0, dtype=cp.float32)
    indices = cp.empty(0, dtype=cp.int64)
    indptr = cp.zeros(2, dtype=cp.int64)
    codes = cp.zeros(1, dtype=cp.int32)
    hist = cp.full((1, 1, 3), 17, dtype=cp.uint32)
    with pytest.raises(ValueError, match="column range overflow"):
        getattr(binned, f"{fmt}_hist")(
            data,
            indices,
            indptr,
            codes,
            hist,
            n_cells=1,
            n_genes=1,
            n_groups=1,
            n_bins=2,
            bin_low=0.0,
            inv_bin_width=1.0,
            gene_start=(1 << 64) - 2,
        )
    cp.testing.assert_array_equal(hist, 17)
