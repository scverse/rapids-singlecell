"""Exact native rank regressions for tiled reductions and validation."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from scipy.stats import rankdata, tiecorrect

from rapids_singlecell._cuda import _wilcoxon_cuda

pytestmark = pytest.mark.skipif(
    getattr(_wilcoxon_cuda, "__backend__", None) != "rust",
    reason="requires Rust backend",
)


@pytest.mark.parametrize("tied", [False, True])
@pytest.mark.parametrize("group_sizes", [(0, 1, 17), (257, 4097, 3)])
def test_ovo_rank_tiles_match_scipy_on_explicit_stream(tied, group_sizes):
    rng = np.random.default_rng(37)
    reference = rng.normal(size=(1027, 5)).astype(np.float32)
    group = rng.normal(size=(sum(group_sizes), 5)).astype(np.float32)
    if tied:
        reference = np.round(reference)
        group = np.round(group)
    offsets = np.r_[0, np.cumsum(group_sizes)].astype(np.int32)
    expected_ranks = np.empty((len(group_sizes), 5))
    expected_ties = np.empty_like(expected_ranks)
    for i, size in enumerate(group_sizes):
        values = np.concatenate([group[offsets[i] : offsets[i + 1]], reference])
        expected_ranks[i] = rankdata(values, axis=0)[:size].sum(
            axis=0, dtype=np.float64
        )
        expected_ties[i] = [tiecorrect(values[:, j]) for j in range(values.shape[1])]
    caller = cp.cuda.get_current_stream()
    work = cp.cuda.Stream(non_blocking=True)
    with work:
        ref = cp.asarray(reference, order="F")
        grp = cp.asarray(group, order="F")
        off = cp.asarray(offsets)
        ranks = cp.empty_like(cp.asarray(expected_ranks))
        ties = cp.empty_like(ranks)
    _wilcoxon_cuda.ovo_rank_dense_tiered_unsorted_ref(
        ref,
        grp,
        off,
        ranks,
        ties,
        compute_tie_corr=True,
        sub_batch_cols=3,
        stream=work.ptr,
    )
    assert cp.cuda.get_current_stream().ptr == caller.ptr
    np.testing.assert_array_equal(cp.asnumpy(ranks), expected_ranks)
    np.testing.assert_allclose(cp.asnumpy(ties), expected_ties, atol=1e-14)


def test_host_ovo_rejects_invalid_rows_before_mutation():
    data = np.ones((5, 3), dtype=np.float64)
    ranks = cp.full((1, 3), 37, dtype=np.float64)
    ties = cp.full_like(ranks, 37)
    sums = cp.full((2, 3), 37, dtype=np.float64)
    nnz = cp.full_like(sums, 37)
    with pytest.raises(ValueError, match="row range"):
        _wilcoxon_cuda.ovo_rank_dense_host_streaming(
            data,
            np.array([0, 5], dtype=np.int64),
            np.array([1, 2], dtype=np.int64),
            np.array([0, 2], dtype=np.int32),
            ranks,
            ties,
            sums,
            nnz,
            compute_tie_corr=True,
            compute_nnz=True,
        )
    for output in (ranks, ties, sums, nnz):
        cp.testing.assert_array_equal(output, 37)


def test_host_ovr_rejects_statistics_alias_before_mutation():
    data = np.ones((5, 3), dtype=np.float64)
    ranks = cp.full((1, 3), 37, dtype=np.float64)
    ties = cp.full(3, 37, dtype=np.float64)
    codes = cp.zeros(5, dtype=np.int32)
    with pytest.raises(ValueError, match="overlap"):
        _wilcoxon_cuda.ovr_rank_dense_host_streaming(
            data,
            codes,
            ranks,
            ties,
            ranks,
            None,
            None,
            None,
            compute_tie_corr=True,
            compute_nnz=False,
            compute_totals=False,
        )
    cp.testing.assert_array_equal(ranks, 37)
    cp.testing.assert_array_equal(ties, 37)


def test_ovr_radix_preserves_ieee_order_and_zero_ties():
    column = np.array(
        [0.0, -0.0, np.inf, -np.inf, 1e-45, -1e-45, 3e38, -3e38, 1, -1],
        dtype=np.float32,
    )
    data = np.column_stack([column, column[::-1]])
    codes = np.arange(len(data), dtype=np.int32) % 3
    expected = rankdata(data, axis=0)
    ranks = cp.empty((3, 2), dtype=np.float64)
    ties = cp.empty(2, dtype=np.float64)
    _wilcoxon_cuda.ovr_rank_dense_streaming(
        cp.asarray(data, order="F"),
        cp.asarray(codes),
        ranks,
        ties,
        compute_tie_corr=True,
        sub_batch_cols=1,
    )
    np.testing.assert_array_equal(
        cp.asnumpy(ranks),
        [expected[codes == group].sum(axis=0, dtype=np.float64) for group in range(3)],
    )
    np.testing.assert_allclose(cp.asnumpy(ties), [tiecorrect(c) for c in data.T])
