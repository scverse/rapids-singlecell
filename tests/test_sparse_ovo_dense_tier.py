"""Adaptive CSR reference caching keeps precision and the sparse fallback."""

from __future__ import annotations

import random

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sparse
from scipy.stats import rankdata, tiecorrect

from rapids_singlecell._cuda import _wilcoxon_cuda as dense_kernel
from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as kernel


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("group_size", [512, 513, 2500])
def test_csr_analytic_zero_tier_and_expanded_slab(dtype, group_size, monkeypatch):
    rng = np.random.default_rng(1453)
    cols, nref = 257, 83
    values = rng.integers(0, 6, (nref + group_size, cols)).astype(dtype)
    values[rng.random(values.shape) < 0.96] = 0
    values[:, 0] = 0
    values[:, 1] = 2
    values[::7, 2] = 1e-37 if dtype == np.float32 else 1e-50
    values[::11, 3] = np.nextafter(dtype(0), dtype(1))
    values[::13, 4] = 1e20 if dtype == np.float32 else 1e40
    source = sparse.csr_matrix(values)
    source.indices = source.indices.astype(np.int64)
    reference = np.arange(nref, dtype=np.int32)[::-1].copy()
    grouped = np.arange(nref, len(values), dtype=np.int32)[::-1].copy()
    offsets = np.asarray([0, 0, group_size], np.int32)
    ranks, ties = cp.empty((2, cols), cp.float64), cp.empty((2, cols), cp.float64)
    sums, counts = cp.empty((3, cols), cp.float64), cp.empty((3, cols), cp.float64)
    original_empty = cp.empty
    group_slabs = []

    def allocate(shape, *args, **kwargs):
        dimensions = (shape,) if isinstance(shape, int) else tuple(shape)
        if dimensions == (group_size * cols,):
            group_slabs.append(group_size * cols)
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", allocate)
    with cp.cuda.Stream(non_blocking=True) as stream:
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
            counts,
            n_cols=cols,
            compute_tie_corr=True,
            compute_nnz=True,
            analytic_zeros=True,
            sub_batch_cols=17,
        )
        assert stream.done
    # The requested width is a minimum for the original bounded CSR planner.
    # The full group slab fits here and avoids sixteen redundant dispatches.
    assert group_slabs == [group_size * cols]
    for group, selected in enumerate((grouped[:0], grouped)):
        with np.errstate(over="ignore"):
            combined = values[np.r_[reference, selected]].astype(np.float32)
        ranked = rankdata(combined, axis=0)
        np.testing.assert_array_equal(ranks[group].get(), ranked[nref:].sum(axis=0))
        np.testing.assert_allclose(
            ties[group].get(), [tiecorrect(ranked[:, col]) for col in range(cols)]
        )
    for group, selected in enumerate((grouped[:0], grouped, reference)):
        np.testing.assert_allclose(
            sums[group].get(), values[selected].astype(np.float64).sum(axis=0)
        )
        np.testing.assert_array_equal(
            counts[group].get(), np.count_nonzero(values[selected], axis=0)
        )


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("bounded_fallback", [False, True])
def test_csr_reference_cache_preserves_gapped_spans_and_native_stats(
    monkeypatch, dtype, index_dtype, bounded_fallback
):
    rng = np.random.default_rng(941)
    rows, cols, nref = 521, 263, 173
    full_cols = cols + 8
    source = sparse.random(
        rows, full_cols, density=0.08, format="csr", random_state=rng
    )
    source.data = rng.integers(-3, 5, source.nnz).astype(dtype)
    source.data[::13] = 1e-37 if dtype == np.float32 else 1e-50
    source.indices = source.indices.astype(index_dtype)
    starts, stops = (
        source.indptr[:-1].astype(np.int64),
        source.indptr[1:].astype(np.int64),
    )
    # Mixed offset/index widths and excluded prefix/suffix entries survive the
    # row gather. Reference/member order intentionally differs from CSR order.
    starts += (stops > starts).astype(np.int64)
    stops -= (stops > starts).astype(np.int64)
    expected = np.zeros((rows, full_cols), dtype)
    for row, (first, stop) in enumerate(zip(starts, stops, strict=True)):
        expected[row, source.indices[first:stop]] = source.data[first:stop]
    expected = expected[:, 4 : cols + 4]
    reference = np.arange(nref, dtype=np.int32)[::-1].copy()
    grouped = np.arange(nref, rows, dtype=np.int32)[::-1].copy()
    offsets = np.asarray([0, 181, 181, rows - nref], np.int32)
    ranks, ties = cp.empty((3, cols), cp.float64), cp.empty((3, cols), cp.float64)
    sums, counts = cp.empty((4, cols), cp.float64), cp.empty((4, cols), cp.float64)
    if bounded_fallback:
        monkeypatch.setattr(
            cp.cuda.runtime, "memGetInfo", lambda: (3 * 1024**2, 8 * 1024**2)
        )
    original_empty = cp.empty
    reference_allocations = []

    def allocate(shape, *args, **kwargs):
        dimensions = (shape,) if isinstance(shape, int) else tuple(shape)
        if dimensions == (nref, cols):
            reference_allocations.append(dimensions)
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", allocate)
    with cp.cuda.Stream(non_blocking=True) as stream:
        kernel.ovo_streaming_csr_host(
            source.data,
            source.indices,
            starts,
            stops,
            reference,
            grouped,
            offsets,
            ranks,
            ties,
            sums,
            counts,
            n_cols=full_cols,
            col_start=4,
            col_stop=cols + 4,
            compute_tie_corr=True,
            compute_nnz=True,
            sub_batch_cols=17,
        )
        assert stream.done
    assert len(reference_allocations) == (0 if bounded_fallback else 1)
    for group in range(3):
        selected = grouped[offsets[group] : offsets[group + 1]]
        values = expected[np.r_[reference, selected]].astype(np.float32)
        ranked = rankdata(values, axis=0)
        np.testing.assert_array_equal(ranks[group].get(), ranked[nref:].sum(axis=0))
        np.testing.assert_allclose(
            ties[group].get(), [tiecorrect(ranked[:, c]) for c in range(cols)]
        )
    for group, selected in enumerate(
        (grouped[:181], grouped[181:181], grouped[181:], reference)
    ):
        np.testing.assert_allclose(
            sums[group].get(),
            expected[selected].astype(np.float64).sum(axis=0),
            atol=1e-12,
        )
        np.testing.assert_array_equal(
            counts[group].get(), np.count_nonzero(expected[selected], axis=0)
        )


def test_csr_cached_reference_preserves_signaling_nan_sort_bits():
    bits = np.asarray(
        [
            0x7F800001,
            0x7FC00001,
            0x7FA00002,
            0xFFC00001,
            0xFF800001,
            0x3F800000,
            0x00000000,
            0x80000000,
            0xBF800000,
            0x7F800000,
            0x40000000,
            0xFF800000,
        ],
        np.uint32,
    )
    values = np.tile(bits.view(np.float32)[:, None], (1, 257))
    # Explicit CSR coordinates retain signed zeros and NaN payload bits.
    source = sparse.csr_matrix(
        (values.ravel(), np.tile(np.arange(257), 12), np.arange(13) * 257),
        shape=values.shape,
    )
    reference, grouped = np.arange(5, dtype=np.int32), np.arange(5, 12, dtype=np.int32)
    offsets = np.asarray([0, 3, 7], np.int32)
    ranks, ties = cp.empty((2, 257), cp.float64), cp.empty((2, 257), cp.float64)
    expected, expected_ties = cp.empty_like(ranks), cp.empty_like(ties)
    dense_kernel.ovo_rank_dense_tiered_unsorted_ref(
        cp.asarray(values[reference], order="F"),
        cp.asarray(values[grouped], order="F"),
        cp.asarray(offsets),
        expected,
        expected_ties,
        compute_tie_corr=True,
        sub_batch_cols=17,
    )
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
        cp.empty((3, 257), cp.float64),
        cp.empty((3, 257), cp.float64),
        n_cols=257,
        compute_tie_corr=True,
        sub_batch_cols=17,
    )
    np.testing.assert_array_equal(ranks.get(), expected.get())
    np.testing.assert_array_equal(ties.get(), expected_ties.get())


@pytest.mark.parametrize("reverse", [False, True])
def test_csr_large_group_nan_ranks_preserve_supplied_member_order(reverse):
    rng = random.Random(912)
    group = np.asarray([rng.randint(-3, 3) for _ in range(513)], np.float32)
    group[3] = np.nan
    values = np.tile(
        np.r_[np.asarray([-3, -1, 0, 1, 3], np.float32), group][:, None], (1, 257)
    )
    source = sparse.csr_matrix(values)
    reference = np.arange(5, dtype=np.int32)
    grouped = np.arange(5, 518, dtype=np.int32)
    if reverse:
        grouped = grouped[::-1].copy()
    offsets = np.asarray([0, 513], np.int32)
    ranks, ties = cp.empty((1, 257), cp.float64), cp.empty((1, 257), cp.float64)
    expected, expected_ties = cp.empty_like(ranks), cp.empty_like(ties)
    dense_kernel.ovo_rank_dense_tiered_unsorted_ref(
        cp.asarray(values[reference], order="F"),
        cp.asarray(values[grouped], order="F"),
        cp.asarray(offsets),
        expected,
        expected_ties,
        compute_tie_corr=True,
    )
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
        cp.empty((2, 257), cp.float64),
        cp.empty((2, 257), cp.float64),
        n_cols=257,
        compute_tie_corr=True,
    )
    np.testing.assert_array_equal(ranks.get(), expected.get())
    np.testing.assert_array_equal(ties.get(), expected_ties.get())


def test_csr_compact_index_alignment_fails_before_output_mutation(monkeypatch):
    source = sparse.random(
        8, 257, density=0.05, format="csr", random_state=43, dtype=np.float32
    )
    reference, grouped = np.arange(2, dtype=np.int32), np.arange(2, 8, dtype=np.int32)
    offsets = np.asarray([0, 3, 6], np.int32)
    ranks, ties = (
        cp.full((2, 257), 17.0, cp.float64),
        cp.full((2, 257), 19.0, cp.float64),
    )
    sums, counts = (
        cp.full((3, 257), 23.0, cp.float64),
        cp.full((3, 257), 29.0, cp.float64),
    )
    original_empty = cp.empty

    def misaligned(shape, *args, **kwargs):
        dtype = np.dtype(kwargs.get("dtype", args[0] if args else np.float64))
        if dtype == np.uint8:
            length = int(np.prod(shape))
            return original_empty(length + 1, cp.uint8)[1:]
        return original_empty(shape, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", misaligned)
    with cp.cuda.Stream(non_blocking=True) as stream:
        with pytest.raises(ValueError, match="misaligned"):
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
                counts,
                n_cols=257,
                compute_tie_corr=True,
            )
        assert stream.done
    for array, value in ((ranks, 17), (ties, 19), (sums, 23), (counts, 29)):
        np.testing.assert_array_equal(array.get(), value)
