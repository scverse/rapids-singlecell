"""Exact ranks and statistics survive pinned slot reuse on caller streams."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.stats import rankdata, tiecorrect

from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as native


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
@pytest.mark.parametrize("compute_ties", [False, True])
def test_sparse_host_staging_reuses_all_slots(dtype, index_dtype, compute_ties):
    rng = np.random.default_rng(184)
    data = rng.integers(-3, 6, size=(301, 29)).astype(dtype)
    data[rng.random(data.shape) < 0.8] = 0
    if dtype == np.float64:
        data[data != 0] += 2**-29
    data[:, 12:15] = 0  # Empty batches must not retain previous slot contents.
    source = sp.csc_matrix(data)
    source.indices = source.indices.astype(index_dtype)
    source.indptr = source.indptr.astype(index_dtype)
    codes = np.arange(len(data), dtype=np.int32) % 3
    sizes = np.bincount(codes).astype(np.float64)
    expected = rankdata(data.astype(np.float32), axis=0)
    caller = cp.cuda.get_current_stream()
    with cp.cuda.Stream(non_blocking=True) as work:
        ranks = cp.full((3, data.shape[1]), -1, dtype=np.float64)
        ties = cp.full(data.shape[1], 19, dtype=np.float64)
        sums = cp.full_like(ranks, -1)
        counts = cp.full_like(ranks, -1)
        totals = cp.full_like(ties, -1)
        total_counts = cp.full_like(ties, -1)
        native.ovr_sparse_csc_host(
            source.data,
            source.indices,
            source.indptr,
            codes,
            sizes,
            ranks,
            ties,
            sums,
            counts,
            totals,
            total_counts,
            compute_tie_corr=compute_ties,
            compute_nnz=True,
            compute_totals=True,
            sub_batch_cols=3,
        )
        assert cp.cuda.get_current_stream().ptr == work.ptr
        # The streaming API completes before returning. Synchronous reads on a
        # different stream therefore need no added user event or device barrier.
    assert cp.cuda.get_current_stream().ptr == caller.ptr
    np.testing.assert_array_equal(
        ranks.get(), [expected[codes == g].sum(axis=0) for g in range(3)]
    )
    np.testing.assert_allclose(
        sums.get(),
        [data[codes == g].sum(axis=0, dtype=np.float64) for g in range(3)],
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_array_equal(
        counts.get(), [np.count_nonzero(data[codes == g], axis=0) for g in range(3)]
    )
    np.testing.assert_allclose(
        totals.get(), data.sum(axis=0, dtype=np.float64), rtol=1e-13, atol=1e-13
    )
    np.testing.assert_array_equal(total_counts.get(), np.count_nonzero(data, axis=0))
    if compute_ties:
        np.testing.assert_allclose(
            ties.get(), [tiecorrect(c) for c in data.astype(np.float32).T], atol=1e-14
        )
    else:
        cp.testing.assert_array_equal(ties, 19)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_reference_staging_preserves_output_windows(dtype):
    rng = np.random.default_rng(218)
    data = rng.integers(-2, 7, size=(217, 31)).astype(dtype)
    data[rng.random(data.shape) < 0.85] = 0
    source = sp.csc_matrix(data)
    offsets = np.array([0, 71, 174], dtype=np.int64)
    ref_map = np.full(len(data), -1, dtype=np.int64)
    grp_map = np.full(len(data), -1, dtype=np.int64)
    ref_map[:43] = np.arange(43)
    grp_map[43:] = np.arange(174)
    stats_codes = np.r_[np.full(43, 2), np.zeros(71), np.ones(103)].astype(np.int32)
    with cp.cuda.Stream(non_blocking=True) as work:
        ranks = cp.full((2, 31), -1, dtype=np.float64)
        ties = cp.full_like(ranks, -1)
        sums = cp.full((3, 31), -1, dtype=np.float64)
        counts = cp.full_like(sums, -1)
        native.ovo_streaming_csc_host(
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
            counts,
            n_ref=43,
            n_all_grp=174,
            compute_tie_corr=True,
            compute_nnz=True,
            sub_batch_cols=3,
        )
        assert cp.cuda.get_current_stream().ptr == work.ptr
    populations = [data[43:114], data[114:], data[:43]]
    for group, values in enumerate(populations[:2]):
        combined = np.concatenate((values, populations[2])).astype(np.float32)
        np.testing.assert_array_equal(
            ranks[group].get(), rankdata(combined, axis=0)[: len(values)].sum(axis=0)
        )
        np.testing.assert_allclose(
            ties[group].get(), [tiecorrect(c) for c in combined.T], atol=1e-14
        )
    np.testing.assert_array_equal(
        sums.get(), [x.sum(axis=0, dtype=np.float64) for x in populations]
    )
    np.testing.assert_array_equal(
        counts.get(), [np.count_nonzero(x, axis=0) for x in populations]
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_dense_staging_exception_waits_and_restores_caller_stream(order):
    from rapids_singlecell._cuda import _wilcoxon_cuda

    class FailingWindow(np.ndarray):
        def __getitem__(self, key):
            if (
                isinstance(key, tuple)
                and isinstance(key[1], slice)
                and key[1].start >= 6
            ):
                raise RuntimeError("injected staging failure")
            return super().__getitem__(key)

    data = np.ones((4097, 13), dtype=np.float64, order=order).view(FailingWindow)
    caller = cp.cuda.get_current_stream()
    with cp.cuda.Stream(non_blocking=True) as work:
        codes = cp.arange(len(data), dtype=np.int32) % 2
        ranks = cp.zeros((2, data.shape[1]), dtype=np.float64)
        ties = cp.zeros(data.shape[1], dtype=np.float64)
        sums = cp.zeros_like(ranks)
        with pytest.raises(RuntimeError, match="injected staging failure"):
            _wilcoxon_cuda.ovr_rank_dense_host_streaming(
                data,
                codes,
                ranks,
                ties,
                sums,
                None,
                None,
                None,
                compute_tie_corr=True,
                compute_nnz=False,
                compute_totals=False,
                sub_batch_cols=3,
            )
        assert cp.cuda.get_current_stream().ptr == work.ptr
    assert cp.cuda.get_current_stream().ptr == caller.ptr
    np.testing.assert_array_equal(sums[:, :6].get(), [[2049] * 6, [2048] * 6])
    # Releasing unused allocator blocks is safe after the exceptional return.
    cp.get_default_memory_pool().free_all_blocks()
    cp.testing.assert_array_equal(cp.arange(31) * 2, np.arange(31) * 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize(
    "layout", ["c_reverse_rows", "f_reverse_columns", "f_gaps", "both_strided"]
)
def test_dense_staging_preserves_strided_axes_and_native_statistics(dtype, layout):
    from rapids_singlecell._cuda import _wilcoxon_cuda

    rng = np.random.default_rng(947)
    source = rng.integers(-3, 7, size=(2062, 38)).astype(dtype)
    if dtype == np.float64:
        source[source != 0] += 2**-30
    if layout == "c_reverse_rows":
        window = source[1030::-1, 2:19]
    elif layout == "f_reverse_columns":
        window = np.asfortranarray(source)[:1031, 18:1:-1]
    elif layout == "f_gaps":
        window = np.asfortranarray(source)[:1031, 2:36:2]
    else:
        window = source[::2, 2:36:2]

    # The public matrix must be contiguous; ndarray subclasses can return
    # strided bounded windows from the streaming callback.
    class WindowSource(np.ndarray):
        def __getitem__(self, key):
            return window[key]

    staged_source = np.empty(window.shape, dtype=dtype).view(WindowSource)
    codes = np.arange(len(window), dtype=np.int32) % 3
    expected = rankdata(window.astype(np.float32), axis=0)
    with cp.cuda.Stream(non_blocking=True) as work:
        ranks = cp.empty((3, window.shape[1]), dtype=np.float64)
        ties = cp.empty(window.shape[1], dtype=np.float64)
        sums = cp.empty_like(ranks)
        _wilcoxon_cuda.ovr_rank_dense_host_streaming(
            staged_source,
            cp.asarray(codes),
            ranks,
            ties,
            sums,
            None,
            None,
            None,
            compute_tie_corr=True,
            compute_nnz=False,
            compute_totals=False,
            sub_batch_cols=3,
        )
        assert cp.cuda.get_current_stream().ptr == work.ptr
    np.testing.assert_array_equal(
        ranks.get(), [expected[codes == group].sum(axis=0) for group in range(3)]
    )
    np.testing.assert_allclose(
        sums.get(),
        [window[codes == group].sum(axis=0, dtype=np.float64) for group in range(3)],
        rtol=1e-13,
        atol=1e-13,
    )
    np.testing.assert_allclose(
        ties.get(),
        [tiecorrect(column) for column in window.astype(np.float32).T],
        rtol=1e-13,
        atol=1e-13,
    )


def test_dense_gpu_transpose_exception_retains_both_buffers(monkeypatch):
    from rapids_singlecell._cuda import _wilcoxon_cuda

    real_copy = cp.copyto

    def failing_copy(destination, source):
        real_copy(destination, source)
        raise RuntimeError("injected GPU transpose failure")

    monkeypatch.setattr(cp, "copyto", failing_copy)
    data = np.ones((4097, 13), dtype=np.float64, order="C")
    caller = cp.cuda.get_current_stream()
    with cp.cuda.Stream(non_blocking=True) as work:
        ranks = cp.zeros((2, 13), dtype=np.float64)
        ties = cp.zeros_like(ranks)
        with pytest.raises(RuntimeError, match="injected GPU transpose failure"):
            _wilcoxon_cuda.ovo_rank_dense_host_streaming(
                data,
                np.arange(2048, dtype=np.int32),
                np.arange(2048, len(data), dtype=np.int32),
                np.array([0, 1024, 2049], dtype=np.int32),
                ranks,
                ties,
                cp.zeros((3, 13), dtype=np.float64),
                None,
                compute_tie_corr=True,
                compute_nnz=False,
                sub_batch_cols=3,
            )
        assert cp.cuda.get_current_stream().ptr == work.ptr
    assert cp.cuda.get_current_stream().ptr == caller.ptr
    cp.get_default_memory_pool().free_all_blocks()
    cp.testing.assert_array_equal(cp.arange(31) * 2, np.arange(31) * 2)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_gpu_transpose_preserves_all_ieee_bits(monkeypatch, dtype):
    from rapids_singlecell._cuda import _wilcoxon_cuda

    integer = np.uint32 if dtype == np.float32 else np.uint64
    rng = np.random.default_rng(953)
    bits = rng.integers(0, np.iinfo(integer).max, size=(33, 17), dtype=integer)
    bits[:2] = np.array([0, 1 << (np.dtype(integer).itemsize * 8 - 1)], dtype=integer)[
        :, None
    ]
    data = bits.view(dtype)
    real_copy = cp.copyto
    start = 0
    population = 0

    def checked_copy(destination, source):
        nonlocal start, population
        real_copy(destination, source)
        if destination.dtype != source.dtype:
            return
        width = destination.shape[1]
        rows = slice(0, 16) if population == 0 else slice(16, 33)
        np.testing.assert_array_equal(
            cp.asnumpy(destination).view(integer), bits[rows, start : start + width]
        )
        if population == 1:
            start += width
        population = 1 - population

    monkeypatch.setattr(cp, "copyto", checked_copy)
    ranks = cp.empty((2, 17), dtype=np.float64)
    _wilcoxon_cuda.ovo_rank_dense_host_streaming(
        data,
        np.arange(16, dtype=np.int32),
        np.arange(16, len(data), dtype=np.int32),
        np.array([0, 8, 17], dtype=np.int32),
        ranks,
        cp.empty_like(ranks),
        cp.empty((3, 17), dtype=np.float64),
        None,
        compute_tie_corr=True,
        compute_nnz=False,
        sub_batch_cols=3,
    )
    assert start == data.shape[1]


def test_dense_fused_cast_exception_waits_and_restores_stream(monkeypatch):
    from rapids_singlecell._cuda import _wilcoxon_cuda

    convert = cp.copyto
    pending = []

    def failing_cast(destination, source):
        convert(destination, source)
        if (
            isinstance(source, cp.ndarray)
            and source.ndim == 2
            and source.dtype == np.float64
            and source.flags.c_contiguous
            and destination.dtype == np.float32
            and destination.flags.f_contiguous
        ):
            pending.append(destination)
            raise RuntimeError("injected fused ranking cast failure")

    data = np.ones((4097, 13), dtype=np.float64, order="C")
    caller = cp.cuda.get_current_stream()
    with cp.cuda.Stream(non_blocking=True) as work:
        ranks = cp.zeros((2, 13), dtype=np.float64)
        codes = cp.arange(len(data), dtype=np.int32) % 2
        monkeypatch.setattr(cp, "copyto", failing_cast)
        with pytest.raises(RuntimeError, match="injected fused ranking cast failure"):
            _wilcoxon_cuda.ovr_rank_dense_host_streaming(
                data,
                codes,
                ranks,
                cp.zeros(13, dtype=np.float64),
                cp.zeros_like(ranks),
                None,
                None,
                None,
                compute_tie_corr=True,
                compute_nnz=False,
                compute_totals=False,
                sub_batch_cols=3,
            )
        assert cp.cuda.get_current_stream().ptr == work.ptr
        assert len(pending) == 1
        cp.testing.assert_array_equal(pending[0], 1)
    assert cp.cuda.get_current_stream().ptr == caller.ptr
    pending.clear()
    cp.get_default_memory_pool().free_all_blocks()


@pytest.mark.parametrize("api", ["ovr", "ovo"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_dense_cast_storage_is_reused_across_batches(monkeypatch, api, dtype):
    from rapids_singlecell._cuda import _wilcoxon_cuda

    data = np.arange(67 * 19, dtype=dtype).reshape(67, 19) % 11
    codes = np.arange(len(data), dtype=np.int32) % 2
    real_copy = cp.copyto
    targets = {}
    calls = []

    def capture_cast(destination, source):
        if destination.dtype != source.dtype or (
            api == "ovr"
            and source.flags.c_contiguous
            and destination.flags.f_contiguous
        ):
            key = (cp.cuda.get_current_stream().ptr, len(source), destination.dtype.str)
            targets.setdefault(key, set()).add(destination.data.ptr)
            calls.append(key)
        real_copy(destination, source)

    monkeypatch.setattr(cp, "copyto", capture_cast)
    ranks = cp.empty((2, 19), dtype=np.float64)
    ties = cp.empty(19 if api == "ovr" else ranks.shape, dtype=np.float64)
    sums = cp.empty((2 if api == "ovr" else 3, 19), dtype=np.float64)
    if api == "ovr":
        _wilcoxon_cuda.ovr_rank_dense_host_streaming(
            data,
            cp.asarray(codes),
            ranks,
            ties,
            sums,
            None,
            None,
            None,
            compute_tie_corr=True,
            compute_nnz=False,
            compute_totals=False,
            sub_batch_cols=3,
        )
    else:
        _wilcoxon_cuda.ovo_rank_dense_host_streaming(
            data,
            np.arange(31, dtype=np.int32),
            np.arange(31, len(data), dtype=np.int32),
            np.array([0, 17, 36], dtype=np.int32),
            ranks,
            ties,
            sums,
            None,
            compute_tie_corr=True,
            compute_nnz=False,
            sub_batch_cols=3,
        )
    assert len(calls) >= 6
    assert len(targets) == (2 if api == "ovr" else 4)
    assert all(len(addresses) == 1 for addresses in targets.values())
