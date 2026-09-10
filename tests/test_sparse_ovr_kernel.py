"""Sparse ranks preserve implicit-zero ties without allocating dense windows."""

from __future__ import annotations

import weakref

import cupy as cp
import cupyx.scipy.sparse as gpu_sparse
import numpy as np
import pytest
import scipy.sparse as sparse
from scipy.stats import rankdata, tiecorrect

from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as kernel


@pytest.mark.parametrize("comparison", ["ovr", "ovo"])
@pytest.mark.parametrize("index_dtype", [cp.int32, cp.int64])
def test_sparse_mixed_nan_bits_match_archived_cub_ranks(comparison, index_dtype):
    # Captured from the archived C++ kernels: signaling NaNs must retain their
    # float32 sort bits instead of being quieted by a float64 round trip.
    bits = np.asarray(
        [
            0x7F800001,
            0x7FC00001,
            0x7FA00002,
            0xFFC00001,
            0xFF800001,
            0x3F800000,
            0,
            0x80000000,
            0xBF800000,
            0x7F800000,
            0x40000000,
            0xFF800000,
        ],
        np.uint32,
    )
    data = cp.asarray(bits.view(np.float32))
    indices = cp.arange(12, dtype=index_dtype)
    pointers = cp.asarray([0, 12], index_dtype)
    ranks = cp.empty((2, 1), cp.float64)
    ties = cp.empty((1,) if comparison == "ovr" else (2, 1), cp.float64)
    if comparison == "ovr":
        kernel.ovr_sparse_csc_device(
            data,
            indices,
            pointers,
            cp.arange(12, dtype=cp.int32) % 2,
            cp.asarray([6, 6], cp.float64),
            ranks,
            ties,
            compute_tie_corr=True,
        )
        expected_ranks, expected_ties = [40.5, 37.5], [0.9965034965034965]
    else:
        refs = cp.asarray([0, 1, 2, 3, 4, -1, -1, -1, -1, -1, -1, -1], cp.int32)
        groups = cp.asarray([-1, -1, -1, -1, -1, 0, 1, 2, 3, 4, 5, 6], cp.int32)
        kernel.ovo_streaming_csc_device(
            data,
            indices,
            pointers,
            refs,
            groups,
            cp.asarray([0, 3, 7], cp.int32),
            ranks,
            ties,
            n_ref=5,
            n_all_grp=7,
            compute_tie_corr=True,
        )
        expected_ranks, expected_ties = [6, 10], [0.9880952380952381, 1]
    np.testing.assert_array_equal(ranks.get().ravel(), expected_ranks)
    np.testing.assert_allclose(ties.get().ravel(), expected_ties, rtol=1e-15)


@pytest.mark.parametrize("comparison", ["ovr", "ovo"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_host_sparse_scratch_uses_caller_pool_with_bounded_arena_growth(
    monkeypatch, comparison, asynchronous
):
    if asynchronous and not cp.cuda.Device().attributes.get("MemoryPoolsSupported", 0):
        pytest.skip("device does not support stream-ordered allocation")
    pool = cp.cuda.MemoryAsyncPool() if asynchronous else cp.cuda.MemoryPool()
    rows, cols = 4099, 13
    # Each of four worker slots grows its arena. Strongly skewed columns also
    # split windows, requiring old arenas to be released before replacement.
    stored = np.asarray([1, 3, 11, 71, 201, 603, 1809, 4099, 21, 9, 3, 1, 0])
    indptr = np.r_[0, stored.cumsum()].astype(np.int64)
    indices = np.concatenate([np.arange(n, dtype=np.int32) for n in stored])
    values = (indices + np.repeat(np.arange(cols), stored)) % 7 - 3
    source = sparse.csc_matrix(
        (values.astype(np.float32), indices, indptr), (rows, cols)
    )
    codes = np.arange(rows, dtype=np.int32) % 3
    sizes = np.bincount(codes).astype(np.float64)
    reference = np.flatnonzero(codes == 2).astype(np.int32)
    grouped = np.concatenate([np.flatnonzero(codes == group) for group in range(2)])
    offsets = np.asarray([0, sizes[0], sizes[0] + sizes[1]], dtype=np.int32)
    ref_map, grp_map = np.full(rows, -1, np.int32), np.full(rows, -1, np.int32)
    ref_map[reference], grp_map[grouped] = (
        np.arange(len(reference)),
        np.arange(len(grouped)),
    )
    dense = source.toarray()
    expected = []
    for group in range(3 if comparison == "ovr" else 2):
        selection = (
            np.ones(rows, bool)
            if comparison == "ovr"
            else (codes == 2) | (codes == group)
        )
        ranked = rankdata(dense[selection], axis=0)
        expected.append(ranked[codes[selection] == group].sum(axis=0))
    original_empty = cp.empty
    arenas = []
    allocations = []
    with (
        cp.cuda.Stream(non_blocking=True) as work,
        cp.cuda.using_allocator(pool.malloc),
    ):
        ranks = cp.empty((len(expected), cols), cp.float64)
        ties = cp.empty(cols if comparison == "ovr" else ranks.shape, cp.float64)
        sums, counts = cp.empty((3, cols), cp.float64), cp.empty((3, cols), cp.float64)
        totals, total_counts = cp.empty(cols, cp.float64), cp.empty(cols, cp.float64)

        def allocate(*args, **kwargs):
            assert cp.cuda.get_current_stream().ptr == work.ptr
            output = original_empty(*args, **kwargs)
            output.fill(17)
            allocations.append(weakref.ref(output))
            if output.dtype == cp.uint8:
                # At most one surviving sort arena belongs to each slot.
                assert sum(owner() is not None for owner in arenas) < 4
                arenas.append(weakref.ref(output))
            return output

        monkeypatch.setattr(cp, "empty", allocate)
        if comparison == "ovr":

            def unexpected_zero_allocation(*args, **kwargs):
                pytest.fail("one-row sparse sorts must reuse the existing index arena")

            # The final empty column has a one-row padded sort window. Its
            # compact zero indices must also come from the existing arena.
            monkeypatch.setattr(cp, "zeros", unexpected_zero_allocation)
        # Warm both allocator bookkeeping and scratch-size buckets before
        # checking that repeated calls stop reserving device storage.
        for iteration in range(3):
            if comparison == "ovr":
                kernel.ovr_sparse_csc_host(
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
                    compute_tie_corr=True,
                    compute_totals=True,
                    sub_batch_cols=3,
                )
            else:
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
                    counts,
                    n_ref=len(reference),
                    n_all_grp=len(grouped),
                    compute_tie_corr=True,
                    sub_batch_cols=3,
                )
            np.testing.assert_array_equal(ranks.get(stream=work), expected)
            np.testing.assert_array_equal(
                sums.get(stream=work),
                np.stack([dense[codes == group].sum(axis=0) for group in range(3)]),
            )
            assert cp.cuda.get_current_stream().ptr == work.ptr
            if not asynchronous:
                if iteration == 1:
                    reserved = pool.total_bytes()
                elif iteration == 2:
                    assert pool.total_bytes() == reserved
        assert len(arenas) > 4
        assert allocations and all(owner() is None for owner in allocations)


@pytest.mark.parametrize(
    "values,expected_ranks,expected_tie",
    [
        ([1.0, 0.0, -0.0, -2.0], [6.5, 3.5], 0.9),
        ([np.nan, np.nan, 0.0, 1.0], [4.0, 6.0], 1.0),
        ([-np.nan, -np.nan, 0.0, 1.0], [4.0, 6.0], 0.6),
        ([-np.inf, np.inf, 0.0, 1.0], [3.0, 7.0], 1.0),
        ([np.nan, np.nan, np.nan, np.nan], [4.0, 6.0], 1.0),
    ],
)
def test_sparse_ovr_preserves_special_value_ranking(
    values, expected_ranks, expected_tie
):
    # These expectations preserve the original sparse backend's ordering and
    # analytic-zero rules, including singleton positive-NaN ranks.
    data = cp.asarray(values, dtype=cp.float32)
    indices = cp.arange(4, dtype=cp.int32)
    indptr = cp.asarray([0, 4], dtype=cp.int32)
    codes = cp.asarray([0, 1, 0, 1], dtype=cp.int32)
    sizes = cp.asarray([2, 2], dtype=cp.float64)
    ranks = cp.empty((2, 1), dtype=cp.float64)
    ties = cp.empty(1, dtype=cp.float64)
    kernel.ovr_sparse_csc_device(
        data, indices, indptr, codes, sizes, ranks, ties, compute_tie_corr=True
    )
    np.testing.assert_array_equal(ranks.get().ravel(), expected_ranks)
    np.testing.assert_allclose(ties.get(), [expected_tie])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_sparse_ovr_keeps_sort_storage_sparse(monkeypatch, dtype, index_dtype):
    rng = np.random.default_rng(42)
    rows, cols = 20001, 67
    source = sparse.random(rows, cols, density=0.003, format="csc", random_state=rng)
    source.data = rng.integers(-3, 7, source.nnz).astype(dtype)
    # One dense column beside sparse columns exercises skewed padding budgets.
    source = sparse.hstack(
        [sparse.csc_matrix(np.ones((rows, 1))), source], format="csc"
    )
    source.data = source.data.astype(dtype)
    source.indices = source.indices.astype(index_dtype)
    source.indptr = source.indptr.astype(index_dtype)
    matrix = gpu_sparse.csc_matrix(source)
    matrix.indices = matrix.indices.astype(index_dtype)
    matrix.indptr = matrix.indptr.astype(index_dtype)
    codes_host = np.arange(rows, dtype=np.int32) % 4 - 1
    codes = cp.asarray(codes_host)
    sizes = cp.asarray(
        [(codes_host == group).sum() for group in range(3)], dtype=cp.float64
    )
    ranks = cp.empty((3, cols + 1), dtype=cp.float64)
    ties = cp.empty(cols + 1, dtype=cp.float64)
    allocations = []
    original_empty = cp.empty
    original_zeros = cp.zeros

    def record_shape(shape, args, kwargs):
        dimensions = (shape,) if isinstance(shape, (int, np.integer)) else tuple(shape)
        dtype = np.dtype(kwargs.get("dtype", args[0] if args else np.float64))
        allocation_bytes = int(np.prod(dimensions)) * dtype.itemsize
        # An arena combines values, permutations and radix metadata in one
        # uint8 allocation. Allow the documented up-to-fourfold sparse padding
        # around a skewed dense column, including its combined radix arena,
        # while still rejecting a dense 32- or 64-column float32 window.
        allocations.append(allocation_bytes)
        assert allocation_bytes < rows * 128, (
            "sparse ranking exceeded its padding budget"
        )

    def bounded_empty(shape, *args, **kwargs):
        record_shape(shape, args, kwargs)
        return original_empty(shape, *args, **kwargs)

    def bounded_zeros(shape, *args, **kwargs):
        record_shape(shape, args, kwargs)
        return original_zeros(shape, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", bounded_empty)
    monkeypatch.setattr(cp, "zeros", bounded_zeros)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        kernel.ovr_sparse_csc_device(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            codes,
            sizes,
            ranks,
            ties,
            compute_tie_corr=True,
            sub_batch_cols=64,
        )
    stream.synchronize()
    dense = source.toarray().astype(np.float32)
    ranked = rankdata(dense, axis=0)
    expected = np.stack(
        [
            ranked[codes_host == group].sum(axis=0, dtype=np.float64)
            for group in range(3)
        ]
    )
    np.testing.assert_array_equal(ranks.get(), expected)
    np.testing.assert_allclose(
        ties.get(), [tiecorrect(ranked[:, col]) for col in range(cols + 1)]
    )
    # The heavily populated column must not force every sparse column to use
    # rows entries of sorting scratch.
    assert allocations
    assert max(allocations) < rows * 128


@pytest.mark.parametrize("groups", [512, 513, 3056, 3057])
@pytest.mark.parametrize("compute_tie_corr", [False, True])
def test_sparse_ovr_shared_and_global_group_accumulators(groups, compute_tie_corr):
    rng = np.random.default_rng(481)
    rows, cols = max(2053, groups + 41), 5
    source = sparse.random(rows, cols, density=0.35, format="csc", random_state=rng)
    source.data = rng.integers(-4, 6, source.nnz).astype(np.float64)
    matrix = gpu_sparse.csc_matrix(source)
    matrix.indices = matrix.indices.astype(cp.int64)
    matrix.indptr = matrix.indptr.astype(cp.int64)
    codes = np.arange(rows, dtype=np.int32) % groups
    sizes = np.bincount(codes, minlength=groups).astype(np.float64)
    ranks = cp.empty((groups, cols), dtype=cp.float64)
    ties = cp.full(cols, -13, dtype=cp.float64)
    d_codes, d_sizes = cp.asarray(codes), cp.asarray(sizes)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        kernel.ovr_sparse_csc_device(
            matrix.data,
            matrix.indices,
            matrix.indptr,
            d_codes,
            d_sizes,
            ranks,
            ties,
            compute_tie_corr=compute_tie_corr,
            sub_batch_cols=3,
        )
    stream.synchronize()
    expected = rankdata(source.toarray().astype(np.float32), axis=0)
    np.testing.assert_array_equal(
        ranks.get(),
        np.stack([expected[codes == group].sum(axis=0) for group in range(groups)]),
    )
    if compute_tie_corr:
        np.testing.assert_allclose(
            ties.get(), [tiecorrect(expected[:, col]) for col in range(cols)]
        )
    else:
        np.testing.assert_array_equal(ties.get(), -13)


@pytest.mark.parametrize("comparison", ["ovr", "ovo"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_device_csr_windows_preserve_unsorted_rows_and_empty_columns(
    monkeypatch, comparison, dtype, index_dtype
):
    rng = np.random.default_rng(507)
    rows, cols = 4099, 13
    dense = np.zeros((rows, cols), dtype=dtype)
    dense[:, 4] = rng.integers(-3, 4, rows)
    selected = rng.choice(rows, 300, replace=False)
    dense[selected, 7] = rng.integers(-3, 4, selected.size)
    dense[selected[::3], 8] = 2
    dense[selected[::7], 12] = -1
    source = sparse.csr_matrix(dense)
    # Native conversion must accept unsorted rows and independently sized
    # index/offset types. Alternate empty and populated windows force buffer
    # growth and reuse, including unused capacity after a dense column.
    for row in range(rows):
        start, stop = source.indptr[row : row + 2]
        source.indices[start:stop] = source.indices[start:stop][::-1]
        source.data[start:stop] = source.data[start:stop][::-1]
    data = cp.asarray(source.data)
    indices = cp.asarray(source.indices, dtype=index_dtype)
    pointer_dtype = np.int64 if index_dtype == np.int32 else np.int32
    indptr = cp.asarray(source.indptr, dtype=pointer_dtype)

    def reject_python_conversion(*args, **kwargs):
        pytest.fail("device CSR staging used Python sparse conversion")

    monkeypatch.setattr(gpu_sparse.csr_matrix, "tocsc", reject_python_conversion)
    codes_host = np.arange(rows, dtype=np.int32) % 3
    codes = cp.asarray(codes_host)
    sizes = cp.asarray(np.bincount(codes_host), dtype=cp.float64)
    groups = 3 if comparison == "ovr" else 2
    ranks = cp.empty((groups, cols), dtype=cp.float64)
    ties = cp.empty(cols if comparison == "ovr" else (groups, cols), dtype=cp.float64)
    reference = np.flatnonzero(codes_host == 2).astype(np.int32)
    members = [
        np.flatnonzero(codes_host == group).astype(np.int32) for group in range(2)
    ]
    grouped = cp.asarray(np.concatenate(members))
    offsets = cp.asarray([0, len(members[0]), sum(map(len, members))], dtype=cp.int32)
    refs = cp.asarray(reference)
    stream = cp.cuda.Stream(non_blocking=True)
    cp.cuda.get_current_stream().synchronize()
    with stream:
        if comparison == "ovr":
            kernel.ovr_sparse_csr_device(
                data,
                indices,
                indptr,
                codes,
                sizes,
                ranks,
                ties,
                compute_tie_corr=True,
                sub_batch_cols=3,
            )
        else:
            kernel.ovo_streaming_csr_device(
                data,
                indices,
                indptr,
                refs,
                grouped,
                offsets,
                ranks,
                ties,
                n_ref=len(reference),
                n_all_grp=sum(map(len, members)),
                compute_tie_corr=True,
                sub_batch_cols=3,
            )
    stream.synchronize()
    if comparison == "ovr":
        ranked = rankdata(dense.astype(np.float32), axis=0)
        np.testing.assert_array_equal(
            ranks.get(),
            np.stack([ranked[codes_host == group].sum(axis=0) for group in range(3)]),
        )
        np.testing.assert_allclose(
            ties.get(), [tiecorrect(ranked[:, col]) for col in range(cols)]
        )
    else:
        for group in range(2):
            selected = np.concatenate((reference, members[group]))
            ranked = rankdata(dense[selected].astype(np.float32), axis=0)
            np.testing.assert_array_equal(
                ranks[group].get(), ranked[len(reference) :].sum(axis=0)
            )
            np.testing.assert_allclose(
                ties[group].get(), [tiecorrect(ranked[:, col]) for col in range(cols)]
            )


@pytest.mark.parametrize("comparison", ["ovr", "ovo"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_host_csr_respects_each_supplied_row_span(comparison, dtype, index_dtype):
    rng = np.random.default_rng(718)
    rows, cols = 11, 7
    values = rng.choice([-3, -1, 1, 2, 4], (rows, cols)).astype(dtype)
    source = sparse.csr_matrix(values)
    source.indices = source.indices.astype(index_dtype)
    source.indptr = source.indptr.astype(index_dtype)
    starts = source.indptr[:-1] + (1 + np.arange(rows) % 2).astype(index_dtype)
    stops = starts + 3
    selected = np.zeros_like(values)
    for row, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        selected[row, source.indices[start:stop]] = source.data[start:stop]
    first, stop = 1, 6
    selected = selected[:, first:stop]
    ranks = cp.empty((3 if comparison == "ovr" else 2, stop - first), dtype=cp.float64)
    ties = cp.empty(
        stop - first if comparison == "ovr" else ranks.shape, dtype=cp.float64
    )
    sums = cp.empty((3, stop - first), dtype=cp.float64)
    counts = cp.empty_like(sums)
    codes = np.arange(rows, dtype=np.int32) % 3
    sizes = np.bincount(codes).astype(np.float64)
    reference = np.arange(3, dtype=np.int32)
    grouped = np.arange(3, rows, dtype=np.int32)
    offsets = np.asarray([0, 4, 8], dtype=np.int32)
    totals = cp.empty((1, stop - first), dtype=cp.float64)
    total_counts = cp.empty_like(totals)
    if comparison == "ovr":
        kernel.ovr_sparse_csr_host(
            source.data,
            source.indices,
            source.indptr,
            starts,
            stops,
            codes,
            sizes,
            ranks,
            ties,
            sums,
            counts,
            totals,
            total_counts,
            n_cols=cols,
            compute_tie_corr=True,
            compute_nnz=True,
            compute_totals=True,
            col_start=first,
            col_stop=stop,
            sub_batch_cols=2,
        )
        ranked = rankdata(selected.astype(np.float32), axis=0)
        np.testing.assert_array_equal(
            ranks.get(),
            np.stack([ranked[codes == group].sum(axis=0) for group in range(3)]),
        )
        np.testing.assert_allclose(
            ties.get(), [tiecorrect(ranked[:, col]) for col in range(stop - first)]
        )
        np.testing.assert_array_equal(totals.get()[0], selected.sum(axis=0))
        np.testing.assert_array_equal(
            total_counts.get()[0], np.count_nonzero(selected, axis=0)
        )
    else:
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
            n_cols=cols,
            compute_tie_corr=True,
            compute_nnz=True,
            col_start=first,
            col_stop=stop,
            sub_batch_cols=2,
        )
        codes = np.r_[np.full(3, 2), np.repeat([0, 1], 4)]
        for group in range(2):
            members = grouped[offsets[group] : offsets[group + 1]]
            ranked = rankdata(
                selected[np.r_[reference, members]].astype(np.float32), axis=0
            )
            np.testing.assert_array_equal(
                ranks[group].get(), ranked[len(reference) :].sum(axis=0)
            )
            np.testing.assert_allclose(
                ties[group].get(),
                [tiecorrect(ranked[:, col]) for col in range(stop - first)],
            )
    for group in range(3):
        np.testing.assert_array_equal(
            sums[group].get(), selected[codes == group].sum(axis=0)
        )
        np.testing.assert_array_equal(
            counts[group].get(), np.count_nonzero(selected[codes == group], axis=0)
        )
