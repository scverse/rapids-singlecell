"""Host aggregation preserves layouts and drains private uploads on failure."""

from __future__ import annotations

import weakref

import cupy as cp
import numpy as np
import pytest
from scipy import sparse

from rapids_singlecell._cuda import _rank_stream_cuda as backend

pytestmark = pytest.mark.skipif(
    backend is None or getattr(backend, "__backend__", None) != "rust",
    reason="requires the Rust host-streaming backend",
)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("histogram", [False, True])
def test_dense_ndarray_subclass_uses_validated_buffer_storage(order, histogram):
    class BufferOnly(np.ndarray):
        def __getitem__(self, index):
            raise AssertionError(
                "native streaming must slice normalized buffer storage"
            )

    source = np.ones((11, 7), np.float32, order=order).view(BufferOnly)
    labels = cp.zeros(11, cp.int32)
    sums = cp.full((1, 7), 17.0)
    if histogram:
        output = cp.zeros((5, 1, 4), cp.uint32)
        backend.hist_dense_host(
            source,
            labels,
            output,
            group_sums=sums,
            n_groups=1,
            n_bins=3,
            bin_low=0.0,
            inv_bin_width=1.0,
            col_start=1,
            col_stop=6,
            sub_batch_cols=2,
        )
        expected = np.zeros((5, 1, 4), np.uint32)
        expected[:, 0, 2] = 11
        np.testing.assert_array_equal(output.get(), expected)
        np.testing.assert_array_equal(sums.get(), [[17, 11, 11, 11, 11, 11, 17]])
    else:
        backend.aggr_dense_host(source, labels, out_sum=sums, sub_batch=2)
        np.testing.assert_array_equal(sums.get(), 28 if order == "C" else 11)


@pytest.mark.parametrize("shape", [(9, 1), (1, 9), (0, 9), (9, 0)])
def test_degenerate_dense_aggregation_preserves_additive_outputs(shape):
    values = np.ones(shape, dtype=np.float32, order="F")
    labels = cp.zeros(shape[0], cp.int32)
    outputs = [cp.full((1, shape[1]), 17.0) for _ in range(3)]
    backend.aggr_dense_host(
        values,
        labels,
        out_sum=outputs[0],
        out_count=outputs[1],
        out_sqsum=outputs[2],
        sub_batch=2,
    )
    for output in outputs:
        np.testing.assert_array_equal(output.get(), 17.0 + shape[0])


@pytest.mark.parametrize("layout", ["C", "F", "csr", "csc"])
def test_streamed_upload_failure_waits_and_restores_stream(monkeypatch, layout):
    source = np.ones((1003, 13), np.float64, order="F" if layout == "F" else "C")
    if layout in ("csr", "csc"):
        source = getattr(sparse, f"{layout}_matrix")(source)
    outputs = cp.zeros((1, 13), cp.float64)
    cats = cp.zeros(1003, cp.int32)
    cp.cuda.get_current_stream().synchronize()
    original_allocator = cp.cuda.alloc_pinned_memory
    original_stream = cp.cuda.Stream
    streams = []
    allocations = 0

    def allocate(size):
        nonlocal allocations
        allocations += 1
        if allocations == 2:
            raise RuntimeError("injected pinned allocation failure")
        return original_allocator(size)

    def create_stream(*args, **kwargs):
        stream = original_stream(*args, **kwargs)
        streams.append(stream)
        return stream

    caller = cp.cuda.get_current_stream()
    with original_stream(non_blocking=True) as work:
        monkeypatch.setattr(cp.cuda, "alloc_pinned_memory", allocate)
        monkeypatch.setattr(cp.cuda, "Stream", create_stream)
        with pytest.raises(RuntimeError, match="injected pinned allocation failure"):
            if layout in ("csr", "csc"):
                getattr(backend, f"aggr_{layout}_host")(
                    source.data,
                    source.indices,
                    source.indptr,
                    cats,
                    out_sum=outputs,
                    n_cells=1003,
                    n_genes=13,
                    **{f"sub_batch_{'rows' if layout == 'csr' else 'cols'}": 3},
                )
            else:
                backend.aggr_dense_host(source, cats, out_sum=outputs, sub_batch=3)
        assert cp.cuda.get_current_stream().ptr == work.ptr
        assert streams and all(stream.done for stream in streams)
    assert cp.cuda.get_current_stream().ptr == caller.ptr
    expected = np.zeros((1, 13))
    if layout == "C":
        expected[:] = 3
    elif layout == "F":
        expected[:, :3] = 1003
    np.testing.assert_array_equal(outputs.get(), expected)
    cp.get_default_memory_pool().free_all_blocks()
    cp.testing.assert_array_equal(cp.arange(31) * 2, np.arange(31) * 2)


@pytest.mark.parametrize("layout", ["C", "F", "csc"])
def test_histogram_flat_buffer_window_preserves_tail(layout):
    values = np.arange(77, dtype=np.float32).reshape(7, 11) % 5
    labels = np.arange(7, dtype=np.int32) % 2
    source = (
        sparse.csc_matrix(values) if layout == "csc" else np.array(values, order=layout)
    )
    start, stop, bins, groups = 2, 9, 5, 2
    size = (stop - start) * groups * (bins + 1)
    output = cp.zeros(size + 13, cp.uint32)
    output[size:] = 91
    options = {
        "n_groups": groups,
        "n_bins": bins,
        "bin_low": 0.0,
        "inv_bin_width": 1.0,
        "col_start": start,
        "col_stop": stop,
        "sub_batch_cols": 2,
    }
    if layout == "csc":
        backend.hist_csc_host(
            source.data,
            source.indices,
            source.indptr,
            cp.asarray(labels),
            output,
            n_cells=7,
            n_genes=11,
            **options,
        )
    else:
        backend.hist_dense_host(source, cp.asarray(labels), output, **options)
    expected = np.zeros((stop - start, groups, bins + 1), np.uint32)
    for row, group in enumerate(labels):
        for col in range(start, stop):
            value = values[row, col]
            if layout != "csc" or value != 0:
                expected[col - start, group, int(value) + 1] += 1
    np.testing.assert_array_equal(output[:size].get().reshape(expected.shape), expected)
    np.testing.assert_array_equal(output[size:].get(), 91)


def test_invalid_histogram_bins_fail_before_output_mutation():
    values = np.ones((7, 11), np.float32, order="F")
    sums = cp.full((2, 11), 17.0)
    with pytest.raises(ValueError, match="bin"):
        backend.hist_dense_host(
            values,
            cp.arange(7, dtype=cp.int32) % 2,
            cp.zeros(1, cp.uint32),
            group_sums=sums,
            n_groups=2,
            n_bins=2**64 - 1,
            bin_low=0.0,
            inv_bin_width=1.0,
            col_start=2,
            col_stop=9,
        )
    np.testing.assert_array_equal(sums.get(), 17.0)


def test_dense_zero_skip_preserves_signed_zero_accumulator():
    values = np.zeros((13, 7), np.float64)
    sums = cp.full((1, 7), -0.0)
    squares = cp.full_like(sums, -0.0)
    backend.aggr_dense_host(
        values, cp.zeros(13, cp.int32), out_sum=sums, out_sqsum=squares
    )
    np.testing.assert_array_equal(np.signbit(sums.get()), True)
    np.testing.assert_array_equal(np.signbit(squares.get()), True)


def test_empty_sparse_batches_budget_pointer_storage(monkeypatch):
    rows = 4097
    pointers = np.zeros(rows + 1, np.int64)
    outputs = cp.full((1, 3), 17.0)
    cats = cp.zeros(rows, cp.int32)
    original_empty = cp.empty
    sizes = []

    def allocate(shape, dtype=float, *args, **kwargs):
        sizes.append(int(np.prod(shape)) * np.dtype(dtype).itemsize)
        return original_empty(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (20480, 20480))
    monkeypatch.setattr(cp, "empty", allocate)
    backend.aggr_csr_host(
        np.empty(0, np.float32),
        np.empty(0, np.int32),
        pointers,
        cats,
        out_sum=outputs,
        n_cells=rows,
        n_genes=3,
        sub_batch_rows=1 << 20,
    )
    # Both slots' indptr allocations count toward the 20% device budget even
    # though the data and index uploads have no nonzero values.
    assert sum(sizes) <= 4096
    np.testing.assert_array_equal(outputs.get(), 17.0)


@pytest.mark.parametrize("layout", ["csr", "csc"])
def test_sparse_histogram_promotes_wide_indices_with_narrow_offsets(layout):
    values = np.arange(77, dtype=np.float32).reshape(7, 11) % 5
    source = getattr(sparse, f"{layout}_matrix")(values)
    source.indices = source.indices.astype(np.int64)
    source.indptr = source.indptr.astype(np.int32)
    labels = np.arange(7, dtype=np.int32) % 2
    output = cp.zeros((11, 2, 6), cp.uint32)
    getattr(backend, f"hist_{layout}_host")(
        source.data,
        source.indices,
        source.indptr,
        cp.asarray(labels),
        output,
        n_cells=7,
        n_genes=11,
        n_groups=2,
        n_bins=5,
        bin_low=0.0,
        inv_bin_width=1.0,
        col_start=0,
        col_stop=11,
        **{f"sub_batch_{'rows' if layout == 'csr' else 'cols'}": 2},
    )
    expected = np.zeros(output.shape, np.uint32)
    for row, group in enumerate(labels):
        for col, value in enumerate(values[row]):
            if value != 0:
                expected[col, group, int(value) + 1] += 1
    np.testing.assert_array_equal(output.get(), expected)


def test_sparse_staging_growth_respects_live_device_budget(monkeypatch):
    stored = np.array([2, 2, 40, 40, 80, 80, 120, 120], dtype=np.int32)
    pointers = np.r_[np.int32(0), stored.cumsum(dtype=np.int32)]
    indices = np.concatenate([np.arange(n, dtype=np.int32) for n in stored])
    values = np.ones(indices.size, dtype=np.float32)
    cats = cp.zeros(stored.size, cp.int32)
    output = cp.zeros((1, 128), cp.float64)
    original_empty = cp.empty
    allocations = []
    peaks = []
    budget = 2048

    def allocate(shape, dtype=float, *args, **kwargs):
        size = int(np.prod(shape)) * np.dtype(dtype).itemsize
        live = sum(n for reference, n in allocations if reference() is not None)
        peaks.append(live + size)
        assert live + size <= budget, "staging growth exceeded the live device budget"
        result = original_empty(shape, dtype, *args, **kwargs)
        allocations.append((weakref.ref(result), size))
        return result

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (5 * budget, 5 * budget))
    monkeypatch.setattr(cp, "empty", allocate)
    backend.aggr_csr_host(
        values,
        indices,
        pointers,
        cats,
        out_sum=output,
        n_cells=stored.size,
        n_genes=128,
        sub_batch_rows=1,
    )
    # Both slots grow repeatedly; completed storage must be released before
    # replacement allocations even when their individual windows fit the cap.
    assert len(allocations) > 6
    assert max(peaks) <= budget
    expected = (np.arange(128)[None, :] < stored[:, None]).sum(axis=0)
    np.testing.assert_array_equal(output.get()[0], expected)


@pytest.mark.parametrize("order", ["C", "F"])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_staging_reuses_caller_pool_and_orders_allocator_work(
    monkeypatch, order, asynchronous
):
    if asynchronous and not cp.cuda.Device().attributes.get("MemoryPoolsSupported", 0):
        pytest.skip("device does not support stream-ordered allocation")
    pool = cp.cuda.MemoryAsyncPool() if asynchronous else cp.cuda.MemoryPool()
    source = np.arange(101 * 13, dtype=np.float32).reshape(101, 13) % 5
    source = np.array(source, order=order)
    expected = np.zeros((13, 1, 6), np.uint32)
    for col in range(13):
        expected[col, 0, 1:] = np.bincount(source[:, col].astype(int), minlength=5)
    original_empty = cp.empty
    allocations = []
    caller = cp.cuda.get_current_stream()
    with (
        cp.cuda.Stream(non_blocking=True) as work,
        cp.cuda.using_allocator(pool.malloc),
    ):
        labels = cp.zeros(101, cp.int32)
        histogram = cp.zeros(expected.shape, cp.uint32)

        def allocate(*args, **kwargs):
            assert cp.cuda.get_current_stream().ptr == work.ptr
            output = original_empty(*args, **kwargs)
            # Custom allocators may enqueue initialization on their stream.
            # The upload and transpose must run after it, not race this fill.
            output.fill(-17)
            allocations.append(weakref.ref(output))
            return output

        monkeypatch.setattr(cp, "empty", allocate)
        for iteration in range(2):
            histogram.fill(0)
            backend.hist_dense_host(
                source,
                labels,
                histogram,
                n_groups=1,
                n_bins=5,
                bin_low=0.0,
                inv_bin_width=1.0,
                col_start=0,
                col_stop=13,
                sub_batch_cols=3,
            )
            assert cp.cuda.get_current_stream().ptr == work.ptr
            np.testing.assert_array_equal(histogram.get(stream=work), expected)
            if not asynchronous:
                if iteration == 0:
                    reserved = pool.total_bytes()
                else:
                    assert pool.total_bytes() == reserved
        assert allocations and all(reference() is None for reference in allocations)
    assert cp.cuda.get_current_stream().ptr == caller.ptr


def test_failed_allocation_handoff_drains_before_releasing_storage(monkeypatch):
    source = np.ones((100_003, 13), np.float32, order="F")
    original_empty = cp.empty
    original_event = cp.cuda.Event
    freed_after_completion = []
    allocations = []
    caller = cp.cuda.get_current_stream()
    with cp.cuda.Stream(non_blocking=True) as work:
        labels = cp.zeros(len(source), cp.int32)
        sums = cp.zeros((1, 13), cp.float64)

        def allocate(*args, **kwargs):
            assert cp.cuda.get_current_stream().ptr == work.ptr
            output = original_empty(*args, **kwargs)
            output.fill(-17)
            allocations.append(weakref.ref(output))
            weakref.finalize(output, lambda: freed_after_completion.append(work.done))
            return output

        def create_event(*args, **kwargs):
            if allocations:
                raise RuntimeError("injected allocator handoff failure")
            return original_event(*args, **kwargs)

        monkeypatch.setattr(cp, "empty", allocate)
        monkeypatch.setattr(cp.cuda, "Event", create_event)
        with pytest.raises(RuntimeError, match="injected allocator handoff failure"):
            backend.aggr_dense_host(source, labels, out_sum=sums, sub_batch=3)
        assert cp.cuda.get_current_stream().ptr == work.ptr
        assert work.done
        assert allocations and all(reference() is None for reference in allocations)
        assert freed_after_completion and all(freed_after_completion)
        np.testing.assert_array_equal(sums.get(stream=work), 0)
    assert cp.cuda.get_current_stream().ptr == caller.ptr
