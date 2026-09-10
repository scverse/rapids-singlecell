"""Memory bounds and completion guarantees for sparse NaN rank windows."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as kernel


def _prepare(adapter, *, columns=7, finite=False):
    reference = np.arange(1, 6, dtype=np.float32)
    if not finite:
        reference.fill(np.nan)
    values = np.r_[reference, np.float32(6 if finite else 1), 0.0, -0.0].astype(
        np.float32
    )
    data = np.tile(values, columns)
    indices = np.tile(np.arange(8, dtype=np.int32), columns)
    pointers = np.arange(0, data.size + 1, 8, dtype=np.int32)
    refs = np.r_[np.arange(5, dtype=np.int32), np.full(3, -1, np.int32)]
    groups = np.r_[np.full(5, -1, np.int32), np.arange(3, dtype=np.int32)]
    offsets = np.asarray([0, 3], np.int32)
    ranks = cp.full((1, columns), -11, cp.float64)
    ties = cp.full_like(ranks, -13)
    statistics = [cp.empty((2, columns), cp.float64) for _ in range(2)]
    if adapter == "device":
        data, indices, pointers, refs, groups, offsets = map(
            cp.asarray, (data, indices, pointers, refs, groups, offsets)
        )
    cp.cuda.get_current_stream().synchronize()

    def run():
        options = {
            "n_ref": 5,
            "n_all_grp": 3,
            "compute_tie_corr": True,
            "sub_batch_cols": columns,
        }
        if adapter == "host":
            kernel.ovo_streaming_csc_host(
                data,
                indices,
                pointers,
                refs,
                groups,
                offsets,
                np.r_[np.ones(5, np.int32), np.zeros(3, np.int32)],
                ranks,
                ties,
                *statistics,
                compute_nnz=True,
                **options,
            )
        else:
            kernel.ovo_streaming_csc_device(
                data, indices, pointers, refs, groups, offsets, ranks, ties, **options
            )

    return run, ranks, ties, statistics


@pytest.mark.parametrize("adapter", ["host", "device"])
def test_sparse_nan_fallback_splits_the_window_within_budget(monkeypatch, adapter):
    run, ranks, ties, statistics = _prepare(adapter)
    # The ordinary compact window fits all seven columns, while the bounded
    # dense workspace can hold only two. Every final partial window is reused.
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (500_000, 8_000_000))
    original = cp.empty
    float_allocations = []

    def allocate(shape, dtype=float, *args, **kwargs):
        if np.dtype(dtype) == np.dtype("float32"):
            dims = (shape,) if isinstance(shape, int) else tuple(shape)
            float_allocations.append(dims)
            # Host CSC uploads its56 stored source values once. Dense
            # reference/group slabs must still stay at two columns each;
            # a full reference/group window would need35/21 values.
            if adapter != "host" or dims != (56,):
                assert np.prod(dims) <= 10
        return original(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", allocate)
    with cp.cuda.Stream(non_blocking=True) as stream:
        run()
    assert stream.done
    assert float_allocations.count((56,)) == (1 if adapter == "host" else 0)
    assert (10,) in float_allocations and (6,) in float_allocations
    cp.testing.assert_array_equal(ranks, 6)
    cp.testing.assert_allclose(ties, 1 - 6 / (8**3 - 8), rtol=1e-15)
    if adapter == "host":
        cp.testing.assert_array_equal(statistics[0][0], 1)
        assert bool(cp.isnan(statistics[0][1]).all())
        cp.testing.assert_array_equal(statistics[1][0], 1)
        cp.testing.assert_array_equal(statistics[1][1], 5)


@pytest.mark.parametrize("adapter", ["host", "device"])
def test_sparse_nan_allocation_failure_drains_pending_work(monkeypatch, adapter):
    run, _, _, _ = _prepare(adapter)
    pending = cp.ones(1_000_003, cp.float64)
    cp.cuda.get_current_stream().synchronize()
    original = cp.empty
    failed = False

    def allocate(shape, dtype=float, *args, **kwargs):
        nonlocal failed
        if np.dtype(dtype) == np.dtype("float32"):
            failed = True
            for _ in range(16):
                cp.sin(pending, out=pending)
            raise RuntimeError("injected sparse NaN allocation failure")
        return original(shape, dtype, *args, **kwargs)

    monkeypatch.setattr(cp, "empty", allocate)
    caller = cp.cuda.get_current_stream()
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        with stream, pytest.raises(RuntimeError, match="sparse NaN allocation failure"):
            run()
        assert failed and stream.done
        assert cp.cuda.get_current_stream().ptr == caller.ptr
    finally:
        stream.synchronize()


def test_finite_host_population_plan_does_not_read_device_data(monkeypatch):
    run, ranks, _, _ = _prepare("host", finite=True)

    def forbidden(*args, **kwargs):
        raise AssertionError("finite host planning must not read device data")

    with monkeypatch.context() as context:
        context.setattr(cp, "asnumpy", forbidden)
        run()
    cp.testing.assert_array_equal(ranks, 11)


def test_segmented_metadata_failure_keeps_prior_upload_alive(monkeypatch):
    from rapids_singlecell._cuda import _wilcoxon_cuda

    reference = cp.ones((5, 3), cp.float32, order="F")
    group = cp.ones((2501, 3), cp.float32, order="F")
    offsets = cp.asarray([0, 2501], dtype=cp.int64)
    ranks = cp.empty((1, 3), cp.float64)
    ties = cp.empty_like(ranks)
    pending = cp.ones(1_000_003, cp.float64)
    cp.cuda.get_current_stream().synchronize()
    original = cp.asarray
    uploads = []

    def upload(source, *args, **kwargs):
        if (
            isinstance(source, list)
            and len(source) == 1
            and kwargs.get("dtype") == "uint64"
        ):
            if uploads:
                raise RuntimeError("injected segmented metadata failure")
            result = original(source, *args, **kwargs)
            # Keep only its pointer in Python; native constructor ownership
            # must protect the successful upload through the next failure.
            uploads.append(result.data.ptr)
            for _ in range(16):
                cp.sin(pending, out=pending)
            return result
        return original(source, *args, **kwargs)

    monkeypatch.setattr(cp, "asarray", upload)
    stream = cp.cuda.Stream(non_blocking=True)
    try:
        with pytest.raises(RuntimeError, match="segmented metadata failure"):
            _wilcoxon_cuda.ovo_rank_dense_tiered_unsorted_ref(
                reference,
                group,
                offsets,
                ranks,
                ties,
                compute_tie_corr=True,
                sub_batch_cols=2,
                stream=stream.ptr,
            )
        assert len(uploads) == 1 and stream.done
    finally:
        stream.synchronize()
