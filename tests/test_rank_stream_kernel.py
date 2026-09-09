"""Native host-streaming kernels: batching, masks, and global column offsets."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from scipy import sparse

from rapids_singlecell._cuda import _rank_stream_cuda as backend

pytestmark = pytest.mark.skipif(
    backend is None or getattr(backend, "__backend__", None) != "rust",
    reason="requires the Rust host-streaming backend",
)


def _input(layout, dtype):
    rng = np.random.default_rng(140)
    x = rng.integers(-3, 7, size=(23, 11)).astype(dtype)
    x[rng.random(x.shape) < 0.65] = 0
    if layout in ("csr", "csc"):
        matrix = getattr(sparse, f"{layout}_matrix")(x)
        # Exercise mixed host index widths without changing the logical data.
        matrix.indptr = matrix.indptr.astype(np.int64)
        return x, matrix
    return x, np.array(x, order=layout)


@pytest.mark.parametrize("layout", ["C", "F", "csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_streamed_aggregation_masked_batches(layout, dtype):
    x, host = _input(layout, dtype)
    labels = np.arange(len(x), dtype=np.int32) % 4
    labels[2] = -1
    mask = np.arange(len(x)) % 3 != 0
    cats = cp.asarray(labels)
    outputs = [cp.zeros((4, x.shape[1]), dtype=cp.float64) for _ in range(3)]
    options = {
        "out_sum": outputs[0],
        "out_count": outputs[1],
        "out_sqsum": outputs[2],
        "mask": cp.asarray(mask),
    }
    if sparse.issparse(host):
        getattr(backend, f"aggr_{layout}_host")(
            host.data,
            host.indices,
            host.indptr,
            cats,
            n_cells=x.shape[0],
            n_genes=x.shape[1],
            **{f"sub_batch_{'rows' if layout == 'csr' else 'cols'}": 2},
            **options,
        )
    else:
        backend.aggr_dense_host(host, cats, sub_batch=2, **options)
    for group in range(4):
        values = x[(labels == group) & mask].astype(np.float64)
        for result, expected in zip(
            outputs,
            [values.sum(0), np.count_nonzero(values, axis=0), (values * values).sum(0)],
            strict=True,
        ):
            np.testing.assert_allclose(
                result[group].get(), expected, rtol=1e-12, atol=1e-12
            )


@pytest.mark.parametrize("layout", ["C", "F", "csr", "csc"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_streamed_histogram_window_and_fused_stats(layout, dtype):
    x, host = _input(layout, dtype)
    labels = np.arange(len(x), dtype=np.int32) % 4
    labels[2] = -1
    cats = cp.asarray(labels)
    start, stop, bins = 3, 10, 8
    hist = cp.zeros((stop - start, 4, bins + 1), dtype=cp.uint32)
    sums = cp.zeros((4, x.shape[1]), dtype=cp.float64)
    counts = cp.zeros_like(sums)
    options = {
        "group_sums": sums,
        "group_nnz": counts,
        "n_groups": 4,
        "n_bins": bins,
        "bin_low": -2.0,
        "inv_bin_width": 1.0,
        "col_start": start,
        "col_stop": stop,
    }
    if sparse.issparse(host):
        getattr(backend, f"hist_{layout}_host")(
            host.data,
            host.indices,
            host.indptr,
            cats,
            hist,
            n_cells=x.shape[0],
            n_genes=x.shape[1],
            **{f"sub_batch_{'rows' if layout == 'csr' else 'cols'}": 2},
            **options,
        )
    else:
        backend.hist_dense_host(host, cats, hist, sub_batch_cols=2, **options)
    expected = np.zeros(hist.shape, dtype=np.uint32)
    for row, group in enumerate(labels):
        if group < 0:
            continue
        for col in range(start, stop):
            value = x[row, col]
            if sparse.issparse(host) and value == 0:
                continue
            b = int(np.clip(int(value + 2), 0, bins - 1)) + 1
            expected[col - start, group, b] += 1
    np.testing.assert_array_equal(hist.get(), expected)
    for group in range(4):
        values = x[labels == group].astype(np.float64)
        expected_sum = values.sum(0)
        expected_nnz = np.count_nonzero(values, axis=0)
        if layout != "csr":
            expected_sum[:start] = expected_sum[stop:] = 0
            expected_nnz[:start] = expected_nnz[stop:] = 0
        np.testing.assert_allclose(
            sums[group].get(), expected_sum, rtol=1e-12, atol=1e-12
        )
        np.testing.assert_array_equal(counts[group].get(), expected_nnz)


def test_streamed_missing_output_and_invalid_indptr():
    x = np.ones((3, 2), dtype=np.float32)
    cats = cp.zeros(3, dtype=cp.int32)
    with pytest.raises(ValueError, match="output"):
        backend.aggr_dense_host(x, cats)
    with pytest.raises(ValueError, match="indptr"):
        backend.aggr_csr_host(
            np.ones(2, np.float32),
            np.zeros(2, np.int32),
            np.array([0, 3, 2, 2], np.int64),
            cats,
            out_sum=cp.zeros((1, 2)),
            n_cells=3,
            n_genes=2,
        )


def test_host_worker_limit_returns_previous_value():
    previous = backend._set_host_worker_limit(3)
    try:
        assert backend._set_host_worker_limit(-2) == 3
        assert backend._set_host_worker_limit(1) == 0
    finally:
        backend._set_host_worker_limit(previous)


def test_invalid_output_fails_before_clearing_valid_plane():
    values = np.ones((3, 2), dtype=np.float32, order="F")
    sums = cp.full((1, 2), 17.0, dtype=cp.float64)
    with pytest.raises(TypeError, match="CuPy"):
        backend.aggr_dense_host(
            values,
            cp.zeros(3, dtype=cp.int32),
            out_sum=sums,
            out_count=np.zeros((1, 2)),
        )
    np.testing.assert_array_equal(sums.get(), 17.0)
