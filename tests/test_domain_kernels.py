"""Boundary and precision checks for native domain kernel implementations."""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from scipy.special import logsumexp

from rapids_singlecell._cuda import _cooc_cuda, _edistance_cuda, _kde_cuda


@pytest.mark.parametrize("n_thresholds", [1, 257, 513])
def test_cooccurrence_threshold_windows_and_large_counters(n_thresholds):
    rng = np.random.default_rng(14)
    counts_by_group = [2, 129, 5]
    offsets = np.r_[0, np.cumsum(counts_by_group)].astype(np.int32)
    xy = rng.integers(0, 13, size=(offsets[-1], 2)).astype(np.float32)
    thresholds = np.linspace(0, 300, n_thresholds, dtype=np.float32)
    left, right = np.triu_indices(len(counts_by_group))
    initial = np.uint64(2**40)
    counts = cp.full((3, 3, n_thresholds), initial, dtype=cp.uint64)
    cell_tile, _, block_size, shared_mem = _cooc_cuda.get_kernel_config(
        n_thresholds, len(xy), 3
    )
    _cooc_cuda.count_csr_catpairs(
        cp.asarray(xy),
        thresholds=cp.asarray(thresholds),
        cat_offsets=cp.asarray(offsets),
        cell_indices=cp.arange(len(xy), dtype=cp.int32),
        pair_left=cp.asarray(left, dtype=cp.int32),
        pair_right=cp.asarray(right, dtype=cp.int32),
        counts=counts,
        num_pairs=len(left),
        k=3,
        l_val=n_thresholds,
        blocks_per_pair=3,
        cell_tile=cell_tile,
        block_size=block_size,
        shared_mem=shared_mem,
    )
    expected = np.full(counts.shape, initial, dtype=np.uint64)
    for a, b in zip(left, right, strict=True):
        delta = (
            xy[offsets[a] : offsets[a + 1], None]
            - xy[None, offsets[b] : offsets[b + 1]]
        )
        distance = (delta * delta).sum(axis=-1)
        if a == b:
            distance = distance[np.triu_indices(len(distance), 1)]
        expected[a, b] += (distance.reshape(-1, 1) <= thresholds).sum(
            axis=0, dtype=np.uint64
        )
    np.testing.assert_array_equal(counts.get(), expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_kde_stable_logsumexp_on_external_stream(dtype):
    rng = np.random.default_rng(20)
    xy = rng.normal(size=(257, 2)).astype(dtype)
    a, b, c = -150.0, 35.0, -90.0
    delta = xy[:, None].astype(np.float64) - xy[None].astype(np.float64)
    dx, dy = delta[..., 0], delta[..., 1]
    expected = logsumexp(a * dx * dx + b * dx * dy + c * dy * dy, axis=1)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        device_xy = cp.asarray(xy)
        output = cp.empty(len(xy), dtype=dtype)
        _kde_cuda.gaussian_kde_2d(
            device_xy, out=output, n=len(xy), a=a, b=b, c=c, stream=stream.ptr
        )
    stream.synchronize()
    tolerance = 2e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(output.get(), expected, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize("n_features", [1, 33, 65])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_energy_distance_feature_and_cell_tile_tails(n_features, dtype):
    rng = np.random.default_rng(32)
    x = rng.normal(size=(64, n_features)).astype(dtype)
    offsets = np.array([0, 31, 64], dtype=np.int32)
    left, right = np.triu_indices(2)
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        args = [
            cp.asarray(x),
            cp.asarray(offsets),
            cp.arange(len(x), dtype=cp.int32),
            cp.asarray(left, dtype=cp.int32),
            cp.asarray(right, dtype=cp.int32),
            cp.full(3, 7, dtype=dtype),
        ]
        cell_tile, feat_tile, block_size, shared_mem = (
            _edistance_cuda.get_kernel_config(n_features, dtype == np.float64)
        )
        _edistance_cuda.compute_distances(
            *args,
            num_pairs=3,
            n_features=n_features,
            blocks_per_pair=3,
            cell_tile=cell_tile,
            feat_tile=feat_tile,
            block_size=block_size,
            shared_mem=shared_mem,
            stream=stream.ptr,
        )
    stream.synchronize()
    expected = []
    for a, b in zip(left, right, strict=True):
        delta = x[offsets[a] : offsets[a + 1], None].astype(np.float64) - x[
            None, offsets[b] : offsets[b + 1]
        ].astype(np.float64)
        distance = np.sqrt((delta * delta).sum(axis=-1))
        if a == b:
            distance = distance[np.triu_indices(len(distance), 1)]
        expected.append(7 + distance.sum())
    tolerance = 3e-7 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(args[-1].get(), expected, rtol=tolerance)
