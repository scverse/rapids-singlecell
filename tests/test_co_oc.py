from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData, read_h5ad

from rapids_singlecell._cuda import _cooc_cuda
from rapids_singlecell.squidpy_gpu import _co_oc
from rapids_singlecell.squidpy_gpu._co_oc import (
    _count_spatial_tiles,
    _find_min_max,
    co_occurrence,
)


@pytest.fixture
def adata():
    file = Path(__file__).parent / Path("_data/dummy.h5ad")
    dummy_adata = read_h5ad(file)
    dummy_adata.obs["leiden"] = dummy_adata.obs.cluster.astype("category")

    return dummy_adata


def test_co_occurrence(adata: AnnData):
    """
    check co_occurrence score and shape
    """
    co_occurrence(adata, cluster_key="leiden")

    # assert occurrence in adata.uns
    assert "leiden_co_occurrence" in adata.uns.keys()
    assert "occ" in adata.uns["leiden_co_occurrence"].keys()
    assert "interval" in adata.uns["leiden_co_occurrence"].keys()

    # assert shapes
    arr = adata.uns["leiden_co_occurrence"]["occ"]
    assert arr.ndim == 3
    assert arr.shape[2] == 49
    assert arr.shape[1] == arr.shape[0] == adata.obs["leiden"].unique().shape[0]


def test_co_occurrence_reproducibility(adata: AnnData):
    """Check co_occurrence reproducibility results."""
    arr_1, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True)
    arr_2, interval_2 = co_occurrence(adata, cluster_key="leiden", copy=True)

    np.testing.assert_array_equal(sorted(interval_1), sorted(interval_2))
    np.testing.assert_allclose(arr_1, arr_2)


@pytest.mark.parametrize("size", [1, 3])
def test_co_occurrence_explicit_interval(adata: AnnData, size: int):
    minn, maxx = _find_min_max(cp.array(adata.obsm["spatial"]))
    interval = np.linspace(minn, maxx, size)
    if size == 1:
        with pytest.raises(ValueError, match=r"Expected interval to be of length"):
            _ = co_occurrence(adata, cluster_key="leiden", copy=True, interval=interval)
    else:
        _, interval_1 = co_occurrence(adata, cluster_key="leiden", copy=True, interval=interval)

        assert interval is not interval_1
        cp.testing.assert_allclose(interval, interval_1)  # allclose because in the func, we use f32


@pytest.mark.parametrize("n", [1, 2])
def test_co_occurrence_smallest_inputs(n):
    adata = AnnData(shape=(n, 0))
    adata.obsm["spatial"] = np.arange(n, dtype=np.float32)[:, None]
    adata.obs["cluster"] = pd.Categorical(np.zeros(n, dtype=int))
    occ, interval = co_occurrence(adata, "cluster", interval=3, multi_gpu=False, copy=True)

    expected = (interval[1:] >= 1).astype(np.float32) if n == 2 else np.zeros(2)
    np.testing.assert_array_equal(occ[0, 0], expected)


@pytest.mark.parametrize(
    ("spatial", "labels", "lower", "upper", "keys"),
    [
        ((0, 3), 0, 3, 3, 0),
        ((2, 0), 2, 0, 0, 2),
        ((2, 3), 1, 3, 3, 2),
        ((2, 3), 2, 2, 3, 2),
        ((2, 3), 2, 3, 2, 2),
        ((2, 3), 2, 3, 3, 1),
    ],
)
def test_spatial_keys_rejects_inconsistent_shapes(spatial, labels, lower, upper, keys):
    with pytest.raises(ValueError, match="spatial_keys"):
        _cooc_cuda.spatial_keys(
            cp.zeros(spatial, cp.float32),
            cp.zeros(labels, cp.int32),
            cp.zeros(lower, cp.float32),
            cp.ones(upper, cp.float32),
            cp.empty(keys, cp.uint64),
            cp.cuda.get_current_stream().ptr,
        )


@pytest.fixture
def tile_inputs():
    points = cp.zeros((1, 3), cp.float32)
    offsets = cp.asarray([0, 1], cp.int32)
    tiles = cp.empty((1, _cooc_cuda.TILE_BYTES), cp.uint8)
    bounds = cp.empty((1, 3, 2), cp.float32)
    return points, offsets, tiles, bounds, cp.cuda.get_current_stream().ptr


def test_build_tiles_rejects_inconsistent_shapes(tile_inputs):
    points, offsets, tiles, bounds, stream = tile_inputs
    with pytest.raises(ValueError, match="build_tiles"):
        _cooc_cuda.build_tiles(points, offsets, offsets, tiles[:, :-1], bounds, stream)
    with pytest.raises(ValueError, match="build_tiles"):
        _cooc_cuda.build_tiles(points, offsets, offsets, tiles, bounds.reshape(1, 2, 3), stream)


@pytest.mark.parametrize(
    ("bins", "count_shape"),
    [(0, (2, 0)), (3, (2, 2)), (3, (2, 2, 2)), (3, (2, 1, 3)), (3, (6,))],
)
def test_count_tiles_rejects_inconsistent_shapes(tile_inputs, bins, count_shape):
    points, offsets, tiles, bounds, stream = tile_inputs
    _cooc_cuda.build_tiles(points, offsets, offsets, tiles, bounds, stream)
    thresholds = cp.zeros(bins, cp.float32)
    counts = cp.zeros(count_shape, cp.uint64)
    with pytest.raises(ValueError, match="count_tiles"):
        _cooc_cuda.count_tiles(points, tiles, bounds, thresholds, counts, 0, 1, stream)


@pytest.mark.parametrize(
    ("buffer", "shape"),
    [
        ("counts", (4, 3)),
        ("counts", (2, 1, 3)),
        ("counts", (2, 2, 1)),
        ("order", (2,)),
        ("order", (1, 3)),
        ("marginals", (6,)),
        ("marginals", (2, 3)),
        ("out", (4, 3)),
        ("out", (3, 2, 2)),
    ],
)
def test_normalize_rejects_inconsistent_shapes(buffer, shape):
    shapes = {"counts": (2, 2, 3), "order": (3,), "marginals": (3, 2), "out": (2, 2, 3)}
    shapes[buffer] = shape
    with pytest.raises(ValueError, match="normalize"):
        _cooc_cuda.normalize(
            cp.zeros(shapes["counts"], cp.uint64),
            cp.arange(np.prod(shapes["order"]), dtype=cp.int64).reshape(shapes["order"]),
            cp.empty(shapes["marginals"], cp.float32),
            cp.empty(shapes["out"], cp.float32),
            cp.cuda.get_current_stream().ptr,
        )


def _reference_counts(points, labels, thresholds):
    i, j = np.triu_indices(len(points), k=1)
    with np.errstate(invalid="ignore"):
        distances = np.square(points[i] - points[j]).sum(axis=1)
    k = int(labels.max()) + 1
    pairs = np.minimum(labels[i], labels[j]) * k + np.maximum(labels[i], labels[j])
    counts = [np.bincount(pairs[distances <= threshold], minlength=k * k) for threshold in thresholds]
    return np.stack(counts, axis=-1).reshape(k, k, len(thresholds))


@pytest.mark.parametrize("dims", [1, 2, 3, 7])
def test_spatial_tiles_count_partial_and_empty_categories(dims):
    rng = np.random.default_rng(42)
    points = rng.integers(0, 32, size=(75, dims)).astype(np.float32)
    labels = rng.permutation(np.repeat([0, 2], [34, 41])).astype(np.int32)
    thresholds = np.arange(513, dtype=np.float32)
    counts = _count_spatial_tiles(cp.asarray(points), cp.asarray(labels), cp.asarray(thresholds), 3, [0]).get()

    np.testing.assert_array_equal(counts, _reference_counts(points, labels, thresholds))


def test_spatial_tiles_scan_with_one_count_buffer(cupy_pool):
    k = 64
    thresholds = cp.arange(1024, dtype=cp.float32)
    points = cp.arange(3, dtype=cp.float32)[:, None]
    labels = cp.asarray([0, k - 1, k - 1], dtype=cp.int32)
    count_bytes = k * k * len(thresholds) * np.dtype(np.uint64).itemsize
    cupy_pool.set_limit(size=count_bytes * 3 // 2)

    counts = _count_spatial_tiles(points, labels, thresholds, k, [points.device.id])

    expected = np.searchsorted([1, 4], np.arange(len(thresholds)), side="right")
    np.testing.assert_array_equal(counts[0, -1].get(), expected)


@pytest.mark.parametrize("scan_limit", [512, 1026])
def test_spatial_tiles_scan_in_batches(monkeypatch, scan_limit):
    thresholds = np.arange(513, dtype=np.float32)
    points = np.arange(6, dtype=np.float32)[:, None]
    labels = np.repeat(np.arange(3, dtype=np.int32), 2)
    monkeypatch.setattr(_co_oc, "_MAX_SCAN_ELEMENTS", scan_limit)
    cumsum = cp.cumsum

    def bounded_cumsum(a, axis=None, out=None):
        if a.dtype == cp.uint64:
            assert out.data.ptr == a.data.ptr
            if axis is not None:
                assert a.size <= scan_limit
        return cumsum(a, axis=axis, out=out)

    monkeypatch.setattr(cp, "cumsum", bounded_cumsum)
    counts = _count_spatial_tiles(cp.asarray(points), cp.asarray(labels), cp.asarray(thresholds), 3, [0])
    np.testing.assert_array_equal(counts.get(), _reference_counts(points, labels, thresholds))


def test_spatial_tiles_skip_idle_devices(monkeypatch):
    points = cp.zeros((2, 2), dtype=cp.float32)
    labels = cp.zeros(2, dtype=cp.int32)
    source = points.device.id
    copy_inputs = _co_oc._copies_on_device

    def copy_to_active_device(arrays, device):
        assert device == source, "copied inputs to a GPU with no tile rows"
        return copy_inputs(arrays, device)

    monkeypatch.setattr(_co_oc, "_copies_on_device", copy_to_active_device)
    counts = _count_spatial_tiles(points, labels, cp.zeros(1, dtype=cp.float32), 1, [source, source + 1])

    np.testing.assert_array_equal(counts.get(), [[[1]]])


@pytest.mark.parametrize("dims", [2, 3])
def test_co_occurrence_nonfinite_coordinates_and_signed_radii(dims):
    points = np.zeros((9, dims), dtype=np.float32)
    points[:, -1] = [0, 0, 1, 2, 4, np.inf, np.inf, -np.inf, np.nan]
    labels = np.array([0, 1, 1, 0, 128, 0, 128, 1, 128], dtype=np.int32)
    radii = np.array([-8, -4, -1, 0, 1, np.inf, np.nan], dtype=np.float32)
    adata = AnnData(shape=(len(points), 0))
    adata.obsm["spatial"] = points
    adata.obs["cluster"] = pd.Categorical.from_codes(labels, categories=range(labels.max() + 1))
    occ, interval = co_occurrence(adata, "cluster", interval=radii, multi_gpu=False, copy=True)

    np.testing.assert_array_equal(interval, radii)
    counts = _reference_counts(points, labels, radii[1:] ** 2).astype(np.float64)
    directed = counts + counts.swapaxes(0, 1)
    marginal = directed.sum(axis=0)
    denominator = marginal[:, None, :] * marginal[None, :, :]
    expected = np.divide(
        directed * directed.sum(axis=(0, 1)), denominator, out=np.zeros_like(directed), where=denominator != 0
    )
    np.testing.assert_allclose(occ, expected, rtol=1e-6)


def test_co_occurrence_coincident_cells_and_radius_ties():
    # The first cluster contributes more than 2**32 unordered pairs at radius zero.
    n = 100_001
    spatial = cp.zeros((n + 2, 2), dtype=cp.float32)
    spatial[n:] = cp.asarray([3, 4])
    labels = cp.zeros(n + 2, dtype=cp.int32)
    labels[n:] = 1
    below_five = np.nextafter(np.float32(5), np.float32(0))
    radii = cp.asarray([0, 0, below_five, 5, 5], dtype=cp.float32)
    counts = _count_spatial_tiles(spatial, labels, radii[1:] ** 2, 2, [0]).get()
    for i, cross_pairs in enumerate([0, 0, 2 * n, 2 * n]):
        expected = np.array([[n * (n - 1) // 2, cross_pairs], [0, 1]], dtype=np.uint64)
        np.testing.assert_array_equal(counts[:, :, i], expected)


@pytest.mark.two_gpus
@pytest.mark.parametrize("devices", [[1], [1, 0]])
def test_co_occurrence_uses_requested_devices(adata: AnnData, devices, monkeypatch):
    single, _ = co_occurrence(adata, "leiden", copy=True, multi_gpu=False)
    count_tiles = _cooc_cuda.count_tiles
    launched = []

    def record_device(*args):
        launched.append(cp.cuda.Device().id)
        return count_tiles(*args)

    monkeypatch.setattr(_cooc_cuda, "count_tiles", record_device)
    actual, _ = co_occurrence(adata, "leiden", copy=True, multi_gpu=devices)

    assert sorted(launched) == sorted(devices)
    np.testing.assert_array_equal(actual, single)


def test_co_occurrence_with_multi_gpu_param(adata: AnnData):
    """Test co_occurrence function with multi_gpu parameter."""
    # Test with multi_gpu=False (single GPU)
    arr_single, interval_single = co_occurrence(adata, cluster_key="leiden", copy=True, multi_gpu=False)

    arr_multi, interval_multi = co_occurrence(adata, cluster_key="leiden", copy=True, multi_gpu=True)

    # Results should match
    np.testing.assert_array_equal(interval_single, interval_multi)
    np.testing.assert_allclose(arr_single, arr_multi)


def test_find_min_max():
    """Test _find_min_max computes valid threshold bounds."""
    # Create random spatial data
    np.random.seed(42)
    spatial = cp.asarray(np.random.rand(100, 2).astype(np.float32))

    thresh_min, thresh_max = _find_min_max(spatial)

    # Both should be positive floats
    assert thresh_min > 0
    assert thresh_max > 0
    # thresh_min is distance between two closest points (by coordinate sum)
    # thresh_max is half the max pairwise distance
    assert np.isfinite(float(thresh_min))
    assert np.isfinite(float(thresh_max))
