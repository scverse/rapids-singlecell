from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from anndata import AnnData, read_h5ad

from rapids_singlecell.squidpy_gpu._co_oc import (
    _co_occurrence_helper,
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
        _, interval_1 = co_occurrence(
            adata, cluster_key="leiden", copy=True, interval=interval
        )

        assert interval is not interval_1
        cp.testing.assert_allclose(
            interval, interval_1
        )  # allclose because in the func, we use f32


def test_co_occurrence_fast_kernel():
    n = 100  # number of points
    k = 5  # number of labels
    # Generate random data
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    thresholds = cp.asarray(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))
    occ_prob_fast = _co_occurrence_helper(spatial, thresholds, label_idx, fast=True)
    cp.cuda.Stream.null.synchronize()
    occ_prob_slow = _co_occurrence_helper(spatial, thresholds, label_idx, fast=False)
    cp.testing.assert_allclose(occ_prob_fast, occ_prob_slow)


@pytest.mark.skipif(
    cp.cuda.runtime.getDeviceCount() < 2,
    reason="Requires multiple GPUs",
)
def test_co_occurrence_multi_gpu_matches_single_gpu():
    """Multi-GPU results should match single GPU with large dataset (>=100k cells)."""
    n = 100_000  # Must be >= 100k to trigger multi-GPU dispatch
    k = 6  # number of labels
    # Generate random data
    np.random.seed(42)
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    thresholds = cp.asarray(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))

    # Single GPU result (explicitly request 1 GPU)
    occ_prob_single = _co_occurrence_helper(
        spatial, thresholds, label_idx, fast=True, device_ids=[0]
    )
    cp.cuda.Stream.null.synchronize()

    # Multi-GPU result (uses all available GPUs)
    n_gpus = cp.cuda.runtime.getDeviceCount()
    device_ids = list(range(n_gpus))
    occ_prob_multi = _co_occurrence_helper(
        spatial, thresholds, label_idx, fast=True, device_ids=device_ids
    )
    cp.cuda.Stream.null.synchronize()

    cp.testing.assert_allclose(occ_prob_multi, occ_prob_single)


@pytest.mark.skipif(
    cp.cuda.runtime.getDeviceCount() < 2,
    reason="Requires multiple GPUs",
)
def test_co_occurrence_multi_gpu_with_device_list():
    """Test with specific device IDs on large dataset."""
    n = 100_000  # Must be >= 100k to trigger multi-GPU dispatch
    k = 4  # number of labels
    np.random.seed(123)
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    thresholds = cp.asarray(np.array([0.15, 0.25, 0.35], dtype=np.float32))
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))

    # Use multiple GPUs with explicit device list
    n_gpus = cp.cuda.runtime.getDeviceCount()
    device_ids = list(range(n_gpus))
    occ_prob = _co_occurrence_helper(
        spatial, thresholds, label_idx, fast=True, device_ids=device_ids
    )
    cp.cuda.Stream.null.synchronize()

    # Verify output shape
    assert occ_prob.shape == (k, k, len(thresholds) - 1)

    # Verify results are finite
    assert cp.all(cp.isfinite(occ_prob))


def test_co_occurrence_small_dataset_uses_single_gpu():
    """Test that small datasets (<100k cells) use single GPU path."""
    n = 1000  # Small dataset
    k = 5
    np.random.seed(456)
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    thresholds = cp.asarray(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))

    # Even with multiple GPUs requested, small dataset should work correctly
    n_gpus = cp.cuda.runtime.getDeviceCount()
    device_ids = list(range(n_gpus))
    occ_prob = _co_occurrence_helper(
        spatial, thresholds, label_idx, fast=True, device_ids=device_ids
    )
    cp.cuda.Stream.null.synchronize()

    # Verify output shape
    assert occ_prob.shape == (k, k, len(thresholds) - 1)
    # Verify results are finite
    assert cp.all(cp.isfinite(occ_prob))


def test_co_occurrence_with_multi_gpu_param(adata: AnnData):
    """Test co_occurrence function with multi_gpu parameter."""
    # Test with multi_gpu=False (single GPU)
    arr_single, interval_single = co_occurrence(
        adata, cluster_key="leiden", copy=True, multi_gpu=False
    )

    # Test with multi_gpu=True (all GPUs - but small dataset so uses single GPU path)
    arr_multi, interval_multi = co_occurrence(
        adata, cluster_key="leiden", copy=True, multi_gpu=True
    )

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


def test_co_occurrence_large_interval():
    """Test co_occurrence with large interval (many bins) to test kernel adaptation."""
    n = 500
    k = 3
    np.random.seed(789)
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    # Large number of bins - kernel should adapt or fallback gracefully
    thresholds = cp.linspace(0.05, 0.95, 200, dtype=np.float32)
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))

    occ_prob = _co_occurrence_helper(spatial, thresholds, label_idx, fast=True)
    cp.cuda.Stream.null.synchronize()

    # Verify output shape (l_val = len(thresholds) - 1 = 199)
    assert occ_prob.shape == (k, k, len(thresholds) - 1)
    # Verify results are finite
    assert cp.all(cp.isfinite(occ_prob))


def test_co_occurrence_single_gpu_large_dataset():
    """Test single GPU path with large dataset (>=100k cells)."""
    n = 100_000
    k = 5
    np.random.seed(999)
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    thresholds = cp.asarray(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))

    # Explicitly use single GPU
    occ_prob = _co_occurrence_helper(
        spatial, thresholds, label_idx, fast=True, device_ids=[0]
    )
    cp.cuda.Stream.null.synchronize()

    # Verify output shape
    assert occ_prob.shape == (k, k, len(thresholds) - 1)
    # Verify results are finite
    assert cp.all(cp.isfinite(occ_prob))


@pytest.mark.parametrize(
    "n,k,expected_block_size",
    [
        # <2.5k cells/cat → 128 threads
        (500, 5, 128),
        # ~3k cells/cat → 256 threads
        (15000, 5, 256),
        # ~6k cells/cat → 512 threads
        (30000, 5, 512),
        # ~20k cells/cat → 1024 threads
        (100000, 5, 1024),
    ],
)
def test_co_occurrence_block_sizes(n, k, expected_block_size):
    """Test kernel works correctly with different block sizes.

    Block size is determined by cells per category:
    - >=10k cells/cat -> 1024 threads
    - >=5k cells/cat -> 512 threads
    - >=2.5k cells/cat -> 256 threads
    - <2.5k cells/cat -> 128 threads
    """
    np.random.seed(42)
    spatial = cp.asarray(np.random.rand(n, 2).astype(np.float32))
    thresholds = cp.asarray(np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32))
    label_idx = cp.asarray(np.random.randint(0, k, size=n, dtype=np.int32))

    # Run fast kernel
    occ_prob_fast = _co_occurrence_helper(
        spatial, thresholds, label_idx, fast=True, device_ids=[0]
    )
    cp.cuda.Stream.null.synchronize()

    # Run slow (pairwise) kernel as reference
    occ_prob_slow = _co_occurrence_helper(
        spatial, thresholds, label_idx, fast=False, device_ids=[0]
    )
    cp.cuda.Stream.null.synchronize()

    # Verify results match
    cp.testing.assert_allclose(occ_prob_fast, occ_prob_slow, rtol=1e-5, atol=1e-5)

    # Verify output shape
    assert occ_prob_fast.shape == (k, k, len(thresholds) - 1)
    # Verify results are finite
    assert cp.all(cp.isfinite(occ_prob_fast))
