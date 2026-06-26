"""Tests that nanobind kernels work with RMM managed memory (kDLCUDAManaged=13).

See https://github.com/scverse/rapids_singlecell/issues/591
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import rmm
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix
from rmm.allocators.cupy import rmm_cupy_allocator

import rapids_singlecell as rsc


@pytest.fixture()
def managed_memory():
    """Enable RMM managed memory for the duration of a test."""
    old_allocator = cp.cuda.get_allocator()
    rmm.reinitialize(managed_memory=True, pool_allocator=False)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    yield
    # Restore default allocator
    rmm.reinitialize(managed_memory=False, pool_allocator=False)
    cp.cuda.set_allocator(old_allocator)


def _make_adata(sparse, dtype=np.float32):
    """Create a small AnnData with managed-memory arrays."""
    X = cp.array([[1.0, 0.0, 3.0], [0.0, 2.0, 4.0], [5.0, 6.0, 0.0]], dtype=dtype)
    if sparse:
        X = csr_matrix(X)
    adata = AnnData(X)
    adata.obs_names = [f"cell{i}" for i in range(3)]
    adata.var_names = [f"gene{i}" for i in range(3)]
    return adata


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("sparse", [True, False])
def test_normalize_total(managed_memory, dtype, sparse):
    """Regression test for GH-591: normalize_total with managed memory."""
    adata = _make_adata(sparse, dtype)
    rsc.pp.normalize_total(adata, target_sum=1e4)
    sums = cp.ravel(adata.X.sum(axis=1))
    cp.testing.assert_allclose(sums, cp.full(3, 1e4, dtype=dtype), rtol=1e-5)


@pytest.mark.parametrize("sparse", [True, False])
def test_normalize_total_exclude_highly_expressed(managed_memory, sparse):
    adata = _make_adata(sparse, np.float64)
    rsc.pp.normalize_total(
        adata,
        target_sum=1e4,
        exclude_highly_expressed=True,
        max_fraction=0.5,
    )
    # Just verify it doesn't raise
    assert adata.X is not None


@pytest.mark.parametrize("sparse", [True, False])
def test_scale(managed_memory, sparse):
    adata = _make_adata(sparse, np.float64)
    rsc.pp.scale(adata)
    assert adata.X is not None


def test_qc_metrics(managed_memory):
    adata = _make_adata(sparse=True)
    rsc.pp.calculate_qc_metrics(adata, log1p=False)
    assert "total_counts" in adata.obs.columns
    assert "n_genes_by_counts" in adata.obs.columns
