"""Tests that nanobind kernels work with RMM managed memory (kDLCUDAManaged=13).

See https://github.com/scverse/rapids_singlecell/issues/591
"""

from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
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


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_harmony_integrate(managed_memory, dtype):
    """Regression test for GH-763: harmony_integrate with managed memory."""
    rng = np.random.default_rng(0)
    adata = AnnData(rng.standard_normal((300, 5)).astype(dtype))
    adata.obs["batch"] = pd.Categorical(rng.choice(["a", "b"], 300))
    adata.obsm["X_pca"] = rng.standard_normal((300, 10)).astype(dtype)
    rsc.pp.harmony_integrate(adata, key="batch", dtype=dtype)
    assert adata.obsm["X_pca_harmony"].shape == (300, 10)
    assert adata.obsm["X_pca_harmony"].dtype == dtype


def test_wnn(managed_memory):
    rng = np.random.default_rng(0)
    adata = AnnData(obsm={"A": rng.normal(size=(300, 8)), "B": rng.normal(size=(300, 5))})
    rsc.pp.wnn(adata, ["A", "B"], knn_range=40)
    assert adata.obsp["snn"].nnz > 0
    np.testing.assert_allclose(adata.obs[["A_weight", "B_weight"]].sum(axis=1), 1)


def test_leiden(managed_memory, monkeypatch):
    """The native Leiden bindings take one memory kind per call: device input
    and a workspace cached before the switch, a split run with an empty
    partition and an edgeless graph all run, with the bits of CUDA memory."""
    from scanpy.datasets import pbmc68k_reduced

    from rapids_singlecell.tools import _leiden

    a = pbmc68k_reduced().obsp["connectivities"]
    n = a.shape[0]
    start = np.arange(n) % 7

    def run(adjacency, **kwargs):
        rng = np.random.default_rng(0)
        return _leiden._leiden_rapids(adjacency, [1.0], rng=rng, n_iterations=2, use_weights=True, **kwargs)[0]

    monkeypatch.setattr(_leiden, "_WORKSPACE_CACHE", {})
    with cp.cuda.using_allocator(cp.get_default_memory_pool().malloc):  # CUDA, not managed, memory
        device_input = csr_matrix(a)
        expected = [run(a), run(a, initial_membership=start)]
        _leiden._WORKSPACE_CACHE[cp.cuda.Device().id] = cp.empty(1 << 24, dtype=cp.uint8)
    results = [run(device_input), run(a, initial_membership=start)]
    monkeypatch.setattr(_leiden, "_N_PARTITIONS", 2)
    monkeypatch.setattr(_leiden, "_MAX_REPLICAS", 1)
    monkeypatch.setattr(_leiden, "_row_bounds", lambda indptr, n_parts: [0, 0, n])
    results.append(run(a))
    for got, want in zip(results, [*expected, expected[0]], strict=True):
        assert got.membership.tobytes() == want.membership.tobytes()
        assert float.hex(got.modularity) == float.hex(want.modularity)
    assert run(csr_matrix((5, 5), dtype=np.float32)).n_clusters == 5
