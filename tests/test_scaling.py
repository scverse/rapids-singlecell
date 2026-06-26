from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from scanpy.datasets import pbmc3k
from scipy.sparse import csc_matrix, csr_matrix

import rapids_singlecell as rsc

# test "data" for 3 cells * 4 genes
X_original = np.array(
    [
        [-1, 2, 0, 0],
        [1, 2, 4, 0],
        [0, 2, 2, 0],
    ]
)  # with gene std 1,0,2,0 and center 0,2,2,0
X_scaled_original = np.array(
    [
        [-1, 2, 0, 0],
        [1, 2, 2, 0],
        [0, 2, 1, 0],
    ]
)  # with gene std 1,0,1,0 and center 0,2,1,0
X_centered_original = np.array(
    [
        [-1, 0, -1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
    ]
)  # with gene std 1,0,1,0 and center 0,0,0,0
X_scaled_original_clipped = np.array(
    [
        [-1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
    ]
)  # with gene std 1,0,1,0 and center 0,2,1,0

X_for_mask = np.array(
    [
        [27, 27, 27, 27],
        [27, 27, 27, 27],
        [-1, 2, 0, 0],
        [1, 2, 4, 0],
        [0, 2, 2, 0],
        [27, 27, 27, 27],
        [27, 27, 27, 27],
    ]
)
X_scaled_for_mask = np.array(
    [
        [27, 27, 27, 27],
        [27, 27, 27, 27],
        [-1, 2, 0, 0],
        [1, 2, 2, 0],
        [0, 2, 1, 0],
        [27, 27, 27, 27],
        [27, 27, 27, 27],
    ]
)
X_centered_for_mask = np.array(
    [
        [27, 27, 27, 27],
        [27, 27, 27, 27],
        [-1, 0, -1, 0],
        [1, 0, 1, 0],
        [0, 0, 0, 0],
        [27, 27, 27, 27],
        [27, 27, 27, 27],
    ]
)
X_scaled_for_mask_clipped = np.array(
    [
        [27, 27, 27, 27],
        [27, 27, 27, 27],
        [-1, 1, 0, 0],
        [1, 1, 1, 0],
        [0, 1, 1, 0],
        [27, 27, 27, 27],
        [27, 27, 27, 27],
    ]
)


def _get_anndata():
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)
    return adata.copy()


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_scale_simple(dtype):
    adata = sc.datasets.pbmc68k_reduced()
    adata.X = adata.raw.X.todense()

    adata[:, 0 : adata.shape[1] // 2]
    adata.X = cp.array(adata.X, dtype=dtype)

    rsc.pp.scale(adata)

    cp.testing.assert_allclose(adata.X.var(axis=0), cp.ones(adata.shape[1]), atol=0.01)
    cp.testing.assert_allclose(
        adata.X.mean(axis=0), cp.zeros(adata.shape[1]), atol=0.00001
    )


@pytest.mark.parametrize(
    "typ", [np.array, csr_matrix, csc_matrix], ids=lambda x: x.__name__
)
def test_mask(typ):
    adata = _get_anndata()
    adata.X = typ(adata.X.toarray(), dtype=np.float64)
    rsc.get.anndata_to_GPU(adata)
    mask = np.random.randint(0, 2, adata.shape[0], dtype=bool)
    adata_mask = adata[mask].copy()
    rsc.pp.scale(adata_mask, zero_center=False)
    rsc.pp.scale(adata, mask_obs=mask, zero_center=False)
    adata = adata[mask].copy()
    cp.testing.assert_allclose(
        cp_csr_matrix(adata_mask.X).toarray(), cp_csr_matrix(adata.X).toarray()
    )


@pytest.mark.parametrize(
    "typ", [np.array, csr_matrix, csc_matrix], ids=lambda x: x.__name__
)
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize(
    ("mask_obs", "X", "X_centered", "X_scaled"),
    [
        (None, X_original, X_centered_original, X_scaled_original),
        (
            np.array((0, 0, 1, 1, 1, 0, 0), dtype=bool),
            X_for_mask,
            X_centered_for_mask,
            X_scaled_for_mask,
        ),
    ],
)
def test_scale(*, typ, dtype, mask_obs, X, X_centered, X_scaled):
    # test AnnData arguments
    # test scaling with default zero_center == True
    adata = AnnData(typ(X, dtype=dtype))
    adata0 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata0, mask_obs=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata0.X).toarray(), X_centered)
    """
    # test scaling with explicit zero_center == True
    adata1 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata1, zero_center=True, mask_obs=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata1.X).toarray(), X_centered)
    # test scaling with explicit zero_center == False
    adata2 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata2, zero_center=False, mask_obs=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata2.X).toarray(), X_scaled)
    """


def test_mask_string():
    adata = AnnData(np.array(X_for_mask, dtype="float32"))
    rsc.get.anndata_to_GPU(adata)
    with pytest.raises(ValueError):
        rsc.pp.scale(adata, mask_obs="mask")
    adata.obs["some cells"] = np.array((0, 0, 1, 1, 1, 0, 0), dtype=bool)
    rsc.pp.scale(adata, mask_obs="some cells")
    cp.testing.assert_allclose(adata.X, X_centered_for_mask)
    assert "mean of some cells" in adata.var.keys()


@pytest.mark.parametrize("zero_center", [True, False])
def test_clip(zero_center):
    adata = sc.datasets.pbmc3k()
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.scale(adata, max_value=1, zero_center=zero_center)
    if zero_center:
        assert adata.X.min() >= -1
    assert adata.X.max() <= 1


@pytest.mark.parametrize(
    ("mask_obs", "X", "X_scaled", "X_clipped"),
    [
        (None, X_original, X_scaled_original, X_scaled_original_clipped),
        (
            np.array((0, 0, 1, 1, 1, 0, 0), dtype=bool),
            X_for_mask,
            X_scaled_for_mask,
            X_scaled_for_mask_clipped,
        ),
    ],
)
def test_scale_sparse(*, mask_obs, X, X_scaled, X_clipped):
    adata = AnnData(csr_matrix(X).astype(np.float32))
    adata0 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata0, mask_obs=mask_obs, zero_center=False)
    cp.testing.assert_allclose(adata0.X.toarray(), X_scaled)
    # test scaling with explicit zero_center == True
    adata1 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata1, zero_center=False, mask_obs=mask_obs, max_value=1)
    cp.testing.assert_allclose(adata1.X.toarray(), X_clipped)
