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
    cp.testing.assert_allclose(adata.X.mean(axis=0), cp.zeros(adata.shape[1]), atol=0.00001)


@pytest.mark.parametrize("dtype", [np.int32, np.int64])
def test_scale_promotes_dense_integers(dtype):
    adata = AnnData(cp.array(X_original, dtype=dtype))

    rsc.pp.scale(adata)

    assert adata.X.dtype == cp.float64
    cp.testing.assert_allclose(adata.X, X_centered_original)


def test_scale_obsm_does_not_write_var_statistics():
    adata = AnnData(cp.ones((3, 4), dtype=cp.float32))
    adata.obsm["X_embedding"] = cp.array([[1, 2], [2, 4], [3, 6]], dtype=cp.float32)

    rsc.pp.scale(adata, use="obsm.X_embedding")

    cp.testing.assert_allclose(
        adata.obsm["X_embedding"],
        cp.array([[-1, -1], [0, 0], [1, 1]], dtype=cp.float32),
    )
    assert "mean" not in adata.var
    assert "std" not in adata.var


@pytest.mark.parametrize("typ", [np.array, csr_matrix, csc_matrix], ids=lambda x: x.__name__)
def test_mask(typ):
    adata = _get_anndata()
    adata.X = typ(adata.X.toarray(), dtype=np.float64)
    rsc.get.anndata_to_GPU(adata)
    mask = np.random.randint(0, 2, adata.shape[0], dtype=bool)
    adata_mask = adata[mask].copy()
    rsc.pp.scale(adata_mask, zero_center=False)
    rsc.pp.scale(adata, mask=mask, zero_center=False)
    adata = adata[mask].copy()
    cp.testing.assert_allclose(cp_csr_matrix(adata_mask.X).toarray(), cp_csr_matrix(adata.X).toarray())


@pytest.mark.parametrize("use_array", [False, True])
@pytest.mark.parametrize("typ", [np.array, csr_matrix, csc_matrix], ids=lambda x: x.__name__)
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
def test_scale(*, use_array, typ, dtype, mask_obs, X, X_centered, X_scaled):
    # test AnnData arguments
    # test scaling with default zero_center == True
    adata = AnnData(typ(X, dtype=dtype))
    adata0 = rsc.get.anndata_to_GPU(adata, copy=True)
    if use_array:
        # feeding the matrix directly should match feeding the AnnData
        out = rsc.pp.scale(adata0.X, mask=mask_obs)
        result = out.toarray() if hasattr(out, "toarray") else out
    else:
        rsc.pp.scale(adata0, mask=mask_obs)
        result = cp_csr_matrix(adata0.X).toarray()
    cp.testing.assert_allclose(result, X_centered)
    """
    # test scaling with explicit zero_center == True
    adata1 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata1, zero_center=True, mask=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata1.X).toarray(), X_centered)
    # test scaling with explicit zero_center == False
    adata2 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata2, zero_center=False, mask=mask_obs)
    cp.testing.assert_allclose(cp_csr_matrix(adata2.X).toarray(), X_scaled)
    """


def test_mask_string():
    adata = AnnData(np.array(X_for_mask, dtype="float32"))
    rsc.get.anndata_to_GPU(adata)
    with pytest.raises(ValueError):
        rsc.pp.scale(adata, mask="obs.mask")
    adata.obs["some cells"] = np.array((0, 0, 1, 1, 1, 0, 0), dtype=bool)
    with pytest.warns(FutureWarning, match="mask_obs"):
        rsc.pp.scale(adata, mask_obs="some cells")
    cp.testing.assert_allclose(adata.X, X_centered_for_mask)
    assert "mean of some cells" in adata.var.keys()
    with rsc.settings.override(preset=rsc.Preset.ScanpyV2Preview), pytest.warns(FutureWarning, match="mask_obs"):
        rsc.pp.scale(adata, zero_center=True, mask_obs="some cells")
    assert "mean of A.obs['some cells']" in adata.var.keys()


def test_scale_use_out():
    from anndata.acc import A

    adata = AnnData(cp.array(X_original, dtype=cp.float32))
    adata.obs["m"] = np.arange(adata.n_obs) > 0
    res = rsc.pp.scale(adata, out=True)
    cp.testing.assert_allclose(res.scaled, X_centered_original)
    cp.testing.assert_allclose(adata.X, X_original)
    rsc.pp.scale(adata, out=A.layers["s"])
    cp.testing.assert_allclose(adata.layers["s"], X_centered_original)
    np.testing.assert_allclose(adata.var["s_mean"], res.mean.get())
    rsc.pp.scale(adata, use=A.layers["s"], out={"std": A.var["sd"]}, mask="obs.m")
    assert "sd" in adata.var
    cp.testing.assert_allclose(adata.layers["s"], X_centered_original)


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
    rsc.pp.scale(adata0, mask=mask_obs, zero_center=False)
    cp.testing.assert_allclose(adata0.X.toarray(), X_scaled)
    # test scaling with explicit zero_center == True
    adata1 = rsc.get.anndata_to_GPU(adata, copy=True)
    rsc.pp.scale(adata1, zero_center=False, mask=mask_obs, max_value=1)
    cp.testing.assert_allclose(adata1.X.toarray(), X_clipped)
