from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData

import rapids_singlecell as rsc


@pytest.fixture
def adata_grid():
    """3x3 grid embedding where center cell (index 4) should have highest density."""
    adata = AnnData(X=np.ones((9, 10), dtype=np.float32))
    adata.obsm["X_test"] = np.array(
        [[x, y] for x in range(3) for y in range(3)], dtype=np.float32
    )
    return adata


@pytest.fixture
def adata_grouped():
    """AnnData with two groups and distinct spatial clusters."""
    n = 200
    rng = np.random.default_rng(42)
    adata = AnnData(X=np.ones((n, 10), dtype=np.float32))

    # Group A clustered around (0, 0), Group B clustered around (5, 5)
    embed = np.zeros((n, 2), dtype=np.float32)
    embed[: n // 2] = rng.normal(loc=0.0, scale=0.5, size=(n // 2, 2)).astype(
        np.float32
    )
    embed[n // 2 :] = rng.normal(loc=5.0, scale=0.5, size=(n // 2, 2)).astype(
        np.float32
    )
    adata.obsm["X_umap"] = embed

    adata.obs["group"] = pd.Categorical(["A"] * (n // 2) + ["B"] * (n // 2))
    return adata


def test_basic_density(adata_grid):
    """Density values are scaled [0, 1] with max at the center of a grid."""
    rsc.tl.embedding_density(adata_grid, "test")

    density = adata_grid.obs["test_density"]
    assert density.max() == pytest.approx(1.0)
    assert density.min() == pytest.approx(0.0)
    assert density.idxmax() == "4"


def test_uns_params(adata_grid):
    """Check that uns metadata is stored correctly."""
    rsc.tl.embedding_density(adata_grid, "test")

    params = adata_grid.uns["test_density_params"]
    assert params["covariate"] is None
    assert params["components"] == [1, 2]


def test_groupby(adata_grouped):
    """Density is computed per group and stored correctly."""
    rsc.tl.embedding_density(adata_grouped, "umap", groupby="group")

    density = adata_grouped.obs["umap_density_group"]
    assert density.min() >= 0.0
    assert density.max() <= 1.0

    # Each group should have at least one cell at density 1.0
    for cat in ["A", "B"]:
        mask = adata_grouped.obs["group"] == cat
        group_dens = density[mask]
        assert group_dens.max() == pytest.approx(1.0)
        assert group_dens.min() == pytest.approx(0.0)

    params = adata_grouped.uns["umap_density_group_params"]
    assert params["covariate"] == "group"


def test_key_added(adata_grid):
    """Custom key_added is respected."""
    rsc.tl.embedding_density(adata_grid, "test", key_added="my_density")

    assert "my_density" in adata_grid.obs.columns
    assert "my_density_params" in adata_grid.uns


def test_custom_components(adata_grouped):
    """Specifying components selects the right embedding dimensions."""
    # Add a 3D embedding
    rng = np.random.default_rng(0)
    adata_grouped.obsm["X_pca"] = rng.standard_normal((200, 3)).astype(np.float32)

    rsc.tl.embedding_density(adata_grouped, "pca", components="2,3")

    assert "pca_density" in adata_grouped.obs.columns
    params = adata_grouped.uns["pca_density_params"]
    assert params["components"] == [2, 3]


def test_missing_basis_raises():
    """Error raised when the embedding doesn't exist."""
    adata = AnnData(X=np.ones((5, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Cannot find the embedded representation"):
        rsc.tl.embedding_density(adata, "umap")


def test_wrong_component_count():
    """Error raised when not exactly 2 components are specified."""
    adata = AnnData(X=np.ones((5, 3), dtype=np.float32))
    adata.obsm["X_umap"] = np.ones((5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="Please specify exactly 2 components"):
        rsc.tl.embedding_density(adata, "umap", components="1,2,3")


def test_groupby_not_found():
    """Error raised when groupby column doesn't exist."""
    adata = AnnData(X=np.ones((5, 3), dtype=np.float32))
    adata.obsm["X_umap"] = np.ones((5, 2), dtype=np.float32)
    with pytest.raises(KeyError):
        rsc.tl.embedding_density(adata, "umap", groupby="nonexistent")


def test_groupby_not_categorical():
    """Error raised when groupby column is not categorical."""
    adata = AnnData(X=np.ones((5, 3), dtype=np.float32))
    adata.obsm["X_umap"] = np.ones((5, 2), dtype=np.float32)
    adata.obs["numeric_col"] = [1.0, 2.0, 3.0, 4.0, 5.0]
    with pytest.raises(ValueError, match="does not contain categorical data"):
        rsc.tl.embedding_density(adata, "umap", groupby="numeric_col")


def test_diffmap_component_offset():
    """diffmap components are offset by 1 internally."""
    adata = AnnData(X=np.ones((9, 3), dtype=np.float32))
    adata.obsm["X_diffmap"] = np.array(
        [[0, x, y] for x in range(3) for y in range(3)], dtype=np.float32
    )

    rsc.tl.embedding_density(adata, "diffmap")
    assert "diffmap_density" in adata.obs.columns
    # Default components "1,2" map to columns 1,2 for diffmap (offset applied)
    params = adata.uns["diffmap_density_params"]
    assert params["components"] == [1, 2]


def test_fa_alias():
    """'fa' basis is aliased to 'draw_graph_fa'."""
    adata = AnnData(X=np.ones((9, 3), dtype=np.float32))
    adata.obsm["X_draw_graph_fa"] = np.array(
        [[x, y] for x in range(3) for y in range(3)], dtype=np.float32
    )

    rsc.tl.embedding_density(adata, "fa")
    assert "draw_graph_fa_density" in adata.obs.columns


@pytest.fixture
def pbmc68k():
    return sc.datasets.pbmc68k_reduced()


@pytest.mark.parametrize("groupby", [None, "louvain"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_matches_scanpy(pbmc68k, groupby, dtype):
    """GPU density matches scanpy on pbmc68k_reduced."""
    adata_sc = pbmc68k.copy()
    adata_sc.obsm["X_umap"] = adata_sc.obsm["X_umap"].astype(dtype)
    sc.tl.embedding_density(adata_sc, "umap", groupby=groupby)

    adata_gpu = pbmc68k.copy()
    adata_gpu.obsm["X_umap"] = adata_gpu.obsm["X_umap"].astype(dtype)
    rsc.tl.embedding_density(adata_gpu, "umap", groupby=groupby)

    key = "umap_density" if groupby is None else f"umap_density_{groupby}"
    atol = 1e-6 if dtype == np.float32 else 1e-12
    np.testing.assert_allclose(
        adata_gpu.obs[key].values,
        adata_sc.obs[key].values,
        atol=atol,
    )
