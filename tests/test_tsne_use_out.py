from __future__ import annotations

import pytest
from scanpy.datasets import pbmc68k_reduced

import rapids_singlecell as rsc

A = pytest.importorskip("anndata.acc").A


def test_tsne_use_out():
    pbmc = pbmc68k_reduced()
    pbmc.layers["x"] = pbmc.X.copy()
    rsc.tl.tsne(pbmc, use=A.obsm["X_pca"], out=A.obsm["t"])
    assert pbmc.obsm["t"].shape == (700, 2)
    assert pbmc.uns["t"]["params"]["use_rep"] == "X_pca"
    res = rsc.tl.tsne(pbmc, use="layers.x", out=True)
    assert res.shape == (700, 2)
    assert "X_tsne" not in pbmc.obsm


def test_tsne_deprecated():
    pbmc = pbmc68k_reduced()
    with pytest.warns(FutureWarning, match="use_rep"), pytest.warns(FutureWarning, match="key_added"):
        rsc.tl.tsne(pbmc, use_rep="X_pca", key_added="t")
    assert "t" in pbmc.obsm
    with pytest.warns(FutureWarning, match="copy"):
        res = rsc.tl.tsne(pbmc, copy=True)
    assert "X_tsne" in res.obsm
    assert "X_tsne" not in pbmc.obsm
    with pytest.raises(TypeError, match="not both"), pytest.warns(FutureWarning, match="use_rep"):
        rsc.tl.tsne(pbmc, use=A.obsm["X_pca"], use_rep="X_pca")


def test_tsne_obsm_named_x():
    """`use=A.obsm['X']` needs that `.obsm` key, it is not the deprecated `use_rep='X'`."""
    with pytest.raises(KeyError, match="^'X'$"):
        rsc.tl.tsne(pbmc68k_reduced(), use=A.obsm["X"], out=True)
