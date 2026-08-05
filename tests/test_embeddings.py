from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from scanpy.datasets import pbmc68k_reduced

from rapids_singlecell.tools import draw_graph, tsne, umap
from testing.rapids_singlecell._pytest import needs


def test_umap():
    pbmc = pbmc68k_reduced()
    del pbmc.obsm["X_umap"]
    umap(pbmc)
    assert pbmc.obsm["X_umap"].shape == (700, 2)


def test_tsne():
    pbmc = pbmc68k_reduced()
    tsne(pbmc)
    assert pbmc.obsm["X_tsne"].shape == (700, 2)


@needs.igraph
def test_umap_init_paga():
    pbmc = pbmc68k_reduced()[:100, :].copy()
    sc.tl.paga(pbmc)
    sc.pl.paga(pbmc, show=False)
    umap(pbmc, init_pos="paga")


@pytest.mark.parametrize("init_pos", ["X_pca", "X_tsne", "numpy", "cupy"])
@pytest.mark.parametrize("func", [umap, draw_graph])
def test_umap_init_pos(init_pos, func):
    pbmc = pbmc68k_reduced()[:100, :].copy()
    if init_pos == "X_pca":
        with pytest.raises(ValueError, match="Expected 2 columns but got 50 columns."):
            func(pbmc, init_pos=init_pos)
    elif init_pos == "X_tsne":
        tsne(pbmc)
        func(pbmc, init_pos=init_pos)
    else:
        if init_pos == "numpy":
            init_pos = np.random.random((100, 2))
        else:
            init_pos = cp.random.random((100, 2))
        func(pbmc, init_pos=init_pos)


def test_draw_graph():
    pbmc = pbmc68k_reduced()[:100, :].copy()
    draw_graph(pbmc)
