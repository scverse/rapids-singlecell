from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from scanpy.datasets import pbmc68k_reduced
from scipy import sparse

from rapids_singlecell.tools import draw_graph, tsne, umap
from rapids_singlecell.tools._umap import _device_coo
from testing.rapids_singlecell._pytest import needs


@pytest.mark.parametrize("kwargs", [{}, {"rng": None}])
def test_umap(kwargs):
    pbmc = pbmc68k_reduced()
    del pbmc.obsm["X_umap"]
    umap(pbmc, **kwargs)
    assert pbmc.obsm["X_umap"].shape == (700, 2)


@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_umap_device_coo(index_dtype):
    graph = sparse.random(50, 50, density=0.2, format="csr", rng=0, dtype=np.float32)
    # empty rows at the start, middle and end
    graph = sparse.csr_matrix(
        graph.multiply(np.isin(np.arange(50), [0, 1, 25, 49], invert=True)[:, None])
    )
    graph.indices = graph.indices.astype(index_dtype)
    graph.indptr = graph.indptr.astype(index_dtype)
    ref = graph.tocoo()
    coo = _device_coo(graph)
    assert coo.has_canonical_format
    np.testing.assert_array_equal(coo.row.get(), ref.row)
    np.testing.assert_array_equal(coo.col.get(), ref.col)
    np.testing.assert_array_equal(coo.data.get(), ref.data)


def test_tsne():
    pbmc = pbmc68k_reduced()
    tsne(pbmc)
    assert pbmc.obsm["X_tsne"].shape == (700, 2)


@needs.igraph
@pytest.mark.parametrize("func", [umap, draw_graph])
def test_init_paga(func):
    pbmc = pbmc68k_reduced()[:100, :].copy()
    sc.tl.paga(pbmc)
    sc.pl.paga(pbmc, show=False)
    func(pbmc, init_pos="paga")


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
