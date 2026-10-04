from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from scanpy.datasets import pbmc68k_reduced
from scipy import sparse

from rapids_singlecell.tools import _umap, draw_graph, tsne, umap
from rapids_singlecell.tools._umap import _device_coo
from testing.rapids_singlecell._pytest import needs


@pytest.mark.parametrize("kwargs", [{}, {"rng": None}])
def test_umap(kwargs):
    pbmc = pbmc68k_reduced()
    del pbmc.obsm["X_umap"]
    umap(pbmc, **kwargs)
    assert pbmc.obsm["X_umap"].shape == (700, 2)


@pytest.mark.parametrize("upper", [False, True])
@pytest.mark.parametrize("index_dtype", [np.int32, np.int64])
def test_umap_device_coo(monkeypatch, index_dtype, upper):
    monkeypatch.setattr(_umap, "_H2D_CHUNK", 7)  # several blocks
    graph = sparse.random(50, 50, density=0.2, format="csr", rng=0, dtype=np.float32)
    # empty rows at the start, middle and end
    graph = sparse.csr_matrix(
        graph.multiply(np.isin(np.arange(50), [0, 1, 25, 49], invert=True)[:, None])
    )
    graph.indices = graph.indices.astype(index_dtype)
    graph.indptr = graph.indptr.astype(index_dtype)
    ref = (sparse.triu(graph) if upper else graph).tocoo()
    coo = _device_coo(graph, upper=upper)
    assert coo.has_canonical_format
    np.testing.assert_array_equal(coo.row.get(), ref.row)
    np.testing.assert_array_equal(coo.col.get(), ref.col)
    np.testing.assert_array_equal(coo.data.get(), ref.data)


def test_umap_upper_triangle_if_graph_does_not_fit(monkeypatch):
    pbmc = pbmc68k_reduced()
    monkeypatch.setattr(_umap, "_MEMORY_FRACTION", 0)  # as if the graph did not fit
    seen = {}
    embed = _umap.simplicial_set_embedding

    def spy(**kwargs):
        seen.update(nnz=kwargs["graph"].nnz, n_epochs=kwargs["n_epochs"])
        return embed(**kwargs)

    monkeypatch.setattr(_umap, "simplicial_set_embedding", spy)
    umap(pbmc)
    upper_nnz = sparse.triu(pbmc.obsp["connectivities"]).nnz
    # every pair once, with twice the epochs (500 for <= 10,000 cells)
    assert seen == {"nnz": upper_nnz, "n_epochs": 1000}
    assert np.isfinite(pbmc.obsm["X_umap"]).all()


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
