from __future__ import annotations

import numpy as np
import pytest
from anndata import AnnData
from scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell.tools import _leiden

pytestmark = pytest.mark.skipif(_leiden._ld is None or not hasattr(_leiden._ld, "leiden"), reason="no native Leiden")


@pytest.fixture
def layers():
    # Two stars on the same four cells, with different hubs and strong edges.
    rna = sparse.csr_matrix([[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 3], [1, 1, 3, 0]], dtype=np.float64)
    atac = sparse.csr_matrix([[0, 3, 0, 0], [3, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
    return rna, atac


def _rb_quality(layer, labels, resolution):
    adjacency = layer.toarray()
    strengths = adjacency.sum(axis=1)
    quality = adjacency - resolution * np.outer(strengths, strengths) / strengths.sum()
    return quality[np.equal.outer(labels, labels)].sum()


def test_multiplex_differs_from_summed_graph(layers):
    acc = pytest.importorskip("anndata.acc").A
    rna, atac = layers
    adata = AnnData(shape=(4, 0), obsp={"rna": rna, "atac": atac, "sum": rna + atac})
    use = [acc.obsp["rna"], "obsp.atac"]
    rsc.tl.leiden(adata, use=use, rng=1)
    labels = adata.obs["leiden"].cat.codes.to_numpy()
    # These are the unique optima over all 15 partitions of the four cells.
    np.testing.assert_array_equal(labels, [0, 0, 0, 0])
    summed = rsc.tl.leiden(adata, use="obsp.sum", out=True, rng=1)
    np.testing.assert_array_equal(summed.codes, [0, 0, 1, 1])
    assert adata.uns["leiden"]["quality"] == pytest.approx(0.0)
    np.testing.assert_array_equal(rsc.tl.leiden(adata, use=use, out=True, rng=1).codes, labels)


def test_multiplex_mudata_weighted_objective(layers):
    md = pytest.importorskip("mudata")
    rna, atac = layers
    graphs = {"rna": 4 * rna, "atac": atac}
    modalities = {
        name: AnnData(shape=(4, 0), obsp={"connectivities": graph}, uns={"neighbors": {}})
        for name, graph in graphs.items()
    }
    mdata = md.MuData(modalities)
    resolution = {"atac": 0.5, "rna": 1.5}
    weights = {"atac": 1.0, "rna": 2.0}
    rsc.tl.leiden(mdata, use={"rna": None, "atac": None}, layer_weights=weights, resolution=resolution, rng=0)
    labels = mdata.obs["leiden"].cat.codes.to_numpy()
    # Normalizing each layer before adding its quality would put all four cells together.
    np.testing.assert_array_equal(labels, [0, 0, 1, 1])
    rb = [_rb_quality(graph, labels, resolution[name]) for name, graph in graphs.items()]
    np.testing.assert_allclose(mdata.uns["leiden"]["modularity"], rb / np.array([g.sum() for g in graphs.values()]))
    assert mdata.uns["leiden"]["quality"] == pytest.approx(np.dot([weights[name] for name in graphs], rb))

    modalities["atac"] = modalities["atac"][[2, 1, 0, 3]].copy()
    with pytest.raises(ValueError, match="order|align"):
        rsc.tl.leiden(md.MuData(modalities), use={"rna": None, "atac": None})


def test_multiplex_empty_and_zero_weight_layers(layers):
    from rapids_singlecell.tools._leiden_multiplex import leiden_multiplex

    rna, atac = layers
    empty = sparse.csr_matrix(rna.shape, dtype=np.float32)
    result = leiden_multiplex(
        [empty, rna, atac], [1, 1, 1], [1, 0, 0], use_weights=True, n_iterations=2, rng=np.random.default_rng(0)
    )
    np.testing.assert_array_equal(result.membership, np.arange(4))
    assert result.quality == 0
    expected = [0, *(_rb_quality(g, result.membership, 1) / g.sum() for g in (rna, atac))]
    np.testing.assert_allclose(result.modularity, expected)


@pytest.mark.parametrize(
    "kwargs",
    [{"resolution": [1]}, {"layer_weights": [1]}, {"layer_weights": [0, 0]}, {"layer_weights": [np.nan, 1]}],
)
def test_multiplex_invalid_layer_parameters(kwargs):
    adata = AnnData(shape=(4, 0))
    with pytest.raises(ValueError, match=next(iter(kwargs))):
        rsc.tl.leiden(adata, use=["obsp.a", "obsp.b"], **kwargs)
