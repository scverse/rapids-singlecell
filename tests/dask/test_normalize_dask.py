from __future__ import annotations

import cupy as cp
import pytest
import scanpy as sc
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_normalize_total(client, data_kind):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    dask_data = adata.copy()

    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
        adata.X = cusparse.csr_matrix(adata.X)
    elif data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)
        adata.X = cp.array(adata.X.toarray())
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.normalize_total(adata)
    rsc.pp.normalize_total(dask_data)

    if data_kind == "sparse":
        adata_X = adata.X.toarray()
        dask_X = dask_data.X.compute().toarray()
    else:
        adata_X = adata.X
        dask_X = dask_data.X.compute()

    cp.testing.assert_allclose(adata_X, dask_X)


def _norm_factor_case(data_kind, target_sum):
    """Counts with an empty cell in three uneven row chunks, and the in-memory result."""
    x = (cp.arange(51 * 20).reshape(51, 20) % 7).astype(cp.float32)
    x[7] = 0
    to_dask = as_sparse_cupy_dask_array if data_kind == "sparse" else as_dense_cupy_dask_array
    expected = rsc.pp.normalize_total(sc.AnnData(x.copy()), target_sum=target_sum, out=True)
    return sc.AnnData(to_dask(x.get())), expected


def _fail_on_compute(*args, **kwargs):
    raise AssertionError("normalize_total computed a Dask array")


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("legacy", [False, True], ids=["out", "inplace"])
def test_normalize_total_norm_factor(client, data_kind, legacy):
    import dask

    adata, expected = _norm_factor_case(data_kind, 100)
    with dask.config.set(scheduler=_fail_on_compute):  # both outputs stay lazy
        if legacy:
            with pytest.warns(FutureWarning, match="inplace"):
                res = rsc.pp.normalize_total(adata, target_sum=100, inplace=False)
            normalized, norm_factor = res["X"], res["norm_factor"]
        else:
            normalized, norm_factor = rsc.pp.normalize_total(adata, target_sum=100, out=True)
        other = rsc.pp.normalize_total(adata, target_sum=10, out=True)  # a second result from the same blocks
    assert norm_factor.chunks == ((25, 25, 1),)
    assert norm_factor.dtype == cp.float32

    normalized, norm_factor, *other = dask.compute(normalized, norm_factor, *other)
    results = [(normalized, norm_factor, expected), (*other, _norm_factor_case(data_kind, 10)[1])]
    for normalized, norm_factor, expected in results:
        cp.testing.assert_allclose(norm_factor, expected.norm_factor, rtol=1e-5)
        normalized = normalized.toarray() if data_kind == "sparse" else normalized
        cp.testing.assert_allclose(normalized, expected.normalized, rtol=1e-5)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("target_sum", [None, 100])
def test_normalize_total_norm_factor_obs(client, data_kind, target_sum):
    A = pytest.importorskip("anndata.acc").A
    adata, expected = _norm_factor_case(data_kind, target_sum)
    rsc.pp.normalize_total(adata, target_sum=target_sum, out={"norm_factor": A.obs["nf"]})
    out = {"normalized": A.layers["norm"], "norm_factor": A.obs["nf2"]}
    rsc.pp.normalize_total(adata, target_sum=target_sum, out=out)

    for key in ["nf", "nf2"]:
        assert adata.obs[key].dtype == "float32"
        cp.testing.assert_allclose(adata.obs[key].to_numpy(), expected.norm_factor, rtol=1e-5)
    normalized = adata.layers["norm"].compute()
    normalized = normalized.toarray() if data_kind == "sparse" else normalized
    cp.testing.assert_allclose(normalized, expected.normalized, rtol=1e-5)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_normalize_total_keeps_source(client, data_kind):
    """Results that aren't written back to their source leave its blocks alone, also within one compute."""
    import dask

    A = pytest.importorskip("anndata.acc").A
    adata, expected = _norm_factor_case(data_kind, 100)

    def compute(*arrays):
        return [x.toarray() if data_kind == "sparse" else x for x in dask.compute(*arrays)]

    (source,) = compute(adata.X)
    rsc.pp.normalize_total(adata, target_sum=100, out=A.layers["norm"])
    normalized = rsc.pp.normalize_total(adata, target_sum=100, out=True).normalized
    X, layer, normalized = compute(adata.X, adata.layers["norm"], normalized)
    cp.testing.assert_array_equal(X, source)
    cp.testing.assert_allclose(layer, expected.normalized, rtol=1e-5)
    cp.testing.assert_allclose(normalized, expected.normalized, rtol=1e-5)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_log1p(client, data_kind):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    dask_data = adata.copy()

    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
        adata.X = cusparse.csr_matrix(adata.X)
    elif data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)
        adata.X = cp.array(adata.X.toarray())
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.log1p(adata)
    rsc.pp.log1p(dask_data)

    if data_kind == "sparse":
        adata_X = adata.X.toarray()
        dask_X = dask_data.X.compute().toarray()
    else:
        adata_X = adata.X
        dask_X = dask_data.X.compute()

    cp.testing.assert_allclose(adata_X, dask_X)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("base", [2, 10])
def test_log1p_base(client, data_kind, base):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    dask_data = adata.copy()

    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
        adata.X = cusparse.csr_matrix(adata.X)
    elif data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)
        adata.X = cp.array(adata.X.toarray())
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.log1p(adata, base=base)
    rsc.pp.log1p(dask_data, base=base)

    if data_kind == "sparse":
        adata_X = adata.X.toarray()
        dask_X = dask_data.X.compute().toarray()
    else:
        adata_X = adata.X
        dask_X = dask_data.X.compute()

    cp.testing.assert_allclose(adata_X, dask_X)


def test_normalize_total_exclude_dask_not_implemented(client):
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    adata.X = as_sparse_cupy_dask_array(adata.X)

    with pytest.raises(NotImplementedError, match="`exclude_highly_expressed`"):
        rsc.pp.normalize_total(adata, exclude_highly_expressed=True)
