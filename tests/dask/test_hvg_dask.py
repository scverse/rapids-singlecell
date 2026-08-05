from __future__ import annotations

import cupy as cp
import pandas as pd
import pytest
import scanpy as sc
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


def _get_anndata():
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    return adata


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
def test_highly_variable_genes(client, data_kind, flavor):
    adata = _get_anndata()
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()

    if data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        adata.X = cp.array(adata.X.toarray())
    elif data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        adata.X = cusparse.csr_matrix(adata.X)
    else:
        raise ValueError(f"Unknown data_kind: {data_kind}")

    rsc.pp.highly_variable_genes(adata, flavor=flavor)
    rsc.pp.highly_variable_genes(dask_data, flavor=flavor)

    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(
        adata.var["dispersions_norm"], dask_data.var["dispersions_norm"]
    )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("flavor", ["seurat", "cell_ranger"])
def test_highly_variable_genes_batched(client, data_kind, flavor):
    adata = _get_anndata()
    adata.obs["batch"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: adata.n_obs]
    dask_data = adata.copy()

    if data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        adata.X = cp.array(adata.X.toarray())
    elif data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        adata.X = cusparse.csr_matrix(adata.X)
    else:
        raise ValueError(f"Unknown data_kind: {data_kind}")

    rsc.pp.highly_variable_genes(adata, batch_key="batch")
    rsc.pp.highly_variable_genes(dask_data, batch_key="batch")
    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["dispersions"], dask_data.var["dispersions"])
    cp.testing.assert_allclose(
        adata.var["dispersions_norm"], dask_data.var["dispersions_norm"]
    )


def _get_anndata_raw():
    """Get raw count data for poisson_gene_selection and seurat_v3."""
    adata = pbmc3k()
    sc.pp.filter_cells(adata, min_genes=100)
    sc.pp.filter_genes(adata, min_cells=3)
    return adata


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("flavor", ["seurat_v3", "seurat_v3_paper"])
def test_highly_variable_genes_seurat_v3(client, data_kind, flavor):
    """Test seurat_v3 HVG with Dask arrays."""
    adata = _get_anndata_raw()
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()

    if data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        adata.X = cp.array(adata.X.toarray())
    elif data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        adata.X = cusparse.csr_matrix(adata.X)

    n_top_genes = 1000
    rsc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_top_genes)
    rsc.pp.highly_variable_genes(dask_data, flavor=flavor, n_top_genes=n_top_genes)

    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["variances"], dask_data.var["variances"])
    cp.testing.assert_allclose(
        adata.var["variances_norm"], dask_data.var["variances_norm"]
    )
    cp.testing.assert_array_equal(
        adata.var["highly_variable"], dask_data.var["highly_variable"]
    )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("flavor", ["seurat_v3", "seurat_v3_paper"])
def test_highly_variable_genes_seurat_v3_batched(client, data_kind, flavor):
    """Test seurat_v3 HVG with Dask arrays and batch_key."""
    adata = _get_anndata_raw()
    adata.X = adata.X.astype("float64")
    adata.obs["batch"] = (
        "source_" + pd.array([*range(1, 6), 5]).repeat(500).astype("string")
    )[: adata.n_obs]
    dask_data = adata.copy()

    if data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        adata.X = cp.array(adata.X.toarray())
    elif data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        adata.X = cusparse.csr_matrix(adata.X)

    n_top_genes = 1000
    rsc.pp.highly_variable_genes(
        adata, flavor=flavor, n_top_genes=n_top_genes, batch_key="batch"
    )
    rsc.pp.highly_variable_genes(
        dask_data, flavor=flavor, n_top_genes=n_top_genes, batch_key="batch"
    )

    cp.testing.assert_allclose(adata.var["means"], dask_data.var["means"])
    cp.testing.assert_allclose(adata.var["variances"], dask_data.var["variances"])
    cp.testing.assert_allclose(
        adata.var["variances_norm"], dask_data.var["variances_norm"]
    )
    cp.testing.assert_array_equal(
        adata.var["highly_variable"], dask_data.var["highly_variable"]
    )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_poisson_gene_selection_dask(client, data_kind):
    """Test poisson_gene_selection with Dask arrays."""
    adata = _get_anndata_raw()
    adata.X = adata.X.astype("float64")
    dask_data = adata.copy()

    if data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        adata.X = cp.array(adata.X.toarray())
    elif data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        adata.X = cusparse.csr_matrix(adata.X)
    else:
        raise ValueError(f"Unknown data_kind: {data_kind}")

    n_top_genes = 1000
    rsc.pp.highly_variable_genes(
        adata, flavor="poisson_gene_selection", n_top_genes=n_top_genes
    )
    rsc.pp.highly_variable_genes(
        dask_data, flavor="poisson_gene_selection", n_top_genes=n_top_genes
    )

    # Check all columns match
    cp.testing.assert_allclose(
        adata.var["observed_fraction_zeros"],
        dask_data.var["observed_fraction_zeros"],
        rtol=1e-5,
        atol=1e-6,
    )
    cp.testing.assert_allclose(
        adata.var["expected_fraction_zeros"],
        dask_data.var["expected_fraction_zeros"],
        rtol=1e-5,
        atol=1e-6,
    )
    cp.testing.assert_allclose(
        adata.var["prob_zero_enrichment"],
        dask_data.var["prob_zero_enrichment"],
        rtol=1e-5,
        atol=1e-6,
    )
    cp.testing.assert_array_equal(
        adata.var["highly_variable"], dask_data.var["highly_variable"]
    )
