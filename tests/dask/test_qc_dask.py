from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_qc_metrics(client, data_kind):
    adata = pbmc3k()
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    dask_data = adata.copy()
    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
        adata.X = cusparse.csr_matrix(adata.X)
    elif data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)
        adata.X = cp.array(adata.X.toarray())
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], log1p=True)
    rsc.pp.calculate_qc_metrics(dask_data, qc_vars=["mt"], log1p=True)
    np.testing.assert_allclose(adata.obs["n_genes_by_counts"], dask_data.obs["n_genes_by_counts"])
    np.testing.assert_allclose(adata.obs["total_counts"], dask_data.obs["total_counts"])
    np.testing.assert_allclose(adata.obs["log1p_n_genes_by_counts"], dask_data.obs["log1p_n_genes_by_counts"])
    np.testing.assert_allclose(adata.obs["log1p_total_counts"], dask_data.obs["log1p_total_counts"])
    np.testing.assert_allclose(adata.obs["pct_counts_mt"], dask_data.obs["pct_counts_mt"])
    np.testing.assert_allclose(adata.obs["total_counts_mt"], dask_data.obs["total_counts_mt"])
    np.testing.assert_allclose(adata.obs["log1p_total_counts_mt"], dask_data.obs["log1p_total_counts_mt"])
    np.testing.assert_allclose(adata.var["n_cells_by_counts"], dask_data.var["n_cells_by_counts"])
    np.testing.assert_allclose(adata.var["total_counts"], dask_data.var["total_counts"])
    np.testing.assert_allclose(adata.var["mean_counts"], dask_data.var["mean_counts"])
    np.testing.assert_allclose(adata.var["pct_dropout_by_counts"], dask_data.var["pct_dropout_by_counts"])
    np.testing.assert_allclose(adata.var["log1p_total_counts"], dask_data.var["log1p_total_counts"])
    np.testing.assert_allclose(adata.var["log1p_mean_counts"], dask_data.var["log1p_mean_counts"])
