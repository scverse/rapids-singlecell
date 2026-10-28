from __future__ import annotations

import cupy as cp
import pytest
from scanpy.datasets import pbmc3k, pbmc68k_reduced

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_mean_var(client, data_kind, dtype):
    if data_kind == "dense":
        adata = pbmc68k_reduced()
        adata.X = adata.X.astype(dtype)
        dask_data = adata.copy()
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)
    elif data_kind == "sparse":
        adata = pbmc3k()
        adata.X = adata.X.astype(dtype)
        dask_data = adata.copy()
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)

    rsc.tl.score_genes(adata, gene_list=adata.var_names[:100], score_name="Test_gpu")
    rsc.tl.score_genes(dask_data, gene_list=adata.var_names[:100], score_name="Test_gpu")
    cp.testing.assert_allclose(adata.obs["Test_gpu"], dask_data.obs["Test_gpu"])
