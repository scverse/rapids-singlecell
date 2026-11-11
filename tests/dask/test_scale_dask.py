from __future__ import annotations

import cupy as cp
import numpy as np
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
    sc.pp.highly_variable_genes(adata, n_top_genes=1000, subset=True)
    return adata.copy()


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("zero_center", [True, False])
def test_scale(client, data_kind, zero_center):
    adata = _get_anndata()
    mask = np.random.randint(0, 2, adata.shape[0], dtype=bool)
    dask_data = adata.copy()

    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X.astype(np.float64))
        adata.X = cusparse.csr_matrix(adata.X.astype(np.float64))
    elif data_kind == "dense":
        dask_data.X = as_dense_cupy_dask_array(dask_data.X.astype(np.float64))
        adata.X = cp.array(adata.X.toarray().astype(np.float64))
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.scale(adata, zero_center=zero_center, mask=mask, max_value=10)
    rsc.pp.scale(dask_data, zero_center=zero_center, mask=mask, max_value=10)
    if data_kind == "sparse" and not zero_center:
        adata_X = adata.X.toarray()
        dask_X = dask_data.X.compute().toarray()
    else:
        adata_X = adata.X
        dask_X = dask_data.X.compute()
    cp.testing.assert_allclose(adata_X, dask_X)
