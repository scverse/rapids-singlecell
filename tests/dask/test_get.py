from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from scanpy.datasets import pbmc3k_processed
from scipy import sparse

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_get_anndata(client, data_kind):
    adata = pbmc3k_processed()
    dask_adata = adata.copy()
    if data_kind == "sparse":
        adata.X = rsc.get.X_to_GPU(sparse.csr_matrix(adata.X.astype(np.float64)))
        dask_adata.X = as_sparse_cupy_dask_array(dask_adata.X.astype(np.float64))
    elif data_kind == "dense":
        adata.X = cp.array(adata.X.astype(np.float64))
        dask_adata.X = as_dense_cupy_dask_array(dask_adata.X.astype(np.float64))
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    assert type(adata.X) is type(dask_adata.X._meta)

    if data_kind == "sparse":
        cp.testing.assert_array_equal(adata.X.toarray(), dask_adata.X.compute().toarray())
    else:
        cp.testing.assert_array_equal(adata.X, dask_adata.X.compute())

    rsc.get.anndata_to_CPU(dask_adata)
    rsc.get.anndata_to_CPU(adata)

    assert type(adata.X) is type(dask_adata.X._meta)

    if data_kind == "sparse":
        cp.testing.assert_array_equal(adata.X.toarray(), dask_adata.X.compute().toarray())
    else:
        cp.testing.assert_array_equal(adata.X, dask_adata.X.compute())
    rsc.get.anndata_to_GPU(dask_adata)
    rsc.get.anndata_to_GPU(adata)

    assert type(adata.X) is type(dask_adata.X._meta)

    if data_kind == "sparse":
        cp.testing.assert_array_equal(adata.X.toarray(), dask_adata.X.compute().toarray())
    else:
        cp.testing.assert_array_equal(adata.X, dask_adata.X.compute())
