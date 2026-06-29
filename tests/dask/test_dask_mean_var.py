from __future__ import annotations

import cupy as cp
import pytest
from scanpy.datasets import pbmc3k, pbmc68k_reduced

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._utils import _get_mean_var
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)

from ..test_score_genes import _create_sparse_nan_matrix  # noqa: TID252


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("correction", [0, 1])
def test_mean_var(client, data_kind, axis, dtype, correction):
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

    mean, var = _get_mean_var(adata.X, axis=axis, correction=correction)
    dask_mean, dask_var = _get_mean_var(dask_data.X, axis=axis, correction=correction)
    dask_mean, dask_var = dask_mean.compute(), dask_var.compute()

    cp.testing.assert_allclose(mean, dask_mean)
    cp.testing.assert_allclose(var, dask_var)


@pytest.mark.parametrize("array_type", ["csr", "dense"])
@pytest.mark.parametrize("percent_nan", [0, 0.3])
def test_sparse_nanmean(client, array_type, percent_nan):
    """Needs to be fixed"""
    from rapids_singlecell.tools._utils import _nan_mean

    R, C = 100, 50

    # sparse matrix with nan
    S = _create_sparse_nan_matrix(R, C, percent_zero=0.3, percent_nan=percent_nan)
    S = S.astype(cp.float64)
    A = S.toarray()
    A = rsc.get.X_to_GPU(A)

    if array_type == "dense":
        S = as_dense_cupy_dask_array(A).persist()
    else:
        S = as_sparse_cupy_dask_array(S).persist()

    cp.testing.assert_allclose(
        _nan_mean(A, 1).ravel(), (_nan_mean(S, 1)).ravel().compute()
    )
    cp.testing.assert_allclose(
        _nan_mean(A, 0).ravel(), (_nan_mean(S, 0)).ravel().compute()
    )
