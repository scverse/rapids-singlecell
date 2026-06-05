from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from cupyx.scipy import sparse as cusparse
from scanpy.datasets import pbmc3k, pbmc3k_processed
from scipy import sparse

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("zero_center", [True, False])
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_pca_dask(dist_client, data_kind, zero_center):
    adata_1 = pbmc3k_processed()
    adata_2 = pbmc3k_processed()

    if data_kind == "sparse":
        adata_1.X = sparse.csr_matrix(adata_1.X.astype(np.float64))
        adata_2.X = as_sparse_cupy_dask_array(adata_2.X.astype(np.float64))
    elif data_kind == "dense":
        adata_1.X = cp.array(adata_1.X.astype(np.float64))
        adata_2.X = as_dense_cupy_dask_array(adata_2.X.astype(np.float64))
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.pca(adata_1, svd_solver="full", zero_center=zero_center)
    rsc.pp.pca(adata_2, svd_solver="full", zero_center=zero_center)

    cp.testing.assert_allclose(
        np.abs(adata_1.obsm["X_pca"]),
        cp.abs(adata_2.obsm["X_pca"].compute()),
        rtol=1e-7,
        atol=1e-6,
    )

    cp.testing.assert_allclose(
        np.abs(adata_1.varm["PCs"]),
        np.abs(adata_2.varm["PCs"]),
        rtol=1e-7,
        atol=1e-6,
    )

    cp.testing.assert_allclose(
        np.abs(adata_1.uns["pca"]["variance_ratio"]),
        np.abs(adata_2.uns["pca"]["variance_ratio"]),
        rtol=1e-5,
        atol=1e-5,
    )


# ("full", 1e-7, 1e-6), ("jacobi", 1e-5, 1e-5),
@pytest.mark.parametrize(
    ("svd_solver", "rtol", "atol"),
    [("covariance_eigh", 1e-7, 1e-6)],
)
def test_pca_dask_dense_svd_solver(client, svd_solver, rtol, atol):
    adata_1 = pbmc3k_processed()
    adata_2 = pbmc3k_processed()

    adata_1.X = cp.array(adata_1.X.astype(np.float64))
    adata_2.X = as_dense_cupy_dask_array(adata_2.X.astype(np.float64))

    # Test results of full Dense PCA against DaskDensePCA with different svd solvers
    rsc.pp.pca(adata_1, svd_solver="full", zero_center=True)
    rsc.pp.pca(adata_2, svd_solver=svd_solver, zero_center=True)

    cp.testing.assert_allclose(
        np.abs(adata_1.obsm["X_pca"]),
        cp.abs(adata_2.obsm["X_pca"].compute()),
        rtol=rtol,
        atol=atol,
    )

    cp.testing.assert_allclose(
        np.abs(adata_1.varm["PCs"]),
        np.abs(adata_2.varm["PCs"]),
        rtol=rtol,
        atol=atol,
    )

    cp.testing.assert_allclose(
        np.abs(adata_1.uns["pca"]["variance_ratio"]),
        np.abs(adata_2.uns["pca"]["variance_ratio"]),
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_pca_dask_full_pipeline(client, data_kind):
    adata_1 = pbmc3k()
    adata_2 = pbmc3k()

    if data_kind == "sparse":
        adata_1.X = cusparse.csr_matrix(sparse.csr_matrix(adata_1.X.astype(np.float64)))
        adata_2.X = as_sparse_cupy_dask_array(adata_2.X.astype(np.float64))
    elif data_kind == "dense":
        adata_1.X = cp.array(adata_1.X.astype(np.float64).toarray())
        adata_2.X = as_dense_cupy_dask_array(adata_2.X.astype(np.float64).toarray())
    else:
        raise ValueError(f"Unknown data_kind {data_kind}")

    rsc.pp.filter_genes(adata_1, min_cells=500)
    rsc.pp.filter_genes(adata_2, min_cells=500)

    rsc.pp.normalize_total(adata_1, target_sum=1e4)
    rsc.pp.normalize_total(adata_2, target_sum=1e4)

    rsc.pp.log1p(adata_1)
    rsc.pp.log1p(adata_2)

    rsc.pp.pca(adata_1, svd_solver="full")
    rsc.pp.pca(adata_2, svd_solver="covariance_eigh")

    cp.testing.assert_allclose(
        np.abs(adata_1.obsm["X_pca"]),
        cp.abs(adata_2.obsm["X_pca"].compute()),
        rtol=1e-7,
        atol=1e-6,
    )

    cp.testing.assert_allclose(
        np.abs(adata_1.varm["PCs"]), np.abs(adata_2.varm["PCs"]), rtol=1e-7, atol=1e-6
    )

    cp.testing.assert_allclose(
        np.abs(adata_1.uns["pca"]["variance_ratio"]),
        np.abs(adata_2.uns["pca"]["variance_ratio"]),
        rtol=1e-7,
        atol=1e-6,
    )
