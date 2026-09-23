from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from scipy.sparse import random as sp_random

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


def _make_adata(n_obs=1000, n_genes=100, density=0.6, dtype="float32"):
    from anndata import AnnData

    rng = np.random.default_rng(42)
    X = sp_random(
        n_obs, n_genes, density=density, format="csr", random_state=rng
    ).astype(dtype)
    adata = AnnData(X)
    adata.obs["percent_mito"] = rng.random(n_obs).astype(dtype)
    adata.obs["n_counts"] = np.array(X.sum(axis=1)).ravel()
    adata.obs["batch"] = pd.Categorical(rng.choice(["a", "b", "c"], size=n_obs))
    return adata


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_regress_out_categorical_dask(client, data_kind, dtype):
    adata = _make_adata(dtype=dtype)

    # Non-dask reference
    ref = adata.copy()
    rsc.get.anndata_to_GPU(ref)
    rsc.pp.regress_out(ref, keys=["batch"])

    # Dask version
    dask_data = adata.copy()
    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    else:
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)

    rsc.pp.regress_out(dask_data, keys=["batch"])

    dask_X = dask_data.X.compute()

    atol = 5e-5 if dtype == "float32" else 1e-7
    cp.testing.assert_allclose(dask_X, ref.X, atol=atol)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.flaky(reruns=2, reruns_delay=5)
def test_regress_out_continuous_dask(client, data_kind, dtype):
    adata = _make_adata(dtype=dtype)

    # Non-dask reference
    ref = adata.copy()
    rsc.get.anndata_to_GPU(ref)
    rsc.pp.regress_out(ref, keys=["n_counts", "percent_mito"], batchsize="all")

    # Dask version
    dask_data = adata.copy()
    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    else:
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)

    rsc.pp.regress_out(dask_data, keys=["n_counts", "percent_mito"])

    dask_X = dask_data.X.compute()

    atol = 5e-5 if dtype == "float32" else 1e-7
    cp.testing.assert_allclose(dask_X, ref.X, atol=atol)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_regress_out_reproducible_dask(client, data_kind, dtype):
    import scanpy as sc

    adata = sc.datasets.pbmc68k_reduced()
    adata = adata.raw.to_adata()[:200, :200].copy()
    adata.X = adata.X.astype(dtype)
    if data_kind == "sparse":
        adata.X = as_sparse_cupy_dask_array(adata.X)
    else:
        adata.X = as_dense_cupy_dask_array(adata.X)

    rsc.pp.regress_out(adata, keys=["n_counts", "percent_mito"])

    # This file was generated for scanpy version 1.10.3
    tester = np.load(Path(__file__).parent.parent / "_data/regress_test_small.npy")
    atol = 1e-5 if dtype == "float32" else 1e-7
    cp.testing.assert_allclose(adata.X.compute(), tester, atol=atol)


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_regress_out_singular_dask(client, data_kind):
    """Linearly dependent regressors have to be reported, not inverted anyway."""
    adata = _make_adata()
    adata.obs["n_counts_copy"] = adata.obs["n_counts"].to_numpy()
    if data_kind == "sparse":
        adata.X = as_sparse_cupy_dask_array(adata.X)
    else:
        adata.X = as_dense_cupy_dask_array(adata.X)

    with pytest.raises(ValueError, match="singular"):
        rsc.pp.regress_out(adata, keys=["n_counts", "n_counts_copy"])


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_regress_out_inplace_false_dask(client, data_kind):
    adata = _make_adata()

    dask_data = adata.copy()
    if data_kind == "sparse":
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X)
    else:
        dask_data.X = as_dense_cupy_dask_array(dask_data.X)

    result = rsc.pp.regress_out(
        dask_data, keys=["n_counts", "percent_mito"], inplace=False
    )
    assert result is not None
    # Result should be a DaskArray
    from dask.array import Array as DaskArray

    assert isinstance(result, DaskArray)
