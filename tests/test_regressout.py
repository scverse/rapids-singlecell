from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData

import rapids_singlecell as rsc

HERE = Path(__file__).parent


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_regress_out_ordinal(dtype):
    from scipy.sparse import random

    adata = AnnData(random(1000, 100, density=0.6, format="csr"))
    rsc.get.anndata_to_GPU(adata)
    adata.X = adata.X.astype(dtype)
    adata.obs["percent_mito"] = np.random.rand(adata.X.shape[0])
    adata.obs["n_counts"] = adata.X.sum(axis=1).get()

    # results using only one processor
    rapids = rsc.pp.regress_out(
        adata, keys=["n_counts", "percent_mito"], batchsize=100, inplace=False
    )
    assert adata.X.shape == rapids.shape

    # results using 8 processors
    cupy = rsc.pp.regress_out(
        adata, keys=["n_counts", "percent_mito"], batchsize="all", inplace=False
    )
    if dtype == "float32":
        cp.testing.assert_allclose(cupy, rapids, atol=1e-5)
    else:
        cp.testing.assert_allclose(cupy, rapids, atol=1e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("batchsize", ["all", 100])
def test_regress_out_layer(dtype, batchsize):
    from scipy.sparse import random

    adata = AnnData(random(1000, 100, density=0.6, format="csr"))
    rsc.get.anndata_to_GPU(adata)
    adata.X = adata.X.astype(dtype)
    adata.obs["percent_mito"] = np.random.rand(adata.X.shape[0])
    adata.obs["n_counts"] = adata.X.sum(axis=1).get()
    adata.layers["counts"] = adata.X.copy()

    single = rsc.pp.regress_out(
        adata, keys=["n_counts", "percent_mito"], batchsize=batchsize, inplace=False
    )
    assert adata.X.shape == single.shape

    layer = rsc.pp.regress_out(
        adata,
        layer="counts",
        keys=["n_counts", "percent_mito"],
        batchsize=batchsize,
        inplace=False,
    )

    cp.testing.assert_array_equal(single, layer)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("sparse_format", ["csr", "csc", "dense"])
def test_regress_out_categorical(dtype, sparse_format):
    import pandas as pd
    from scipy.sparse import random as sp_random

    if sparse_format == "dense":
        adata = AnnData(sp_random(1000, 100, density=0.6, format="csr").toarray())
    else:
        adata = AnnData(sp_random(1000, 100, density=0.6, format=sparse_format))
    adata.X = adata.X.astype(dtype)
    adata.obs["batch"] = pd.Categorical(
        np.random.choice(["a", "b", "c"], size=adata.n_obs)
    )

    # scanpy reference
    adata_sc = adata.copy()
    sc.pp.regress_out(adata_sc, keys=["batch"])

    # rapids
    rsc.get.anndata_to_GPU(adata)
    rsc.pp.regress_out(adata, keys=["batch"])

    if dtype == "float32":
        cp.testing.assert_allclose(adata.X, cp.array(adata_sc.X), atol=1e-5)
    else:
        cp.testing.assert_allclose(adata.X, cp.array(adata_sc.X), atol=1e-7)


@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("batchsize", ["all", 100])
def test_regress_out_reproducible(dtype, batchsize):
    adata = sc.datasets.pbmc68k_reduced()
    adata = adata.raw.to_adata()[:200, :200].copy()
    rsc.get.anndata_to_GPU(adata)
    adata.X = adata.X.astype(dtype)
    rsc.pp.regress_out(adata, keys=["n_counts", "percent_mito"], batchsize=batchsize)
    # This file was generated for scanpy version 1.10.3
    tester = np.load(HERE / "_data/regress_test_small.npy")
    if dtype == "float32":
        cp.testing.assert_allclose(adata.X, tester, atol=1e-5)
    else:
        cp.testing.assert_allclose(adata.X, tester, atol=1e-7)
