from __future__ import annotations

import decoupler as dc
import numpy as np
import pandas as pd
import pytest
import scanpy as sc


@pytest.fixture
def rng():
    rng = np.random.default_rng(seed=42)
    return rng


@pytest.fixture
def adata():
    adata, _ = dc.ds.toy(nobs=40, nvar=20, bval=2, seed=42, verbose=False)
    adata.layers["counts"] = adata.X.round()
    return adata


@pytest.fixture
def tdata():
    tdata, _ = dc.ds.toy(nobs=40, nvar=20, bval=2, seed=42, verbose=False, pstime=True)
    return tdata


@pytest.fixture
def tdata_obsm(
    tdata,
    net,
    rng,
):
    sc.tl.pca(tdata)
    tdata.obsm["X_umap"] = tdata.obsm["X_pca"][:, :2] + rng.random(tdata.obsm["X_pca"][:, :2].shape)
    dc.mt.ulm(data=tdata, net=net, tmin=0)
    return tdata


@pytest.fixture
def pdata(
    adata,
    rng,
):
    adata.X = adata.X.round() * (rng.random(adata.shape) > 0.75)
    return dc.pp.pseudobulk(adata=adata, sample_col="sample", groups_col="group")


@pytest.fixture
def bdata():
    adata, _ = dc.ds.toy_bench(nobs=100, nvar=20, bval=2, seed=42, verbose=False)
    adata.obs["bm_group"] = adata.obs.apply(lambda x: [x["sample"], x["group"]], axis=1)
    return adata


@pytest.fixture
def deg():
    deg = pd.DataFrame(
        data=[
            [1, 0.5],
            [-2, 0.25],
            [3, 0.125],
            [-4, 0.05],
            [5, 0.025],
        ],
        columns=["stat", "padj"],
        index=["G01", "G02", "G03", "G04", "G05"],
    )
    return deg


@pytest.fixture
def net():
    _, net = dc.ds.toy(nobs=2, nvar=12, bval=2, seed=42, verbose=False)
    net = dc.pp.prune(features=net["target"].unique(), net=net, tmin=3)
    return net


@pytest.fixture
def unwnet(net):
    return net.drop(columns=["weight"], inplace=False)


@pytest.fixture
def mat(adata):
    return dc.pp.extract(adata)


@pytest.fixture
def idxmat(mat, net):
    X, obs, var = mat
    sources, cnct, starts, offsets = dc.pp.idxmat(features=var, net=net, verbose=False)
    return cnct, starts, offsets


@pytest.fixture
def adjmat(mat, net):
    X, obs, var = mat
    sources, targets, adjmat = dc.pp.adjmat(features=var, net=net, verbose=False)
    return adjmat
