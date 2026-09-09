from __future__ import annotations

import cupy as cp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from cupyx.scipy import sparse as sparse_gpu

import rapids_singlecell as rsc


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qc_metrics(dtype):
    cudata = AnnData(
        X=sparse_gpu.csr_matrix(
            cp.random.binomial(100, 0.005, (1000, 1000), dtype=dtype), dtype=dtype
        )
    )
    cudata.var["mito"] = np.concatenate(
        (np.ones(100, dtype=bool), np.zeros(900, dtype=bool))
    )
    cudata.var["negative"] = False
    rsc.pp.calculate_qc_metrics(cudata, qc_vars=["mito", "negative"])
    assert (cudata.obs["n_genes_by_counts"] < cudata.shape[1]).all()
    assert (
        cudata.obs["n_genes_by_counts"] >= cudata.obs["log1p_n_genes_by_counts"]
    ).all()
    assert (cudata.obs["total_counts"] == cp.ravel(cudata.X.sum(axis=1)).get()).all()
    assert (cudata.obs["total_counts"] >= cudata.obs["log1p_total_counts"]).all()
    assert (
        cudata.obs["total_counts_mito"] >= cudata.obs["log1p_total_counts_mito"]
    ).all()
    assert (cudata.obs["total_counts_negative"] == 0).all()

    for col in filter(lambda x: "negative" not in x, cudata.obs.columns):
        assert (cudata.obs[col] >= 0).all()  # Values should be positive or zero
        assert (cudata.obs[col] != 0).any().all()  # Nothing should be all zeros
        if col.startswith("pct_counts_in_top"):
            assert (cudata.obs[col] <= 100).all()
            assert (cudata.obs[col] >= 0).all()
    for col in cudata.var.columns:
        assert (cudata.var[col] >= 0).all()
    assert (
        cudata.var["mean_counts"] < cp.ravel(cudata.X.max(axis=0).toarray()).get()
    ).all()
    assert (cudata.var["mean_counts"] >= cudata.var["log1p_mean_counts"]).all()
    assert (cudata.var["total_counts"] >= cudata.var["log1p_total_counts"]).all()
    # Should return the same thing if run again
    old_obs, old_var = cudata.obs.copy(), cudata.var.copy()
    rsc.pp.calculate_qc_metrics(cudata, qc_vars=["mito", "negative"])
    assert set(cudata.obs.columns) == set(old_obs.columns)
    assert set(cudata.var.columns) == set(old_var.columns)
    for col in cudata.obs:
        assert np.allclose(cudata.obs[col], old_obs[col])
    for col in cudata.var:
        assert np.allclose(cudata.var[col], old_var[col])
    # inplace=False returns the metrics and leaves adata untouched
    plain = AnnData(X=cudata.X.copy(), var=cudata.var[["mito", "negative"]].copy())
    obs_metrics, var_metrics = rsc.pp.calculate_qc_metrics(
        plain, qc_vars=["mito", "negative"], inplace=False
    )
    assert list(plain.obs.columns) == []
    assert list(plain.var.columns) == ["mito", "negative"]
    pd.testing.assert_frame_equal(obs_metrics, cudata.obs[obs_metrics.columns])
    pd.testing.assert_frame_equal(var_metrics, cudata.var[var_metrics.columns])
    # with log1p=False
    cudata = AnnData(
        X=sparse_gpu.csr_matrix(
            cp.random.binomial(100, 0.005, (1000, 1000), dtype=dtype), dtype=dtype
        )
    )
    cudata.var["mito"] = np.concatenate(
        (np.ones(100, dtype=bool), np.zeros(900, dtype=bool))
    )
    cudata.var["negative"] = False
    rsc.pp.calculate_qc_metrics(cudata, qc_vars=["mito", "negative"], log1p=False)
    assert not np.any(cudata.obs.columns.str.startswith("log1p_"))
    assert not np.any(cudata.var.columns.str.startswith("log1p_"))


def adata_mito(dtype):
    a = cp.random.binomial(100, 0.005, (1000, 1000), dtype=dtype)
    init_var = pd.DataFrame(
        {"mito": np.concatenate((np.ones(100, dtype=bool), np.zeros(900, dtype=bool)))}
    )
    adata_dense = AnnData(X=a, var=init_var.copy())
    return adata_dense


@pytest.mark.parametrize(
    "cls", [cp.array, sparse_gpu.csc_matrix, sparse_gpu.csr_matrix]
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_qc_metrics_format(cls, dtype):
    cudata_dense = adata_mito(dtype)
    rsc.pp.calculate_qc_metrics(cudata_dense, qc_vars=["mito"])
    cudata = cudata_dense.copy()
    cudata.X = cls(cudata.X)

    rsc.pp.calculate_qc_metrics(cudata, qc_vars=["mito"])
    assert np.allclose(cudata.obs, cudata_dense.obs)
    for col in cudata.var:  # np.allclose doesn't like mix of types
        assert np.allclose(cudata.var[col], cudata_dense.var[col])
