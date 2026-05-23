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
        # batchsize="all" now uses `cp.linalg.solve` (more stable than the
        # previous `inv() @ ...`); batchsize=100 uses cuML's batched SVD.
        # The two converge to within ~3e-7 at f64 — slightly looser than the
        # previous atol=1e-7 which compared two `inv`-based paths.
        cp.testing.assert_allclose(cupy, rapids, atol=1e-6)


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


def test_regress_out_near_singular_regressors_dask_raises():
    """The Dask path doesn't have a clean batched fallback, so when the Gram
    matrix is numerically singular it should raise ValueError with a useful
    message (rather than silently producing garbage from an unstable ``inv``).
    """
    import dask.array as da

    from rapids_singlecell.preprocessing._regress_out import (
        _regress_out_continuous_dask,
    )

    rng = np.random.default_rng(0)
    n_cells, n_genes = 5000, 20
    X_cp = cp.asarray(rng.standard_normal((n_cells, n_genes)).astype("float32"))
    X_dask = da.from_array(X_cp, chunks=(1000, n_genes))
    k1 = rng.uniform(0.5, 10.0, size=n_cells).astype("float32")
    # Tight collinearity to ensure cond > 1e12 (same shape as the dense test)
    k2 = k1 + (1e-6 * rng.standard_normal(n_cells)).astype("float32")

    # Build a minimal AnnData stand-in: only obs is used by the dask helper.
    adata = AnnData(X=cp.asarray(X_cp))
    adata.obs["k1"] = k1
    adata.obs["k2"] = k2

    with pytest.raises(ValueError, match="numerically singular"):
        _regress_out_continuous_dask(X_dask, adata, ["k1", "k2"])


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_regress_out_near_singular_regressors_no_garbage(dtype):
    """When the user passes near-collinear regressors, the Gram matrix is
    numerically singular and ``cp.linalg.det == 0`` is a vacuous guard
    (e.g. cond=1e12 → det≈90, not 0). Before the fix, the
    ``batchsize="all"`` path called ``cp.linalg.inv`` on the near-singular
    Gram and produced residuals on the order of 1e+5 (vs ~1 from the input).

    The fix detects ill-conditioning via the condition number and falls back
    to the batched SVD path. This test confirms:
      1. No NaN / Inf in the output.
      2. Residuals stay bounded (max |residual| ≤ some-multiple-of-input-stdev),
         instead of exploding to 1e+5 as the bug produced.
    """
    rng = np.random.default_rng(0)
    n_cells, n_genes = 5000, 50
    X = rng.standard_normal((n_cells, n_genes)).astype(dtype)
    k1 = rng.uniform(0.5, 10.0, size=n_cells).astype(dtype)
    # Near-collinear with k1 — cond(R^T R) ~ 1e12, det ~ 1e2 (so `det == 0`
    # would NOT have fired — the bug condition).
    k2 = k1 + (1e-5 * rng.standard_normal(n_cells)).astype(dtype)

    adata = AnnData(X=cp.asarray(X))
    adata.obs["k1"] = k1
    adata.obs["k2"] = k2

    rsc.pp.regress_out(adata, keys=["k1", "k2"], batchsize="all")
    out = cp.asnumpy(adata.X)

    # Property 1: no NaN/Inf
    assert np.isfinite(out).all(), (
        "regress_out produced NaN/Inf on near-singular regressors — the "
        "`det == 0` guard let an unstable `cp.linalg.inv` slip through. The "
        "fix should detect this via `cp.linalg.cond` and fall back."
    )

    # Property 2: residuals stay in the same order as the input. The input has
    # std ≈ 1, so residuals should be O(1). Before the fix, |residuals| reached
    # ~1.6e+5; after the fix they're bounded by < 100x input.
    max_abs_res = float(np.abs(out).max())
    max_abs_in = float(np.abs(X).max())
    assert max_abs_res < 100 * max_abs_in, (
        f"regress_out residuals exploded: max |residual| = {max_abs_res:.2e} "
        f"vs max |input| = {max_abs_in:.2e}. The `det == 0` guard at "
        f"_regress_out.py was vacuous for near-singular regressors. After the "
        f"fix this should drop dramatically (residuals stay O(input))."
    )


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
