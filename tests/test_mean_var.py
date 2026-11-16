from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from fast_array_utils.stats import mean_var as sc_get_mean_var
from scanpy.datasets import pbmc3k, pbmc68k_reduced

import rapids_singlecell as rsc
from rapids_singlecell.preprocessing._utils import _get_mean_var as rsc_get_mean_var


@pytest.mark.parametrize("data_kind", ["csc", "csr", "dense"])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("correction", [0, 1])
def test_mean_var(data_kind, axis, dtype, correction):
    if data_kind == "dense":
        adata = pbmc68k_reduced()
    else:
        adata = pbmc3k()
        if data_kind == "csc":
            adata.X = adata.X.tocsc()

    adata.X = adata.X.astype(dtype)
    cudata = rsc.get.anndata_to_GPU(adata, copy=True)

    mean, var = sc_get_mean_var(adata.X, axis=axis, correction=correction)
    rsc_mean, rsc_var = rsc_get_mean_var(cudata.X, axis=axis, correction=correction)

    cp.testing.assert_allclose(mean, rsc_mean)
    cp.testing.assert_allclose(var, rsc_var)
