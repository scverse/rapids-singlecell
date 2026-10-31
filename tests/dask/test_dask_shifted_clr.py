from __future__ import annotations

from pathlib import Path

import cupy as cp
import numpy as np
import pytest
from anndata import AnnData

import rapids_singlecell as rsc
from rapids_singlecell._compat import DaskArray
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)

SCANPY = np.load(Path(__file__).parents[1] / "_data/shifted_clr_scanpy.npz")


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("case", ["estimated", "alpha_0.3"])
def test_normalize_shifted_clr(client, data_kind, case):
    kwargs = {} if case == "estimated" else {"alpha": 0.3}
    to_dask = as_sparse_cupy_dask_array if data_kind == "sparse" else as_dense_cupy_dask_array
    adata = AnnData(to_dask(SCANPY["counts"]))
    assert adata.X.numblocks[0] > 1

    rsc.pp.normalize_shifted_clr(adata, **kwargs)

    assert isinstance(adata.X, DaskArray)
    assert isinstance(adata.X._meta, cp.ndarray)
    cp.testing.assert_allclose(adata.X.compute(), SCANPY[case], rtol=1e-12, atol=1e-12)
