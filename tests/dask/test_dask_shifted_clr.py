from __future__ import annotations

import cupy as cp
import pytest
from anndata import AnnData

import rapids_singlecell as rsc
from rapids_singlecell._compat import DaskArray
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)

from ..test_shifted_clr import _counts, _scanpy_shifted_clr, needs_scanpy_shifted_clr  # noqa: TID252


@needs_scanpy_shifted_clr
@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("alpha", [None, 0.3], ids=["estimated", "explicit"])
def test_normalize_shifted_clr(client, data_kind, alpha):
    x = _counts()
    expected = _scanpy_shifted_clr(x, alpha=alpha)
    to_dask = as_sparse_cupy_dask_array if data_kind == "sparse" else as_dense_cupy_dask_array
    adata = AnnData(to_dask(x))
    assert adata.X.numblocks[0] > 1

    rsc.pp.normalize_shifted_clr(adata, alpha=alpha)

    assert isinstance(adata.X, DaskArray)
    assert isinstance(adata.X._meta, cp.ndarray)
    cp.testing.assert_allclose(adata.X.compute(), expected, rtol=1e-12, atol=1e-12)
