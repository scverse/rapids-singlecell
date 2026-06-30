from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
import scanpy as sc
from anndata import AnnData
from cupy.testing import assert_allclose
from cupyx.scipy.sparse import csc_matrix, csr_matrix, issparse
from scanpy.datasets import pbmc68k_reduced

import rapids_singlecell as rsc


@pytest.mark.parametrize("array_type", (csr_matrix, csc_matrix, cp.ndarray))
@pytest.mark.parametrize(
    ("max_cells", "max_counts", "min_cells", "min_counts"),
    [
        (100, None, None, None),
        (None, 100, None, None),
        (None, None, 20, None),
        (None, None, None, 20),
    ],
)
def test_filter_genes(array_type, max_cells, max_counts, min_cells, min_counts):
    adata = pbmc68k_reduced()
    adata.X = adata.raw.X.todense()
    rsc.get.anndata_to_GPU(adata)
    adata_casted = adata.copy()
    if array_type is not cp.ndarray:
        adata_casted.X = array_type(adata_casted.X)
    rsc.pp.filter_genes(
        adata,
        max_cells=max_cells,
        max_counts=max_counts,
        min_cells=min_cells,
        min_counts=min_counts,
    )
    rsc.pp.filter_genes(
        adata_casted,
        max_cells=max_cells,
        max_counts=max_counts,
        min_cells=min_cells,
        min_counts=min_counts,
    )
    X = adata_casted.X
    if issparse(X):
        X = X.todense()

    assert_allclose(X, adata.X, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("array_type", (csr_matrix, csc_matrix, cp.ndarray))
@pytest.mark.parametrize(
    ("max_genes", "max_counts", "min_genes", "min_counts"),
    [
        (100, None, None, None),
        (None, 100, None, None),
        (None, None, 20, None),
        (None, None, None, 20),
    ],
)
def test_filter_cells(array_type, max_genes, max_counts, min_genes, min_counts):
    adata = pbmc68k_reduced()
    adata.X = adata.raw.X.todense()
    rsc.get.anndata_to_GPU(adata)
    adata_casted = adata.copy()
    if array_type is not cp.ndarray:
        adata_casted.X = array_type(adata_casted.X)
    rsc.pp.filter_cells(
        adata,
        max_genes=max_genes,
        max_counts=max_counts,
        min_genes=min_genes,
        min_counts=min_counts,
    )
    rsc.pp.filter_cells(
        adata_casted,
        max_genes=max_genes,
        max_counts=max_counts,
        min_genes=min_genes,
        min_counts=min_counts,
    )
    X = adata_casted.X
    if issparse(X):
        X = X.todense()
    assert_allclose(X, adata.X, rtol=1e-5, atol=1e-5)


# integer counts so n_counts (value sum) differs from n_cells (nonzero count)
_COUNTS = (np.random.default_rng(0).random((40, 12)) > 0.4) * np.random.default_rng(
    1
).integers(1, 5, (40, 12))


@pytest.mark.parametrize(
    ("kwargs", "expected", "unexpected"),
    [
        ({"min_counts": 2}, "n_counts", "n_cells"),
        ({"min_cells": 2}, "n_cells", "n_counts"),
    ],
)
def test_filter_genes_single_column(kwargs, expected, unexpected):
    # only the threshold-appropriate column is written, matching scanpy
    X = _COUNTS.astype("float32")
    rsc_ad = AnnData(cp.asarray(X.copy()))
    rsc.pp.filter_genes(rsc_ad, **kwargs)
    sc_ad = AnnData(X.copy())
    sc.pp.filter_genes(sc_ad, **kwargs)
    assert expected in rsc_ad.var.columns
    assert unexpected not in rsc_ad.var.columns
    np.testing.assert_allclose(
        rsc_ad.var[expected].to_numpy(), sc_ad.var[expected].to_numpy()
    )


@pytest.mark.parametrize(
    ("kwargs", "expected", "unexpected"),
    [
        ({"min_counts": 2}, "n_counts", "n_genes"),
        ({"min_genes": 2}, "n_genes", "n_counts"),
    ],
)
def test_filter_cells_single_column(kwargs, expected, unexpected):
    # only the threshold-appropriate column is written, matching scanpy
    X = _COUNTS.astype("float32")
    rsc_ad = AnnData(cp.asarray(X.copy()))
    rsc.pp.filter_cells(rsc_ad, **kwargs)
    sc_ad = AnnData(X.copy())
    sc.pp.filter_cells(sc_ad, **kwargs)
    assert expected in rsc_ad.obs.columns
    assert unexpected not in rsc_ad.obs.columns
    np.testing.assert_allclose(
        rsc_ad.obs[expected].to_numpy(), sc_ad.obs[expected].to_numpy()
    )
