from __future__ import annotations

import cupy as cp
import pytest
from scanpy.datasets import pbmc3k_processed, pbmc68k_reduced

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_sparse_cupy_dask_array,
)


def _compare_top_genes(result1, result2, top_n=10, min_overlap=9):
    """
    Compare top N genes between two rank_genes_groups results.

    Parameters
    ----------
    result1, result2 : dict
        Results from rank_genes_groups with 'names' key
    top_n : int
        Number of top genes to compare
    min_overlap : int
        Minimum number of overlapping genes required

    Returns
    -------
    bool
        True if overlap meets minimum threshold for all groups
    """
    groups1 = result1["names"].dtype.names
    groups2 = result2["names"].dtype.names

    if set(groups1) != set(groups2):
        return False

    for group in groups1:
        top_genes1 = set(result1["names"][group][:top_n])
        top_genes2 = set(result2["names"][group][:top_n])
        overlap = len(top_genes1.intersection(top_genes2))

        if overlap < min_overlap:
            return False

    return True


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_rank_genes_groups_logreg(dist_client, data_kind, dtype):
    if data_kind == "dense":
        adata = pbmc68k_reduced()
        adata.X = adata.X.astype(dtype)
        dask_data = adata.copy()
        dask_data.X = as_dense_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)
        groupby = "bulk_labels"
    elif data_kind == "sparse":
        adata = pbmc3k_processed()
        org_var_names = adata.var_names
        adata = adata.raw.to_adata()
        adata = adata[:, org_var_names].copy()
        adata.X = adata.X.astype(dtype)
        dask_data = adata.copy()
        dask_data.X = as_sparse_cupy_dask_array(dask_data.X).persist()
        rsc.get.anndata_to_GPU(adata)
        groupby = "louvain"

    rsc.tl.rank_genes_groups_logreg(adata, groupby=groupby, use_raw=False)
    rsc.tl.rank_genes_groups_logreg(dask_data, groupby=groupby, use_raw=False)

    assert _compare_top_genes(
        adata.uns["rank_genes_groups"], dask_data.uns["rank_genes_groups"]
    )
