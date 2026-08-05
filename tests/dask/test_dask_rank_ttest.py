from __future__ import annotations

import cupy as cp
import numpy as np
import pytest
from scanpy.datasets import pbmc3k_processed, pbmc68k_reduced

import rapids_singlecell as rsc
from testing.rapids_singlecell._helper import (
    as_dense_cupy_dask_array,
    as_dense_dask_array,
    as_sparse_cupy_dask_array,
    as_sparse_dask_matrix,
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
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_dask(client, data_kind, dtype, method):
    """Test t-test methods with dask arrays."""
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

    rsc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False)
    rsc.tl.rank_genes_groups(dask_data, groupby=groupby, method=method, use_raw=False)

    # Compare top genes overlap
    assert _compare_top_genes(
        adata.uns["rank_genes_groups"], dask_data.uns["rank_genes_groups"]
    )

    # Compare scores by gene name (accounts for tie-breaking differences in sorting)
    # Genes with tied scores may appear in different order, but scores should match
    for group in adata.uns["rank_genes_groups"]["scores"].dtype.names:
        gpu_names = list(adata.uns["rank_genes_groups"]["names"][group])
        gpu_scores = np.asarray(
            adata.uns["rank_genes_groups"]["scores"][group], dtype=float
        )
        dask_names = list(dask_data.uns["rank_genes_groups"]["names"][group])
        dask_scores = np.asarray(
            dask_data.uns["rank_genes_groups"]["scores"][group], dtype=float
        )

        # Create gene->score mappings
        gpu_score_map = dict(zip(gpu_names, gpu_scores))
        dask_score_map = dict(zip(dask_names, dask_scores))

        # All genes should have identical scores
        for gene in gpu_score_map:
            np.testing.assert_allclose(
                gpu_score_map[gene],
                dask_score_map[gene],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Score mismatch for gene {gene} in group {group}",
            )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
def test_rank_genes_groups_wilcoxon_dask_errors(client, data_kind):
    """Test that wilcoxon raises an error with dask arrays."""
    if data_kind == "dense":
        adata = pbmc68k_reduced()
        adata.X = as_dense_cupy_dask_array(adata.X).persist()
        groupby = "bulk_labels"
    elif data_kind == "sparse":
        adata = pbmc3k_processed()
        org_var_names = adata.var_names
        adata = adata.raw.to_adata()
        adata = adata[:, org_var_names].copy()
        adata.X = as_sparse_cupy_dask_array(adata.X).persist()
        groupby = "louvain"

    with pytest.raises(ValueError, match="Wilcoxon test is not supported for Dask"):
        rsc.tl.rank_genes_groups(
            adata, groupby=groupby, method="wilcoxon", use_raw=False
        )


@pytest.mark.parametrize("data_kind", ["sparse", "dense"])
@pytest.mark.parametrize("method", ["t-test", "t-test_overestim_var"])
def test_rank_genes_groups_ttest_cpu_dask(client, data_kind, method):
    """Test t-test methods with CPU dask arrays (auto-converted to GPU).

    Compares CPU dask arrays against GPU cupy arrays to ensure conversion works.
    """
    if data_kind == "dense":
        adata = pbmc68k_reduced()
        cpu_dask = adata.copy()
        X = as_dense_dask_array(cpu_dask.X)
        # Rechunk to only have row-wise chunks (matching GPU dask array convention)
        X = X.rechunk((X.shape[0] // 2, X.shape[1]))
        cpu_dask.X = X.persist()
        # Convert original to GPU for reference comparison
        rsc.get.anndata_to_GPU(adata)
        groupby = "bulk_labels"
    elif data_kind == "sparse":
        adata = pbmc3k_processed()
        org_var_names = adata.var_names
        adata = adata.raw.to_adata()
        adata = adata[:, org_var_names].copy()
        cpu_dask = adata.copy()
        X = as_sparse_dask_matrix(cpu_dask.X)
        # Rechunk to only have row-wise chunks (matching GPU dask array convention)
        X = X.rechunk((X.shape[0] // 2, X.shape[1]))
        cpu_dask.X = X.persist()
        # Convert original to GPU for reference comparison
        rsc.get.anndata_to_GPU(adata)
        groupby = "louvain"

    # Run rsc on GPU cupy array for reference
    rsc.tl.rank_genes_groups(adata, groupby=groupby, method=method, use_raw=False)

    # Run rsc on CPU dask array (gets converted to GPU internally)
    rsc.tl.rank_genes_groups(cpu_dask, groupby=groupby, method=method, use_raw=False)

    # Compare top genes overlap
    assert _compare_top_genes(
        adata.uns["rank_genes_groups"], cpu_dask.uns["rank_genes_groups"]
    )

    # Compare scores by gene name
    for group in adata.uns["rank_genes_groups"]["scores"].dtype.names:
        gpu_names = list(adata.uns["rank_genes_groups"]["names"][group])
        gpu_scores = np.asarray(
            adata.uns["rank_genes_groups"]["scores"][group], dtype=float
        )
        dask_names = list(cpu_dask.uns["rank_genes_groups"]["names"][group])
        dask_scores = np.asarray(
            cpu_dask.uns["rank_genes_groups"]["scores"][group], dtype=float
        )

        # Create gene->score mappings
        gpu_score_map = dict(zip(gpu_names, gpu_scores))
        dask_score_map = dict(zip(dask_names, dask_scores))

        # All genes should have matching scores
        for gene in gpu_score_map:
            np.testing.assert_allclose(
                gpu_score_map[gene],
                dask_score_map[gene],
                rtol=1e-5,
                atol=1e-8,
                err_msg=f"Score mismatch for gene {gene} in group {group}",
            )
