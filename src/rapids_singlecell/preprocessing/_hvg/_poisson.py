from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from scanpy.get import _get_obs_rep

from rapids_singlecell._compat import DaskArray
from rapids_singlecell._cuda import _hvg_cuda as _hvg
from rapids_singlecell.preprocessing._qc import _basic_qc
from rapids_singlecell.preprocessing._utils import (
    _check_gpu_X,
    _check_nonnegative_integers,
)

if TYPE_CHECKING:
    from anndata import AnnData


def _poisson_gene_selection(
    adata: AnnData,
    *,
    layer: str | None = None,
    n_top_genes: int | None = None,
    batch_key: str | None = None,
    check_values: bool = True,
) -> None:
    """
    Rank and select genes based on the enrichment of zero counts.

    Enrichment is considered by comparing data to a Poisson count model.
    This is based on M3Drop: https://github.com/tallulandrews/M3Drop

    The probability of zero enrichment is computed as:
    P(observed > expected) = p_obs * (1 - p_exp)

    Parameters
    ----------
    adata
        AnnData object
    layer
        If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
    n_top_genes
        How many variable genes to select.
    batch_key
        Key in adata.obs for batch information. If None, treat as single batch.

    Returns
    -------
    Updates `.var` with highly_variable, observed_fraction_zeros,
    expected_fraction_zeros, prob_zero_enrichment, and related fields.
    """
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='poisson_gene_selection'` expects `n_top_genes` to be defined, "
            "defaulting to 2000 HVGs",
            UserWarning,
        )

    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X, allow_dask=True)
    check_values = False if isinstance(X, DaskArray) else check_values
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='poisson_gene_selection'` expects raw count data, "
            "but non-integers were found.",
            UserWarning,
        )

    # Handle batches
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key].values

    n_genes = adata.n_vars
    batches = np.unique(batch_info)
    n_batches = len(batches)

    # Use input dtype (float32 or float64)
    dtype = X.dtype

    # Pre-allocate result arrays
    all_prob_enrichment = cp.zeros((n_batches, n_genes), dtype=dtype)
    all_obs_frac_zeros = cp.zeros((n_batches, n_genes), dtype=dtype)
    all_exp_frac_zeros = cp.zeros((n_batches, n_genes), dtype=dtype)

    for batch_idx, b in enumerate(batches):
        batch_mask = batch_info == b
        X_batch = X[batch_mask]

        # Single-pass computation of all statistics using optimized kernels
        # Returns: (sums_cells, sums_genes, genes_per_cell, cells_per_gene)
        total_counts, gene_sums, _, cells_per_gene = _basic_qc(X_batch)

        n_cells = X_batch.shape[0]

        # Observed fraction of zeros = 1 - (cells expressing gene / total cells)
        observed_frac_zeros = 1.0 - cells_per_gene.astype(dtype) / n_cells

        # Scaled means for Poisson model
        total_sum = gene_sums.sum()
        scaled_means = gene_sums.astype(dtype) / total_sum

        # Expected fraction of zeros under Poisson model (single kernel call)
        expected_frac_zeros = _compute_expected_zeros_kernel(
            scaled_means,
            total_counts.astype(dtype),
            n_genes,
            n_cells,
            dtype,
        )

        # Exact probability: P(Bern(p_obs) > Bern(p_exp)) = p_obs * (1 - p_exp)
        prob_enrichment = observed_frac_zeros * (1.0 - expected_frac_zeros)

        # Store results
        all_prob_enrichment[batch_idx] = prob_enrichment
        all_obs_frac_zeros[batch_idx] = observed_frac_zeros
        all_exp_frac_zeros[batch_idx] = expected_frac_zeros

    # Aggregate and write results
    _write_results(
        adata,
        all_prob_enrichment,
        all_obs_frac_zeros,
        all_exp_frac_zeros,
        n_top_genes,
    )


def _compute_expected_zeros_kernel(
    scaled_means: cp.ndarray,
    total_counts: cp.ndarray,
    n_genes: int,
    n_cells: int,
    dtype,
) -> cp.ndarray:
    """
    Compute expected fraction of zeros under Poisson model using CUDA kernel.

    E[zeros_g] = (1/n_cells) * sum_c exp(-scaled_means[g] * total_counts[c])

    Single kernel call, no intermediate matrices, no Python loops.
    """
    expected = cp.zeros(n_genes, dtype=dtype)

    _hvg.expected_zeros(
        scaled_means,
        total_counts,
        expected,
        n_genes,
        n_cells,
        stream=cp.cuda.get_current_stream().ptr,
    )

    return expected


def _write_results(
    adata: AnnData,
    prob_enrichment: np.ndarray,
    obs_frac_zeros: np.ndarray,
    exp_frac_zeros: np.ndarray,
    n_top_genes: int,
) -> None:
    """Aggregate batch results and update adata.var."""
    n_genes = prob_enrichment.shape[1]

    # Rank genes within each batch (higher enrichment = higher rank)
    ranks = prob_enrichment.argsort(axis=1).argsort(axis=1)

    # Median across batches
    median_prob = cp.median(prob_enrichment, axis=0).get()
    median_obs = cp.median(obs_frac_zeros, axis=0).get()
    median_exp = cp.median(exp_frac_zeros, axis=0).get()
    median_rank = cp.median(ranks, axis=0).get()

    # Count batches where gene is in top n_top_genes
    rank_threshold = n_genes - n_top_genes
    n_batches_enriched = cp.sum(ranks >= rank_threshold, axis=0).get()

    # Build DataFrame and select top genes
    df = pd.DataFrame(
        {
            "observed_fraction_zeros": median_obs,
            "expected_fraction_zeros": median_exp,
            "prob_zero_enriched_nbatches": n_batches_enriched,
            "prob_zero_enrichment": median_prob,
            "prob_zero_enrichment_rank": median_rank,
            "highly_variable": False,
        },
        index=adata.var_names,
    )

    top_genes = df.nlargest(
        n_top_genes, ["prob_zero_enriched_nbatches", "prob_zero_enrichment_rank"]
    ).index
    df.loc[top_genes, "highly_variable"] = True

    # Update adata
    adata.uns["hvg"] = {"flavor": "poisson_zeros"}
    for col in df.columns:
        adata.var[col] = df[col].values
