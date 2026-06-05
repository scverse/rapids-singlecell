from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import issparse
from scanpy.get import _get_obs_rep

from rapids_singlecell._cuda import _pr_cuda
from rapids_singlecell.preprocessing._utils import (
    _check_gpu_X,
    _check_nonnegative_integers,
    _get_mean_var,
)

if TYPE_CHECKING:
    from anndata import AnnData


def _highly_variable_pearson_residuals(
    adata: AnnData,
    *,
    theta: float = 100,
    clip: float | None = None,
    n_top_genes: int = 2000,
    batch_key: str | None = None,
    check_values: bool = True,
    layer: str | None = None,
):
    """
    Select highly variable genes using analytic Pearson residuals.
    Pearson residuals of a negative binomial offset model are computed
    (with overdispersion `theta` shared across genes). By default, overdispersion
    `theta=100` is used and residuals are clipped to `sqrt(n_obs)`. Finally, genes
    are ranked by residual variance.
    Expects raw count input.
    """
    X = _get_obs_rep(adata, layer=layer)
    _check_gpu_X(X, require_cf=True)
    if check_values and not _check_nonnegative_integers(X):
        warnings.warn(
            "`flavor='pearson_residuals'` expects raw count data, but non-integers were found.",
            UserWarning,
        )
    if n_top_genes is None:
        n_top_genes = 2000
        warnings.warn(
            "`flavor='pearson_residuals'` expects `n_top_genes`  to be defined, defaulting to 2000 HVGs",
            UserWarning,
        )
    if theta <= 0:
        raise ValueError("Pearson residuals require theta > 0")
    if batch_key is None:
        batch_info = pd.Categorical(np.zeros(adata.shape[0], dtype=int))
    else:
        batch_info = adata.obs[batch_key].values

    n_batches = len(np.unique(batch_info))
    residual_gene_vars = []
    dtype = X.dtype

    for b in np.unique(batch_info):
        if issparse(X):
            X_batch = X[batch_info == b].tocsc()
            nnz_per_gene = cp.diff(X_batch.indptr).ravel()
        else:
            X_batch = cp.array(X[batch_info == b], dtype=dtype)
            nnz_per_gene = cp.sum(X_batch != 0, axis=0).ravel()
        nonzero_genes = cp.array(nnz_per_gene >= 1)
        X_batch = X_batch[:, nonzero_genes]
        if clip is None:
            n = X_batch.shape[0]
            clip_batch = cp.sqrt(n, dtype=dtype)
        else:
            clip_batch = clip
        if clip_batch < 0:
            raise ValueError("Pearson residuals require `clip>=0` or `clip=None`.")

        n_cells = X_batch.shape[0]
        n_genes = X_batch.shape[1]
        clip_val = float(clip_batch)
        inv_theta = 1.0 / theta
        residual_gene_var = cp.zeros(n_genes, dtype=dtype, order="C")
        stream = cp.cuda.get_current_stream().ptr

        if issparse(X_batch):
            # Compute sums using custom kernel (single pass, no CSC->CSR conversion)
            sums_genes = cp.zeros(n_genes, dtype=dtype)
            sums_cells = cp.zeros(n_cells, dtype=dtype)
            _pr_cuda.sparse_sum_csc(
                X_batch.indptr,
                X_batch.indices,
                X_batch.data,
                sums_genes=sums_genes,
                sums_cells=sums_cells,
                n_genes=n_genes,
                stream=stream,
            )
            sum_total = float(sums_genes.sum())
            inv_sum_total = 1.0 / sum_total

            _pr_cuda.csc_hvg_res(
                X_batch.indptr,
                X_batch.indices,
                X_batch.data,
                sums_genes=sums_genes,
                sums_cells=sums_cells,
                residuals=residual_gene_var,
                inv_sum_total=inv_sum_total,
                clip=clip_val,
                inv_theta=inv_theta,
                n_genes=n_genes,
                n_cells=n_cells,
                stream=stream,
            )
        else:
            sums_genes = cp.sum(X_batch, axis=0, dtype=dtype).ravel()
            sums_cells = cp.sum(X_batch, axis=1, dtype=dtype).ravel()
            sum_total = float(sums_genes.sum())
            inv_sum_total = 1.0 / sum_total

            _pr_cuda.dense_hvg_res(
                cp.asfortranarray(X_batch),
                sums_genes=sums_genes,
                sums_cells=sums_cells,
                residuals=residual_gene_var,
                inv_sum_total=inv_sum_total,
                clip=clip_val,
                inv_theta=inv_theta,
                n_genes=n_genes,
                n_cells=n_cells,
                stream=stream,
            )

        unmasked_residual_gene_var = cp.zeros(len(nonzero_genes))
        unmasked_residual_gene_var[nonzero_genes] = residual_gene_var
        residual_gene_vars.append(unmasked_residual_gene_var.reshape(1, -1))

    residual_gene_vars = cp.concatenate(residual_gene_vars, axis=0)
    # Get rank per gene within each batch
    # argsort twice gives ranks, small rank means most variable
    ranks_residual_var = cp.argsort(cp.argsort(-residual_gene_vars, axis=1), axis=1)
    ranks_residual_var = ranks_residual_var.astype(X.dtype)
    # count in how many batches a genes was among the n_top_genes
    highly_variable_nbatches = cp.sum(
        (ranks_residual_var < n_top_genes).astype(int), axis=0
    ).get()
    ranks_residual_var[ranks_residual_var >= n_top_genes] = np.nan
    ranks_residual_var = ranks_residual_var.get()
    ranks_masked_array = np.ma.masked_invalid(ranks_residual_var)
    # Median rank across batches, ignoring batches in which gene was not selected
    medianrank_residual_var = np.ma.median(ranks_masked_array, axis=0).filled(np.nan)
    means, variances = _get_mean_var(X, axis=0)
    means, variances = means.get(), variances.get()
    df = pd.DataFrame.from_dict(
        {
            "means": means,
            "variances": variances,
            "residual_variances": cp.mean(residual_gene_vars, axis=0).get(),
            "highly_variable_rank": medianrank_residual_var,
            "highly_variable_nbatches": highly_variable_nbatches.astype(np.int64),
            "highly_variable_intersection": highly_variable_nbatches == n_batches,
        }
    )
    df = df.set_index(adata.var_names)
    df.sort_values(
        ["highly_variable_nbatches", "highly_variable_rank"],
        ascending=[False, True],
        na_position="last",
        inplace=True,
    )
    high_var = np.zeros(df.shape[0], dtype=bool)
    high_var[:n_top_genes] = True
    df["highly_variable"] = high_var
    df = df.loc[adata.var_names, :]

    computed_on = layer if layer else "adata.X"
    adata.uns["hvg"] = {"flavor": "pearson_residuals", "computed_on": computed_on}
    adata.var["means"] = df["means"].values
    adata.var["variances"] = df["variances"].values
    adata.var["residual_variances"] = df["residual_variances"]
    adata.var["highly_variable_rank"] = df["highly_variable_rank"].values
    if batch_key is not None:
        adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"].values
        adata.var["highly_variable_intersection"] = df[
            "highly_variable_intersection"
        ].values
    adata.var["highly_variable"] = df["highly_variable"].values
