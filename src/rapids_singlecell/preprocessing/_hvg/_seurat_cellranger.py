from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csr_matrix, issparse

from rapids_singlecell._compat import DaskArray, _meta_dense, _meta_sparse
from rapids_singlecell.get import _get_obs_rep
from rapids_singlecell.preprocessing._qc import _basic_qc
from rapids_singlecell.preprocessing._utils import (
    _check_gpu_X,
    _get_mean_var,
    _sanitize_column,
)

if TYPE_CHECKING:
    from anndata import AnnData
    from numpy.typing import NDArray

    from ._cutoffs import _Cutoffs


def _hvg_expm1(X):
    if isinstance(X, DaskArray):
        meta = _meta_sparse if isinstance(X._meta, csr_matrix) else _meta_dense
        X = X.map_blocks(_hvg_expm1, meta=meta(X.dtype))
    else:
        X = X.copy()
        if issparse(X):
            X = X.expm1()
        else:
            X = cp.expm1(X)
    return X


def _highly_variable_genes_single_batch(
    adata: AnnData,
    *,
    layer: str | None = None,
    cutoff: _Cutoffs | int,
    n_bins: int = 20,
    flavor: Literal["seurat", "cell_ranger"] = "seurat",
) -> pd.DataFrame:
    """\
    See `highly_variable_genes`.

    Returns
    -------
    A DataFrame that contains the columns
    `highly_variable`, `means`, `dispersions`, and `dispersions_norm`.
    """
    X = _get_obs_rep(adata, layer=layer)

    _check_gpu_X(X, allow_dask=True)
    if hasattr(X, "_view_args"):  # AnnData array view
        X = X.copy()

    if flavor == "seurat":
        X = _hvg_expm1(X)

    mean, var = _get_mean_var(X, axis=0)
    if isinstance(X, DaskArray):
        import dask

        mean, var = dask.compute(mean, var)
    mean[mean == 0] = 1e-12
    disp = var / mean
    if flavor == "seurat":  # logarithmized mean as in Seurat
        disp[disp == 0] = np.nan
        disp = cp.log(disp)
        mean = cp.log1p(mean)
    df = pd.DataFrame()
    mean = mean.get()
    disp = disp.get()
    df["means"] = mean
    df["dispersions"] = disp
    df["mean_bin"] = _get_mean_bins(df["means"], flavor, n_bins)
    disp_stats = _get_disp_stats(df, flavor)

    # actually do the normalization
    df["dispersions_norm"] = (df["dispersions"] - disp_stats["avg"]) / disp_stats["dev"]
    df["highly_variable"] = _subset_genes(
        adata,
        mean=mean,
        dispersion_norm=df["dispersions_norm"].to_numpy(),
        cutoff=cutoff,
    )
    df.index = adata.var_names
    return df


def _get_mean_bins(
    means: pd.Series, flavor: Literal["seurat", "cell_ranger"], n_bins: int
) -> pd.Series:
    if flavor == "seurat":
        bins = n_bins
    elif flavor == "cell_ranger":
        bins = np.r_[-np.inf, np.percentile(means, np.arange(10, 105, 5)), np.inf]
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')

    return pd.cut(means, bins=bins)


def _get_disp_stats(
    df: pd.DataFrame, flavor: Literal["seurat", "cell_ranger"]
) -> pd.DataFrame:
    disp_grouped = df.groupby("mean_bin", observed=True)["dispersions"]
    if flavor == "seurat":
        disp_bin_stats = disp_grouped.agg(avg="mean", dev="std")
        _postprocess_dispersions_seurat(disp_bin_stats, df["mean_bin"])
    elif flavor == "cell_ranger":
        disp_bin_stats = disp_grouped.agg(avg="median", dev=_mad)
    else:
        raise ValueError('`flavor` needs to be "seurat" or "cell_ranger"')
    return disp_bin_stats.loc[df["mean_bin"]].set_index(df.index)


def _postprocess_dispersions_seurat(
    disp_bin_stats: pd.DataFrame, mean_bin: pd.Series
) -> None:
    # retrieve those genes that have nan std, these are the ones where
    # only a single gene fell in the bin and implicitly set them to have
    # a normalized disperion of 1
    one_gene_per_bin = disp_bin_stats["dev"].isnull()
    gen_indices = np.flatnonzero(one_gene_per_bin.loc[mean_bin])
    if len(gen_indices) == 0:
        return
    disp_bin_stats.loc[one_gene_per_bin, "dev"] = disp_bin_stats.loc[
        one_gene_per_bin, "avg"
    ]
    disp_bin_stats.loc[one_gene_per_bin, "avg"] = 0


def _mad(a):
    from statsmodels.robust import mad

    with warnings.catch_warnings():
        # MAD calculation raises the warning: "Mean of empty slice"
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return mad(a)


def _subset_genes(
    adata: AnnData,
    *,
    mean: NDArray[np.float64],
    dispersion_norm: NDArray[np.float64],
    cutoff: _Cutoffs | int,
) -> NDArray[np.bool_]:
    """Get boolean mask of genes with normalized dispersion in bounds."""
    from ._cutoffs import _Cutoffs

    if isinstance(cutoff, _Cutoffs):
        dispersion_norm = np.nan_to_num(dispersion_norm)  # similar to Seurat
        return cutoff.in_bounds(mean, dispersion_norm)
    n_top_genes = cutoff
    del cutoff

    if n_top_genes > adata.n_vars:
        n_top_genes = adata.n_vars
    disp_cut_off = _nth_highest(dispersion_norm, n_top_genes)
    return np.nan_to_num(dispersion_norm) >= disp_cut_off


def _nth_highest(x: NDArray[np.float64], n: int) -> float:
    x = x[~np.isnan(x)]
    if n > x.size:
        msg = "`n_top_genes` > number of normalized dispersions, returning all genes with normalized dispersions."
        warnings.warn(msg, UserWarning)
        n = x.size
    # interestingly, np.argpartition is slightly slower
    x[::-1].sort()
    return x[n - 1]


def _highly_variable_genes_batched(
    adata: AnnData,
    batch_key: str,
    *,
    layer: str | None,
    n_bins: int,
    flavor: Literal["seurat", "cell_ranger"],
    cutoff: _Cutoffs | int,
) -> pd.DataFrame:
    _sanitize_column(adata, batch_key)
    batches = adata.obs[batch_key].cat.categories
    dfs = []
    gene_list = adata.var_names
    for batch in batches:
        adata_subset = adata[adata.obs[batch_key] == batch]

        X = _get_obs_rep(adata_subset, layer=layer)
        _check_gpu_X(X, allow_dask=True)
        _, _, _, n_cells_per_gene = _basic_qc(X=X)
        filt = (n_cells_per_gene > 0).get()
        adata_subset = adata_subset[:, filt]

        hvg = _highly_variable_genes_single_batch(
            adata_subset,
            layer=layer,
            cutoff=cutoff,
            n_bins=n_bins,
            flavor=flavor,
        )
        hvg.reset_index(drop=False, inplace=True, names=["gene"])

        if (n_removed := np.sum(~filt)) > 0:
            # Add 0 values for genes that were filtered out
            missing_hvg = pd.DataFrame(
                np.zeros((n_removed, len(hvg.columns))),
                columns=hvg.columns,
            )
            missing_hvg["highly_variable"] = missing_hvg["highly_variable"].astype(bool)
            missing_hvg["gene"] = gene_list[~filt]
            hvg = pd.concat([hvg, missing_hvg], ignore_index=True)

        dfs.append(hvg)

    df = pd.concat(dfs, axis=0)

    df["highly_variable"] = df["highly_variable"].astype(int)
    df = df.groupby("gene", observed=True).agg(
        {
            "means": "mean",
            "dispersions": "mean",
            "dispersions_norm": "mean",
            "highly_variable": "sum",
        }
    )
    df["highly_variable_nbatches"] = df["highly_variable"]
    df["highly_variable_intersection"] = df["highly_variable_nbatches"] == len(batches)

    if isinstance(cutoff, int):
        # sort genes by how often they selected as hvg within each batch and
        # break ties with normalized dispersion across batches
        df.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
            inplace=True,
        )
        df["highly_variable"] = np.arange(df.shape[0]) < cutoff
    else:
        df["dispersions_norm"] = df["dispersions_norm"].fillna(0)  # similar to Seurat
        df["highly_variable"] = cutoff.in_bounds(df["means"], df["dispersions_norm"])

    return df
