from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from anndata import AnnData
from scverse_misc import deprecated_arg

from rapids_singlecell._settings import Default, resolve_default
from rapids_singlecell.get import _get_arr, _get_vec
from rapids_singlecell.get._out import DEPR_BATCH_KEY, DEPR_INPLACE, DEPR_USE, _legacy_out, _resolve, _use_args
from rapids_singlecell.get._utils import LayerAcc
from rapids_singlecell.preprocessing._utils import _sanitize_column

from ._cutoffs import _Cutoffs
from ._pearson_residuals import _highly_variable_pearson_residuals
from ._poisson import _poisson_gene_selection
from ._seurat_cellranger import (
    _highly_variable_genes_batched,
    _highly_variable_genes_single_batch,
)
from ._seurat_v3 import _highly_variable_genes_seurat_v3

if TYPE_CHECKING:
    from rapids_singlecell.get._utils import AdRef

flavors = Literal[
    "seurat",
    "cell_ranger",
    "seurat_v3",
    "seurat_v3_paper",
    "pearson_residuals",
    "poisson_gene_selection",
]


@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("inplace", DEPR_INPLACE)
@deprecated_arg("batch_key", DEPR_BATCH_KEY)
def highly_variable_genes(
    adata: AnnData,
    *,
    use: LayerAcc | str | None = None,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    n_top_genes: int = None,
    flavor: flavors | Default = Default(("highly_variable_genes", "flavor")),
    n_bins: int = 20,
    span: float = 0.3,
    check_values: bool = True,
    theta: int = 100,
    clip: bool = None,
    chunksize: int = 1000,
    n_samples: int = 10000,
    batch: AdRef | str | None = None,
    out: bool = False,
    # deprecated
    layer: str = None,
    inplace: bool = True,
    batch_key: str | None = None,
) -> pd.DataFrame | None:
    """Annotate highly variable genes :cite:p:`Satija2015,Zheng2017,Stuart2019,Lause2021,Andrews2019`.

    Expects logarithmized data, except when `flavor='seurat_v3','seurat_v3_paper','pearson_residuals','poisson_gene_selection'`, in which count data is expected.

    Reimplementation of scanpy's function.
    Depending on flavor, this reproduces the R-implementations of Seurat, Cell Ranger, Seurat v3 and Pearson Residuals.
    Flavor `poisson_gene_selection` calculates analytical Poisson gene selection based on M3Drop using CuPy with CUDA kernels.

    For these dispersion-based methods, the normalized dispersion is obtained by scaling
    with the mean and standard deviation of the dispersions for genes falling into a given
    bin for mean expression of genes. This means that for each bin of mean expression,
    highly variable genes are selected.

    For `flavor='seurat_v3'`/`'seurat_v3_paper'`, a normalized variance for each gene
    is computed. First, the data are standardized (i.e., z-score normalization
    per feature) with a regularized standard deviation. Next, the normalized variance
    is computed as the variance of each gene after the transformation. Genes are ranked
    by the normalized variance.
    Only if `batch` is not `None`, the two flavors differ: For `flavor='seurat_v3'`, genes are first sorted by the median (across batches) rank, with ties broken by the number of batches a gene is a HVG.
    For `flavor='seurat_v3_paper'`, genes are first sorted by the number of batches a gene is a HVG, with ties broken by the median (across batches) rank.

    The following may help when comparing to Seurat's naming:
    If `batch=None` and `flavor='seurat'`, this mimics Seurat's `FindVariableFeatures(…, method='mean.var.plot')`.
    If `batch=None` and `flavor='seurat_v3'`/`flavor='seurat_v3_paper'`, this mimics Seurat's `FindVariableFeatures(..., method='vst')`.
    If `batch` is not `None` and `flavor='seurat_v3_paper'`, this mimics Seurat's `SelectIntegrationFeatures`.

    Parameters
    ----------
        adata
            AnnData object
        use
            Which matrix to use for expression values, e.g. `A.X` or `A.layers['counts']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.
        min_mean
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        max_mean
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        min_disp
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        max_disp
            If n_top_genes unequals None, this and all other cutoffs for the means and the normalized dispersions are ignored.
        n_top_genes
            Number of highly-variable genes to keep.
        flavor :
            Choose the flavors for identifying highly variable genes. For the dispersion based methods
            in their default workflows, Seurat passes the cutoffs whereas Cell Ranger passes n_top_genes.
        n_bins
            Number of bins for binning the mean gene expression. Normalization is done with respect to each bin. If just a single gene falls into a bin, the normalized dispersion is artificially set to 1.
        span
            The fraction of the data (cells) used when estimating the variance in the loess
            model fit if `flavor='seurat_v3'`.
        check_values
            Check if counts in selected layer are integers. A Warning is returned if set to True.
            Only used if `flavor='seurat_v3'` or `'pearson_residuals'`.
        theta
            The negative binomial overdispersion parameter `theta` for Pearson residuals.
            Higher values correspond to less overdispersion (`var = mean + mean^2/theta`), and `theta=np.inf` corresponds to a Poisson model.
        clip
            Only used if `flavor='pearson_residuals'`. Determines if and how residuals are clipped:
                * If `None`, residuals are clipped to the interval `[-sqrt(n_obs), sqrt(n_obs)]`, where `n_obs` is the number of cells in the dataset (default behavior).
                * If any scalar `c`, residuals are clipped to the interval `[-c, c]`. Set `clip=np.inf` for no clipping.
        chunksize
            If `'poisson_gene_selection'`, this dertermines how many genes are processed at
            once. Choosing a smaller value will reduce the required memory.
        n_samples
            The number of Binomial samples to use to estimate posterior probability
            of enrichment of zeros for each gene (only for `flavor='poisson_gene_selection'`).
        batch
            If specified (e.g. `A.obs['batch']` or `'obs.batch'`),
            highly-variable genes are selected within each batch separately and merged.
        out
            If `True`, return the metrics as a :class:`~pandas.DataFrame` instead of writing them.

    Returns
    -------
        A :class:`~pandas.DataFrame` if `out=True`, else updates `adata.var` with the following fields:

            `highly_variable` : bool
                boolean indicator of highly-variable genes
            `means`: float
                means per gene
            `dispersions`: float
                For dispersion-based flavors, dispersions per gene
            `dispersions_norm`: float
                For dispersion-based flavors, normalized dispersions per gene
            `variances`: float
                For `flavor='seurat_v3','pearson_residuals'`, variance per gene
            `variances_norm`: float
                For `flavor='seurat_v3'`, normalized variance per gene, averaged in
                the case of multiple batches
            `residual_variances` : float
                For `flavor='pearson_residuals'`, residual variance per gene. Averaged in the
                case of multiple batches.
            `highly_variable_rank` : float
                For `flavor='seurat_v3','pearson_residuals'`, rank of the gene according to normalized
                variance, median rank in the case of multiple batches
            `highly_variable_nbatches` : int
                If batch_key is given, this denotes in how many batches genes are detected as HVG
            `highly_variable_intersection` : bool
                If batch_key is given, this denotes the genes that are highly variable in all batches
    """
    flavor = resolve_default(flavor)
    out = _legacy_out(out, "inplace", None if inplace else True)
    src = _use_args(use, kinds=(LayerAcc,), layer=layer)
    if batch_key is not None and batch is not None:
        raise TypeError("Pass either `batch` or `batch_key`, not both.")
    # run on a proxy holding only the input matrix and batches, then copy the results
    target = adata
    # share string obs_names: generating new ones takes 0.2 s per million cells
    obs = pd.DataFrame(index=target.obs_names) if pd.api.types.is_string_dtype(target.obs_names) else None
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Observation names are not unique")
        adata = AnnData(_get_arr(target, dim="obs", **src), obs=obs)
    if batch_key is not None or batch is not None:
        vals = target.obs[batch_key] if batch is None else _get_vec(target, _resolve(batch, vec=True), dim="obs")
        adata.obs["batch"] = vals.values if isinstance(vals, pd.Series) else vals
        batch_key = "batch"
        _sanitize_column(adata, batch_key)
        adata.obs[batch_key] = adata.obs[batch_key].astype("category")
    layer = None

    if flavor == "seurat_v3" or flavor == "seurat_v3_paper":
        _highly_variable_genes_seurat_v3(
            adata=adata,
            layer=layer,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            span=span,
            check_values=check_values,
            flavor=flavor,
        )
    elif flavor == "pearson_residuals":
        _highly_variable_pearson_residuals(
            adata=adata,
            theta=theta,
            clip=clip,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            layer=layer,
        )
    elif flavor == "poisson_gene_selection":
        _poisson_gene_selection(
            adata=adata,
            n_top_genes=n_top_genes,
            batch_key=batch_key,
            check_values=check_values,
            layer=layer,
        )
    else:
        cutoff = _Cutoffs.validate(
            n_top_genes=n_top_genes,
            min_disp=min_disp,
            max_disp=max_disp,
            min_mean=min_mean,
            max_mean=max_mean,
        )
        del min_disp, max_disp, min_mean, max_mean, n_top_genes

        if batch_key is None:
            df = _highly_variable_genes_single_batch(
                adata,
                layer=layer,
                cutoff=cutoff,
                n_bins=n_bins,
                flavor=flavor,
            )
        else:
            df = _highly_variable_genes_batched(
                adata,
                batch_key,
                layer=layer,
                cutoff=cutoff,
                n_bins=n_bins,
                flavor=flavor,
            )

        adata.uns["hvg"] = {"flavor": flavor}

        adata.var["highly_variable"] = df["highly_variable"]
        adata.var["means"] = df["means"]
        adata.var["dispersions"] = df["dispersions"]
        adata.var["dispersions_norm"] = df["dispersions_norm"].astype(np.float32, copy=False)

        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"]
            adata.var["highly_variable_intersection"] = df["highly_variable_intersection"]

    res = adata.var.set_axis(target.var_names)
    if out:
        return res
    target.var[res.columns] = res
    target.uns.update(adata.uns)
    if "computed_on" in target.uns.get("hvg", {}):
        target.uns["hvg"]["computed_on"] = src.get("layer", "adata.X") if use is None else repr(use)
    return None
