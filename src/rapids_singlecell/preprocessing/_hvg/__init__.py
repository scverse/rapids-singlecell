from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

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
    from anndata import AnnData

flavors = Literal[
    "seurat",
    "cell_ranger",
    "seurat_v3",
    "seurat_v3_paper",
    "pearson_residuals",
    "poisson_gene_selection",
]


def highly_variable_genes(
    adata: AnnData,
    *,
    layer: str = None,
    min_mean: float = 0.0125,
    max_mean: float = 3,
    min_disp: float = 0.5,
    max_disp: float = np.inf,
    n_top_genes: int = None,
    flavor: flavors = "seurat",
    n_bins: int = 20,
    span: float = 0.3,
    check_values: bool = True,
    theta: int = 100,
    clip: bool = None,
    chunksize: int = 1000,
    n_samples: int = 10000,
    batch_key: str | None = None,
) -> None:
    """\
    Annotate highly variable genes :cite:p:`Satija2015,Zheng2017,Stuart2019,Lause2021,Andrews2019`.

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
    Only if `batch_key` is not `None`, the two flavors differ: For `flavor='seurat_v3'`, genes are first sorted by the median (across batches) rank, with ties broken by the number of batches a gene is a HVG.
    For `flavor='seurat_v3_paper'`, genes are first sorted by the number of batches a gene is a HVG, with ties broken by the median (across batches) rank.

    The following may help when comparing to Seurat's naming:
    If `batch_key=None` and `flavor='seurat'`, this mimics Seurat's `FindVariableFeatures(…, method='mean.var.plot')`.
    If `batch_key=None` and `flavor='seurat_v3'`/`flavor='seurat_v3_paper'`, this mimics Seurat's `FindVariableFeatures(..., method='vst')`.
    If `batch_key` is not `None` and `flavor='seurat_v3_paper'`, this mimics Seurat's `SelectIntegrationFeatures`.

    Parameters
    ----------
        adata
            AnnData object
        layer
            If provided, use `adata.layers[layer]` for expression values instead of `adata.X`.
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
            Higher values correspond to less overdispersion (`var = mean + mean^2/theta`), and `theta=np.Inf` corresponds to a Poisson model.
        clip
            Only used if `flavor='pearson_residuals'`. Determines if and how residuals are clipped:
                * If `None`, residuals are clipped to the interval `[-sqrt(n_obs), sqrt(n_obs)]`, where `n_obs` is the number of cells in the dataset (default behavior).
                * If any scalar `c`, residuals are clipped to the interval `[-c, c]`. Set `clip=np.Inf` for no clipping.
        chunksize
            If `'poisson_gene_selection'`, this dertermines how many genes are processed at
            once. Choosing a smaller value will reduce the required memory.
        n_samples
            The number of Binomial samples to use to estimate posterior probability
            of enrichment of zeros for each gene (only for `flavor='poisson_gene_selection'`).
        batch_key
            If specified, highly-variable genes are selected within each batch separately and merged.

    Returns
    -------
        updates `adata.var` with the following fields:

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
    if batch_key is not None:
        _sanitize_column(adata, batch_key)

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
        adata.var["dispersions_norm"] = df["dispersions_norm"].astype(
            np.float32, copy=False
        )

        if batch_key is not None:
            adata.var["highly_variable_nbatches"] = df["highly_variable_nbatches"]
            adata.var["highly_variable_intersection"] = df[
                "highly_variable_intersection"
            ]
