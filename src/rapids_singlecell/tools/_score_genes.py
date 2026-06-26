from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import dask
import numpy as np
import pandas as pd

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import X_to_GPU, _get_obs_rep
from rapids_singlecell.preprocessing._utils import _check_gpu_X, _check_use_raw

from ._utils import _nan_mean

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from anndata import AnnData


def score_genes(
    adata: AnnData,
    gene_list: Sequence[str] | pd.Index,
    *,
    ctrl_as_ref: bool = True,
    ctrl_size: int = 50,
    gene_pool: Sequence[str] | pd.Index | None = None,
    n_bins: int = 25,
    score_name: str = "score",
    random_state: int | None = 0,
    copy: bool = False,
    use_raw: bool | None = None,
    layer: str | None = None,
) -> AnnData | None:
    """\
    Score a set of genes :cite:p:`Satija2015,Tirosh2016`.

    The score is the average expression of a set of genes subtracted with the
    average expression of a reference set of genes. The reference set is
    randomly sampled from the `gene_pool` for each binned expression value.

    Parameters
    ----------
    adata
        The annotated data matrix.
    gene_list
        The list of gene names used for score calculation.
    ctrl_as_ref
        Allow the algorithm to use the control genes as reference.
        Will be changed to `False` in scanpy 2.0.
    ctrl_size
        Number of reference genes to be sampled from each bin. If `len(gene_list)` is not too
        low, you can set `ctrl_size=len(gene_list)`.
    gene_pool
        Genes for sampling the reference set. Default is all genes.
    n_bins
        Number of expression level bins for sampling.
    score_name
        Name of the field to be added in `.obs`.
    random_state
        The random seed for sampling.
    copy
        Copy `adata` or modify it inplace.
    use_raw
        Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following field:

    `adata.obs[score_name]` : :class:`numpy.ndarray` (dtype `float`)
        Scores of each cell.
    """
    adata = adata.copy() if copy else adata
    use_raw = _check_use_raw(adata, layer, use_raw=use_raw)
    X = _get_obs_rep(adata, layer=layer, use_raw=use_raw)
    X = X_to_GPU(X)
    _check_gpu_X(X, allow_dask=True)
    if random_state is not None:
        np.random.seed(random_state)

    var_names = adata.raw.var_names if use_raw else adata.var_names
    gene_list, gene_pool = _check_score_genes_args(var_names, gene_list, gene_pool)

    control_genes = pd.Index([], dtype="string")
    for r_genes in _score_genes_bins(
        X,
        gene_list,
        gene_pool,
        var_names=var_names,
        ctrl_as_ref=ctrl_as_ref,
        ctrl_size=ctrl_size,
        n_bins=n_bins,
    ):
        control_genes = control_genes.union(r_genes)

    if len(control_genes) == 0:
        msg = "No control genes found in any cut."
        if ctrl_as_ref:
            msg += " Try setting `ctrl_as_ref=False`."
        raise RuntimeError(msg)
    gene_array = cp.array(var_names.isin(gene_list), dtype=cp.bool_)
    control_array = cp.array(var_names.isin(control_genes), dtype=cp.bool_)
    means_list = _nan_mean(X, axis=1, mask=gene_array, n_features=len(gene_list))
    means_control = _nan_mean(
        X, axis=1, mask=control_array, n_features=len(control_genes)
    )
    if isinstance(X, DaskArray):
        means_list, means_control = dask.compute(means_list, means_control)

    score = means_list - means_control

    adata.obs[score_name] = pd.Series(
        score.get().ravel(), index=adata.obs_names, dtype="float64"
    )

    return adata if copy else None


def _check_score_genes_args(
    var_names: pd.Index[str],
    gene_list: pd.Index[str] | Sequence[str],
    gene_pool: pd.Index[str] | Sequence[str] | None,
) -> tuple[pd.Index[str], pd.Index[str]]:
    """Restrict `gene_list` and `gene_pool` to present genes in `adata`.

    Also returns a function to get subset of `adata.X` based on a set of genes passed.
    """
    gene_list = pd.Index([gene_list] if isinstance(gene_list, str) else gene_list)
    genes_to_ignore = gene_list.difference(var_names, sort=False)  # first get missing
    gene_list = gene_list.intersection(var_names)  # then restrict to present
    if len(genes_to_ignore) > 0:
        warnings.warn(f"genes are not in var_names and ignored: {genes_to_ignore}")
    if len(gene_list) == 0:
        raise ValueError("No valid genes were passed for scoring.")

    if gene_pool is None:
        gene_pool = var_names.astype("string")
    else:
        gene_pool = pd.Index(gene_pool, dtype="string").intersection(var_names)
    if len(gene_pool) == 0:
        raise ValueError("No valid genes were passed for reference set.")

    return gene_list, gene_pool


def _score_genes_bins(
    X,
    gene_list: pd.Index[str],
    gene_pool: pd.Index[str],
    *,
    var_names: pd.Index[str],
    ctrl_as_ref: bool,
    ctrl_size: int,
    n_bins: int,
) -> Generator[pd.Index[str], None, None]:
    # average expression of genes
    idx = cp.array(var_names.isin(gene_pool), dtype=cp.bool_)
    nanmeans = _nan_mean(X, axis=0, mask=idx, n_features=len(gene_pool))
    if isinstance(X, DaskArray):
        nanmeans = nanmeans.compute()
    nanmeans = nanmeans.get()
    obs_avg = pd.Series(nanmeans, index=gene_pool)
    # Sometimes (and I don’t know how) missing data may be there, with NaNs for missing entries
    obs_avg = obs_avg[np.isfinite(obs_avg)]

    n_items = int(np.round(len(obs_avg) / (n_bins - 1)))
    obs_cut = obs_avg.rank(method="min") // n_items
    keep_ctrl_in_obs_cut = False if ctrl_as_ref else obs_cut.index.isin(gene_list)

    # now pick `ctrl_size` genes from every cut
    for cut in np.unique(obs_cut.loc[gene_list]):
        r_genes: pd.Index[str] = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        if len(r_genes) == 0:
            msg = (
                f"No control genes for {cut=}. You might want to increase "
                f"gene_pool size (current size: {len(gene_pool)})"
            )
            warnings.warn(msg)
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size).index
        if ctrl_as_ref:  # otherwise `r_genes` is already filtered
            r_genes = r_genes.difference(gene_list)
        yield r_genes


def score_genes_cell_cycle(
    adata: AnnData,
    *,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    copy: bool = False,
    **kwargs,
) -> AnnData | None:
    """\
    Score cell cycle genes :cite:p:`Scialdone2015`.

    Given two lists of genes associated to S phase and G2M phase, calculates
    scores and assigns a cell cycle phase (G1, S or G2M). See
    :func:`~rapids_singlecell.tl.score_genes` for more explanation.

    Parameters
    ----------
    adata
        The annotated data matrix.
    s_genes
        List of genes associated with S phase.
    g2m_genes
        List of genes associated with G2M phase.
    copy
        Copy `adata` or modify it inplace.
    **kwargs
        Are passed to :func:`~rapids_singlecell.tl.score_genes`. `ctrl_size` is not
        possible, as it's set as `min(len(s_genes), len(g2m_genes))`.

    Returns
    -------
    Returns `None` if `copy=False`, else returns an `AnnData` object. Sets the following fields:

    `adata.obs['S_score']` : :class:`pandas.Series` (dtype `object`)
        The score for S phase for each cell.
    `adata.obs['G2M_score']` : :class:`pandas.Series` (dtype `object`)
        The score for G2M phase for each cell.
    `adata.obs['phase']` : :class:`pandas.Series` (dtype `object`)
        The cell cycle phase (`S`, `G2M` or `G1`) for each cell.

    See also
    --------
    score_genes
    """

    adata = adata.copy() if copy else adata
    ctrl_size = min(len(s_genes), len(g2m_genes))
    for genes, name in [(s_genes, "S_score"), (g2m_genes, "G2M_score")]:
        score_genes(adata, genes, score_name=name, ctrl_size=ctrl_size, **kwargs)
    scores = adata.obs[["S_score", "G2M_score"]]

    # default phase is S
    phase = pd.Series("S", index=scores.index)

    # if G2M is higher than S, it's G2M
    phase[scores["G2M_score"] > scores["S_score"]] = "G2M"

    # if all scores are negative, it's G1...
    phase[np.all(scores < 0, axis=1)] = "G1"

    adata.obs["phase"] = phase
    return adata if copy else None
