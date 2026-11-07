from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, NamedTuple, TypedDict

import cupy as cp
import dask
import numpy as np
import pandas as pd
from scverse_misc import deprecated_arg

from rapids_singlecell._compat import DaskArray
from rapids_singlecell._settings import Default, resolve_default
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _if_legacy_apply_global,
)
from rapids_singlecell.get import X_to_GPU, _get_arr
from rapids_singlecell.get._out import (
    DEPR_KEY_ADDED,
    DEPR_RAW,
    DEPR_USE,
    _check_legacy,
    _out_dests,
    _use_args,
    _write_dests,
    _write_out,
    deprecated_copy,
)
from rapids_singlecell.get._utils import LayerAcc
from rapids_singlecell.preprocessing._utils import _check_gpu_X, _check_use_raw

from ._utils import _nan_mean

if TYPE_CHECKING:
    from collections.abc import Generator, Sequence

    from anndata import AnnData

    from rapids_singlecell.get._utils import AdRef


@_accepts_legacy_random_state(0)
@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("score_name", DEPR_KEY_ADDED)
@deprecated_arg("use_raw", DEPR_RAW)
@deprecated_copy
def score_genes(
    adata: AnnData,
    gene_list: Sequence[str] | pd.Index,
    *,
    ctrl_as_ref: bool | Default = Default(("score_genes", "ctrl_as_ref")),
    ctrl_size: int = 50,
    gene_pool: Sequence[str] | pd.Index | None = None,
    n_bins: int = 25,
    use: LayerAcc | str | None = None,
    out: AdRef | str | bool = False,
    rng: SeedLike | RNGLike | None = None,
    # deprecated
    score_name: str | None = None,
    copy: bool = False,
    use_raw: bool | None = None,
    layer: str | None = None,
) -> AnnData | pd.Series | None:
    """Score a set of genes :cite:p:`Satija2015,Tirosh2016`.

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
    use
        Which matrix to score, e.g. `A.X` or `A.layers['counts']`.
        Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.
    out
        Where to write the scores, e.g. `A.obs['k']` (or `'obs.k'`).
        If `True`, the scores are returned instead of written.
        If `False`, they are written to `.obs['score']`.
    rng
        Random seed or :class:`~numpy.random.Generator` for sampling.
        The superseded `random_state` argument is still accepted.
    use_raw
        Whether to use `raw` attribute of `adata`. Defaults to `True` if `.raw` is present
        and neither `use` nor `layer` is given.

    Returns
    -------
    The scores as a :class:`~pandas.Series` if `out=True`, else `None`.
    By default sets `adata.obs['score']` (see `out`).
    """
    if score_name is not None:
        _check_legacy(out, "score_name")
    ctrl_as_ref = resolve_default(ctrl_as_ref)
    src = _use_args(use, kinds=(LayerAcc,), layer=layer)
    use_raw = _check_use_raw(adata, src.get("layer", use), use_raw=use_raw)
    X = _get_arr(adata, use_raw=use_raw, **src)
    X = X_to_GPU(X)
    _check_gpu_X(X, allow_dask=True)
    rng = np.random.default_rng(rng)
    rng = _if_legacy_apply_global(rng)

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
        rng=rng,
    ):
        control_genes = control_genes.union(r_genes)

    if len(control_genes) == 0:
        msg = "No control genes found in any cut."
        if ctrl_as_ref:
            msg += " Try setting `ctrl_as_ref=False`."
        raise RuntimeError(msg)
    gene_array = cp.array(var_names.isin(gene_list), dtype=cp.bool_)
    control_array = cp.array(var_names.isin(control_genes), dtype=cp.bool_)
    # count masked columns, not names: duplicated var_names select several columns
    means_list = _nan_mean(X, axis=1, mask=gene_array, n_features=int(gene_array.sum()))
    means_control = _nan_mean(X, axis=1, mask=control_array, n_features=int(control_array.sum()))
    if isinstance(X, DaskArray):
        means_list, means_control = dask.compute(means_list, means_control)

    score = means_list - means_control

    scores = pd.Series(score.get().ravel(), index=adata.obs_names, dtype="float64")
    return _write_out(adata, scores, out=out, dest={"obs": score_name or "score"})


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
        warnings.warn(f"genes are not in var_names and ignored: {genes_to_ignore}", stacklevel=2)
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
    rng: np.random.Generator,
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
    keep_ctrl_in_obs_cut = np.zeros(len(obs_cut), dtype=bool) if ctrl_as_ref else obs_cut.index.isin(gene_list)

    # now pick `ctrl_size` genes from every cut
    cuts = np.unique(obs_cut.loc[gene_list])
    # spawn sub-rngs so this can maybe be parallelized without changing random
    # number generation
    for cut, sub_rng in zip(cuts, rng.spawn(len(cuts)), strict=True):
        r_genes: pd.Index[str] = obs_cut[(obs_cut == cut) & ~keep_ctrl_in_obs_cut].index
        if len(r_genes) == 0:
            msg = (
                f"No control genes for {cut=}. You might want to increase "
                f"gene_pool size (current size: {len(gene_pool)})"
            )
            warnings.warn(msg, stacklevel=2)
        if ctrl_size < len(r_genes):
            r_genes = r_genes.to_series().sample(ctrl_size, random_state=sub_rng).index
        if ctrl_as_ref:  # otherwise `r_genes` is already filtered
            r_genes = r_genes.difference(gene_list)
        yield r_genes


class CellCycleResult(NamedTuple):
    """Outputs of :func:`~rapids_singlecell.tl.score_genes_cell_cycle`."""

    s_score: pd.Series
    """The score for S phase for each cell."""
    g2m_score: pd.Series
    """The score for G2M phase for each cell."""
    phase: pd.Series
    """The cell cycle phase (`S`, `G2M` or `G1`) for each cell."""


class CellCycleOut(TypedDict, total=False):
    """Where a mapping `out` of :func:`~rapids_singlecell.tl.score_genes_cell_cycle` writes each output."""

    s_score: AdRef | str
    g2m_score: AdRef | str
    phase: AdRef | str


@deprecated_copy
def score_genes_cell_cycle(
    adata: AnnData,
    *,
    s_genes: Sequence[str],
    g2m_genes: Sequence[str],
    out: CellCycleOut | bool = False,
    # deprecated
    copy: bool = False,
    **kwargs,
) -> AnnData | CellCycleResult | None:
    """Score cell cycle genes :cite:p:`Scialdone2015`.

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
    out
        A mapping from output names to accessors writes only the outputs it names,
        e.g. `{'phase': A.obs['cc_phase']}` (see :class:`~rapids_singlecell.tl.CellCycleOut`).
        If `True`, the outputs are returned as a :class:`~rapids_singlecell.tl.CellCycleResult`.
        If `False`, they are written to `.obs['S_score']`, `.obs['G2M_score']` and `.obs['phase']`.
    **kwargs
        Are passed to :func:`~rapids_singlecell.tl.score_genes`. `ctrl_size` is not
        possible, as it's set as `min(len(s_genes), len(g2m_genes))`.

    Returns
    -------
    The outputs as a :class:`~rapids_singlecell.tl.CellCycleResult` if `out=True`, else `None`.
    By default sets the following fields:

    `adata.obs['S_score']` : :class:`pandas.Series` (dtype `object`)
        The score for S phase for each cell.
    `adata.obs['G2M_score']` : :class:`pandas.Series` (dtype `object`)
        The score for G2M phase for each cell.
    `adata.obs['phase']` : :class:`pandas.Series` (dtype `object`)
        The cell cycle phase (`S`, `G2M` or `G1`) for each cell.

    See Also
    --------
    score_genes
    """
    dests = _out_dests(
        out,
        None,
        dests=lambda _: CellCycleResult({"obs": "S_score"}, {"obs": "G2M_score"}, {"obs": "phase"}),
        kinds=CellCycleResult({"obs"}, {"obs"}, {"obs"}),
        primary=False,
    )
    ctrl_size = min(len(s_genes), len(g2m_genes))
    s_score, g2m_score = (
        score_genes(adata, genes, out=True, ctrl_size=ctrl_size, **kwargs) for genes in (s_genes, g2m_genes)
    )

    # default phase is S
    phase = pd.Series("S", index=adata.obs_names)

    # if G2M is higher than S, it's G2M
    phase[g2m_score > s_score] = "G2M"

    # if all scores are negative, it's G1...
    phase[(s_score < 0) & (g2m_score < 0)] = "G1"

    vals = CellCycleResult(s_score, g2m_score, phase)
    if dests is True:
        return vals
    _write_dests(adata, vals, dests[1])
    return None
