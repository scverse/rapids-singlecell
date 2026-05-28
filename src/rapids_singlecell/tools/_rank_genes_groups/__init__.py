from __future__ import annotations

import sys
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from rapids_singlecell import logging as logg

from ._core import _RankGenes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from numpy.typing import NDArray

type _CorrMethod = Literal["benjamini-hochberg", "bonferroni"]
type _Method = Literal[
    "logreg", "t-test", "t-test_overestim_var", "wilcoxon", "wilcoxon_binned"
]


def rank_genes_groups(
    adata: AnnData,
    groupby: str,
    *,
    mask_var: NDArray[np.bool_] | str | None = None,
    use_raw: bool | None = None,
    groups: Literal["all"] | Iterable[str] = "all",
    reference: str = "rest",
    n_genes: int | None = None,
    rankby_abs: bool = False,
    pts: bool = False,
    key_added: str | None = None,
    method: _Method | None = None,
    corr_method: _CorrMethod = "benjamini-hochberg",
    tie_correct: bool = False,
    use_continuity: bool = False,
    layer: str | None = None,
    chunk_size: int | None = None,
    pre_load: bool = False,
    n_bins: int | None = None,
    bin_range: Literal["log1p", "auto"] | None = None,
    **kwds,
) -> None:
    """
    Rank genes for characterizing groups using GPU acceleration.

    Expects logarithmized data.

    .. note::
        **Dask support:** `'t-test'`, `'t-test_overestim_var'`, and
        `'wilcoxon_binned'` support Dask arrays. The `'wilcoxon'` and
        `'logreg'` methods do not support Dask arrays.

    Parameters
    ----------
    adata
        Annotated data matrix.
    groupby
        The key of the observations grouping to consider.
    mask_var
        Select subset of genes to use in statistical tests.
        Can be a boolean array of shape `(n_vars,)` or a key in `adata.var`.
    use_raw
        Use `raw` attribute of `adata` if present.
    groups
        Subset of groups, e.g. [`'g1'`, `'g2'`, `'g3'`], to which comparison
        shall be restricted, or `'all'` (default), for all groups.
    reference
        If `'rest'`, compare each group to the union of the rest of the group.
        If a group identifier, compare with respect to this group.
    n_genes
        The number of genes that appear in the returned tables.
        Defaults to all genes.
    rankby_abs
        Rank genes by the absolute value of the score, not by the
        score. The returned scores are never the absolute values.
    pts
        Compute the fraction of cells expressing the genes.
    key_added
        The key in `adata.uns` information is saved to.
    method
        `'t-test'` uses Welch's t-test (default),
        `'t-test_overestim_var'` overestimates variance of each group,
        `'wilcoxon'` uses Wilcoxon rank-sum,
        `'wilcoxon_binned'` uses histogram-based approximate Wilcoxon rank-sum
        (faster for large datasets, supports Dask arrays),
        `'logreg'` uses logistic regression.
    corr_method
        p-value correction method. Used only for `'t-test'`,
        `'t-test_overestim_var'`, `'wilcoxon'`, and `'wilcoxon_binned'`.
    tie_correct
        Use tie correction for `'wilcoxon'` and `'wilcoxon_binned'` scores.
        Adjusts the variance of the rank-sum statistic for tied values.
        For `'wilcoxon_binned'`, each histogram bin acts as a tie group
        and the correction is derived from the bin counts.
    use_continuity
        Apply continuity correction to `'wilcoxon'` and `'wilcoxon_binned'`
        z-scores. Subtracts 0.5 from ``|R - E[R]|`` before dividing by the
        standard deviation, matching :func:`scipy.stats.mannwhitneyu`
        default behavior.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    chunk_size
        Number of genes to process at once for `'wilcoxon'` and
        `'wilcoxon_binned'`. Default is 128 for `'wilcoxon'`. For
        `'wilcoxon_binned'` the default is sized dynamically based on
        ``n_groups`` and ``n_bins`` to keep histogram memory stable.
    pre_load
        Pre-load the data into GPU memory. Used only for `'wilcoxon'`.
    n_bins
        Number of histogram bins for `'wilcoxon_binned'`. Higher values give
        a better approximation at slightly increased cost. Default is 1000
        for in-memory arrays and 200 for Dask arrays.
    bin_range
        How to determine the histogram bin range for `'wilcoxon_binned'`.
        ``None`` (default) uses ``'auto'`` for in-memory arrays and
        ``'log1p'`` for Dask arrays (to avoid a costly data scan).
        ``'log1p'`` uses a fixed [0, 15] range suitable for most log1p-normalized data.
        ``'auto'`` computes the actual data range. Use this for z-scored
        or unnormalized data.
    **kwds
        Additional arguments passed to the method. For `'logreg'`, these are
        passed to :class:`cuml.linear_model.LogisticRegression`.

    Returns
    -------
    Updates `adata` with the following fields:

    `adata.uns['rank_genes_groups' | key_added]['names']`
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['scores']`
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['logfoldchanges']`
        Structured array to be indexed by group id storing the log2
        fold change for each gene for each group.
    `adata.uns['rank_genes_groups' | key_added]['pvals']`
        p-values. Only for `'t-test'`, `'t-test_overestim_var'`,
        `'wilcoxon'`, and `'wilcoxon_binned'`.
    `adata.uns['rank_genes_groups' | key_added]['pvals_adj']`
        Corrected p-values. Only for `'t-test'`, `'t-test_overestim_var'`,
        `'wilcoxon'`, and `'wilcoxon_binned'`.
    `adata.uns['rank_genes_groups' | key_added]['pts']`
        Fraction of cells expressing genes per group. Only if `pts=True`.
    `adata.uns['rank_genes_groups' | key_added]['pts_rest']`
        Fraction of cells expressing genes in rest. Only if `pts=True` and `reference='rest'`.
    """
    if corr_method not in {"benjamini-hochberg", "bonferroni"}:
        msg = "corr_method must be either 'benjamini-hochberg' or 'bonferroni'."
        raise ValueError(msg)

    start = logg.info("ranking genes")
    if method is None:
        method = "t-test"

    if method not in {
        "logreg",
        "t-test",
        "t-test_overestim_var",
        "wilcoxon",
        "wilcoxon_binned",
    }:
        msg = (
            "method must be one of 'logreg', 't-test', 't-test_overestim_var', "
            f"'wilcoxon', 'wilcoxon_binned'. Got {method!r}."
        )
        raise ValueError(msg)

    if key_added is None:
        key_added = "rank_genes_groups"

    # Process mask_var: convert string to boolean array
    mask_var_array: NDArray[np.bool_] | None = None
    if mask_var is not None:
        if isinstance(mask_var, str):
            if mask_var not in adata.var.columns:
                msg = f"mask_var key {mask_var!r} not found in adata.var."
                raise KeyError(msg)
            mask_var_array = adata.var[mask_var].values.astype(bool)
        else:
            mask_var_array = np.asarray(mask_var, dtype=bool)
            if mask_var_array.shape[0] != adata.n_vars:
                msg = f"mask_var has wrong shape: {mask_var_array.shape[0]} != {adata.n_vars}"
                raise ValueError(msg)

    test_obj = _RankGenes(
        adata,
        groups,
        groupby,
        mask_var=mask_var_array,
        reference=reference,
        use_raw=use_raw,
        layer=layer,
        comp_pts=pts,
        pre_load=pre_load,
    )

    logg.debug(f"consider {groupby!r} groups:")
    logg.debug(f"with sizes: {list(map(int, test_obj.group_sizes))}")

    # Determine n_genes_user
    n_genes_user = n_genes
    if n_genes_user is None or n_genes_user > test_obj.X.shape[1]:
        n_genes_user = test_obj.X.shape[1]

    test_obj.compute_statistics(
        method,
        corr_method=corr_method,
        n_genes_user=n_genes_user,
        rankby_abs=rankby_abs,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
        chunk_size=chunk_size,
        n_bins=n_bins,
        bin_range=bin_range,
        **kwds,
    )

    # Build output
    test_obj.stats.columns = test_obj.stats.columns.swaplevel()

    dtypes = {
        "names": "U50",
        "scores": "float32",
        "logfoldchanges": "float32",
        "pvals": "float64",
        "pvals_adj": "float64",
    }

    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "groupby": groupby,
        "reference": reference,
        "method": method,
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method,
    }

    # Store pts results if computed
    if test_obj.pts is not None:
        groups_names = [str(name) for name in test_obj.groups_order]
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    if method == "wilcoxon":
        adata.uns[key_added]["params"]["tie_correct"] = tie_correct

    for col in test_obj.stats.columns.levels[0]:
        if col in dtypes:
            adata.uns[key_added][col] = test_obj.stats[col].to_records(
                index=False, column_dtypes=dtypes[col]
            )

    logg.info(
        "    finished",
        time=start,
        deep=(
            f"added to `.uns[{key_added!r}]`\n"
            "    'names', sorted np.recarray to be indexed by group ids\n"
            "    'scores', sorted np.recarray to be indexed by group ids\n"
            + (
                "    'logfoldchanges', sorted np.recarray to be indexed by group ids\n"
                "    'pvals', sorted np.recarray to be indexed by group ids\n"
                "    'pvals_adj', sorted np.recarray to be indexed by group ids"
                if method != "logreg"
                else ""
            )
        ),
    )


if TYPE_CHECKING:
    from warnings import deprecated
else:
    if sys.version_info >= (3, 13):
        from warnings import deprecated as _deprecated
    else:
        from typing_extensions import deprecated as _deprecated
    deprecated = partial(_deprecated, category=FutureWarning)


@deprecated(
    "rank_genes_groups_logreg is deprecated. "
    "Use rank_genes_groups(method='logreg') instead."
)
def rank_genes_groups_logreg(
    adata: AnnData,
    groupby: str,
    *,
    groups: Literal["all"] | Iterable[str] = "all",
    use_raw: bool | None = None,
    reference: str = "rest",
    n_genes: int | None = None,
    key_added: str | None = None,
    layer: str | None = None,
    **kwds,
) -> None:
    rank_genes_groups(
        adata,
        groupby,
        groups=groups,
        use_raw=use_raw,
        reference=reference,
        n_genes=n_genes,
        key_added=key_added,
        method="logreg",
        layer=layer,
        **kwds,
    )
