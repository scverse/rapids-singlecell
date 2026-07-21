from __future__ import annotations

import sys
from functools import partial
from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd

from ._core import _RankGenes

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from numpy.typing import NDArray

type _CorrMethod = Literal["benjamini-hochberg", "bonferroni"]
type _Method = Literal[
    "logreg", "t-test", "t-test_overestim_var", "wilcoxon", "wilcoxon_binned"
]


def _matrix_to_records(
    values: np.ndarray, group_names: Iterable[object], dtype: str | np.dtype
) -> np.ndarray:
    field_dtype = np.dtype(dtype)
    record_dtype = np.dtype(
        [(str(group_name), field_dtype) for group_name in group_names]
    )
    if values.shape[1] == 0:
        return np.empty(0, dtype=record_dtype)
    record_matrix = np.ascontiguousarray(values.T, dtype=field_dtype)
    # Reinterpret rows as records; the returned view retains its backing matrix.
    return np.ndarray(values.shape[1], dtype=record_dtype, buffer=record_matrix)


def _array_result_to_records(
    arrays: dict[str, object], field: str, dtype: str | np.dtype
) -> np.ndarray:
    return _matrix_to_records(np.asarray(arrays[field]), arrays["group_names"], dtype)


def _array_result_to_names(arrays: dict[str, object]) -> np.ndarray:
    var_names = np.asarray(arrays["var_names"], dtype=object)
    gene_indices = np.asarray(arrays["gene_indices"], dtype=np.intp)
    values = np.take(var_names, gene_indices)
    return _matrix_to_records(values, arrays["group_names"], np.dtype(object))


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
    return_u_values: bool = False,
    layer: str | None = None,
    chunk_size: int | None = None,
    multi_gpu: bool | list[int] | str | None = None,
    n_bins: int | None = None,
    bin_range: Literal["log1p", "auto"] | None = None,
    skip_empty_groups: bool = False,
    **kwds,
) -> None:
    """
    Rank genes for characterizing groups using GPU acceleration.

    Log1p/log-normalized data is expected for biologically meaningful log fold
    changes. Exact sparse ``wilcoxon`` versus rest ranks signed stored values
    and implicit zeros directly. Sparse ``wilcoxon`` with an explicit reference
    uses sign-safe dense ranking in bounded CUDA streamer tiles. Dense inputs
    are ranked directly and support any sign.
    (``wilcoxon_binned`` rejects negative Dask sparse input, which it cannot
    bin correctly.)

    .. note::
        **Dask support:** `'t-test'`, `'t-test_overestim_var'`,
        `'wilcoxon_binned'`, and `'logreg'` support Dask arrays. The
        `'wilcoxon'` method does not support Dask arrays.

    .. note::
        **Wilcoxon ranking precision:** `'wilcoxon'` and `'wilcoxon_binned'`
        rank values in float32 on every code path, while means and log fold
        changes are computed in float64. This only diverges from Scanpy when the
        **preprocessing itself ran in float64** — i.e. normalization/log1p
        produced values carrying sub-float32 precision. If preprocessing was
        done in float32 (the common case), the values are float32-exact and
        ranking is bit-identical to Scanpy (~1e-13), even if they are afterward
        stored as float64. For a fully float64 pipeline the rank-derived scores
        and p-values still match Scanpy-on-float64 to ~1e-4 on log-normalized
        data — below any significance threshold and changing no DE calls —
        because the rank-sum normal approximation is insensitive to sub-float32
        tie jitter. If exact float64 ranking matters for your workflow, please
        open an issue at https://github.com/scverse/rapids_singlecell/issues.

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
    return_u_values
        For `'wilcoxon'`, store Mann-Whitney U statistics in `scores` instead
        of z-scores. P-values are still computed from the z-score normal
        approximation using the selected tie and continuity settings.
    layer
        Key from `adata.layers` whose value will be used to perform tests on.
    chunk_size
        Number of genes to process at once for `'wilcoxon'` and
        `'wilcoxon_binned'`. Default is 512 for `'wilcoxon'`. For
        `'wilcoxon_binned'` the default is sized dynamically based on
        ``n_groups`` and ``n_bins`` to keep histogram memory stable.
    multi_gpu
        GPU selection for exact `'wilcoxon'`. ``None`` uses all visible GPUs
        for host input and device OVO, while device OVR stays on its input-owning
        GPU. ``False`` uses one GPU, ``True`` uses all visible GPUs, and a list
        or comma-separated string selects device IDs. Multi-GPU supports
        host/device dense, CSR, and CSC input. Device input must fit on its
        owning GPU; forced multi-GPU may be slower due to transfers.
    n_bins
        Number of histogram bins for `'wilcoxon_binned'`. Higher values give
        a better approximation at slightly increased cost. Default is 1000
        for in-memory arrays and 200 for Dask arrays.
    bin_range
        How to determine the histogram bin range for `'wilcoxon_binned'`.
        ``None`` (default) uses ``'auto'`` for in-memory arrays and
        ``'log1p'`` for Dask arrays (to avoid a costly data scan).
        ``'log1p'`` uses a fixed [0, 15] range suitable for most log1p-normalized data.
        ``'auto'`` computes the actual data range. Use this for nonnegative
        expression data outside the fixed log1p range.
    skip_empty_groups
        Skip selected groups with fewer than two observations after filtering.
        This is useful for perturbation workflows where a per-cell-type slice
        keeps categories that are empty or singleton in that slice.
    **kwds
        Additional arguments passed to the method. For `'logreg'`, these are
        passed to :class:`cuml.linear_model.LogisticRegression`.

    Returns
    -------
    Updates `adata` with the following fields. Rank result fields are
    Scanpy-compatible structured arrays.

    `adata.uns['rank_genes_groups' | key_added]['names']`
        Structured array to be indexed by group id storing the gene
        names. Ordered according to scores.
    `adata.uns['rank_genes_groups' | key_added]['scores']`
        Structured array to be indexed by group id storing the z-score
        underlying the computation of a p-value for each gene for each
        group, or the Mann-Whitney U statistic when
        `return_u_values=True`. Ordered according to scores.
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

    if "return_format" in kwds:
        msg = (
            "return_format has been removed; rank_genes_groups always writes "
            "Scanpy-compatible structured results to adata.uns."
        )
        raise TypeError(msg)

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

    if return_u_values and method != "wilcoxon":
        msg = "return_u_values is only supported for method='wilcoxon'."
        raise ValueError(msg)

    if multi_gpu is not None and multi_gpu is not False and method != "wilcoxon":
        msg = "multi_gpu is only supported for method='wilcoxon'."
        raise ValueError(msg)

    if chunk_size is not None and chunk_size <= 0:
        msg = "chunk_size must be a positive integer."
        raise ValueError(msg)

    if key_added is None:
        key_added = "rank_genes_groups"

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
        skip_empty_groups=skip_empty_groups,
    )

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
        return_u_values=return_u_values,
        chunk_size=chunk_size,
        multi_gpu=multi_gpu,
        n_bins=n_bins,
        bin_range=bin_range,
        **kwds,
    )

    params = {
        "groupby": groupby,
        "reference": reference,
        "method": method,
        "use_raw": use_raw,
        "layer": layer,
        "corr_method": corr_method,
    }
    if method == "wilcoxon":
        params["tie_correct"] = tie_correct
        params["return_u_values"] = return_u_values

    arrays = test_obj.stats_arrays or {}
    adata.uns[key_added] = {"params": params}
    if arrays and len(arrays.get("group_names", ())) > 0:
        adata.uns[key_added]["names"] = _array_result_to_names(arrays)
        for col in ("scores", "logfoldchanges", "pvals", "pvals_adj"):
            if col in arrays:
                values = arrays[col]
                dtype = values.dtype
                adata.uns[key_added][col] = _array_result_to_records(arrays, col, dtype)

    groups_names = [str(name) for name in test_obj.groups_order]
    if test_obj.pts is not None:
        adata.uns[key_added]["pts"] = pd.DataFrame(
            test_obj.pts.T, index=test_obj.var_names, columns=groups_names
        )
    if test_obj.pts_rest is not None:
        adata.uns[key_added]["pts_rest"] = pd.DataFrame(
            test_obj.pts_rest.T, index=test_obj.var_names, columns=groups_names
        )

    return None


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
    return rank_genes_groups(
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
