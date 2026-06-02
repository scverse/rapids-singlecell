from __future__ import annotations

from typing import TYPE_CHECKING, Literal, assert_never

import cupy as cp
import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import X_to_GPU
from rapids_singlecell.get._aggregated import Aggregate
from rapids_singlecell.preprocessing._utils import _check_gpu_X

from ._utils import EPS, _select_groups, _select_top_n

if TYPE_CHECKING:
    from collections.abc import Iterable

    from anndata import AnnData
    from numpy.typing import NDArray

    from . import _CorrMethod, _Method


class _RankGenes:
    """Class for computing differential expression statistics on GPU."""

    def __init__(
        self,
        adata: AnnData,
        groups: Iterable[str] | Literal["all"],
        groupby: str,
        *,
        mask_var: NDArray[np.bool_] | None = None,
        reference: Literal["rest"] | str = "rest",
        use_raw: bool | None = None,
        layer: str | None = None,
        comp_pts: bool = False,
        pre_load: bool = False,
    ) -> None:
        # Handle groups parameter
        if groups == "all" or groups is None:
            selected: list | None = None
        elif isinstance(groups, str | int):
            msg = "Specify a sequence of groups"
            raise ValueError(msg)
        else:
            selected = list(groups)
            if len(selected) > 0 and isinstance(selected[0], int):
                selected = [str(n) for n in selected]
            if reference != "rest" and reference not in set(selected):
                selected.append(reference)

        self.labels = pd.Series(adata.obs[groupby]).reset_index(drop=True)
        all_categories = self.labels.cat.categories

        if reference != "rest" and str(reference) not in {
            str(c) for c in all_categories
        }:
            cats = all_categories.tolist()
            msg = f"reference = {reference} needs to be one of groupby = {cats}."
            raise ValueError(msg)

        self.groups_order, self.group_codes, self.group_sizes = _select_groups(
            self.labels, selected
        )

        # Get data matrix
        if layer is not None:
            if use_raw is True:
                msg = "Cannot specify `layer` and have `use_raw=True`."
                raise ValueError(msg)
            self.X = adata.layers[layer]
            self.var_names = adata.var_names
        elif use_raw is None and adata.raw is not None:
            self.X = adata.raw.X
            self.var_names = adata.raw.var_names
        elif use_raw is True:
            if adata.raw is None:
                msg = "Received `use_raw=True`, but `adata.raw` is empty."
                raise ValueError(msg)
            self.X = adata.raw.X
            self.var_names = adata.raw.var_names
        else:
            self.X = adata.X
            self.var_names = adata.var_names

        # Apply mask_var to select subset of genes
        if mask_var is not None:
            self.X = self.X[:, mask_var]
            self.var_names = self.var_names[mask_var]

        self.pre_load = pre_load

        self.ireference = None
        if reference != "rest":
            self.ireference = int(np.where(self.groups_order == str(reference))[0][0])

        # Set up expm1 function based on log base
        self.is_log1p = "log1p" in adata.uns
        base = adata.uns.get("log1p", {}).get("base")
        if base is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(base))
        else:
            self.expm1_func = np.expm1

        # For basic stats
        self.comp_pts = comp_pts
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.pts: np.ndarray | None = None
        self.means_rest: np.ndarray | None = None
        self.vars_rest: np.ndarray | None = None
        self.pts_rest: np.ndarray | None = None

        self.stats: pd.DataFrame | None = None
        self._compute_stats_in_chunks: bool = False
        self._ref_chunk_computed: set[int] = set()

    def _init_stats_arrays(self, n_genes: int) -> None:
        """Pre-allocate stats arrays before chunk loop."""
        n_groups = len(self.groups_order)

        self.means = np.zeros((n_groups, n_genes), dtype=np.float64)
        self.vars = np.zeros((n_groups, n_genes), dtype=np.float64)
        self.pts = (
            np.zeros((n_groups, n_genes), dtype=np.float64) if self.comp_pts else None
        )

        if self.ireference is None:
            self.means_rest = np.zeros((n_groups, n_genes), dtype=np.float64)
            self.vars_rest = np.zeros((n_groups, n_genes), dtype=np.float64)
            self.pts_rest = (
                np.zeros((n_groups, n_genes), dtype=np.float64)
                if self.comp_pts
                else None
            )
        else:
            self.means_rest = None
            self.vars_rest = None
            self.pts_rest = None

    def _basic_stats(self) -> None:
        """Compute means, vars, and pts for each group.

        If data is already on GPU, uses Aggregate for fast single-pass computation.
        Otherwise, sets flag for chunk-based computation during the wilcoxon loop.
        """
        n_genes = self.X.shape[1]

        # Check if data is already on GPU
        try:
            _check_gpu_X(self.X, allow_dask=True)
        except TypeError:
            is_on_gpu = False
        else:
            is_on_gpu = True

        if not is_on_gpu:
            # Data not on GPU - defer to chunk-based computation
            self._compute_stats_in_chunks = True
            self._init_stats_arrays(n_genes)
            return

        # Data is on GPU - use Aggregate for fast computation
        self._compute_stats_in_chunks = False

        agg = Aggregate(groupby=self.labels.cat, data=self.X)
        # Only sum and sq_sum are needed (means/vars are derived locally below);
        # count_nonzero is only needed when comp_pts is requested.
        funcs = {"sum", "sq_sum"}
        if self.comp_pts:
            funcs.add("count_nonzero")
        result = agg.count_mean_var(funcs, dof=1)

        # Map Aggregate category order → selected groups order
        cat_names = list(self.labels.cat.categories)
        cat_to_idx = {str(name): i for i, name in enumerate(cat_names)}
        order = [cat_to_idx[str(name)] for name in self.groups_order]

        # Aggregate returns stats per ALL categories. Slice to selected groups
        # for per-group means/vars; keep the all-category arrays for "rest"
        # stats so the totals stay correct when ``groups`` is a strict subset.
        sums_all = result["sum"]
        sq_sums_all = result["sq_sum"]
        nnz_all = result["count_nonzero"] if self.comp_pts else None

        n = cp.asarray(self.group_sizes, dtype=cp.float64)[:, None]
        sums = sums_all[order]
        sq_sums = sq_sums_all[order]

        # Compute means and variances from raw sums (all on GPU)
        means = sums / n
        group_ss = sq_sums - n * means**2
        vars_ = cp.maximum(group_ss / cp.maximum(n - 1, 1), 0)

        if self.comp_pts:
            pts = nnz_all[order].astype(cp.float64) / n
        else:
            pts = None

        # Compute rest statistics if reference='rest' — "rest" means every
        # cell in ``groupby`` not in this group, including cells in
        # categories that weren't selected via ``groups=``.
        if self.ireference is None:
            n_total = agg.n_cells.sum()
            n_rest = n_total - n
            means_rest = (sums_all.sum(axis=0) - sums) / n_rest
            rest_ss = (sq_sums_all.sum(axis=0) - sq_sums) - n_rest * means_rest**2
            vars_rest = cp.maximum(rest_ss / cp.maximum(n_rest - 1, 1), 0)

            self.means_rest = cp.asnumpy(means_rest)
            self.vars_rest = cp.asnumpy(vars_rest)

            if self.comp_pts:
                nnz_total = nnz_all.sum(axis=0)
                self.pts_rest = cp.asnumpy(
                    (nnz_total - nnz_all[order]).astype(cp.float64) / n_rest
                )
            else:
                self.pts_rest = None
        else:
            self.means_rest = None
            self.vars_rest = None
            self.pts_rest = None

        # Transfer to CPU
        self.means = cp.asnumpy(means)
        self.vars = cp.asnumpy(vars_)
        self.pts = cp.asnumpy(pts) if pts is not None else None

    def _accumulate_chunk_stats_vs_rest(
        self,
        block: cp.ndarray,
        start: int,
        stop: int,
        *,
        group_matrix: cp.ndarray,
        group_sizes_dev: cp.ndarray,
        n_cells: int,
    ) -> None:
        """Compute and store stats for one gene chunk (vs rest mode)."""
        if not self._compute_stats_in_chunks:
            return  # Stats already computed via Aggregate

        rest_sizes = n_cells - group_sizes_dev

        # Group sums and sum of squares
        group_sums = group_matrix.T @ block
        group_sum_sq = group_matrix.T @ (block**2)

        # Means
        chunk_means = group_sums / group_sizes_dev[:, None]
        self.means[:, start:stop] = cp.asnumpy(chunk_means)

        # Variances (with Bessel correction)
        chunk_vars = group_sum_sq / group_sizes_dev[:, None] - chunk_means**2
        chunk_vars *= group_sizes_dev[:, None] / (group_sizes_dev[:, None] - 1)
        self.vars[:, start:stop] = cp.asnumpy(chunk_vars)

        # Pts (fraction expressing)
        if self.comp_pts:
            group_nnz = group_matrix.T @ (block != 0).astype(cp.float64)
            self.pts[:, start:stop] = cp.asnumpy(group_nnz / group_sizes_dev[:, None])

        # Rest statistics
        if self.ireference is None:
            total_sum = block.sum(axis=0)
            total_sum_sq = (block**2).sum(axis=0)

            rest_sums = total_sum[None, :] - group_sums
            rest_means = rest_sums / rest_sizes[:, None]
            self.means_rest[:, start:stop] = cp.asnumpy(rest_means)

            rest_sum_sq = total_sum_sq[None, :] - group_sum_sq
            rest_vars = rest_sum_sq / rest_sizes[:, None] - rest_means**2
            rest_vars *= rest_sizes[:, None] / (rest_sizes[:, None] - 1)
            self.vars_rest[:, start:stop] = cp.asnumpy(rest_vars)

            if self.comp_pts:
                total_nnz = (block != 0).sum(axis=0)
                rest_nnz = total_nnz[None, :] - group_nnz
                self.pts_rest[:, start:stop] = cp.asnumpy(
                    rest_nnz / rest_sizes[:, None]
                )

    def _accumulate_chunk_stats_with_ref(
        self,
        block: cp.ndarray,
        start: int,
        stop: int,
        *,
        group_index: int,
        group_mask_gpu: cp.ndarray,
        n_group: int,
        n_ref: int,
    ) -> None:
        """Compute and store stats for one gene chunk (with reference mode)."""
        if not self._compute_stats_in_chunks:
            return  # Stats already computed via Aggregate

        # Group stats
        group_data = block[group_mask_gpu]
        group_mean = group_data.mean(axis=0)
        self.means[group_index, start:stop] = cp.asnumpy(group_mean)

        if n_group > 1:
            group_var = group_data.var(axis=0, ddof=1)
            self.vars[group_index, start:stop] = cp.asnumpy(group_var)

        if self.comp_pts:
            group_nnz = (group_data != 0).sum(axis=0)
            self.pts[group_index, start:stop] = cp.asnumpy(group_nnz / n_group)

        # Reference stats (only compute once, on first non-reference group)
        if start not in self._ref_chunk_computed:
            self._ref_chunk_computed.add(start)
            ref_data = block[~group_mask_gpu]
            ref_mean = ref_data.mean(axis=0)
            self.means[self.ireference, start:stop] = cp.asnumpy(ref_mean)

            if n_ref > 1:
                ref_var = ref_data.var(axis=0, ddof=1)
                self.vars[self.ireference, start:stop] = cp.asnumpy(ref_var)

            if self.comp_pts:
                ref_nnz = (ref_data != 0).sum(axis=0)
                self.pts[self.ireference, start:stop] = cp.asnumpy(ref_nnz / n_ref)

    def t_test(
        self, method: Literal["t-test", "t-test_overestim_var"]
    ) -> list[tuple[int, NDArray, NDArray]]:
        """Compute t-test statistics using Welch's t-test."""
        from ._ttest import t_test

        return t_test(self, method)

    def wilcoxon(
        self,
        *,
        tie_correct: bool,
        use_continuity: bool = False,
        chunk_size: int | None = None,
    ) -> list[tuple[int, NDArray, NDArray]]:
        """Compute Wilcoxon rank-sum test statistics."""
        from ._wilcoxon import wilcoxon

        return wilcoxon(
            self,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            chunk_size=chunk_size,
        )

    def wilcoxon_binned(
        self,
        *,
        tie_correct: bool = False,
        use_continuity: bool = False,
        n_bins: int | None = None,
        chunk_size: int | None = None,
        bin_range: Literal["log1p", "auto"] | None = None,
    ) -> list[tuple[int, NDArray, NDArray]]:
        """Histogram-based approximate Wilcoxon rank-sum test."""
        from ._wilcoxon_binned import wilcoxon_binned

        return wilcoxon_binned(
            self,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
            n_bins=n_bins,
            chunk_size=chunk_size,
            bin_range=bin_range,
        )

    def logreg(self, **kwds) -> list[tuple[int, NDArray, None]]:
        """Compute logistic regression scores."""
        from ._logreg import logreg

        return logreg(self, **kwds)

    def compute_statistics(
        self,
        method: _Method,
        *,
        corr_method: _CorrMethod = "benjamini-hochberg",
        n_genes_user: int | None = None,
        rankby_abs: bool = False,
        tie_correct: bool = False,
        use_continuity: bool = False,
        chunk_size: int | None = None,
        n_bins: int | None = None,
        bin_range: Literal["log1p", "auto"] | None = None,
        **kwds,
    ) -> None:
        """Compute statistics for all groups."""
        if self.pre_load or method in {
            "t-test",
            "t-test_overestim_var",
            "wilcoxon_binned",
        }:
            self.X = X_to_GPU(self.X)

        if method in {"t-test", "t-test_overestim_var"}:
            test_results = self.t_test(method)
        elif method == "wilcoxon":
            if isinstance(self.X, DaskArray):
                msg = "Wilcoxon test is not supported for Dask arrays. Please convert your data to CuPy arrays."
                raise ValueError(msg)
            test_results = self.wilcoxon(
                tie_correct=tie_correct,
                use_continuity=use_continuity,
                chunk_size=chunk_size,
            )
        elif method == "wilcoxon_binned":
            test_results = self.wilcoxon_binned(
                tie_correct=tie_correct,
                use_continuity=use_continuity,
                n_bins=n_bins,
                chunk_size=chunk_size,
                bin_range=bin_range,
            )
        elif method == "logreg":
            test_results = self.logreg(**kwds)
        else:
            assert_never(method)

        n_genes = self.X.shape[1]

        # Collect all stats data first to avoid DataFrame fragmentation
        stats_data: dict[tuple[str, str], np.ndarray] = {}

        for group_index, scores, pvals in test_results:
            group_name = str(self.groups_order[group_index])

            if n_genes_user is not None:
                scores_sort = np.abs(scores) if rankby_abs else scores
                global_indices = _select_top_n(scores_sort, n_genes_user)
            else:
                global_indices = slice(None)

            if n_genes_user is not None:
                stats_data[group_name, "names"] = np.asarray(self.var_names)[
                    global_indices
                ]

            stats_data[group_name, "scores"] = scores[global_indices]

            if pvals is not None:
                stats_data[group_name, "pvals"] = pvals[global_indices]
                if corr_method == "benjamini-hochberg":
                    pvals_clean = np.array(pvals, copy=True)
                    pvals_clean[np.isnan(pvals_clean)] = 1.0
                    _, pvals_adj, _, _ = multipletests(
                        pvals_clean, alpha=0.05, method="fdr_bh"
                    )
                elif corr_method == "bonferroni":
                    pvals_adj = np.minimum(pvals * n_genes, 1.0)
                stats_data[group_name, "pvals_adj"] = pvals_adj[global_indices]

            # Compute logfoldchanges
            if self.means is not None:
                mean_group = self.means[group_index]
                if self.ireference is None:
                    mean_rest = self.means_rest[group_index]
                else:
                    mean_rest = self.means[self.ireference]
                foldchanges = (self.expm1_func(mean_group) + EPS) / (
                    self.expm1_func(mean_rest) + EPS
                )
                stats_data[group_name, "logfoldchanges"] = np.log2(
                    foldchanges[global_indices]
                )

        # Create DataFrame all at once to avoid fragmentation
        if stats_data:
            self.stats = pd.DataFrame(stats_data)
            self.stats.columns = pd.MultiIndex.from_tuples(self.stats.columns)
            if n_genes_user is None:
                self.stats.index = self.var_names
        else:
            self.stats = None
