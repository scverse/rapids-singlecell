from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Literal, assert_never

import cupy as cp
import numpy as np
import pandas as pd
import scipy.sparse as sp

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.get import X_to_GPU
from rapids_singlecell.get._aggregated import Aggregate

from ._utils import (
    EPS,
    _canonicalize_sparse,
    _select_groups,
    _sparse_has_negative,
)

_RANK_SORT_MIN_ELEMENTS = 1_000_000
_RANK_SORT_MAX_WORKERS = 64

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
        skip_empty_groups: bool = False,
    ) -> None:
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
            self.labels,
            selected,
            reference=reference,
            skip_empty_groups=skip_empty_groups,
        )

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

        if mask_var is not None:
            self.X = self.X[:, mask_var]
            self.var_names = self.var_names[mask_var]

        self.ireference = None
        if reference != "rest":
            self.ireference = int(np.where(self.groups_order == str(reference))[0][0])

        # expm1 function depends on the log base used by log1p
        self.is_log1p = "log1p" in adata.uns
        base = adata.uns.get("log1p", {}).get("base")
        self._log1p_base = base
        if base is not None:
            self.expm1_func = lambda x: np.expm1(x * np.log(base))
        else:
            self.expm1_func = np.expm1

        self.comp_pts = comp_pts
        self.means: np.ndarray | None = None
        self.vars: np.ndarray | None = None
        self.pts: np.ndarray | None = None
        self.means_rest: np.ndarray | None = None
        self.vars_rest: np.ndarray | None = None
        self.pts_rest: np.ndarray | None = None

        self.stats_arrays: dict[str, object] | None = None
        self._sparse_negative_fallback = False
        self._score_dtype = np.dtype(np.float32)
        self._multi_gpu: bool | list[int] | str | None = None

    def _accumulate_planes(
        self,
    ) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray | None]:
        """Sum / sq_sum / (count_nonzero) over ALL categories → (n_cats, n_genes).

        Host input (numpy / scipy) streams blocks to the GPU (no full copy);
        device / Dask uses the device ``Aggregate``. ``count_nonzero`` only when
        ``comp_pts``.
        """
        X = self.X
        if isinstance(X, np.ndarray) or sp.issparse(X):
            return self._stream_planes()
        agg = Aggregate(groupby=self.labels.cat, data=X)
        funcs = {"sum", "sq_sum"}
        if self.comp_pts:
            funcs.add("count_nonzero")
        result = agg.count_mean_var(funcs, dof=1)
        return (
            result["sum"],
            result["sq_sum"],
            result["count_nonzero"] if self.comp_pts else None,
        )

    def _stream_planes(self) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray | None]:
        """Host-streaming accumulation of sum / sq_sum / (count_nonzero)."""
        from ._stream_multi_gpu import (
            aggr_host_planes,
            resolve_stream_devices,
            stream_planes_multi,
        )

        device_ids = resolve_stream_devices(multi_gpu=self._multi_gpu)
        n_cats = len(self.labels.cat.categories)
        if len(device_ids) > 1:
            return stream_planes_multi(self, device_ids)
        cats = cp.asarray(self.labels.cat.codes.to_numpy(), dtype=cp.int32)
        return aggr_host_planes(self.X, cats, n_cats, comp_pts=self.comp_pts)

    def _basic_stats(self) -> None:
        """Compute means, vars, and pts (host input streams, device uses Aggregate)."""
        sums_all, sq_sums_all, nnz_all = self._accumulate_planes()

        # Map category order → selected groups order.
        cat_names = list(self.labels.cat.categories)
        cat_to_idx = {str(name): i for i, name in enumerate(cat_names)}
        order = [cat_to_idx[str(name)] for name in self.groups_order]

        n = cp.asarray(self.group_sizes, dtype=cp.float64)[:, None]
        sums = sums_all[order]
        sq_sums = sq_sums_all[order]

        means = sums / n
        group_ss = sq_sums - n * means**2
        vars_ = cp.maximum(group_ss / cp.maximum(n - 1, 1), 0)

        if self.comp_pts:
            pts = nnz_all[order].astype(cp.float64) / n
        else:
            pts = None

        # For reference='rest', rest includes every category not in this group.
        # That includes categories omitted by a strict ``groups=`` selection.
        if self.ireference is None:
            n_total = cp.float64(self.X.shape[0])
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

        self.means = cp.asnumpy(means)
        self.vars = cp.asnumpy(vars_)
        self.pts = cp.asnumpy(pts) if pts is not None else None

    def t_test(
        self, method: Literal["t-test", "t-test_overestim_var"]
    ) -> list[tuple[int, NDArray, NDArray]]:
        """Compute t-test statistics using Welch's t-test."""
        from ._ttest import t_test

        return t_test(self, method)

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
        multi_gpu: bool | list[int] | str | None = None,
        n_bins: int | None = None,
        bin_range: Literal["log1p", "auto"] | None = None,
        return_u_values: bool = False,
        **kwds,
    ) -> None:
        """Compute statistics for all groups."""
        # Devices for the host-streaming t-test / binned shards.
        self._multi_gpu = multi_gpu
        # Exact OVR inserts implicit zeros between negative and positive stored
        # values analytically. Binned and host OVO sparse paths still need the
        # sign check; device OVO does not use its result.
        self._sparse_negative_fallback = False
        if method in {"wilcoxon", "wilcoxon_binned"}:
            # Fast paths rank each stored coordinate once, so they must see
            # scanpy's summed duplicate view even when no sign scan is needed.
            self.X = _canonicalize_sparse(self.X)
            needs_signed_fallback = method == "wilcoxon_binned" or (
                self.ireference is not None and sp.issparse(self.X)
            )
            if needs_signed_fallback:
                self._sparse_negative_fallback = _sparse_has_negative(self.X)
        if method in {"t-test", "t-test_overestim_var", "wilcoxon_binned"}:
            # Host input streams (no full copy); device / Dask move to the GPU.
            if not (isinstance(self.X, np.ndarray) or sp.issparse(self.X)):
                self.X = X_to_GPU(self.X)

        n_genes = self.X.shape[1]
        if n_genes_user is None:
            n_genes_user = n_genes

        wilcoxon_result = None
        if method in {"t-test", "t-test_overestim_var"}:
            test_results = self.t_test(method)
        elif method == "wilcoxon":
            from ._wilcoxon import wilcoxon

            if isinstance(self.X, DaskArray):
                msg = "Wilcoxon test is not supported for Dask arrays. Please convert your data to CuPy arrays."
                raise ValueError(msg)
            self._score_dtype = np.dtype(np.float64 if return_u_values else np.float32)
            wilcoxon_result = wilcoxon(
                self,
                tie_correct=tie_correct,
                use_continuity=use_continuity,
                chunk_size=chunk_size,
                multi_gpu=multi_gpu,
                return_u_values=return_u_values,
            )
            test_results = []
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

        if not test_results and wilcoxon_result is None:
            self.stats_arrays = {
                "group_indices": np.empty(0, dtype=np.intp),
                "group_names": np.empty(0, dtype=object),
                "var_names": np.asarray(self.var_names),
                "gene_indices": np.empty((0, n_genes_user), dtype=np.intp),
            }
            return

        if wilcoxon_result is not None:
            group_indices, scores_gpu, pvals_gpu, logfoldchanges_gpu = wilcoxon_result
            with cp.cuda.Device(scores_gpu.device.id):
                self._compute_statistics_gpu_arrays(
                    group_indices,
                    scores_gpu,
                    pvals_gpu,
                    logfoldchanges_gpu,
                    corr_method=corr_method,
                    n_genes_user=n_genes_user,
                    n_genes=n_genes,
                    rankby_abs=rankby_abs,
                )
            return

        self._compute_statistics_arrays(
            test_results,
            corr_method=corr_method,
            n_genes_user=n_genes_user,
            n_genes=n_genes,
            rankby_abs=rankby_abs,
        )

    @staticmethod
    def _rank_indices_matrix(scores: np.ndarray, n_top: int) -> np.ndarray:
        if n_top >= scores.shape[1]:
            return _RankGenes._argsort_desc_matrix(scores)
        partition = np.argpartition(scores, -n_top, axis=1)[:, -n_top:]
        row_ids = np.arange(scores.shape[0])[:, None]
        order = np.argsort(scores[row_ids, partition], axis=1)[:, ::-1]
        return partition[row_ids, order]

    @staticmethod
    def _argsort_desc_matrix(scores: np.ndarray) -> np.ndarray:
        n_rows, n_cols = scores.shape
        n_elements = n_rows * n_cols
        n_workers = min(_RANK_SORT_MAX_WORKERS, os.cpu_count() or 1, n_rows)
        if n_workers <= 1 or n_elements < _RANK_SORT_MIN_ELEMENTS:
            return np.argsort(scores, axis=1)[:, ::-1]

        chunks = np.linspace(0, n_rows, n_workers + 1, dtype=np.intp)
        indices = np.empty((n_rows, n_cols), dtype=np.intp)

        def sort_chunk(chunk_index: int) -> None:
            start = int(chunks[chunk_index])
            stop = int(chunks[chunk_index + 1])
            if start < stop:
                indices[start:stop] = np.argsort(scores[start:stop], axis=1)[:, ::-1]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            list(executor.map(sort_chunk, range(n_workers)))
        return indices

    @staticmethod
    def _fdr_bh_matrix(pvals: np.ndarray) -> np.ndarray:
        pvals_clean = np.array(pvals, copy=True)
        pvals_clean[np.isnan(pvals_clean)] = 1.0
        order = np.argsort(pvals_clean, axis=1)
        sorted_p = np.take_along_axis(pvals_clean, order, axis=1)
        n_tests = sorted_p.shape[1]
        scale = n_tests / np.arange(1, n_tests + 1, dtype=np.float64)
        corrected_sorted = sorted_p * scale
        corrected_sorted = np.minimum.accumulate(corrected_sorted[:, ::-1], axis=1)[
            :, ::-1
        ]
        corrected_sorted[corrected_sorted > 1.0] = 1.0
        corrected = np.empty_like(corrected_sorted)
        np.put_along_axis(corrected, order, corrected_sorted, axis=1)
        return corrected

    @staticmethod
    def _fdr_bh_matrix_gpu(pvals: cp.ndarray) -> cp.ndarray:
        pvals_clean = cp.nan_to_num(pvals, nan=1.0)
        order = cp.argsort(pvals_clean, axis=1)
        corrected_sorted = cp.take_along_axis(pvals_clean, order, axis=1)
        corrected_sorted *= corrected_sorted.shape[1] / cp.arange(
            1, corrected_sorted.shape[1] + 1, dtype=cp.float64
        )
        from rapids_singlecell._cuda import _rank_stats_cuda as _rs

        _rs.fdr_bh_reverse_cummin(
            corrected_sorted, stream=cp.cuda.get_current_stream().ptr
        )
        corrected = cp.empty_like(corrected_sorted)
        cp.put_along_axis(corrected, order, corrected_sorted, axis=1)
        return corrected

    def _logfoldchanges_into(
        self, arrays: dict, group_indices: np.ndarray, top_idx: np.ndarray
    ) -> None:
        mean_group = self.means[group_indices]
        if self.ireference is None:
            mean_rest = self.means_rest[group_indices]
        else:
            mean_rest = self.means[self.ireference][None, :]
        foldchanges = (self.expm1_func(mean_group) + EPS) / (
            self.expm1_func(mean_rest) + EPS
        )
        logfoldchanges = np.log2(foldchanges)
        arrays["logfoldchanges"] = np.take_along_axis(
            logfoldchanges, top_idx, axis=1
        ).astype(np.float32, copy=False)

    def _compute_statistics_arrays(
        self,
        test_results: list[tuple[int, NDArray, NDArray]],
        *,
        corr_method: _CorrMethod,
        n_genes_user: int,
        n_genes: int,
        rankby_abs: bool,
    ) -> None:
        group_indices = np.asarray([r[0] for r in test_results], dtype=np.intp)
        scores = np.vstack([r[1] for r in test_results])
        sort_scores = np.abs(scores) if rankby_abs else scores
        top_idx = self._rank_indices_matrix(sort_scores, n_genes_user)

        arrays: dict[str, object] = {
            "group_indices": group_indices,
            "group_names": np.asarray(
                [str(self.groups_order[i]) for i in group_indices], dtype=object
            ),
            "var_names": np.asarray(self.var_names),
            "gene_indices": top_idx.astype(np.intp, copy=False),
            "scores": np.take_along_axis(scores, top_idx, axis=1).astype(
                self._score_dtype, copy=False
            ),
        }

        if test_results[0][2] is not None:
            pvals = np.vstack([r[2] for r in test_results])
            arrays["pvals"] = np.take_along_axis(pvals, top_idx, axis=1)
            if corr_method == "benjamini-hochberg":
                pvals_adj = self._fdr_bh_matrix(pvals)
            elif corr_method == "bonferroni":
                pvals_adj = np.minimum(pvals * n_genes, 1.0)
            else:
                msg = f"Unsupported correction method: {corr_method!r}."
                raise ValueError(msg)
            arrays["pvals_adj"] = np.take_along_axis(pvals_adj, top_idx, axis=1)

        if self.means is not None:
            self._logfoldchanges_into(arrays, group_indices, top_idx)

        self.stats_arrays = arrays

    def _compute_statistics_gpu_arrays(
        self,
        group_indices: np.ndarray,
        scores_gpu: cp.ndarray,
        pvals_gpu: cp.ndarray,
        logfoldchanges_gpu: cp.ndarray | None,
        *,
        corr_method: _CorrMethod,
        n_genes_user: int,
        n_genes: int,
        rankby_abs: bool,
    ) -> None:
        group_indices = np.asarray(group_indices, dtype=np.intp)
        scores = cp.asnumpy(scores_gpu)
        sort_scores = np.abs(scores) if rankby_abs else scores
        top_idx = self._rank_indices_matrix(sort_scores, n_genes_user)
        top_idx_gpu = cp.asarray(top_idx, dtype=cp.int32)

        arrays: dict[str, object] = {
            "group_indices": group_indices,
            "group_names": np.asarray(
                [str(self.groups_order[i]) for i in group_indices], dtype=object
            ),
            "var_names": np.asarray(self.var_names),
            "gene_indices": top_idx.astype(np.intp, copy=False),
            "scores": cp.asnumpy(
                cp.take_along_axis(scores_gpu, top_idx_gpu, axis=1).astype(
                    self._score_dtype, copy=False
                ),
                order="F",
            ),
            "pvals": cp.asnumpy(
                cp.take_along_axis(pvals_gpu, top_idx_gpu, axis=1), order="F"
            ),
        }

        if corr_method == "benjamini-hochberg":
            pvals_adj_gpu = self._fdr_bh_matrix_gpu(pvals_gpu)
        elif corr_method == "bonferroni":
            pvals_adj_gpu = cp.minimum(pvals_gpu * n_genes, 1.0)
        else:
            msg = f"Unsupported correction method: {corr_method!r}."
            raise ValueError(msg)
        arrays["pvals_adj"] = cp.asnumpy(
            cp.take_along_axis(pvals_adj_gpu, top_idx_gpu, axis=1), order="F"
        )

        if logfoldchanges_gpu is not None:
            arrays["logfoldchanges"] = cp.asnumpy(
                cp.take_along_axis(logfoldchanges_gpu, top_idx_gpu, axis=1).astype(
                    cp.float32, copy=False
                ),
                order="F",
            )
        elif self.means is not None:
            self._logfoldchanges_into(arrays, group_indices, top_idx)

        self.stats_arrays = arrays
