from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import issparse as cp_issparse
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell.get import X_to_GPU

if TYPE_CHECKING:
    from anndata import AnnData

_PACKED_READOUT_MAX_GUIDES = 32
_PACKED_READOUT_MIN_ASSIGNABLE_CELLS = 250_000
_GROUPED_READOUT_MAX_GUIDES = 128
_SUBSET_READOUT_MAX_ASSIGNABLE_FRACTION = 0.25
_MIXTURE_MODEL_VAR_COLUMNS = [
    "poisson_rate",
    "gaussian_mean",
    "gaussian_std",
    "mix_probs_0",
    "mix_probs_1",
    "threshold",
    "weight_Poisson",
    "weight_Normal",
    "lambda",
    "mu",
    "scale",
]


class GuideAssignment:
    """GPU-accelerated guide RNA assignment.

    Provides threshold-based and mixture-model-based methods for assigning
    cells to guide RNAs, compatible with pertpy's ``GuideAssignment`` API.
    The mixture model follows crispat's Poisson-Gaussian assignment rule
    while using batched EM on GPU instead of per-guide Pyro SVI, yielding
    orders-of-magnitude speedup.
    """

    def assign_by_threshold(
        self,
        adata: AnnData,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        output_layer: str = "assigned_guides",
    ) -> None:
        """Assign cells to gRNAs exceeding a count threshold.

        Each cell is assigned to every gRNA with at least
        ``assignment_threshold`` counts. Expects unnormalized count data.

        Parameters
        ----------
        adata
            Annotated data matrix of shape ``n_obs x n_vars``.
        assignment_threshold
            Minimum count for a viable assignment.
        layer
            Layer with raw counts. Uses ``adata.X`` if ``None``.
        output_layer
            Key under which the binary assignment matrix is stored
            in ``adata.layers``.
        """
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))

        if cp_issparse(X):
            new_data = cp.where(
                X.data >= assignment_threshold,
                X.dtype.type(1),
                X.dtype.type(0),
            )
            result = X.copy()
            result.data = new_data
        else:
            result = cp.where(X >= assignment_threshold, cp.int8(1), cp.int8(0))

        _set_obs_rep(adata, result, layer=output_layer)

    def assign_to_max_guide(
        self,
        adata: AnnData,
        *,
        assignment_threshold: float,
        layer: str | None = None,
        obs_key: str = "assigned_guide",
        no_grna_assigned_key: str = "Negative",
    ) -> None:
        """Assign each cell to its most expressed gRNA.

        Each cell is assigned to the gRNA with the highest count, provided
        that count is at least ``assignment_threshold``. Expects
        unnormalized count data.

        Parameters
        ----------
        adata
            Annotated data matrix of shape ``n_obs x n_vars``.
        assignment_threshold
            Minimum count for a viable assignment.
        layer
            Layer with raw counts. Uses ``adata.X`` if ``None``.
        obs_key
            Column in ``adata.obs`` where the assignment is stored.
        no_grna_assigned_key
            Label for cells with no guide above threshold.
        """
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        var_names = np.asarray(adata.var_names)

        if cp_issparse(X):
            X_dense = X.toarray()
        else:
            X_dense = X

        max_vals = X_dense.max(axis=1)
        max_idx = X_dense.argmax(axis=1)

        max_vals_cpu = cp.asnumpy(max_vals).ravel()
        max_idx_cpu = cp.asnumpy(max_idx).ravel()

        assigned = np.full(adata.n_obs, no_grna_assigned_key, dtype=object)
        above = max_vals_cpu >= assignment_threshold
        assigned[above] = var_names[max_idx_cpu[above]]

        adata.obs[obs_key] = assigned

    def assign_mixture_model(
        self,
        adata: AnnData,
        *,
        layer: str | None = None,
        assigned_guides_key: str = "assigned_guide",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        only_return_results: bool = False,
        max_iter: int = 90,
        tol: float = 1e-4,
        posterior_threshold: float = 0.5,
    ) -> np.ndarray | None:
        """Assign gRNAs using a GPU-accelerated Poisson–Gaussian mixture model.

        Fits a two-component mixture (Poisson background + Gaussian signal)
        to the log₂-transformed non-zero counts of each guide simultaneously
        using batched Expectation-Maximization on GPU. Like crispat's
        Poisson-Gaussian assignment, the fitted model is converted to an
        integer raw-count threshold. The default posterior cutoff matches
        pertpy's crispat-style threshold rule.

        Parameters
        ----------
        adata
            Annotated data matrix with guide RNA counts.
        layer
            Layer with raw counts. Uses ``adata.X`` if ``None``.
        assigned_guides_key
            Key in ``adata.obs`` for storing the assignment result.
        no_grna_assigned_key
            Label for cells negative for all gRNAs.
        max_assignments_per_cell
            Maximum number of gRNAs a cell can be assigned to.
        multiple_grna_assigned_key
            Label for cells exceeding ``max_assignments_per_cell``.
        multiple_grna_assignment_string
            Delimiter for joining multiple guide names.
        only_return_results
            If ``True``, return assignments without modifying ``adata``.
        max_iter
            Maximum number of EM iterations.
        tol
            Convergence tolerance on parameter changes.
        posterior_threshold
            Minimum posterior probability of the Gaussian component required
            for a raw UMI count to define the assignment threshold.
        Returns
        -------
        If ``only_return_results`` is ``True``, returns an array of
            assignments. Otherwise modifies ``adata`` in-place and returns
            ``None``.
        """
        _validate_mixture_model_args(
            max_assignments_per_cell=max_assignments_per_cell,
            max_iter=max_iter,
            tol=tol,
            posterior_threshold=posterior_threshold,
        )

        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        if cp_issparse(X):
            from rapids_singlecell.preprocessing._utils import _sparse_to_dense

            if X.dtype != cp.float32:
                X = X.astype(cp.float32)
            X = _sparse_to_dense(X, order="F")
        else:
            X = X.astype(cp.float32, copy=False)
            if not (X.flags.c_contiguous or X.flags.f_contiguous):
                X = cp.ascontiguousarray(X)

        var_names = np.asarray(adata.var_names)
        assignments, thresholds, lam, mu, sigma, pi0, valid_guides = _fit_assign_cuda(
            X,
            max_iter=max_iter,
            tol=tol,
            posterior_threshold=posterior_threshold,
        )

        if len(valid_guides) == 0:
            warnings.warn(
                "No guides have enough expressing cells for mixture model fitting.",
                UserWarning,
                stacklevel=2,
            )
            series = pd.Series(
                no_grna_assigned_key,
                index=adata.obs_names,
            )
            if only_return_results:
                return series.values
            adata.obs[assigned_guides_key] = series.values
            return None

        lam_cpu = cp.asnumpy(lam.ravel())
        mu_cpu = cp.asnumpy(mu.ravel())
        sigma_cpu = cp.asnumpy(sigma.ravel())
        pi0_cpu = cp.asnumpy(pi0.ravel())
        thresholds_cpu = cp.asnumpy(thresholds.ravel())

        valid_var_names = var_names[np.asarray(valid_guides, dtype=np.intp)]
        series_values = _assignments_to_labels(
            assignments,
            valid_var_names,
            max_assignments_per_cell=max_assignments_per_cell,
            no_grna_assigned_key=no_grna_assigned_key,
            multiple_grna_assigned_key=multiple_grna_assigned_key,
            multiple_grna_assignment_string=multiple_grna_assignment_string,
        )

        if only_return_results:
            return series_values

        adata.var[_MIXTURE_MODEL_VAR_COLUMNS] = np.nan
        valid_var_index = adata.var_names[np.asarray(valid_guides, dtype=np.intp)]
        adata.var.loc[valid_var_index, "poisson_rate"] = lam_cpu
        adata.var.loc[valid_var_index, "gaussian_mean"] = mu_cpu
        adata.var.loc[valid_var_index, "gaussian_std"] = sigma_cpu
        adata.var.loc[valid_var_index, "mix_probs_0"] = pi0_cpu
        adata.var.loc[valid_var_index, "mix_probs_1"] = 1.0 - pi0_cpu
        adata.var.loc[valid_var_index, "threshold"] = thresholds_cpu
        adata.var.loc[valid_var_index, "weight_Poisson"] = pi0_cpu
        adata.var.loc[valid_var_index, "weight_Normal"] = 1.0 - pi0_cpu
        adata.var.loc[valid_var_index, "lambda"] = lam_cpu
        adata.var.loc[valid_var_index, "mu"] = mu_cpu
        adata.var.loc[valid_var_index, "scale"] = sigma_cpu

        adata.obs[assigned_guides_key] = series_values
        return None


def _assignments_to_labels(
    assignments: cp.ndarray | np.ndarray,
    guide_names: np.ndarray,
    *,
    max_assignments_per_cell: int,
    no_grna_assigned_key: str,
    multiple_grna_assigned_key: str,
    multiple_grna_assignment_string: str,
) -> np.ndarray:
    """Convert guide-by-cell assignments to pertpy-style cell labels.

    The dispatcher keeps all label creation in one place:
    high-guide GPU count gating, packed small-guide patterns, grouped
    medium-guide readout, and the exact fallback.
    """
    n_guides, n_cells = assignments.shape

    if isinstance(assignments, cp.ndarray) and n_guides > _GROUPED_READOUT_MAX_GUIDES:
        num_guides_assigned_gpu = assignments.sum(axis=0)
        assignable_gpu = (num_guides_assigned_gpu > 0) & (
            num_guides_assigned_gpu <= max_assignments_per_cell
        )
        assignable_count_gpu = cp.count_nonzero(assignable_gpu)
        assignable_count = int(assignable_count_gpu.item())

        negative_count = int(cp.count_nonzero(num_guides_assigned_gpu == 0).item())
        if assignable_count == 0:
            if negative_count == 0:
                return np.full(n_cells, multiple_grna_assigned_key, dtype=object)
            if negative_count == n_cells:
                return np.full(n_cells, no_grna_assigned_key, dtype=object)

            num_guides_assigned = cp.asnumpy(num_guides_assigned_gpu)
            labels = np.empty(n_cells, dtype=object)
            labels[num_guides_assigned == 0] = no_grna_assigned_key
            labels[num_guides_assigned > max_assignments_per_cell] = (
                multiple_grna_assigned_key
            )
            return labels

        assignable_fraction = assignable_count / n_cells
        if assignable_fraction <= _SUBSET_READOUT_MAX_ASSIGNABLE_FRACTION:
            assignable_idx_gpu = cp.flatnonzero(assignable_gpu)
            assignable_idx = cp.asnumpy(assignable_idx_gpu)
            subset_assignments = cp.asnumpy(assignments[:, assignable_idx_gpu])

            num_guides_assigned = cp.asnumpy(num_guides_assigned_gpu)
            labels = np.empty(n_cells, dtype=object)
            labels[num_guides_assigned == 0] = no_grna_assigned_key
            labels[num_guides_assigned > max_assignments_per_cell] = (
                multiple_grna_assigned_key
            )
            labels[assignable_idx] = _assignments_to_labels(
                subset_assignments,
                guide_names,
                no_grna_assigned_key=no_grna_assigned_key,
                max_assignments_per_cell=max_assignments_per_cell,
                multiple_grna_assigned_key=multiple_grna_assigned_key,
                multiple_grna_assignment_string=multiple_grna_assignment_string,
            )
            return labels

    if isinstance(assignments, cp.ndarray):
        assignments = cp.asnumpy(assignments)

    num_guides_assigned = assignments.sum(axis=0)

    assignable = (num_guides_assigned > 0) & (
        num_guides_assigned <= max_assignments_per_cell
    )
    assignable_count = int(assignable.sum())
    if (
        guide_names.size <= _PACKED_READOUT_MAX_GUIDES
        and assignable_count >= _PACKED_READOUT_MIN_ASSIGNABLE_CELLS
    ):
        labels = np.empty(n_cells, dtype=object)
        labels[num_guides_assigned == 0] = no_grna_assigned_key
        labels[(num_guides_assigned > 0) & ~assignable] = multiple_grna_assigned_key

        packed = np.packbits(assignments[:, assignable].T, axis=1, bitorder="little")
        padded = np.zeros(
            (packed.shape[0], np.dtype(np.uint64).itemsize), dtype=np.uint8
        )
        padded[:, : packed.shape[1]] = packed
        codes = padded.view(np.uint64).ravel()

        unique_codes, inverse = np.unique(codes, return_inverse=True)
        unique_labels = np.empty(unique_codes.size, dtype=object)
        guide_bits = np.arange(guide_names.size, dtype=np.uint64)
        one = np.uint64(1)
        for i, code in enumerate(unique_codes):
            selected = ((code >> guide_bits) & one).astype(bool)
            unique_labels[i] = multiple_grna_assignment_string.join(
                guide_names[selected].tolist()
            )

        labels[assignable] = unique_labels[inverse]
        return labels

    labels = np.empty(n_cells, dtype=object)
    labels[num_guides_assigned == 0] = no_grna_assigned_key

    single_assignment = num_guides_assigned == 1
    if single_assignment.any():
        labels[single_assignment] = guide_names[
            np.argmax(assignments[:, single_assignment], axis=0)
        ]

    multi_assignment = assignable & (num_guides_assigned > 1)
    if guide_names.size <= _GROUPED_READOUT_MAX_GUIDES:
        max_join_count = min(max_assignments_per_cell, guide_names.size)
        for n_assigned in range(2, max_join_count + 1):
            cell_indices = np.flatnonzero(num_guides_assigned == n_assigned)
            if cell_indices.size == 0:
                continue
            _rows, guide_indices = np.nonzero(assignments[:, cell_indices].T)
            guide_indices = guide_indices.reshape(cell_indices.size, n_assigned)
            labels[cell_indices] = [
                multiple_grna_assignment_string.join(guide_names[row].tolist())
                for row in guide_indices
            ]
        labels[num_guides_assigned > max_assignments_per_cell] = (
            multiple_grna_assigned_key
        )
        return labels

    for cell_idx in np.flatnonzero(multi_assignment):
        labels[cell_idx] = multiple_grna_assignment_string.join(
            guide_names[assignments[:, cell_idx]].tolist()
        )

    labels[num_guides_assigned > max_assignments_per_cell] = multiple_grna_assigned_key
    return labels


def _validate_fit_args(
    *, max_iter: int, tol: float, posterior_threshold: float
) -> None:
    if max_iter < 1:
        raise ValueError("max_iter must be at least 1.")
    if tol <= 0:
        raise ValueError("tol must be positive.")
    if not 0 < posterior_threshold < 1:
        raise ValueError("posterior_threshold must be between 0 and 1.")


def _validate_mixture_model_args(
    *,
    max_assignments_per_cell: int,
    max_iter: int,
    tol: float,
    posterior_threshold: float,
) -> None:
    _validate_fit_args(
        max_iter=max_iter,
        tol=tol,
        posterior_threshold=posterior_threshold,
    )
    if max_assignments_per_cell < 1:
        raise ValueError("max_assignments_per_cell must be at least 1.")


def _fit_assign_cuda(
    X: cp.ndarray,
    *,
    max_iter: int,
    tol: float,
    posterior_threshold: float,
) -> tuple[
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    cp.ndarray,
    list[int],
]:
    """Fit and assign all guides with the nanobind/CUDA EM kernel."""
    _validate_fit_args(
        max_iter=max_iter,
        tol=tol,
        posterior_threshold=posterior_threshold,
    )

    from rapids_singlecell._cuda import _guide_assignment_cuda

    if _guide_assignment_cuda is None:
        raise ImportError(
            "The _guide_assignment_cuda extension is not available. "
            "Build rapids-singlecell with CUDA extensions."
        )

    if not (X.flags.c_contiguous or X.flags.f_contiguous):
        X = cp.ascontiguousarray(X)

    n_cells, n_guides = X.shape
    assignments_all = cp.empty((n_guides, n_cells), dtype=cp.bool_)
    thresholds_all = cp.empty((n_guides, 1), dtype=cp.float32)
    lam_all = cp.empty((n_guides, 1), dtype=cp.float32)
    mu_all = cp.empty((n_guides, 1), dtype=cp.float32)
    sigma_all = cp.empty((n_guides, 1), dtype=cp.float32)
    pi0_all = cp.empty((n_guides, 1), dtype=cp.float32)
    valid_mask = cp.empty(n_guides, dtype=cp.bool_)
    nonzero_counts = cp.empty(n_guides, dtype=cp.int32)
    max_counts = cp.empty(n_guides, dtype=cp.int32)

    _guide_assignment_cuda.fit_assign_dense(
        X,
        assignments_all,
        thresholds_all,
        lam_all,
        mu_all,
        sigma_all,
        pi0_all,
        valid_mask,
        nonzero_counts,
        max_counts,
        n_cells=n_cells,
        n_guides=n_guides,
        max_iter=int(max_iter),
        tol=float(tol),
        posterior_threshold=float(posterior_threshold),
        stream=cp.cuda.get_current_stream().ptr,
    )

    valid_mask_cpu = cp.asnumpy(valid_mask).astype(bool)
    nonzero_counts_cpu = cp.asnumpy(nonzero_counts)
    max_counts_cpu = cp.asnumpy(max_counts)

    for guide, (nz_count, max_count) in enumerate(
        zip(nonzero_counts_cpu, max_counts_cpu, strict=True)
    ):
        if 0 < nz_count < 2:
            warnings.warn(
                f"Skipping guide index {guide} as there are less than 2 cells "
                "expressing the guide.",
                UserWarning,
                stacklevel=4,
            )
        elif nz_count >= 2 and max_count < 2:
            warnings.warn(
                f"Skipping guide index {guide} as the maximum UMI count is less "
                "than 2.",
                UserWarning,
                stacklevel=4,
            )

    valid_guides = np.flatnonzero(valid_mask_cpu).tolist()
    if len(valid_guides) == 0:
        empty_2d = cp.empty((0, 1), dtype=cp.float32)
        return (
            cp.empty((0, n_cells), dtype=cp.bool_),
            empty_2d,
            empty_2d,
            empty_2d,
            empty_2d,
            empty_2d,
            [],
        )

    if len(valid_guides) == n_guides:
        return (
            assignments_all,
            thresholds_all,
            lam_all,
            mu_all,
            sigma_all,
            pi0_all,
            valid_guides,
        )

    valid_guides_gpu = cp.asarray(valid_guides, dtype=cp.int32)
    return (
        assignments_all[valid_guides_gpu],
        thresholds_all[valid_guides_gpu],
        lam_all[valid_guides_gpu],
        mu_all[valid_guides_gpu],
        sigma_all[valid_guides_gpu],
        pi0_all[valid_guides_gpu],
        valid_guides,
    )
