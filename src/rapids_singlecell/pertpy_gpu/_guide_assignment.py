from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import issparse as cp_issparse
from cupyx.scipy.special import gammaln
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell.get import X_to_GPU

if TYPE_CHECKING:
    from anndata import AnnData

_LOG_2PI = float(np.log(2.0 * np.pi))
_PACKED_READOUT_MAX_GUIDES = 32
_PACKED_READOUT_MIN_ASSIGNABLE_CELLS = 250_000
_GROUPED_READOUT_MAX_GUIDES = 128
_SUBSET_READOUT_MAX_ASSIGNABLE_FRACTION = 0.25


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
            from cupyx.scipy.sparse import csr_matrix as gpu_csr

            new_data = cp.where(
                X.data >= assignment_threshold,
                X.dtype.type(1),
                X.dtype.type(0),
            )
            result = gpu_csr(
                (new_data, X.indices.copy(), X.indptr.copy()), shape=X.shape
            )
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
        assigned_guides_key: str = "assigned_guide",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        only_return_results: bool = False,
        max_iter: int = 90,
        tol: float = 1e-4,
        posterior_threshold: float = 0.645,
        backend: str = "auto",
    ) -> np.ndarray | None:
        """Assign gRNAs using a GPU-accelerated Poisson–Gaussian mixture model.

        Fits a two-component mixture (Poisson background + Gaussian signal)
        to the log₂-transformed non-zero counts of each guide simultaneously
        using batched Expectation-Maximization on GPU. Like crispat's
        Poisson-Gaussian assignment, the fitted model is converted to an
        integer raw-count threshold. The default posterior cutoff is slightly
        conservative to calibrate the GPU EM approximation against crispat's
        Pyro SVI outputs; set ``posterior_threshold=0.5`` for the literal
        crispat threshold rule.

        Parameters
        ----------
        adata
            Annotated data matrix with guide RNA counts.
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
        backend
            Backend for fitting and assignment. ``"auto"`` tries the
            nanobind/CUDA EM + threshold kernel and falls back to the CuPy
            implementation if the extension is unavailable. ``"cuda"``
            requires the CUDA extension, and ``"cupy"`` uses the temporary
            CuPy reference path.

        Returns
        -------
        If ``only_return_results`` is ``True``, returns an array of
        assignments. Otherwise modifies ``adata`` in-place and returns
        ``None``.
        """
        X = X_to_GPU(adata.X)
        if cp_issparse(X):
            X = X.toarray()
        # TODO: The CUDA guide kernel scans one guide column per block. If this
        # path becomes the default, consider densifying sparse inputs directly
        # to F-order with _sparse_to_dense(order="F") and dispatching an
        # F-contiguous kernel to improve memory coalescing.
        X = cp.ascontiguousarray(X.astype(cp.float32, copy=False))
        if not 0 < posterior_threshold < 1:
            raise ValueError("posterior_threshold must be between 0 and 1.")
        if backend not in {"cupy", "cuda", "auto"}:
            raise ValueError("backend must be one of 'cupy', 'cuda', or 'auto'.")

        _, n_guides = X.shape
        var_names = np.asarray(adata.var_names)

        if backend in {"cuda", "auto"}:
            try:
                assignments, thresholds, lam, mu, sigma, pi0, valid_guides = (
                    _fit_assign_cuda(
                        X,
                        max_iter=max_iter,
                        tol=tol,
                        posterior_threshold=posterior_threshold,
                    )
                )
            except ImportError:
                if backend == "cuda":
                    raise
                backend = "cupy"

        if backend == "cupy":
            data_pad, mask, _cell_indices, counts, valid_guides = _prepare_batched_data(
                X, n_guides
            )
            if len(valid_guides) > 0:
                lam, mu, sigma, pi0, _ = _batched_em(
                    data_pad, mask, counts, max_iter=max_iter, tol=tol
                )
                assignments, thresholds = _assign_by_crispat_threshold(
                    X,
                    valid_guides,
                    lam=lam,
                    mu=mu,
                    sigma=sigma,
                    pi0=pi0,
                    posterior_threshold=posterior_threshold,
                )

        for col in [
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
        ]:
            if col not in adata.var.columns:
                adata.var[col] = np.nan

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

        # Store fitted parameters in adata.var
        lam_cpu = cp.asnumpy(lam.ravel())
        mu_cpu = cp.asnumpy(mu.ravel())
        sigma_cpu = cp.asnumpy(sigma.ravel())
        pi0_cpu = cp.asnumpy(pi0.ravel())

        thresholds_cpu = cp.asnumpy(thresholds.ravel())
        for i, g in enumerate(valid_guides):
            adata.var.iloc[g, adata.var.columns.get_loc("poisson_rate")] = lam_cpu[i]
            adata.var.iloc[g, adata.var.columns.get_loc("gaussian_mean")] = mu_cpu[i]
            adata.var.iloc[g, adata.var.columns.get_loc("gaussian_std")] = sigma_cpu[i]
            adata.var.iloc[g, adata.var.columns.get_loc("mix_probs_0")] = pi0_cpu[i]
            adata.var.iloc[g, adata.var.columns.get_loc("mix_probs_1")] = (
                1.0 - pi0_cpu[i]
            )
            adata.var.iloc[g, adata.var.columns.get_loc("threshold")] = thresholds_cpu[
                i
            ]
            adata.var.iloc[g, adata.var.columns.get_loc("weight_Poisson")] = pi0_cpu[i]
            adata.var.iloc[g, adata.var.columns.get_loc("weight_Normal")] = (
                1.0 - pi0_cpu[i]
            )
            adata.var.iloc[g, adata.var.columns.get_loc("lambda")] = lam_cpu[i]
            adata.var.iloc[g, adata.var.columns.get_loc("mu")] = mu_cpu[i]
            adata.var.iloc[g, adata.var.columns.get_loc("scale")] = sigma_cpu[i]

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


def _prepare_batched_data(
    X: cp.ndarray,
    n_guides: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, list[int]]:
    """Build padded arrays for batched EM across guides.

    Returns
    -------
    data_pad
        ``(n_valid_guides, max_nnz)`` log₂ of non-zero counts, zero-padded.
    mask
        ``(n_valid_guides, max_nnz)`` boolean validity mask.
    cell_indices
        ``(n_valid_guides, max_nnz)`` original cell indices.
    counts
        ``(n_valid_guides,)`` number of non-zero cells per guide.
    valid_guides
        Column indices of guides with >= 2 non-zero cells.
    """
    valid_guides = []
    nnz_per_guide = []
    nz_data = []
    nz_indices = []

    for g in range(n_guides):
        col = X[:, g]
        nz_mask = col > 0
        nz_count = int(nz_mask.sum())
        if nz_count < 2:
            if nz_count > 0:
                warnings.warn(
                    f"Skipping guide index {g} as there are less than 2 cells "
                    "expressing the guide.",
                    UserWarning,
                    stacklevel=4,
                )
            continue
        max_count = float(col.max().item())
        if max_count < 2:
            warnings.warn(
                f"Skipping guide index {g} as the maximum UMI count is less than 2.",
                UserWarning,
                stacklevel=4,
            )
            continue
        valid_guides.append(g)
        nnz_per_guide.append(nz_count)
        nz_data.append(cp.log2(col[nz_mask]))
        nz_indices.append(cp.where(nz_mask)[0])

    if len(valid_guides) == 0:
        empty = cp.empty((0, 0), dtype=cp.float32)
        return empty, empty, empty.astype(cp.int64), cp.empty(0, dtype=cp.int32), []

    max_nnz = max(nnz_per_guide)
    n_valid = len(valid_guides)

    data_pad = cp.zeros((n_valid, max_nnz), dtype=cp.float32)
    mask = cp.zeros((n_valid, max_nnz), dtype=cp.bool_)
    cell_indices = cp.zeros((n_valid, max_nnz), dtype=cp.int64)
    counts = cp.array(nnz_per_guide, dtype=cp.int32)

    for i in range(n_valid):
        n = nnz_per_guide[i]
        data_pad[i, :n] = nz_data[i].astype(cp.float32)
        mask[i, :n] = True
        cell_indices[i, :n] = nz_indices[i]

    return data_pad, mask, cell_indices, counts, valid_guides


def _assign_by_crispat_threshold(
    X: cp.ndarray,
    valid_guides: list[int],
    *,
    lam: cp.ndarray,
    mu: cp.ndarray,
    sigma: cp.ndarray,
    pi0: cp.ndarray,
    posterior_threshold: float,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Assign cells using crispat's posterior-derived raw UMI threshold."""
    assignments = cp.zeros((len(valid_guides), X.shape[0]), dtype=cp.bool_)
    thresholds = cp.full((len(valid_guides), 1), cp.nan, dtype=cp.float32)

    for i, g in enumerate(valid_guides):
        col = X[:, g]
        max_count = int(cp.ceil(col.max()).item())
        if max_count < 2:
            continue

        raw_counts = cp.arange(1, max_count + 1, dtype=cp.float32)
        log_counts = cp.log2(raw_counts).reshape(1, -1)
        threshold_mask = cp.ones_like(log_counts, dtype=cp.bool_)
        _, prob_gaussian = _e_step(
            log_counts,
            threshold_mask,
            lam=lam[i : i + 1],
            mu=mu[i : i + 1],
            sigma=sigma[i : i + 1],
            pi0=pi0[i : i + 1],
        )
        positive_counts = raw_counts[cp.ravel(prob_gaussian > posterior_threshold)]
        if positive_counts.size == 0:
            continue

        threshold = positive_counts[0]
        thresholds[i, 0] = threshold
        assignments[i] = col >= threshold

    return assignments, thresholds


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
    from rapids_singlecell._cuda import _guide_assignment_cuda

    if _guide_assignment_cuda is None:
        raise ImportError(
            "The _guide_assignment_cuda extension is not available. "
            "Build rapids-singlecell with CUDA extensions or use backend='cupy'."
        )

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


def _batched_em(
    data: cp.ndarray,
    mask: cp.ndarray,
    counts: cp.ndarray,
    *,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Run batched Poisson–Gaussian EM across all guides simultaneously.

    Parameters
    ----------
    data
        ``(n_guides, max_nnz)`` padded log₂ counts.
    mask
        ``(n_guides, max_nnz)`` validity mask.
    counts
        ``(n_guides,)`` valid cell count per guide.
    max_iter
        Maximum EM iterations.
    tol
        Convergence tolerance.

    Returns
    -------
    lam, mu, sigma, pi0
        Fitted parameters, each ``(n_guides, 1)``.
    assignments
        ``(n_guides, max_nnz)`` boolean array (``True`` = positive).
    """
    n_valid = mask.sum(axis=1, keepdims=True).astype(cp.float32)  # (n_guides, 1)

    # --- Initialization via percentiles ---
    lam, mu, sigma, pi0 = _initialize_params(data, mask, counts)

    # Prior hyperparameters (matching crispat / Braunger et al.)
    # μ ~ Normal(3, 2)
    prior_mu_mean = cp.float32(3.0)
    prior_mu_var = cp.float32(4.0)  # σ₀² = 2² = 4
    # λ ~ LogNormal(0, 1)  — mode ≈ 0.37, mean ≈ 1.65
    prior_lam_mu = cp.float32(0.0)
    prior_lam_sigma2 = cp.float32(1.0)
    # scale ~ LogNormal(2, 1)  — mode ≈ 2.72, mean ≈ 12.2
    prior_scale_mu = cp.float32(2.0)
    prior_scale_sigma2 = cp.float32(1.0)
    # π ~ Dirichlet(0.9, 0.1)
    prior_alpha0 = cp.float32(0.9)
    prior_alpha1 = cp.float32(0.1)

    for _ in range(max_iter):
        # E-step (no separation penalty — crispat doesn't use one)
        r0, r1 = _e_step(data, mask, lam=lam, mu=mu, sigma=sigma, pi0=pi0)

        # MAP M-step with prior regularization
        n0 = r0.sum(axis=1, keepdims=True)
        n1 = r1.sum(axis=1, keepdims=True)

        # λ MAP with LogNormal(μ₀, σ₀²) prior:
        # MLE: λ = Σ(r₀·x) / Σr₀
        # LogNormal prior adds -(log(λ)-μ₀)²/(2σ₀²) - log(λ) to log-posterior
        # We approximate via iterating: use MLE then shrink toward prior mode
        lam_mle = (r0 * data).sum(axis=1, keepdims=True) / cp.maximum(n0, 1e-10)
        log_lam_mle = cp.log(cp.maximum(lam_mle, 1e-10))
        # Bayesian shrinkage: weighted average in log-space
        log_lam_new = (n0 * log_lam_mle + prior_lam_mu / prior_lam_sigma2) / (
            n0 + 1.0 / prior_lam_sigma2
        )
        lam_new = cp.exp(log_lam_new)

        # μ MAP: Normal(μ₀, σ₀²) prior
        sigma_sq = sigma * sigma
        mu_new = (
            (r1 * data).sum(axis=1, keepdims=True) / cp.maximum(sigma_sq, 1e-10)
            + prior_mu_mean / prior_mu_var
        ) / (n1 / cp.maximum(sigma_sq, 1e-10) + 1.0 / prior_mu_var)

        # σ MAP with LogNormal(μ₀, σ₀²) prior:
        # Same approach as λ — MLE then shrink in log-space
        diff = data - mu_new
        sigma_sq_mle = (r1 * diff * diff).sum(axis=1, keepdims=True) / cp.maximum(
            n1, 1e-10
        )
        sigma_mle = cp.maximum(cp.sqrt(sigma_sq_mle), 1e-2)
        log_sig_mle = cp.log(sigma_mle)
        log_sig_new = (n1 * log_sig_mle + prior_scale_mu / prior_scale_sigma2) / (
            n1 + 1.0 / prior_scale_sigma2
        )
        sigma_new = cp.maximum(cp.exp(log_sig_new), 1e-2)

        # π MAP: Dirichlet(α₀, α₁) prior
        denom = n_valid + prior_alpha0 + prior_alpha1 - 2.0
        pi0_new = (n0 + prior_alpha0 - 1.0) / cp.maximum(denom, 1e-10)
        pi0_new = cp.clip(pi0_new, 0.01, 0.99)

        # Convergence check
        max_change = max(
            float(cp.abs(lam_new - lam).max()),
            float(cp.abs(mu_new - mu).max()),
            float(cp.abs(sigma_new - sigma).max()),
            float(cp.abs(pi0_new - pi0).max()),
        )

        lam, mu, sigma, pi0 = lam_new, mu_new, sigma_new, pi0_new

        if max_change < tol:
            break

    # Final assignment: cell is positive if P(Gaussian) > 0.5
    r0, r1 = _e_step(data, mask, lam=lam, mu=mu, sigma=sigma, pi0=pi0)
    assignments = r1 > 0.5

    return lam, mu, sigma, pi0, assignments


def _initialize_params(
    data: cp.ndarray,
    mask: cp.ndarray,
    counts: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Initialize EM parameters using data-driven estimates.

    Uses the data distribution per guide to set starting points that
    help the EM converge to pertpy-compatible solutions.
    """
    n_guides = data.shape[0]
    n_valid_f = mask.sum(axis=1, keepdims=True).astype(cp.float32)

    # Compute per-guide statistics on valid entries
    masked_data = data * mask
    mean_vals = masked_data.sum(axis=1, keepdims=True) / cp.maximum(n_valid_f, 1.0)

    # λ: start small — Exponential(0.2) prior has mode at 0
    # Use the overall mean as a rough guide but keep it modest
    lam = cp.minimum(mean_vals * 0.5, cp.float32(0.5))
    lam = cp.maximum(lam, cp.float32(0.01))

    # μ: Normal(3, 2) prior mean, but shift toward data if there's signal
    # Use mean of top quartile as signal estimate
    data_for_sort = cp.where(mask, data, -cp.inf)
    sorted_data = cp.sort(data_for_sort, axis=1)
    counts_f = counts.astype(cp.float32)
    idx_75 = cp.clip((counts_f * 0.75).astype(cp.int64), 0, data.shape[1] - 1)
    p75 = sorted_data[cp.arange(n_guides), idx_75].reshape(-1, 1)
    # If p75 is 0 (most data at 0), fall back to prior mean
    mu = cp.where(p75 > 0.5, p75, cp.float32(3.0))

    # σ: HalfNormal(1) prior — start at 1.0
    sigma = cp.full((n_guides, 1), 1.0, dtype=cp.float32)

    # π₀: Dirichlet(0.85, 0.15) — start at prior
    pi0 = cp.full((n_guides, 1), 0.85, dtype=cp.float32)

    return lam, mu, sigma, pi0


def _e_step(
    data: cp.ndarray,
    mask: cp.ndarray,
    *,
    lam: cp.ndarray,
    mu: cp.ndarray,
    sigma: cp.ndarray,
    pi0: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute responsibilities for Poisson and Gaussian components."""
    # Poisson log-PMF: x*log(λ) - λ - gammaln(x+1)
    log_lam = cp.log(cp.maximum(lam, 1e-10))
    log_p0 = data * log_lam - lam - gammaln(data + 1.0) + cp.log(cp.maximum(pi0, 1e-10))

    # Gaussian log-PDF: -0.5*((x-μ)/σ)^2 - log(σ) - 0.5*log(2π)
    z = (data - mu) / cp.maximum(sigma, 1e-10)
    log_sigma = cp.log(cp.maximum(sigma, 1e-10))
    log_p1 = (
        -0.5 * z * z - log_sigma - 0.5 * _LOG_2PI + cp.log(cp.maximum(1.0 - pi0, 1e-10))
    )

    # Numerically stable softmax
    log_total = cp.logaddexp(log_p0, log_p1)
    r0 = cp.exp(log_p0 - log_total)
    r1 = 1.0 - r0

    # Zero out padding
    r0 = r0 * mask
    r1 = r1 * mask

    return r0, r1
