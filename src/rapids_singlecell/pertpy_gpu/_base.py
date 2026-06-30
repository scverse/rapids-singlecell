from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import issparse as cp_issparse
from scanpy.get import _get_obs_rep, _set_obs_rep

from rapids_singlecell.get import X_to_GPU

if TYPE_CHECKING:
    from anndata import AnnData

# Below this many variables ``perturbation_signature`` uses ``.X`` rather than a
# PCA representation when ``use_rep`` is not given (mirrors scanpy).
_NN_REPRESENTATION_AUTO_MAX_VARS = 50
# Sentinel groupby label for cells outside the current split during differential
# expression, so they are excluded without copying/subsetting the matrix.
_DE_IGNORE_LABEL = "__mixscape_other__"


class PerturbationEfficacyAnalyzer:
    """Shared substrate for the GPU perturbation-efficacy tools.

    Holds the steps that both the binary
    :class:`~rapids_singlecell.ptg.Mixscape` classification and the continuous
    :class:`~rapids_singlecell.ptg.Mixscale` scoring build on: computing the
    perturbation signature and detecting the differentially expressed marker
    genes per perturbation.
    """

    def perturbation_signature(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        ref_selection_mode: Literal["nn", "split_by"] = "nn",
        split_by: str | None = None,
        n_neighbors: int = 20,
        use_rep: str | None = None,
        n_dims: int | None = 15,
        n_pcs: int | None = None,
        knn_algorithm: str = "brute",
        knn_kwargs: dict | None = None,
        copy: bool = False,
    ) -> AnnData | None:
        """Calculate the perturbation signature.

        The perturbation signature replaces each cell's expression with the
        residual against comparable control cells, removing confounding
        variation so that what remains reflects the perturbation. The result is
        written to ``adata.layers["X_pert"]``. As in the original
        implementation, this is intended to run on unscaled log-normalized data.

        Parameters
        ----------
        adata
            The annotated data object.
        pert_key
            The column of ``.obs`` with perturbation categories; must also
            contain ``control``.
        control
            Name of the control category in ``pert_key``.
        ref_selection_mode
            How reference cells are selected. ``"nn"`` uses the ``n_neighbors``
            nearest control cells in the chosen representation; ``"split_by"``
            uses all control cells within the same ``split_by`` group.
        split_by
            Column of ``.obs`` used to compute the signature separately per
            group (e.g. biological replicate). Required for
            ``ref_selection_mode="split_by"``.
        n_neighbors
            Number of control neighbors used for ``ref_selection_mode="nn"``.
            Capped to the number of control cells available in each split, so a
            split with fewer controls than ``n_neighbors`` still runs (pertpy
            would error).
        use_rep
            Representation to use for neighbor selection. ``"X"`` or any
            ``.obsm`` key. If ``None``, ``.X`` is used when ``n_vars`` is below
            50, otherwise ``"X_pca"`` (computed if absent).
        n_dims
            Number of dimensions of the representation to use. ``None`` uses all.
        n_pcs
            Number of principal components to compute if a PCA representation is
            built.
        knn_algorithm
            Nearest-neighbor backend for ``ref_selection_mode="nn"``: ``"brute"``
            (exact, default), or the approximate cuVS backends ``"ivfflat"``,
            ``"cagra"``, ``"ivfpq"`` which are much faster for large datasets.
        knn_kwargs
            Extra parameters for the approximate backends (e.g. ``n_lists`` /
            ``n_probes`` for ``"ivfflat"``).
        copy
            Whether to return a copy of ``adata``.

        Returns
        -------
        Returns the modified copy if ``copy=True``, otherwise writes
        ``adata.layers["X_pert"]`` in place and returns ``None``.
        """
        if ref_selection_mode not in ("nn", "split_by"):
            raise ValueError("ref_selection_mode must be either 'nn' or 'split_by'.")
        if ref_selection_mode == "split_by" and split_by is None:
            raise ValueError(
                "split_by must be provided if ref_selection_mode is 'split_by'."
            )

        if copy:
            adata = adata.copy()

        X = _to_dense_gpu(_get_obs_rep(adata))
        control_mask = (adata.obs[pert_key] == control).to_numpy()
        X_pert = X.copy()

        if ref_selection_mode == "split_by":
            for split in adata.obs[split_by].unique():
                split_mask = (adata.obs[split_by] == split).to_numpy()
                control_group_np = control_mask & split_mask
                if not control_group_np.any():
                    warnings.warn(
                        f"No control cells in split {split!r}; leaving its "
                        "perturbation signature equal to the input.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                control_mean = X[cp.asarray(control_group_np)].mean(axis=0)
                split_dev = cp.asarray(split_mask)
                X_pert[split_dev] = control_mean[None, :] - X_pert[split_dev]
        else:
            if split_by is None:
                split_masks = [np.ones(adata.n_obs, dtype=bool)]
            else:
                split_obs = adata.obs[split_by]
                split_masks = [
                    (split_obs == cat).to_numpy() for cat in split_obs.unique()
                ]

            from rapids_singlecell.preprocessing._neighbors._neighbors import (
                KNN_ALGORITHMS,
            )

            if knn_algorithm not in KNN_ALGORITHMS:
                raise ValueError(
                    f"knn_algorithm must be one of {sorted(KNN_ALGORITHMS)}, "
                    f"got {knn_algorithm!r}"
                )
            knn = KNN_ALGORITHMS[knn_algorithm]
            algorithm_kwds = dict(knn_kwargs) if knn_kwargs is not None else {}

            representation = _choose_representation_gpu(
                adata, use_rep=use_rep, n_pcs=n_pcs
            )
            if n_dims is not None and n_dims < representation.shape[1]:
                representation = representation[:, :n_dims]
            representation = cp.ascontiguousarray(representation)

            for split_mask in split_masks:
                control_mask_split = control_mask & split_mask
                n_split = int(split_mask.sum())
                n_control = int(control_mask_split.sum())
                if n_control == 0:
                    warnings.warn(
                        "No control cells in a split; leaving its perturbation "
                        "signature equal to the input.",
                        UserWarning,
                        stacklevel=2,
                    )
                    continue
                R_split = representation[cp.asarray(split_mask)]
                R_control = representation[cp.asarray(control_mask_split)]
                k = min(n_neighbors, n_control)

                indices, _ = knn(
                    R_control,
                    R_split,
                    k,
                    metric="euclidean",
                    metric_kwds={},
                    algorithm_kwds=algorithm_kwds,
                )

                X_control = cp.expm1(X[cp.asarray(control_mask_split)])
                col_indices = indices.ravel().astype(cp.int32, copy=False)
                row_indices = cp.repeat(cp.arange(n_split, dtype=cp.int32), k)
                neigh_matrix = cp_csr_matrix(
                    (
                        cp.full(col_indices.size, 1.0 / k, dtype=X.dtype),
                        (row_indices, col_indices),
                    ),
                    shape=(n_split, n_control),
                )
                split_dev = cp.asarray(split_mask)
                X_pert[split_dev] = (
                    cp.log1p(neigh_matrix @ X_control) - X_pert[split_dev]
                )

        _set_obs_rep(adata, X_pert, layer="X_pert")

        if copy:
            return adata
        return None

    def _get_perturbation_markers(
        self,
        adata: AnnData,
        *,
        split_masks: list[np.ndarray],
        categories: list[str],
        pert_key: str,
        control: str,
        layer: str | None,
        pval_cutoff: float,
        min_de_genes: int,
        logfc_threshold: float,
        test_method: str,
    ) -> dict[tuple, np.ndarray]:
        """Differentially expressed genes per split and target gene."""
        from rapids_singlecell.tools import rank_genes_groups

        perturbation_markers: dict[tuple, np.ndarray] = {}
        pert_str = adata.obs[pert_key].astype(str).to_numpy()
        group_key = "_mixscape_de_groupby"
        rgg_key = "_mixscape_rank_genes_groups"
        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            gene_targets = sorted(set(pert_str[split_mask]) - {control})
            if len(gene_targets) == 0:
                continue
            # Mask-based grouping instead of ``adata[split_mask].copy()``: cells
            # outside the split get a sentinel label so they belong to neither a
            # target group nor the control reference. rank_genes_groups computes
            # per-group stats over the full matrix (via Aggregate); nothing is
            # copied or subsetted.
            group_labels = np.where(split_mask, pert_str, _DE_IGNORE_LABEL)
            adata.obs[group_key] = pd.Categorical(group_labels)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    rank_genes_groups(
                        adata,
                        group_key,
                        groups=gene_targets,
                        reference=control,
                        layer=layer,
                        method=test_method,
                        use_raw=False,
                        key_added=rgg_key,
                    )
                rgg = adata.uns[rgg_key]
                for gene in gene_targets:
                    logfc_mask = np.abs(rgg["logfoldchanges"][gene]) >= logfc_threshold
                    de_genes = rgg["names"][gene][logfc_mask]
                    pvals_adj = rgg["pvals_adj"][gene][logfc_mask]
                    de_genes = de_genes[pvals_adj < pval_cutoff]
                    if len(de_genes) < min_de_genes:
                        de_genes = np.array([])
                    perturbation_markers[category, gene] = de_genes
            finally:
                adata.obs.drop(columns=group_key, inplace=True, errors="ignore")
                adata.uns.pop(rgg_key, None)

        return perturbation_markers


def _to_dense_gpu(X) -> cp.ndarray:
    """Return ``X`` as a contiguous dense float CuPy array on the GPU.

    ``float32``/``float64`` are preserved (so ``float64`` input keeps pertpy's
    precision end to end); any other dtype is cast to ``float32``.
    """
    X = X_to_GPU(X)
    if X.dtype not in (cp.float32, cp.float64):
        X = X.astype(cp.float32)
    if cp_issparse(X):
        from rapids_singlecell.preprocessing._utils import _sparse_to_dense

        return _sparse_to_dense(X, order="C")
    return X


def _choose_representation_gpu(
    adata: AnnData, *, use_rep: str | None, n_pcs: int | None
) -> cp.ndarray:
    """GPU analogue of scanpy's ``_choose_representation`` for neighbor search.

    Always returns a float32 representation: neighbor selection does not need
    float64 precision, and the cuVS backends (ivfflat/cagra) only accept float32.
    """
    if use_rep is None:
        if adata.n_vars < _NN_REPRESENTATION_AUTO_MAX_VARS:
            rep = _to_dense_gpu(_get_obs_rep(adata))
        else:
            if "X_pca" not in adata.obsm or (
                n_pcs is not None and adata.obsm["X_pca"].shape[1] < n_pcs
            ):
                from rapids_singlecell.preprocessing import pca

                pca(adata, n_comps=n_pcs)
            rep = X_to_GPU(adata.obsm["X_pca"])
    elif use_rep == "X":
        rep = _to_dense_gpu(_get_obs_rep(adata))
    else:
        rep = X_to_GPU(adata.obsm[use_rep])

    rep = cp.asarray(rep, dtype=cp.float32)
    if n_pcs is not None and n_pcs < rep.shape[1]:
        rep = rep[:, :n_pcs]
    return rep
