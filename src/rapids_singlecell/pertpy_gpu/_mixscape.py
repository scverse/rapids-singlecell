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

# Posterior probability of the perturbed component above which a cell is called
# knocked out (matches Seurat/pertpy).
_KO_POSTERIOR_CUTOFF = 0.5
# Iterations for the per-gene Gaussian mixture fit (pertpy default).
_GMM_MAX_ITER = 100
# Below this many variables ``perturbation_signature`` uses ``.X`` rather than a
# PCA representation when ``use_rep`` is not given (mirrors scanpy).
_NN_REPRESENTATION_AUTO_MAX_VARS = 50
# Sentinel groupby label for cells outside the current split during differential
# expression, so they are excluded without copying/subsetting the matrix.
_DE_IGNORE_LABEL = "__mixscape_other__"


class Mixscape:
    """GPU-accelerated Mixscape for pooled CRISPR screens.

    Identifies cells with a detectable perturbation effect and separates them
    from cells that escaped perturbation, following Seurat's Mixscape and
    pertpy's :class:`~pertpy.tools.Mixscape`. The perturbation signature and the
    iterative Gaussian-mixture classification run on the GPU; every gene's
    spherical, fixed-control-component mixture is fit in a single batched CUDA
    kernel (one block per gene), via ``_gmm_cuda.mixscape_project_em``.
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

    def mixscape(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        new_class_name: str = "mixscape_class",
        layer: str | None = None,
        min_de_genes: int = 5,
        logfc_threshold: float = 0.25,
        de_layer: str | None = None,
        test_method: str = "wilcoxon",
        iter_num: int = 10,
        scale: bool = True,
        split_by: str | None = None,
        pval_cutoff: float = 5e-2,
        perturbation_type: str = "KO",
        random_state: int = 0,
        copy: bool = False,
    ) -> AnnData | None:
        """Identify perturbed and escaping cells per target gene.

        For each target gene, differentially expressed genes are found against
        the control, the perturbation signature is projected onto the
        gene-specific perturbation direction, and a two-component spherical
        Gaussian mixture (with the control component held fixed) iteratively
        separates knocked-out (perturbed) from non-perturbed cells.

        Parameters
        ----------
        adata
            The annotated data object.
        pert_key
            The column of ``.obs`` with target gene labels.
        control
            Control category in ``pert_key``.
        new_class_name
            Name of the ``.obs`` column for the classification result.
        layer
            Layer used for the mixture. Defaults to ``.layers["X_pert"]``.
        min_de_genes
            Minimum number of differentially expressed genes required to test a
            gene for perturbation.
        logfc_threshold
            Minimum absolute log fold change for a gene to count as
            differentially expressed.
        de_layer
            Layer used for differential expression. ``None`` uses ``.X``.
        test_method
            Differential-expression test passed to
            :func:`rapids_singlecell.tl.rank_genes_groups`.
        iter_num
            Maximum number of refinement iterations.
        scale
            Scale the mixture input before fitting.
        split_by
            ``.obs`` column with a condition/cell-type annotation if
            perturbations are condition specific.
        pval_cutoff
            Adjusted p-value cutoff for differentially expressed genes.
        perturbation_type
            Label suffix used for perturbed cells (e.g. ``"KO"``).
        random_state
            Accepted for ``pertpy.tl.Mixscape`` API compatibility; has no
            effect, as the spherical mixture is initialized deterministically
            from the per-gene projection statistics.
        copy
            Whether to return a copy of ``adata``.

        Returns
        -------
        Returns the modified copy if ``copy=True``, otherwise writes the results
        to ``adata`` in place and returns ``None``. The results are
        ``adata.obs[new_class_name]`` (per-gene classification),
        ``adata.obs[f"{new_class_name}_global"]`` (perturbed/NP/NT),
        ``adata.obs[f"{new_class_name}_p_{perturbation_type.lower()}"]``
        (posterior probability) and ``adata.uns["mixscape"]``.
        """
        if copy:
            adata = adata.copy()

        if split_by is None:
            split_masks = [np.ones(adata.n_obs, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = list(split_obs.unique())
            split_masks = [
                (split_obs == category).to_numpy() for category in categories
            ]

        perturbation_markers = self._get_perturbation_markers(
            adata,
            split_masks=split_masks,
            categories=categories,
            pert_key=pert_key,
            control=control,
            layer=de_layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
            test_method=test_method,
        )

        if layer is not None:
            X = X_to_GPU(adata.layers[layer])
        else:
            try:
                X = X_to_GPU(adata.layers["X_pert"])
            except KeyError:
                raise KeyError(
                    "No 'X_pert' found in .layers! Please run perturbation_signature "
                    "first to calculate the perturbation signature!"
                ) from None
        X = _to_dense_gpu(X)

        p_col = f"{new_class_name}_p_{perturbation_type.lower()}"
        gv_list: dict[str, dict] = {}

        var_names = np.asarray(adata.var_names)
        # name -> column index, so per-gene DE markers map to indices in O(k)
        # instead of an O(n_vars) ``np.isin`` scan per gene.
        var_to_idx = {name: i for i, name in enumerate(var_names)}
        # Integer codes let every per-gene cell mask be built with a fast integer
        # comparison over the full obs instead of an object-dtype string compare,
        # and no cells are ever copied/subsetted out of ``adata``.
        pert_cat = adata.obs[pert_key].astype("category")
        codes = pert_cat.cat.codes.to_numpy()
        cat_names = [str(c) for c in pert_cat.cat.categories]
        control_code = {name: i for i, name in enumerate(cat_names)}.get(control, -1)
        obs_names_arr = adata.obs_names.to_numpy()

        # Results accumulate in flat arrays and are written to ``adata.obs`` once
        # at the end, instead of per-gene ``.loc`` assignments into object Series.
        class_arr = np.asarray(adata.obs[pert_key].astype(str), dtype=object).copy()
        p_arr = np.zeros(adata.n_obs, dtype=float)

        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            nt_mask = (codes == control_code) & split_mask
            split_gene_codes = sorted(set(codes[split_mask].tolist()) - {control_code})

            # Build one job per gene with enough DE markers and control cells,
            # then refine all of them together (one GMM kernel launch per outer
            # iteration) instead of a per-gene Python loop of GMM fits.
            gene_jobs: list[dict] = []
            for gene_code in split_gene_codes:
                gene = cat_names[gene_code]
                orig_mask = (codes == gene_code) & split_mask

                de_genes = perturbation_markers[category, gene]
                if len(de_genes) == 0:
                    class_arr[orig_mask] = f"{gene} NP"
                    continue

                all_mask = orig_mask | nt_mask
                nt_in_all_np = nt_mask[all_mask]
                if not nt_in_all_np.any():
                    # No control cells in this split: cannot define a
                    # perturbation direction, so leave every guide cell NP.
                    class_arr[orig_mask] = f"{gene} NP"
                    continue

                de_genes_indices = np.fromiter(
                    (var_to_idx[g] for g in de_genes if g in var_to_idx),
                    dtype=np.int64,
                )
                # Gather the (n_all, k) sub-matrix directly; X[all_mask][:, de]
                # would first materialize the full (n_all, n_vars) row slice.
                all_idx = cp.asarray(np.flatnonzero(all_mask))
                dat = X[all_idx[:, None], cp.asarray(de_genes_indices)[None, :]]
                if scale:
                    dat = _scale_gpu(dat)
                orig_in_all_np = orig_mask[all_mask]
                nt_in_all = cp.asarray(nt_in_all_np)
                gene_jobs.append(
                    {
                        "gene": gene,
                        "category": category,
                        "orig_mask": orig_mask,
                        "dat": dat,
                        "nt_in_all": nt_in_all,
                        "orig_in_all": cp.asarray(orig_in_all_np),
                        "orig_in_all_np": orig_in_all_np,
                        "nt_cells_mean": dat[nt_in_all].mean(axis=0),
                        "n_all": int(orig_in_all_np.shape[0]),
                        "n_orig": int(orig_in_all_np.sum()),
                        "all_index": obs_names_arr[all_mask],
                        "orig_index": obs_names_arr[orig_mask],
                    }
                )

            if not gene_jobs:
                continue
            self._fit_genes_batched(
                gene_jobs,
                iter_num=iter_num,
                min_de_genes=min_de_genes,
                perturbation_type=perturbation_type,
                pert_key=pert_key,
                control=control,
                gv_list=gv_list,
            )
            for job in gene_jobs:
                class_arr[job["orig_mask"]] = job["labels"]
                p_arr[job["orig_mask"]] = job["post_prob"]

        adata.obs[new_class_name] = class_arr
        adata.obs[p_col] = p_arr

        # The global class is just the last token of each class label and is only
        # read after every gene is processed, so derive it once here (pertpy
        # recomputes it inside the per-gene loop, an O(n_obs x n_perts) cost).
        adata.obs[f"{new_class_name}_global"] = (
            adata.obs[new_class_name].astype(str).str.split(" ").str[-1]
        )
        adata.uns["mixscape"] = gv_list

        if copy:
            return adata
        return None

    def mixscale(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        new_class_name: str = "mixscale_score",
        layer: str | None = None,
        min_de_genes: int = 5,
        max_de_genes: int = 100,
        logfc_threshold: float = 0.25,
        de_layer: str | None = None,
        test_method: str = "wilcoxon",
        scale: bool = True,
        split_by: str | None = None,
        pval_cutoff: float = 5e-2,
        perturbation_type: str = "KO",
        copy: bool = False,
    ) -> AnnData | None:
        """Continuous perturbation efficiency scores (Mixscale).

        Unlike :meth:`mixscape`, which performs a binary knocked-out/non-perturbed
        classification with a Gaussian mixture, this assigns each cell a
        *continuous* perturbation-efficiency score: the scalar projection of its
        perturbation signature onto the per-gene perturbation direction
        (mean perturbed minus mean control), z-score standardized relative to the
        non-targeting control distribution. This is useful for CRISPRi/CRISPRa
        screens where cells show a gradient of perturbation strength rather than a
        binary knockout. Control cells receive a score of 0.

        Implements Jiang, Dalgarno et al., "Systematic reconstruction of molecular
        pathway signatures using scalable single-cell perturbation screens",
        Nature Cell Biology (2025), following pertpy's
        :meth:`~pertpy.tools.Mixscape.mixscale`.

        Parameters
        ----------
        adata
            The annotated data object.
        pert_key
            The column of ``.obs`` with target gene labels.
        control
            Control category in ``pert_key``.
        new_class_name
            Name of the ``.obs`` column for the continuous score.
        layer
            Layer used for scoring. Defaults to ``.layers["X_pert"]``.
        min_de_genes
            Minimum number of differentially expressed genes required to score a
            gene; genes with fewer are skipped.
        max_de_genes
            Maximum number of (top-ranked) differentially expressed genes used to
            define the perturbation direction.
        logfc_threshold
            Minimum absolute log fold change for a gene to count as
            differentially expressed.
        de_layer
            Layer used for differential expression. ``None`` uses ``.X``.
        test_method
            Differential-expression test passed to
            :func:`rapids_singlecell.tl.rank_genes_groups`.
        scale
            Scale the per-gene sub-matrix before computing scores.
        split_by
            ``.obs`` column with a condition/cell-type annotation if
            perturbations are condition specific.
        pval_cutoff
            Adjusted p-value cutoff for differentially expressed genes.
        perturbation_type
            Accepted for ``pertpy.tl.Mixscape.mixscale`` API compatibility; has no
            effect on the continuous score.
        copy
            Whether to return a copy of ``adata``.

        Returns
        -------
        Returns the modified copy if ``copy=True``, otherwise writes
        ``adata.obs[new_class_name]`` in place and returns ``None``. Higher
        absolute values indicate a stronger perturbation effect; control cells
        and any gene that cannot be scored receive 0.
        """
        if copy:
            adata = adata.copy()

        if split_by is None:
            split_masks = [np.ones(adata.n_obs, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = list(split_obs.unique())
            split_masks = [
                (split_obs == category).to_numpy() for category in categories
            ]

        perturbation_markers = self._get_perturbation_markers(
            adata,
            split_masks=split_masks,
            categories=categories,
            pert_key=pert_key,
            control=control,
            layer=de_layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
            test_method=test_method,
        )

        if layer is not None:
            X = X_to_GPU(adata.layers[layer])
        else:
            try:
                X = X_to_GPU(adata.layers["X_pert"])
            except KeyError:
                raise KeyError(
                    "No 'X_pert' found in .layers! Please run perturbation_signature "
                    "first to calculate the perturbation signature!"
                ) from None
        X = _to_dense_gpu(X)

        var_names = np.asarray(adata.var_names)
        var_to_idx = {name: i for i, name in enumerate(var_names)}
        # Integer codes let every per-gene cell mask be built with a fast integer
        # comparison over the full obs, and no cells are ever copied out of adata.
        pert_cat = adata.obs[pert_key].astype("category")
        codes = pert_cat.cat.codes.to_numpy()
        cat_names = [str(c) for c in pert_cat.cat.categories]
        control_code = {name: i for i, name in enumerate(cat_names)}.get(control, -1)

        # One job per scorable gene (indices + masks, cheap host work); the
        # gather, scaling, projection and z-score for all genes then run in a
        # single batched kernel that reads each block straight from X.
        gene_jobs: list[dict] = []
        for split, split_mask in enumerate(split_masks):
            category = categories[split]
            nt_mask = (codes == control_code) & split_mask
            split_gene_codes = sorted(set(codes[split_mask].tolist()) - {control_code})

            for gene_code in split_gene_codes:
                gene = cat_names[gene_code]
                de_genes = perturbation_markers[category, gene]
                if len(de_genes) == 0:
                    continue
                # Keep only the top-ranked DE genes (markers are score-ranked).
                if len(de_genes) > max_de_genes:
                    de_genes = de_genes[:max_de_genes]
                de_genes_indices = np.fromiter(
                    (var_to_idx[g] for g in de_genes if g in var_to_idx),
                    dtype=np.int32,
                )
                if de_genes_indices.size == 0:
                    continue

                orig_mask = (codes == gene_code) & split_mask
                all_mask = orig_mask | nt_mask
                nt_in_all_np = nt_mask[all_mask]
                if not nt_in_all_np.any():
                    # No control cells: cannot define a perturbation direction.
                    continue

                gene_jobs.append(
                    {
                        "row_ids": np.flatnonzero(all_mask).astype(np.int32),
                        "col_ids": de_genes_indices,
                        "is_guide": orig_mask[all_mask],
                        "nt_in_all": nt_in_all_np,
                    }
                )

        # Control cells stay 0; the kernel writes only guide cells. float64
        # output matches pertpy's ``obs[...] = 0.0`` column.
        scores_gpu = cp.zeros(adata.n_obs, dtype=X.dtype)
        if gene_jobs:
            _project_scores_batched(X, gene_jobs, scores_gpu, do_scale=scale)
        adata.obs[new_class_name] = cp.asnumpy(scores_gpu).astype(
            np.float64, copy=False
        )

        if copy:
            return adata
        return None

    def lda(
        self,
        adata: AnnData,
        pert_key: str,
        control: str,
        *,
        mixscape_class_global: str = "mixscape_class_global",
        layer: str | None = None,
        n_comps: int = 10,
        min_de_genes: int = 5,
        logfc_threshold: float = 0.25,
        test_method: str = "wilcoxon",
        split_by: str | None = None,
        pval_cutoff: float = 5e-2,
        perturbation_type: str = "KO",
        copy: bool = False,
    ) -> AnnData | None:
        """Linear discriminant analysis on the mixscape result.

        Requires :meth:`mixscape` to have been run. For each perturbed gene, a
        PCA is fit on its differentially expressed genes and all perturbed and
        control cells are projected into that subspace; the concatenated
        projections are then reduced with a GPU linear discriminant analysis
        (a CuPy port of scikit-learn's SVD solver). The embedding is written to
        ``adata.uns["mixscape_lda"]``.

        Parameters
        ----------
        adata
            The annotated data object.
        pert_key
            The column of ``.obs`` with target gene labels.
        control
            Control category in ``pert_key``.
        mixscape_class_global
            The ``.obs`` column with the global mixscape classification.
        layer
            Layer used for differential expression. ``None`` uses ``.X``.
        n_comps
            Number of principal components per gene subspace.
        min_de_genes
            Minimum number of differentially expressed genes to test a gene.
        logfc_threshold
            Minimum absolute log fold change for a differentially expressed gene.
        test_method
            Differential-expression test passed to
            :func:`rapids_singlecell.tl.rank_genes_groups`.
        split_by
            ``.obs`` column with a condition/cell-type annotation if
            perturbations are condition specific.
        pval_cutoff
            Adjusted p-value cutoff for differentially expressed genes.
        perturbation_type
            Label used for perturbed cells (e.g. ``"KO"``).
        copy
            Whether to return a copy of ``adata``.

        Returns
        -------
        Returns the modified copy if ``copy=True``, otherwise writes
        ``adata.uns["mixscape_lda"]`` in place and returns ``None``.
        """
        if copy:
            adata = adata.copy()
        if mixscape_class_global not in adata.obs:
            raise ValueError("Please run the `mixscape` method first.")

        if split_by is None:
            split_masks = [np.ones(adata.n_obs, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = list(split_obs.unique())
            split_masks = [(split_obs == c).to_numpy() for c in categories]

        perturbation_markers = self._get_perturbation_markers(
            adata,
            split_masks=split_masks,
            categories=categories,
            pert_key=pert_key,
            control=control,
            layer=layer,
            pval_cutoff=pval_cutoff,
            min_de_genes=min_de_genes,
            logfc_threshold=logfc_threshold,
            test_method=test_method,
        )

        from cuml.decomposition import PCA

        global_class = adata.obs[mixscape_class_global].to_numpy()
        subset_mask = (global_class == perturbation_type) | (global_class == control)
        X_sub = _to_dense_gpu(_get_obs_rep(adata))[cp.asarray(subset_mask)]
        X_proj_base = X_sub - X_sub.mean(axis=0)
        pert_sub = adata.obs[pert_key].astype(str).to_numpy()[subset_mask]

        projected = []
        for (_category, gene), de_genes in perturbation_markers.items():
            if len(de_genes) == 0:
                continue
            gene_mask = (pert_sub == gene) | (pert_sub == control)
            gene_subset = _scale_gpu(X_sub[cp.asarray(gene_mask)])
            n_pcs = min(n_comps, min(gene_subset.shape) - 1)
            if n_pcs < 1:
                # Too few cells or genes for a meaningful per-gene PCA.
                continue
            pca = PCA(n_components=n_pcs)
            pca.fit(gene_subset)
            loadings = cp.asarray(pca.components_, dtype=X_proj_base.dtype).T
            projected.append(X_proj_base @ loadings)

        if len(projected) == 0:
            raise ValueError(
                "No perturbation had enough differentially expressed genes for LDA."
            )

        projected_pcs = cp.concatenate(projected, axis=1)
        codes = cp.asarray(pd.factorize(pert_sub)[0])
        n_components = len(np.unique(pert_sub)) - 1
        embedding = _lda_fit_transform_gpu(
            projected_pcs, codes, n_components=n_components
        )
        adata.uns["mixscape_lda"] = cp.asnumpy(embedding)

        if copy:
            return adata
        return None

    def _fit_genes_batched(
        self,
        gene_jobs: list[dict],
        *,
        iter_num: int,
        min_de_genes: int,
        perturbation_type: str,
        pert_key: str,
        control: str,
        gv_list: dict,
    ) -> None:
        """Batched per-gene iterative mixture refinement (fully on GPU).

        Each gene's projection (``dat @ vec``), control/guide statistics and
        two-component spherical EM (control component pinned) run in one CUDA
        block via :func:`mixscape_project_em`; a single launch per outer
        iteration refines every still-active gene (an ``active_genes`` index
        list skips converged genes). Only the cheap label update / convergence
        check stays on the host. Each ``gene_jobs`` entry is updated in place
        with ``labels`` and ``post_prob``.
        """
        from rapids_singlecell._cuda import _gmm_cuda as _gc

        n_genes = len(gene_jobs)
        dtype = gene_jobs[0]["dat"].dtype
        n_list = np.array([job["n_all"] for job in gene_jobs], dtype=np.int64)
        k_list = np.array(
            [int(job["dat"].shape[1]) for job in gene_jobs], dtype=np.int64
        )
        cell_off = np.concatenate([[0], np.cumsum(n_list)]).astype(np.int32)
        feat_off = np.concatenate([[0], np.cumsum(k_list)]).astype(np.int32)
        dat_off = np.concatenate([[0], np.cumsum(n_list * k_list)]).astype(np.int64)
        total_cells = int(cell_off[-1])
        max_k = int(k_list.max())

        # Flat, gene-blocked device buffers: gene g owns dat[dat_off[g]:...] as
        # an (n_g, k_g) row-major block and cells [cell_off[g]:cell_off[g+1]].
        dat_flat = cp.empty(int(dat_off[-1]), dtype=dtype)
        ntm_flat = cp.empty(int(feat_off[-1]), dtype=dtype)
        guide_sel = cp.zeros(total_cells, dtype=cp.bool_)
        nt_flat = cp.zeros(total_cells, dtype=cp.bool_)
        orig_pos_list: list[np.ndarray] = []
        n_orig_list: list[int] = []
        for gi, job in enumerate(gene_jobs):
            n, k = int(n_list[gi]), int(k_list[gi])
            sd, sf, sc = int(dat_off[gi]), int(feat_off[gi]), int(cell_off[gi])
            dat_flat[sd : sd + n * k] = cp.ascontiguousarray(job["dat"]).ravel()
            ntm_flat[sf : sf + k] = job["nt_cells_mean"]
            nt_flat[sc : sc + n] = job["nt_in_all"]
            orig_local = np.flatnonzero(job["orig_in_all_np"])
            orig_pos_list.append(sc + orig_local)
            n_orig_list.append(int(orig_local.shape[0]))
            job["dat"] = None  # freed; the matrix now lives in dat_flat

        orig_pos_all = np.concatenate(orig_pos_list).astype(np.int32)
        orig_seg = np.concatenate([[0], np.cumsum(n_orig_list)]).astype(np.int64)
        guide_sel[cp.asarray(orig_pos_all)] = True  # initial: every guide cell
        orig_pos_all_gpu = cp.asarray(orig_pos_all)

        pvec_scratch = cp.empty(total_cells, dtype=dtype)
        resp1 = cp.empty(total_cells, dtype=dtype)
        dat_off_g = cp.asarray(dat_off)
        n_pg = cp.asarray(n_list.astype(np.int32))
        k_pg = cp.asarray(k_list.astype(np.int32))
        cell_off_g = cp.asarray(cell_off)
        feat_off_g = cp.asarray(feat_off)

        converged = np.zeros(n_genes, dtype=bool)
        np_collapse = np.zeros(n_genes, dtype=bool)
        prev_ko = [np.ones(n, dtype=bool) for n in n_orig_list]
        ko_final = [np.ones(n, dtype=bool) for n in n_orig_list]
        post_final: list = [0.0] * n_genes

        for outer in range(iter_num):
            active_idx = np.flatnonzero(~converged)
            if active_idx.size == 0:
                break
            _gc.mixscape_project_em(
                dat_flat,
                dat_off_g,
                n_pg,
                k_pg,
                cell_off_g,
                feat_off_g,
                ntm_flat,
                guide_sel,
                nt_flat,
                cp.asarray(active_idx.astype(np.int32)),
                pvec_scratch,
                resp1,
                n_active=int(active_idx.size),
                max_k=max_k,
                max_iter=_GMM_MAX_ITER,
                tol=1e-3,
                reg_covar=1e-6,
                stream=cp.cuda.get_current_stream().ptr,
            )

            if outer == 0:
                # uns["mixscape"] plotting frame, built once from the first pvec.
                pvec_host = cp.asnumpy(pvec_scratch)
                for gi, job in enumerate(gene_jobs):
                    sc, n = int(cell_off[gi]), int(n_list[gi])
                    pert_col = np.full(n, control, dtype=object)
                    pert_col[job["orig_in_all_np"]] = job["gene"]
                    gv = pd.DataFrame(
                        {"pvec": pvec_host[sc : sc + n], pert_key: pert_col},
                        index=job["all_index"],
                    )
                    gv_list.setdefault(job["gene"], {})[job["category"]] = gv

            # Component-1 posterior on every gene's guide cells (one transfer);
            # undefined entries (0/0 underflow) -> 0, matching pertpy.
            post_all = np.nan_to_num(cp.asnumpy(resp1[orig_pos_all_gpu]), nan=0.0)
            upd_pos, upd_ko = [], []
            for gi in active_idx:
                post = post_all[orig_seg[gi] : orig_seg[gi + 1]]
                ko = post > _KO_POSTERIOR_CUTOFF
                post_final[gi] = post
                if int(ko.sum()) < min_de_genes:
                    np_collapse[gi] = True
                    converged[gi] = True
                elif np.array_equal(ko, prev_ko[gi]):
                    converged[gi] = True
                ko_final[gi] = ko
                prev_ko[gi] = ko
                upd_pos.append(orig_pos_list[gi])
                upd_ko.append(ko)
            if upd_pos:
                guide_sel[cp.asarray(np.concatenate(upd_pos))] = cp.asarray(
                    np.concatenate(upd_ko)
                )

        for gi, job in enumerate(gene_jobs):
            np_label = "NP" if np_collapse[gi] else f"{job['gene']} {perturbation_type}"
            job["labels"] = np.where(ko_final[gi], np_label, f"{job['gene']} NP")
            post = post_final[gi]
            job["post_prob"] = post if not isinstance(post, float) else 0.0

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


def _scale_gpu(dat: cp.ndarray) -> cp.ndarray:
    """Zero-center and unit-scale columns (matches ``scanpy.pp.scale``)."""
    mean = dat.mean(axis=0)
    std = cp.sqrt(dat.var(axis=0, ddof=1))
    std = cp.where(std == 0, dat.dtype.type(1.0), std)
    return (dat - mean) / std


def _project_scores_batched(
    X: cp.ndarray, gene_jobs: list[dict], scores_gpu: cp.ndarray, *, do_scale: bool
) -> None:
    """Batched Mixscale scoring for all genes in one kernel launch.

    Flattens each gene's cell indices, DE-gene indices and guide/control masks
    into gene-blocked buffers and runs ``_mixscale_cuda.project_score`` once,
    writing each guide cell's standardized score into ``scores_gpu`` at its obs
    index (controls stay 0; the scatter is race-free, one gene per guide cell).
    """
    from rapids_singlecell._cuda import _mixscale_cuda as _msc

    # Kernel indexes X row-major; ensure C-order (no-op if already contiguous).
    X = cp.ascontiguousarray(X)

    n_genes = len(gene_jobs)
    n_list = np.array([job["row_ids"].shape[0] for job in gene_jobs], dtype=np.int64)
    k_list = np.array([job["col_ids"].shape[0] for job in gene_jobs], dtype=np.int64)
    cell_off = np.concatenate([[0], np.cumsum(n_list)]).astype(np.int32)
    feat_off = np.concatenate([[0], np.cumsum(k_list)]).astype(np.int32)
    max_k = int(k_list.max())

    row_ids = cp.asarray(np.concatenate([job["row_ids"] for job in gene_jobs]))
    col_ids = cp.asarray(np.concatenate([job["col_ids"] for job in gene_jobs]))
    is_guide = cp.asarray(np.concatenate([job["is_guide"] for job in gene_jobs]))
    nt_flat = cp.asarray(np.concatenate([job["nt_in_all"] for job in gene_jobs]))
    # double scratch: the kernel accumulates projections/stats in 64-bit so
    # float32 inputs keep full precision (matches mean_var's accumulation).
    pvec_scratch = cp.empty(int(cell_off[-1]), dtype=cp.float64)

    _msc.project_score(
        X,
        int(X.shape[1]),
        row_ids,
        col_ids,
        cp.asarray(n_list.astype(np.int32)),
        cp.asarray(k_list.astype(np.int32)),
        cp.asarray(cell_off),
        cp.asarray(feat_off),
        is_guide,
        nt_flat,
        pvec_scratch,
        scores_gpu,
        n_genes=n_genes,
        max_k=max_k,
        do_scale=do_scale,
        stream=cp.cuda.get_current_stream().ptr,
    )


def _lda_fit_transform_gpu(
    X: cp.ndarray, y: cp.ndarray, *, n_components: int, tol: float = 1e-4
) -> cp.ndarray:
    """Fit-transform a linear discriminant analysis on the GPU.

    A CuPy port of scikit-learn's ``LinearDiscriminantAnalysis`` SVD solver
    (BSD-3-Clause): two SVDs (within-class whitening, then between-class
    rotation) followed by the ``(X - xbar) @ scalings`` projection. ``y`` holds
    integer class codes. Embedding columns may be sign-flipped relative to
    scikit-learn (an inherent SVD freedom), but span the same subspace.
    """
    n_samples, _ = X.shape
    class_list = cp.unique(y).tolist()
    n_classes = len(class_list)
    dtype = X.dtype

    means = cp.stack([X[y == c].mean(axis=0) for c in class_list])
    priors = cp.asarray(
        [float((y == c).sum()) / n_samples for c in class_list], dtype=dtype
    )
    xbar = priors @ means
    centered = cp.concatenate(
        [X[y == c] - means[i] for i, c in enumerate(class_list)], axis=0
    )

    std = cp.std(centered, axis=0)
    std = cp.where(std == 0, dtype.type(1.0), std)
    fac = dtype.type(1.0 / (n_samples - n_classes))
    within = cp.sqrt(fac) * (centered / std)
    _, s, vt = cp.linalg.svd(within, full_matrices=False)
    rank = int(cp.sum(s > tol).item())
    scalings = (vt[:rank] / std).T / s[:rank]

    between_fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)
    between = (
        (cp.sqrt((n_samples * priors) * between_fac)) * (means - xbar).T
    ).T @ scalings
    _, s2, vt2 = cp.linalg.svd(between, full_matrices=False)
    rank2 = 0 if s2.size == 0 else int(cp.sum(s2 > tol * s2[0]).item())
    if rank < 1 or rank2 < 1:
        raise ValueError(
            "LDA failed: the within- or between-class scatter is rank-deficient "
            "(no discriminant directions found). This usually means too few "
            "cells, collinear features, or insufficient class separation."
        )
    scalings = scalings @ vt2.T[:, :rank2]

    embedding = (X - xbar) @ scalings
    max_components = min(n_components, embedding.shape[1])
    return embedding[:, :max_components]


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
