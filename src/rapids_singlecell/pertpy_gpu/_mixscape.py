from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd
from scanpy.get import _get_obs_rep

from rapids_singlecell.get import X_to_GPU

from ._base import PerturbationEfficacyAnalyzer, _to_dense_gpu

if TYPE_CHECKING:
    from anndata import AnnData

# Posterior probability of the perturbed component above which a cell is called
# knocked out (matches Seurat/pertpy).
_KO_POSTERIOR_CUTOFF = 0.5
# Iterations for the per-gene Gaussian mixture fit (pertpy default).
_GMM_MAX_ITER = 100


class Mixscape(PerturbationEfficacyAnalyzer):
    """GPU-accelerated Mixscape for pooled CRISPR screens.

    Identifies cells with a detectable perturbation effect and separates them
    from cells that escaped perturbation, following Seurat's Mixscape and
    pertpy's :class:`~pertpy.tools.Mixscape`. The perturbation signature and the
    iterative Gaussian-mixture classification run on the GPU; every gene's
    spherical, fixed-control-component mixture is fit in a single batched CUDA
    kernel (one block per gene), via ``_gmm_cuda.mixscape_project_em``.
    """

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
                if de_genes_indices.size == 0:
                    class_arr[orig_mask] = f"{gene} NP"
                    continue
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
            Number of principal components per gene subspace. Reduced per gene
            to ``min(n_comps, min(cells, genes) - 1)``; genes that leave fewer
            than one component are skipped rather than raising (pertpy errors).
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


def _scale_gpu(dat: cp.ndarray) -> cp.ndarray:
    """Zero-center and unit-scale columns (matches ``scanpy.pp.scale``)."""
    mean = dat.mean(axis=0)
    std = cp.sqrt(dat.var(axis=0, ddof=1))
    std = cp.where(std == 0, dat.dtype.type(1.0), std)
    return (dat - mean) / std


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

    between_fac = dtype.type(1.0 if n_classes == 1 else 1.0 / (n_classes - 1))
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
