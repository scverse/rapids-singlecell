from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell.get import X_to_GPU

from ._base import PerturbationEfficacyAnalyzer, _to_dense_gpu

if TYPE_CHECKING:
    from anndata import AnnData


class Mixscale(PerturbationEfficacyAnalyzer):
    """GPU-accelerated Mixscale for continuous perturbation-efficiency scoring.

    Unlike :class:`~rapids_singlecell.ptg.Mixscape`, which performs a binary
    knocked-out/non-perturbed classification with a Gaussian mixture, Mixscale
    assigns each cell a *continuous* perturbation-efficiency score. It follows
    Seurat's Mixscale and pertpy's :class:`~pertpy.tools.Mixscale`; the
    perturbation signature is computed via
    :meth:`~rapids_singlecell.ptg.Mixscale.perturbation_signature` and the
    per-gene projection and z-score scoring run on the GPU.
    """

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

        Unlike :meth:`~rapids_singlecell.ptg.Mixscape.mixscape`, which performs a
        binary knocked-out/non-perturbed classification with a Gaussian mixture,
        this assigns each cell a *continuous* perturbation-efficiency score: the
        scalar projection of its perturbation signature onto the per-gene
        perturbation direction (mean perturbed minus mean control), z-score
        standardized relative to the non-targeting control distribution. This is
        useful for CRISPRi/CRISPRa screens where cells show a gradient of
        perturbation strength rather than a binary knockout. Control cells
        receive a score of 0.

        Implements Jiang, Dalgarno et al., "Systematic reconstruction of molecular
        pathway signatures using scalable single-cell perturbation screens",
        Nature Cell Biology (2025), following pertpy's
        :meth:`~pertpy.tools.Mixscale.mixscale`.

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
            Accepted for ``pertpy.tl.Mixscale.mixscale`` API compatibility; has no
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
