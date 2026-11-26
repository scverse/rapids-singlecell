from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from ._base import PerturbationEfficacyAnalyzer, _get_scoring_matrix, _target_jobs, _with_marker_blocks

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
        signature_kwargs: dict | None = None,
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
            :func:`rapids_singlecell.tl.rank_genes_groups`. Use ``"t-test"``
            for Dask expression with dense, CSR or CSC chunks.
        scale
            Scale the per-gene sub-matrix before computing scores.
        split_by
            ``.obs`` column with a condition/cell-type annotation if
            perturbations are condition specific.
        signature_kwargs
            Compute signatures directly from unscaled, log-normalized ``.X`` for
            each target's marker genes. Pass ``{}`` for defaults or supply neighbor
            and reference options from :meth:`perturbation_signature`. The
            method uses ``split_by`` for both scoring and reference selection.
            This skips storing ``X_pert`` and requires ``layer=None``.
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
        if layer is not None and signature_kwargs is not None:
            raise ValueError("Pass either layer or signature_kwargs.")
        if copy:
            adata = adata.copy()

        if split_by is None:
            split_masks = [np.ones(adata.n_obs, dtype=bool)]
            categories = ["all"]
        else:
            split_obs = adata.obs[split_by]
            categories = list(split_obs.unique())
            split_masks = [(split_obs == category).to_numpy() for category in categories]

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
        X = _get_scoring_matrix(
            adata, pert_key, control, layer=layer, signature_kwargs=signature_kwargs, split_by=split_by
        )

        jobs = _target_jobs(
            adata, pert_key, control, split_masks, categories, markers=perturbation_markers, max_de_genes=max_de_genes
        )
        scores_gpu = cp.zeros(adata.n_obs, dtype=X.dtype)
        if isinstance(X, cp.ndarray):
            gene_jobs = list(jobs)
            if gene_jobs:
                _project_scores_batched(X, gene_jobs, scores_gpu, do_scale=scale)
        else:
            for job in _with_marker_blocks(X, jobs):
                block = job.pop("dat")
                rows = job["row_ids"]
                job["row_ids"] = np.arange(block.shape[0], dtype=np.int32)
                job["col_ids"] = np.arange(block.shape[1], dtype=np.int32)
                block_scores = cp.zeros(block.shape[0], dtype=X.dtype)
                _project_scores_batched(block, [job], block_scores, do_scale=scale)
                scores_gpu[cp.asarray(rows)] = block_scores
                del block
        adata.obs[new_class_name] = cp.asnumpy(scores_gpu).astype(np.float64, copy=False)

        if copy:
            return adata
        return None


def _project_scores_batched(X: cp.ndarray, gene_jobs: list[dict], scores_gpu: cp.ndarray, *, do_scale: bool) -> None:
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

    row_ids = cp.asarray(np.concatenate([job["row_ids"] for job in gene_jobs]), dtype=cp.int32)
    col_ids = cp.asarray(np.concatenate([job["col_ids"] for job in gene_jobs]))
    is_guide = cp.asarray(np.concatenate([job["is_guide"] for job in gene_jobs]))
    nt_flat = ~is_guide
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
