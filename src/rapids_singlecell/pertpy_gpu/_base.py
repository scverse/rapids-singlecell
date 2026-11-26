from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import issparse as cp_issparse

from rapids_singlecell._compat import DaskArray
from rapids_singlecell._keys import _embedding_keys, _existing_preset_keys
from rapids_singlecell._settings import Preset
from rapids_singlecell.get import X_to_GPU, _get_obs_rep, _set_obs_rep
from rapids_singlecell.get._out import own_deprecations
from rapids_singlecell.tools._utils import _choose_representation

if TYPE_CHECKING:
    from anndata import AnnData

# Sentinel groupby label for cells outside the current split during differential
# expression, so they are excluded without copying/subsetting the matrix.
_DE_IGNORE_LABEL = "__mixscape_other__"
_MARKER_BLOCK_BYTES = 256 * 1024**2


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
        written to ``adata.layers["X_pert"]`` as a dense array. To score sparse
        expression directly, pass ``signature_kwargs`` to ``mixscape`` or ``mixscale``.
        Both workflows use unscaled, log-normalized expression.

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
            50, otherwise the PCA key selected by
            ``rapids_singlecell.settings.preset`` (computed if absent).
        n_dims
            Number of dimensions of the representation to use. ``None`` uses all.
        n_pcs
            Number of principal components to compute if a PCA representation is
            built.
        knn_algorithm
            Nearest-neighbor backend for ``ref_selection_mode="nn"``: ``"brute"``
            (exact, default), or the approximate cuVS backends ``"ivfflat"``,
            ``"cagra"``, ``"ivfpq"``. Brute search supports sparse expression.
            Approximate backends use a dense representation, such as a PCA embedding.
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
        if copy:
            adata = adata.copy()

        signature = _PerturbationSignature(
            adata,
            pert_key,
            control,
            split_by=split_by,
            ref_selection_mode=ref_selection_mode,
            n_neighbors=n_neighbors,
            use_rep=use_rep,
            n_dims=n_dims,
            n_pcs=n_pcs,
            knn_algorithm=knn_algorithm,
            knn_kwargs=knn_kwargs,
        )
        _set_obs_rep(adata, signature.materialize(), layer="X_pert")

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
                with warnings.catch_warnings(), own_deprecations():
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


def _expression_matrix(X):
    if not isinstance(X, DaskArray):
        X = X_to_GPU(X)
    if X.dtype not in (cp.float32, cp.float64):
        X = X.astype(cp.float32)
    return X


def _to_dense_gpu(X) -> cp.ndarray:
    X = _expression_matrix(X)
    if cp_issparse(X):
        from rapids_singlecell.preprocessing._utils import _sparse_to_dense

        return _sparse_to_dense(X, order="C")
    return X


def _get_scoring_matrix(adata, pert_key, control, *, layer, signature_kwargs, split_by):
    if signature_kwargs is not None:
        return _PerturbationSignature(adata, pert_key, control, split_by=split_by, **signature_kwargs)
    if layer is None and "X_pert" not in adata.layers:
        raise KeyError("No 'X_pert' found in .layers! Run perturbation_signature first or pass signature_kwargs={}.")
    return _expression_matrix(adata.layers["X_pert" if layer is None else layer])


def _target_jobs(adata, pert_key, control, split_masks, categories, *, markers, max_de_genes=None):
    """Guide and control rows and the score-ranked marker columns for each target."""
    labels = adata.obs[pert_key].astype("category")
    codes = labels.cat.codes.to_numpy()
    names = labels.cat.categories.astype(str).tolist()
    control_code = names.index(control) if control in names else -1
    var_to_idx = {name: i for i, name in enumerate(adata.var_names)}
    for split, mask in enumerate(split_masks):
        controls = (codes == control_code) & mask
        if not controls.any():
            continue
        for code in np.unique(codes[mask]):
            if code == control_code:
                continue
            gene = names[code]
            de = markers[categories[split], gene][:max_de_genes]
            cols = np.fromiter((var_to_idx[g] for g in de if g in var_to_idx), dtype=np.int32)
            if not cols.size:
                continue
            rows = np.flatnonzero(controls | ((codes == code) & mask))
            yield {"split": split, "gene": gene, "row_ids": rows, "col_ids": cols, "is_guide": ~controls[rows]}


def _with_marker_blocks(X, jobs):
    """Read Dask panels together so targets can share source chunks within a bounded batch."""
    expr = X.expression if isinstance(X, _PerturbationSignature) else X
    if not isinstance(expr, DaskArray):
        for job in jobs:
            job["dat"] = _get_marker_block(X, job["row_ids"], job["col_ids"], job["split"])
            yield job
        return

    from itertools import chain, groupby

    from dask import compute

    def read_batch(batch):
        panels = [_dask_marker_block(expr, job["row_ids"], job["col_ids"]) for job in batch]
        for job, panel in zip(batch, compute(*panels), strict=True):
            block = _dense_marker_values(panel)
            if isinstance(X, _PerturbationSignature):
                block = X.residual(block, job["row_ids"], job["split"])
            job["dat"] = block
            yield job

    def panel_bytes(job):
        return job["row_ids"].size * job["col_ids"].size * expr.dtype.itemsize

    sparse = getattr(expr._meta, "format", None) in ("csr", "csc", "coo")
    row_bytes = None
    for split, group in groupby(
        jobs, key=lambda job: job["split"] if sparse and panel_bytes(job) > _MARKER_BLOCK_BYTES else None
    ):
        if split is not None:
            pending = iter(group)
            first, second = next(pending), next(pending, None)
            if second is None:
                yield from read_batch([first])
                continue
            if row_bytes is None:
                row_bytes = _csr_row_bytes(expr)
            yield from _wide_marker_blocks(X, expr, chain((first, second), pending), row_bytes)
            continue
        batch = []
        size = 0
        for job in group:
            nbytes = panel_bytes(job)
            if batch and size + nbytes > _MARKER_BLOCK_BYTES:
                yield from read_batch(batch)
                batch = []
                size = 0
            batch.append(job)
            size += nbytes
        if batch:
            yield from read_batch(batch)


def _csr_row_bytes(expr):
    """Bound CSR payload per source row using stored entries and 64-bit indices."""
    xp = cp if cp_issparse(expr._meta) else np

    def counts(part):
        if part.format == "csr":
            return xp.diff(part.indptr)[:, None]
        rows = part.indices if part.format == "csc" else part.row
        return xp.bincount(rows, minlength=part.shape[0])[:, None]

    result = expr.map_blocks(
        counts, chunks=(expr.chunks[0], (1,) * expr.numblocks[1]), meta=xp.empty((0, 0), dtype=np.int64)
    )
    nnz = result.sum(axis=1, dtype=np.int64).compute()
    nnz = nnz.get() if isinstance(nnz, cp.ndarray) else nnz
    return 8 + nnz * (expr.dtype.itemsize + 8)


def _wide_marker_blocks(X, expr, jobs, row_bytes):
    """Stage shared sparse rows once, then densify one target's markers at a time."""

    def read(batch):
        first = batch[0]
        if len(batch) == 1:
            first["dat"] = _get_marker_block(X, first["row_ids"], first["col_ids"], first["split"])
            yield first
            return

        controls = first["row_ids"][~first["is_guide"]]
        rows = np.sort(np.concatenate([controls, *(j["row_ids"][j["is_guide"]] for j in batch)]))
        csr = expr.map_blocks(lambda part: part.tocsr(), meta=expr._meta.tocsr())
        staged = X_to_GPU(csr[rows].compute().tocsr())
        for job in batch:
            local = np.searchsorted(rows, job["row_ids"])
            block = _get_marker_block(staged, local, job["col_ids"], job["split"])
            if isinstance(X, _PerturbationSignature):
                block = X.residual(block, job["row_ids"], job["split"])
            job["dat"] = block
            del block
            yield job

    batch = []
    size, n_rows = 8, 0  # final CSR indptr entry
    for job in jobs:
        rows, guides = job["row_ids"], job["is_guide"]
        extra = rows[guides] if batch else rows
        metadata = rows.nbytes + guides.nbytes + job["col_ids"].nbytes
        cost = metadata + row_bytes[extra].sum()
        if batch and (size + cost > _MARKER_BLOCK_BYTES or n_rows + extra.size >= expr.shape[0]):
            yield from read(batch)
            batch = []
            size, n_rows = 8, 0
            extra = rows
            cost = metadata + row_bytes[rows].sum()
        batch.append(job)
        size += cost
        n_rows += extra.size
    if batch:
        yield from read(batch)


class _PerturbationSignature:
    """Reference for each split, evaluated on the requested rows and genes."""

    def __init__(
        self,
        adata,
        pert_key,
        control,
        *,
        split_by=None,
        ref_selection_mode="nn",
        n_neighbors=20,
        use_rep=None,
        n_dims=15,
        n_pcs=None,
        knn_algorithm="brute",
        knn_kwargs=None,
    ):
        if ref_selection_mode not in ("nn", "split_by"):
            raise ValueError("ref_selection_mode must be either 'nn' or 'split_by'.")
        if ref_selection_mode == "split_by" and split_by is None:
            raise ValueError("split_by must be provided if ref_selection_mode is 'split_by'.")

        self.expression = _expression_matrix(_get_obs_rep(adata))
        self.dtype = self.expression.dtype
        if ref_selection_mode == "nn":
            from rapids_singlecell.preprocessing._neighbors._helper import _check_knn_complete
            from rapids_singlecell.preprocessing._neighbors._neighbors import KNN_ALGORITHMS

            if knn_algorithm not in KNN_ALGORITHMS:
                raise ValueError(f"knn_algorithm must be one of {sorted(KNN_ALGORITHMS)}, got {knn_algorithm!r}")
            representation = _choose_representation_gpu(
                adata,
                use_rep=use_rep,
                n_pcs=n_pcs,
                n_dims=n_dims,
                knn_algorithm=knn_algorithm,
            )

        control_mask = (adata.obs[pert_key] == control).to_numpy()
        groups = adata.obs[split_by] if split_by is not None else None
        self.references = []
        for category in groups.unique() if groups is not None else [None]:
            if groups is None:
                rows = np.arange(adata.n_obs)
            elif pd.isna(category):
                rows = np.flatnonzero(groups.isna())
            else:
                rows = np.flatnonzero(groups == category)
            controls = rows[control_mask[rows]]
            weights = None
            if not controls.size or (groups is not None and pd.isna(category)):
                controls = rows[:0]
                warnings.warn(
                    f"No control cells in split {category!r}; leaving its perturbation signature equal to the input.",
                    UserWarning,
                    stacklevel=3,
                )
            elif ref_selection_mode == "nn":
                k = min(n_neighbors, controls.size)
                indices, dist = _control_neighbors(
                    representation, controls, rows, k, knn_algorithm=knn_algorithm, knn_kwargs=knn_kwargs
                )
                _check_knn_complete(indices, dist, n_index=controls.size, algorithm=knn_algorithm)
                indices = indices.ravel()
                index_dtype = cp.int64 if max(indices.size, controls.size) > np.iinfo(np.int32).max else cp.int32
                weights = cp_csr_matrix(
                    (
                        cp.full(indices.size, 1.0 / k, dtype=self.dtype),
                        indices.astype(index_dtype, copy=False),
                        cp.arange(0, indices.size + 1, k, dtype=index_dtype),
                    ),
                    shape=(rows.size, controls.size),
                )
            self.references.append((rows, weights, controls))

    def materialize(self):
        result = cp.empty(self.expression.shape, dtype=self.dtype)
        for split, (rows, _, _) in enumerate(self.references):
            if rows.size:
                # Target 64 MiB per dense tile when writing the full signature.
                width = max(1, (64 * 1024**2) // (rows.size * self.dtype.itemsize))
                row_ids = cp.asarray(rows)
                for start in range(0, result.shape[1], width):
                    cols = slice(start, start + width)
                    result[row_ids, cols] = _get_marker_block(self, rows, cols, split)
        return result

    def residual(self, block, rows, split):
        split_rows, weights, controls = self.references[split]
        if not controls.size:
            return block
        # Both scoring and materialization request sorted rows including every control in the split.
        control_values = block[cp.asarray(np.searchsorted(rows, controls))]
        if weights is None:
            baseline = control_values.mean(axis=0)
        else:
            if rows is not split_rows:
                weights = weights[cp.asarray(np.searchsorted(split_rows, rows))]
            baseline = cp.log1p(weights @ cp.expm1(control_values))
        return cp.subtract(baseline, block, out=block)


def _get_marker_block(X, rows, cols, split):
    """Read the expression for one marker panel, then apply its control reference."""
    signature = X if isinstance(X, _PerturbationSignature) else None
    if signature is not None:
        X = signature.expression
    if isinstance(X, DaskArray):
        block = _dask_marker_block(X, rows, cols).compute()
    else:
        row_ids = cp.asarray(rows)
        block = X[row_ids, cols] if isinstance(cols, slice) else X[row_ids[:, None], cp.asarray(cols)[None, :]]
    block = _dense_marker_values(block)
    return block if signature is None else signature.residual(block, rows, split)


def _dense_marker_values(panel):
    panel = X_to_GPU(panel)
    return panel.toarray(order="C") if cp_issparse(panel) else panel


def _dask_marker_block(X, rows, cols):
    """Gather on workers, retaining sparse storage until the scoring GPU receives the panel."""
    from scipy.sparse import issparse

    if isinstance(cols, slice):
        cols = np.arange(X.shape[1])[cols]

    def local_indices(indices, chunks):
        edges = np.r_[0, np.cumsum(chunks)]
        cuts = np.searchsorted(indices, edges)
        return tuple(indices[lo:hi] - offset for lo, hi, offset in zip(cuts[:-1], cuts[1:], edges[:-1], strict=True))

    row_ids = local_indices(rows, X.chunks[0])
    if X.numblocks[1] == 1:
        col_ids, order = (cols,), None
    else:
        columns, order = np.unique(cols, return_inverse=True)
        col_ids = local_indices(columns, X.chunks[1])
    chunks = tuple(map(len, row_ids)), tuple(map(len, col_ids))
    on_gpu = cp_issparse(X._meta) or isinstance(X._meta, cp.ndarray)
    xp = cp if on_gpu else np

    def gather(part, block_id=None):
        i, j = block_id
        r, c = xp.asarray(row_ids[i]), xp.asarray(col_ids[j])
        if getattr(part, "format", None) == "coo":
            part = part.tocsr()
        if issparse(part) or cp_issparse(part):
            return part[r, :][:, c].tocsr()
        return part[r[:, None], c[None, :]]

    meta = X._meta.tocsr() if issparse(X._meta) or cp_issparse(X._meta) else X._meta
    result = X.map_blocks(gather, chunks=chunks, meta=meta)
    result = result.blocks[np.flatnonzero(chunks[0]), :].blocks[:, np.flatnonzero(chunks[1])]
    if order is None:
        return result

    def reorder(part):
        # Joining sparse feature chunks can yield COO, which cannot be indexed.
        if getattr(part, "format", None) == "coo":
            part = part.tocsr()
        return part[:, order]

    return result.map_blocks(reorder, drop_axis=1, new_axis=1, chunks=(result.chunks[0], (len(cols),)), meta=meta)


def _choose_representation_gpu(adata: AnnData, *, use_rep, n_pcs, n_dims, knn_algorithm):
    try:
        rep = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    except ValueError:
        # A preset PCA with fewer than ``n_pcs`` components is recomputed;
        # an explicit ``use_rep`` or a missing ``.X`` keeps the original error.
        pca_keys = _existing_preset_keys(adata, "pca") if use_rep is None else None
        if pca_keys is None or adata.X is None:
            raise
        from rapids_singlecell.preprocessing import pca

        # a `Preset` as `key_added` gives its key names (`uns` too) without needing `anndata.acc`
        key_added = Preset.ScanpyV1 if pca_keys == _embedding_keys("pca", None) else pca_keys.obsm
        with own_deprecations():
            pca(adata, n_comps=n_pcs, key_added=key_added)
        rep = _choose_representation(adata, use_rep=use_rep, n_pcs=n_pcs)
    if isinstance(rep, DaskArray):
        from rapids_singlecell.preprocessing._pca import _gpu_row_chunks

        raw = isinstance(adata.X, DaskArray) and rep.name == adata.X.name
        rep = _gpu_row_chunks(rep)
    else:
        raw = False
        rep = X_to_GPU(rep)
    rep = rep.astype(cp.float32, copy=False)
    if n_pcs is not None and n_pcs < rep.shape[1]:
        rep = rep[:, :n_pcs]
    if n_dims is not None and n_dims < rep.shape[1]:
        rep = rep[:, :n_dims]
    if isinstance(rep, DaskArray):
        if raw:
            return rep
        rep = rep.compute()
    if cp_issparse(rep):
        if knn_algorithm == "brute":
            return rep.tocsr()
        rep = rep.toarray(order="C")
    return cp.ascontiguousarray(rep)


def _control_neighbors(representation, controls, rows, k, *, knn_algorithm, knn_kwargs):
    from rapids_singlecell.preprocessing._neighbors._neighbors import KNN_ALGORITHMS

    knn = KNN_ALGORITHMS[knn_algorithm]
    kwargs = {"metric": "euclidean", "metric_kwds": {}, "algorithm_kwds": dict(knn_kwargs or {})}
    if not isinstance(representation, DaskArray):
        return knn(representation[cp.asarray(controls)], representation[cp.asarray(rows)], k, **kwargs)

    def gather(row_ids):
        block = representation[row_ids].compute()
        if cp_issparse(block):
            return block.tocsr() if knn_algorithm == "brute" else block.toarray(order="C")
        return cp.ascontiguousarray(block)

    index = gather(controls)
    if knn_algorithm == "brute":
        from rapids_singlecell.preprocessing._neighbors._algorithms._brute import _fit_brute_knn

        index = _fit_brute_knn(index, k, metric="euclidean", metric_kwds={})

    indices, distances = [], []
    batch_size = max(representation.chunks[0])
    for start in range(0, len(rows), batch_size):
        queries = gather(rows[start : start + batch_size])
        if knn_algorithm == "brute":
            dist, ind = index.kneighbors(queries)
        else:
            ind, dist = knn(index, queries, k, **kwargs)
        indices.append(ind)
        distances.append(dist)
    return cp.concatenate(indices), cp.concatenate(distances)
