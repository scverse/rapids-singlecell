from __future__ import annotations

import warnings
from functools import partial
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy.sparse import csc_matrix as cp_csc
from cupyx.scipy.sparse import csr_matrix as cp_csr
from cupyx.scipy.sparse import issparse as cp_issparse
from cupyx.scipy.sparse import isspmatrix_csc, isspmatrix_csr
from scverse_misc import Deprecation, deprecated_arg

from rapids_singlecell._cuda import _guide_assignment_cuda as _ga
from rapids_singlecell.get import X_to_GPU, _get_obs_rep, _set_obs_rep

_HISTOGRAM_CAP = 1024
_HISTOGRAM_BINS = 1 << 24
_COLUMN_CHUNK = 4096
_COLUMN_BLOCKS = 1 << 20
_LABEL_BYTES = 1 << 28
_LEARNING_RATE_DEPRECATION = Deprecation(
    "0.19.0",
    "The mixture model is fit by EM, which has no learning rate. The argument is ignored.",
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from cupyx.scipy.sparse import spmatrix

_MIXTURE_MODEL_VAR_COLUMNS = (
    "poisson_rate",
    "gaussian_mean",
    "gaussian_std",
    "mix_probs_0",
    "mix_probs_1",
    "threshold",
    "final_loss",
)


def _format_guides(names: Sequence[str] | np.ndarray, limit: int = 10) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


def _layout(X: cp.ndarray | spmatrix) -> cp.ndarray | cp_csr | cp_csc:
    """CSR or CSC ``X`` with float32 or float64 values for the sparse kernels, or dense ``X`` likewise.

    Other sparse formats become CSR, other dtypes float32, and matrices with mixed index dtypes or strided arrays are rebuilt.
    """
    if cp_issparse(X):
        if not (isspmatrix_csc(X) or isspmatrix_csr(X)):
            X = X.tocsr()
        arrays = (X.data, X.indices, X.indptr)
        if X.indices.dtype != X.indptr.dtype or not all(a.flags.c_contiguous for a in arrays):
            data, indices, indptr = map(cp.ascontiguousarray, arrays)
            X = type(X)((data, indices, indptr), shape=X.shape)
    return X if X.dtype in (cp.float32, cp.float64) else X.astype(cp.float32)


def _column_chunks(X: cp.ndarray | cp_csr | cp_csc) -> tuple[int, int]:
    """Entries per block and blocks per column of CSC or dense ``X``, so long columns fill the GPU."""
    if isspmatrix_csc(X) and X.shape[1]:
        longest = int(cp.diff(X.indptr).max())
    else:
        longest = X.shape[0]
    # Grid dimension y allows 65535 blocks per column.
    n_chunks = min(-(-longest // _COLUMN_CHUNK), _COLUMN_BLOCKS // max(X.shape[1], 1), 65535)
    n_chunks = max(n_chunks, 1)
    return max(-(-longest // n_chunks), 1), n_chunks


def _distinct_values(
    X: cp.ndarray | cp_csr | cp_csc, n_guides: int
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Sorted distinct non-zero values and their multiplicities per guide, zero padded.

    Integer counts below ``_HISTOGRAM_CAP`` are counted in a per-guide histogram by the CSR, CSC or dense kernel, the few other values are sorted.
    Also returns the number of distinct values, the number of non-zero cells and the maximum count per guide.
    """
    data = X.data if cp_issparse(X) else X
    top = float(data.max()) if data.size else 0.0
    # NaN or large counts get the full histogram, the tail takes the rest.
    cap = int(top) + 1 if top < _HISTOGRAM_CAP else _HISTOGRAM_CAP
    cap = min(cap, max(_HISTOGRAM_BINS // max(n_guides, 1), 2))
    hist = cp.zeros((n_guides, cap), dtype=cp.uint32)
    n_tail = cp.zeros(1, dtype=cp.uint64)
    tail = cp.empty(max(data.size // 64, 1024), dtype=cp.uint64)
    if isspmatrix_csr(X):
        histogram = partial(_ga.value_histogram_csr, X.indices, X.data)
    else:
        chunk, n_chunks = _column_chunks(X)
        histogram = (
            partial(_ga.value_histogram_csc, X.indptr, X.indices, X.data)
            if isspmatrix_csc(X)
            else partial(_ga.value_histogram_dense, X)
        )
        histogram = partial(histogram, chunk=chunk, n_chunks=n_chunks)
    for _ in range(2):
        histogram(hist, n_tail, tail, stream=cp.cuda.get_current_stream().ptr)
        if int(n_tail[0]) <= tail.size:
            break
        tail = cp.empty(int(n_tail[0]), dtype=cp.uint64)
        hist[:] = 0
        n_tail[:] = 0
    tail = tail[: int(n_tail[0])]
    # The bits of non-negative floats sort like the floats, so keys sort by guide, then value.
    bins = cp.flatnonzero(hist)
    keys = cp.concatenate(
        [
            ((bins // cap).astype(cp.uint64) << 32) | (bins % cap).astype(cp.float32).view(cp.uint32),
            tail,
        ]
    )
    counts = cp.concatenate([hist.ravel()[bins], cp.ones(tail.size, dtype=cp.uint32)])
    keys, inverse = cp.unique(keys, return_inverse=True)
    multiplicity = cp.bincount(inverse, weights=counts).astype(cp.float32)
    guide = (keys >> 32).astype(cp.int64)
    value = (keys & 0xFFFFFFFF).astype(cp.uint32).view(cp.float32)
    n_distinct = cp.bincount(guide, minlength=n_guides).astype(cp.int32)
    rank = cp.arange(keys.size) - (cp.cumsum(n_distinct) - n_distinct)[guide]
    width = int(n_distinct.max()) if n_guides else 0
    values_padded = cp.zeros((n_guides, width), dtype=cp.float32)
    weights_padded = cp.zeros((n_guides, width), dtype=cp.float32)
    values_padded[guide, rank] = cp.log2(value)
    weights_padded[guide, rank] = multiplicity
    last = cp.flatnonzero(cp.diff(guide, append=-1))
    max_counts = cp.zeros(n_guides, dtype=cp.float32)
    max_counts[guide[last]] = value[last]
    return (
        values_padded,
        weights_padded,
        n_distinct,
        weights_padded.sum(axis=1),
        max_counts,
    )


def _fit_mixture(
    values: cp.ndarray,
    weights: cp.ndarray,
    n_values: cp.ndarray,
    max_counts: cp.ndarray,
    *,
    n_iter: int,
    n_init_seeds: int,
    seed: int,
) -> np.ndarray:
    """Rate, mean, std, Gaussian weight, loss and threshold of every guide by multi-start MAP-EM, nan without an admissible fit.

    Every guide starts from ``n_init_seeds`` prior medians, as crispat, and from five quantile splits of its values.
    """
    n_guides = values.shape[0]
    rng = cp.random.default_rng(seed)
    draws = cp.stack(
        [rng.standard_normal((n_init_seeds, _ga.PRIOR_SAMPLES, n_guides), dtype=cp.float32) for _ in range(3)]
    )
    n_starts = n_init_seeds + _ga.N_QUANTILES
    out = cp.full((n_guides, _ga.N_OUTPUTS), cp.nan, dtype=cp.float32)
    _ga.fit_poisson_gauss(
        values,
        weights,
        n_values,
        max_counts.astype(cp.int64),
        draws,
        cp.empty((n_starts, n_guides, _ga.N_PARAMS), dtype=cp.float32),
        cp.empty((n_starts, n_guides), dtype=cp.float32),
        out,
        n_iter=n_iter,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return cp.asnumpy(out).astype(np.float64)


def _assigned(X: cp.ndarray | cp_csr | cp_csc, thresholds: cp.ndarray) -> cp.ndarray | cp_csr | cp_csc:
    """Binary matrix in the layout of ``X`` of the entries at or above their guide's threshold.

    Thresholds are rounded to the precision of ``X``, as a scalar would be, and a NaN threshold assigns nothing.
    """
    thresholds = thresholds.astype(X.dtype if X.dtype.kind == "f" else cp.float64)
    if not cp_issparse(X):
        return (X >= thresholds).astype(cp.int8)
    if isspmatrix_csc(X):
        chunk, n_chunks = _column_chunks(X)
        assigned = partial(_ga.assigned_csc, chunk=chunk, n_chunks=n_chunks)
        n_slots = X.shape[1] * n_chunks
    else:
        assigned, n_chunks, n_slots = _ga.assigned_csr, 1, X.shape[0]
    args = (X.indptr, X.indices, X.data, thresholds.astype(cp.float64))
    stream = cp.cuda.get_current_stream().ptr
    offsets = cp.zeros(n_slots + 1, dtype=X.indptr.dtype)
    assigned(*args, offsets[1:], offsets[:0], fill=False, stream=stream)
    cp.cumsum(offsets, out=offsets)
    indices = cp.empty(int(offsets[-1]), dtype=X.indptr.dtype)
    assigned(*args, offsets, indices, fill=True, stream=stream)
    return (cp_csc if isspmatrix_csc(X) else cp_csr)(
        (
            cp.ones(indices.size, dtype=cp.float32),
            indices,
            cp.ascontiguousarray(offsets[::n_chunks]),
        ),
        shape=X.shape,
    )


def _cell_major(binary):
    """CSR form of a binary assignment matrix with sorted guides in every cell; duplicate entries are summed, as in pertpy."""
    if cp_issparse(binary):
        cells = binary.tocsr()
        cells.sum_duplicates()
        return cells
    rows, cols = cp.nonzero(binary)
    indptr = cp.searchsorted(rows, cp.arange(binary.shape[0] + 1))
    return cp_csr((cp.ones(cols.size, dtype=cp.float32), cols, indptr), shape=binary.shape)


def _max_guide(X) -> tuple[np.ndarray, np.ndarray]:
    """Largest stored count of every cell of sparse ``X`` and its guide, ``-1`` for cells without stored counts.

    As in pertpy's sparse path implicit zeros are ignored; ties go to the first guide, which is pertpy's first stored entry for canonical input.
    """
    stream = cp.cuda.get_current_stream().ptr
    keys = cp.zeros(X.shape[0], dtype=cp.uint64)
    guides = cp.full(X.shape[0], -1, dtype=cp.int32)
    if isspmatrix_csc(X):
        chunk, n_chunks = _column_chunks(X)
        _ga.max_guide_csc(
            X.indptr,
            X.indices,
            X.data,
            keys,
            guides,
            chunk=chunk,
            n_chunks=n_chunks,
            stream=stream,
        )
    else:
        _ga.max_guide_csr(X.indptr, X.indices, X.data, keys, guides, stream=stream)
    # Inverse of the kernels' order_key; the key 0 of cells without counts decodes to NaN.
    bits = cp.where((keys >> 63) != 0, keys & 0x7FFFFFFFFFFFFFFF, ~keys)
    return cp.asnumpy(bits.view(cp.float64)).astype(X.dtype), cp.asnumpy(guides)


def _per_cell_assignments(
    indptr: cp.ndarray,
    indices: cp.ndarray,
    guide_names: np.ndarray,
    *,
    no_grna_assigned_key: str,
    max_assignments_per_cell: int,
    multiple_grna_assigned_key: str,
    multiple_grna_assignment_string: str,
) -> pd.Categorical:
    """Per-cell assignment labels from the CSR structure of the binary assignment."""
    n_cells = indptr.size - 1
    n_guides = len(guide_names)
    n_assigned = cp.diff(indptr)
    if any(multiple_grna_assignment_string in name for name in guide_names):
        warnings.warn(
            "Some guide names contain the separator "
            f"{multiple_grna_assignment_string!r}, so assignments of several guides "
            "to one cell cannot be split unambiguously. "
            "Pass a different `multiple_grna_assignment_string`.",
            UserWarning,
            stacklevel=4,
        )
    labels = [
        np.array([no_grna_assigned_key, multiple_grna_assigned_key], dtype=str),
        guide_names,
    ]
    codes = cp.zeros(n_cells, dtype=cp.int64)
    single = n_assigned == 1
    codes[single] = 2 + indices[indptr[:-1][single]]
    multi = (n_assigned > 1) & (n_assigned <= max_assignments_per_cell)
    with_separator = np.char.add(multiple_grna_assignment_string, guide_names)
    offset = 2 + n_guides
    for size in cp.asnumpy(cp.unique(n_assigned[multi])).tolist():
        cells = cp.flatnonzero(multi & (n_assigned == size))
        members = indices[indptr[cells][:, None] + cp.arange(size)]
        order = cp.lexsort(members.T[::-1])
        members = members[order]
        new_row = cp.ones(cells.size, dtype=cp.bool_)
        new_row[1:] = (members[1:] != members[:-1]).any(axis=1)
        inverse = cp.empty(cells.size, dtype=cp.int64)
        inverse[order] = cp.cumsum(new_row) - 1
        codes[cells] = offset + inverse
        unique = cp.asnumpy(members[new_row])
        joined = guide_names[unique[:, 0]]
        for j in range(1, size):
            joined = np.char.add(joined, with_separator[unique[:, j]])
        labels.append(joined)
        offset += len(joined)
    codes[n_assigned > max_assignments_per_cell] = 1
    codes_cpu = cp.asnumpy(codes)
    widest = max(label.dtype.itemsize for label in labels)
    if offset * widest > _LABEL_BYTES:
        labels = [label.astype(object) for label in labels]
    names = np.concatenate(labels)
    # Sorted distinct labels, as the categories of pertpy's pd.Categorical.
    used = np.zeros(names.size, dtype=bool)
    used[codes_cpu] = True
    used = np.flatnonzero(used)
    categories, category_codes = np.unique(names[used], return_inverse=True)
    remap = np.empty(names.size, dtype=np.intp)
    remap[used] = category_codes
    return pd.Categorical.from_codes(remap[codes_cpu], pd.Index(categories))


class GuideAssignment:
    """GPU-accelerated guide RNA assignment.

    Provides the threshold-based and mixture-model-based methods of pertpy's ``GuideAssignment`` with the same parameters and results.
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

        Each cell is assigned to every gRNA with at least ``assignment_threshold`` counts.
        Expects unnormalized count data.

        Parameters
        ----------
        adata
            Annotated data matrix of shape ``n_obs x n_vars``.
        assignment_threshold
            Minimum count for a viable assignment.
        layer
            Layer with raw counts.
            Uses ``adata.X`` if ``None``.
        output_layer
            Key under which the binary assignment matrix is stored in ``adata.layers``.
        """
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        X = _layout(X) if cp_issparse(X) else X
        binary = _assigned(X, cp.full(X.shape[1], assignment_threshold))
        _set_obs_rep(adata, binary, layer=output_layer)

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

        Each cell is assigned to the gRNA with the highest count, provided that count is at least ``assignment_threshold``.
        Ties are resolved in favour of the first of the tied gRNAs.
        For sparse input, as in pertpy, only stored counts are considered.
        Expects unnormalized count data.

        Parameters
        ----------
        adata
            Annotated data matrix of shape ``n_obs x n_vars``.
        assignment_threshold
            Minimum count for a viable assignment.
        layer
            Layer with raw counts.
            Uses ``adata.X`` if ``None``.
        obs_key
            Column in ``adata.obs`` where the assignment is stored.
        no_grna_assigned_key
            Label for cells with no guide above threshold.
        """
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        if not X.shape[1]:
            max_values = max_positions = np.full(X.shape[0], -1)
        elif cp_issparse(X):
            max_values, max_positions = _max_guide(_layout(X))
        else:
            max_values = cp.asnumpy(X.max(axis=1))
            max_positions = cp.asnumpy(X.argmax(axis=1))
        # Position -1 picks the appended key.
        labels = np.append(adata.var_names.to_numpy(), no_grna_assigned_key)
        assigned = (max_positions >= 0) & (max_values >= assignment_threshold)
        adata.obs[obs_key] = labels[np.where(assigned, max_positions, -1)]

    @deprecated_arg("learning_rate", _LEARNING_RATE_DEPRECATION)
    def assign_mixture_model(
        self,
        adata: AnnData,
        *,
        model: Literal["poisson_gauss_mixture"] = "poisson_gauss_mixture",
        layer: str | None = None,
        assigned_guides_key: str = "assigned_guide",
        output_layer: str | None = "assigned_guides",
        no_grna_assigned_key: str = "negative",
        max_assignments_per_cell: int = 5,
        multiple_grna_assigned_key: str = "multiple",
        multiple_grna_assignment_string: str = "+",
        only_return_results: bool = False,
        show_progress: bool = False,
        n_iter: int = 1000,
        learning_rate: float = 0.01,
        n_init_seeds: int = 10,
        seed: int = 2024,
    ) -> np.ndarray | None:
        """Assign gRNAs to cells using a Poisson-Gaussian mixture model.

        The model, priors and per-guide thresholding rule are those of ``crispat.ga_poisson_gauss`` and of pertpy's ``assign_mixture_model``.
        For each guide, the log2 of its non-zero UMI counts is modelled as a mixture of a continuous Poisson background and a Gaussian on-target component with priors ``weights ~ Dirichlet([0.9, 0.1])``, ``mu ~ Normal(3, 2)``, ``scale ~ LogNormal(2, 1)`` and ``lam ~ LogNormal(0, 1)``.
        A cell is assigned to a guide if its UMI count is at least the smallest integer ``t`` for which ``P(Normal | log2(t)) > 0.5``.

        The MAP estimate is found by multi-start EM on the distinct count values of every guide, for all guides in parallel on the GPU.
        The lowest-loss solution is kept after discarding label-switched optima and degenerate optima in which the Gaussian weight vanishes or the Gaussian collapses onto a single count value.

        Parameters
        ----------
        adata
            Annotated data matrix with guide RNA counts.
        model
            The mixture model to use.
            Only ``"poisson_gauss_mixture"`` is supported.
        layer
            Layer with raw counts.
            Uses ``adata.X`` if ``None``.
        assigned_guides_key
            Per-cell assignment is saved on ``adata.obs[assigned_guides_key]``.
        output_layer
            The binary cell-by-guide assignment matrix is saved on ``adata.layers[output_layer]``, or not at all if ``None``.
            Unlike the per-cell assignment it is not capped by ``max_assignments_per_cell``.
        no_grna_assigned_key
            Key to use when a cell is negative for all gRNAs.
        max_assignments_per_cell
            Maximum number of gRNAs that can be assigned to a cell.
        multiple_grna_assigned_key
            Key to use when more than ``max_assignments_per_cell`` gRNAs are assigned.
        multiple_grna_assignment_string
            Separator used to join multiple gRNAs assigned to one cell.
        only_return_results
            If ``True``, do not modify ``adata`` and return the assignment array.
        show_progress
            Whether to print a progress line.
        n_iter
            Maximum number of EM iterations per start.
        learning_rate
            Deprecated and ignored.
        n_init_seeds
            Number of prior-median initialisations per guide.
        seed
            Random seed used for the prior-median initialisations.

        Returns
        -------
        The per-cell assignments if ``only_return_results`` is ``True``, else ``None``.
        """
        if model != "poisson_gauss_mixture":
            raise ValueError("Model not implemented. Please use 'poisson_gauss_mixture'.")
        X = X_to_GPU(_get_obs_rep(adata, layer=layer))
        guide_names = np.asarray(adata.var_names, dtype=str)
        n_guides = X.shape[1]

        X = _layout(X)
        data = X.data if cp_issparse(X) else X
        if bool((data < 0).any()):
            raise ValueError(
                "Data contains negative values. Please use non-negative data for guide "
                "assignment with the Mixture Model."
            )
        values_padded, weights_padded, n_distinct, nonzero_counts, max_counts = _distinct_values(X, n_guides)

        fittable = cp.asnumpy((nonzero_counts >= 2) & (max_counts >= 2))
        skipped = np.flatnonzero(~fittable)
        if skipped.size:
            warnings.warn(
                f"Skipping {skipped.size} guides with fewer than 2 expressing cells or a "
                f"maximum count below 2: {_format_guides(guide_names[skipped])}.",
                UserWarning,
                stacklevel=3,
            )

        fittable_idx = np.flatnonzero(fittable)
        estimates = np.full((n_guides, _ga.N_OUTPUTS), np.nan)
        if fittable_idx.size > 0:
            if show_progress:
                print(f"Fitting Poisson-Gaussian mixture for {fittable_idx.size} guides in parallel.")
            idx = cp.asarray(fittable_idx)
            estimates[fittable_idx] = _fit_mixture(
                values_padded[idx],
                weights_padded[idx],
                n_distinct[idx],
                max_counts[idx],
                n_iter=n_iter,
                n_init_seeds=n_init_seeds,
                seed=seed,
            )
            unassignable = fittable_idx[np.isnan(estimates[fittable_idx, 5])]
            if unassignable.size:
                warnings.warn(
                    f"No positive population found for {unassignable.size} guides, which "
                    "are not assigned to any cell: "
                    f"{_format_guides(guide_names[unassignable])}.",
                    UserWarning,
                    stacklevel=3,
                )

        rate, mean, std, weight, loss, thresholds = estimates.T
        binary = _assigned(X, cp.asarray(thresholds))
        cells = _cell_major(binary)
        assignments = _per_cell_assignments(
            cells.indptr,
            cells.indices,
            guide_names,
            no_grna_assigned_key=no_grna_assigned_key,
            max_assignments_per_cell=max_assignments_per_cell,
            multiple_grna_assigned_key=multiple_grna_assigned_key,
            multiple_grna_assignment_string=multiple_grna_assignment_string,
        )
        if only_return_results:
            return np.asarray(assignments)

        columns = (rate, mean, std, 1.0 - weight, weight, thresholds, loss)
        for column, estimate in zip(_MIXTURE_MODEL_VAR_COLUMNS, columns, strict=True):
            adata.var[column] = estimate
        adata.obs[assigned_guides_key] = assignments
        if output_layer is not None:
            _set_obs_rep(adata, binary, layer=output_layer)
        return None
