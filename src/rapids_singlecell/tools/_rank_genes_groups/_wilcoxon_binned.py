from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import cupyx.scipy.sparse as cpsp
import cupyx.scipy.special as cupyx_special
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._compat import DaskArray
from rapids_singlecell._cuda import _rank_stream_cuda as _rss
from rapids_singlecell._cuda import _wilcoxon_binned_cuda as _wb

from ._utils import MIN_GROUP_SIZE_WARNING, _get_column_block

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes

_AUTO_CHUNK_MEMORY_FRACTION = 0.60
_COUNT_BYTES = np.dtype(np.uint32).itemsize
_FLOAT_BYTES = np.dtype(np.float64).itemsize
_LOG1P_RANGE = (0.0, 15.0)  # covers log1p(x) for raw counts up to ~3.3 million
_DASK_N_BINS = 200
_DEFAULT_N_BINS = 1000


def _auto_chunk_width(
    n_genes: int,
    n_groups: int,
    n_hist_groups: int,
    n_bins_total: int,
    *,
    device_ids: list[int],
    ireference: int | None,
    tie_correct: bool,
    gathers_multi_gpu: bool,
) -> int:
    """Choose a histogram width from the least-free selected device."""
    free_bytes = []
    for device_id in device_ids:
        with cp.cuda.Device(device_id):
            free, _ = cp.cuda.runtime.memGetInfo()
            free_bytes.append(int(free))

    # Histograms are uint32. The gather device can briefly hold both its shard
    # and the reduced histogram; include both even though the memory pool can
    # normally reuse the released shard for downstream temporaries.
    histogram_copies = 2 if gathers_multi_gpu else 1
    bytes_per_gene = histogram_copies * n_hist_groups * n_bins_total * _COUNT_BYTES

    if ireference is None:
        # One-vs-rest keeps a float64 histogram plus per-bin totals, cumulative
        # counts, and midranks. Tie correction adds one more per-bin plane.
        bytes_per_gene += n_groups * n_bins_total * _FLOAT_BYTES
        n_bin_planes = 3 + int(tie_correct)
        bytes_per_gene += n_bin_planes * n_bins_total * _FLOAT_BYTES
    else:
        # Reference comparisons materialize pair totals, cumulative counts,
        # midranks, and the float64 histogram for every group and bin.
        n_group_bin_bytes = _COUNT_BYTES + 3 * _FLOAT_BYTES
        if tie_correct:
            n_group_bin_bytes += _FLOAT_BYTES
        bytes_per_gene += n_groups * n_bins_total * n_group_bin_bytes

    # Account for group-wise statistic/result temporaries, then reserve the
    # remaining 40% for the input, allocator fragmentation, and other clients.
    bytes_per_gene += n_groups * 8 * _FLOAT_BYTES
    usable_bytes = int(min(free_bytes) * _AUTO_CHUNK_MEMORY_FRACTION)
    return min(n_genes, max(1, usable_bytes // max(bytes_per_gene, 1)))


def _fill_sparse_zero_bin(hist: cp.ndarray, group_counts: cp.ndarray) -> None:
    """Fill sparse histogram bin 0 from group size minus nonzero-bin counts."""
    nonzero_per_group = hist.sum(axis=2)  # (n_genes, n_groups)
    hist[:, :, 0] = group_counts[None, :].astype(cp.uint32) - nonzero_per_group


def _data_range(X) -> tuple[float, float]:
    """Compute (min, max) of the data, including implicit zeros for sparse."""
    if isinstance(X, DaskArray):
        if cpsp.issparse(X._meta):
            # Dask sparse: min is 0 (structural zeros).
            # Compute max per block, then global max.
            def _block_max(block, block_info=None):
                if block.nnz > 0:
                    return block.data.max().reshape(1)
                return cp.zeros(1, dtype=block.dtype)

            maxes = X.map_blocks(
                _block_max,
                dtype=X.dtype,
                drop_axis=1,
                chunks=((1,) * len(X.chunks[0]),),
            )
            return 0.0, float(maxes.max().compute())
        import dask

        lo, hi = dask.compute(X.min(), X.max())
        return float(lo), float(hi)
    if cpsp.issparse(X) or sp.issparse(X):
        if X.nnz == 0:
            return 0.0, 0.0
        d = X.data
        return min(0.0, float(d.min())), float(d.max())
    return float(X.min()), float(X.max())


def wilcoxon_binned(
    rg: _RankGenes,
    *,
    tie_correct: bool = False,
    use_continuity: bool = False,
    n_bins: int | None = None,
    chunk_size: int | None = None,
    bin_range: Literal["log1p", "auto"] | None = None,
) -> list[tuple[int, NDArray, NDArray]]:
    """Histogram-based approximate Wilcoxon rank-sum test."""
    if not rg.is_log1p:
        warnings.warn(
            "wilcoxon_binned expects log-normalized data "
            "(adata.uns['log1p'] not found).",
            UserWarning,
            stacklevel=4,
        )

    X = rg.X
    ireference = rg.ireference

    if n_bins is None:
        n_bins = _DASK_N_BINS if isinstance(X, DaskArray) else _DEFAULT_N_BINS

    n_groups = len(rg.groups_order)
    n_cells, n_genes = X.shape
    group_sizes = rg.group_sizes

    # Dask sparse cannot bin negatives correctly because implicit zeros use bin 0.
    # Refuse instead of silently mis-ranking; in-memory sparse uses dense fallback.
    if isinstance(X, DaskArray) and cpsp.issparse(X._meta):

        def _block_data_min(block):
            if block.nnz > 0:
                return block.data.min().reshape(1)
            return cp.zeros(1, dtype=block.dtype)

        data_min = float(
            X.map_blocks(
                _block_data_min,
                dtype=X.dtype,
                drop_axis=1,
                chunks=((1,) * len(X.chunks[0]),),
            )
            .min()
            .compute()
        )
        if data_min < 0:
            raise ValueError(
                "wilcoxon_binned does not support negative values in Dask "
                "sparse input; the binned approximation mis-ranks implicit "
                "zeros. Densify the data or use a nonnegative representation."
            )

    # group_codes use n_groups as sentinel for unselected cells.
    # vs-rest bins sentinels for totals; vs-reference kernels skip them.
    group_codes_np = rg.group_codes
    has_unselected = bool(np.any(group_codes_np == n_groups))

    # One-vs-one only ranks selected groups; filter in-memory rows.
    # Dask keeps rows but kernels skip sentinels, avoiding dummy-group atomics.
    if ireference is not None and has_unselected:
        if isinstance(X, DaskArray):
            has_unselected = False
        else:
            selected = group_codes_np != n_groups
            X = X[selected]
            group_codes_np = group_codes_np[selected]
            n_cells = int(group_sizes.sum())
            has_unselected = False

    if has_unselected:
        n_dummy = n_cells - group_sizes.sum()
        n_cells_per_group_hist = np.concatenate([group_sizes, np.array([n_dummy])])
    else:
        n_cells_per_group_hist = group_sizes

    # Warn for small groups
    if ireference is not None:
        n_ref = int(group_sizes[ireference])
        for gi, (name, size) in enumerate(
            zip(rg.groups_order, group_sizes, strict=True)
        ):
            if gi == ireference:
                continue
            if size <= MIN_GROUP_SIZE_WARNING or n_ref <= MIN_GROUP_SIZE_WARNING:
                warnings.warn(
                    f"Group {name} has size {size} (reference {n_ref}); normal "
                    "approximation of the Wilcoxon statistic may be inaccurate.",
                    RuntimeWarning,
                    stacklevel=4,
                )
    else:
        for name, size in zip(rg.groups_order, group_sizes, strict=True):
            rest = n_cells - size
            if size <= MIN_GROUP_SIZE_WARNING or rest <= MIN_GROUP_SIZE_WARNING:
                warnings.warn(
                    f"Group {name} has size {size} (rest {rest}); normal "
                    "approximation of the Wilcoxon statistic may be inaccurate.",
                    RuntimeWarning,
                    stacklevel=4,
                )

    # Resolve bin range: None → auto for in-memory, log1p for Dask
    if bin_range is None:
        bin_range = "log1p" if isinstance(X, DaskArray) else "auto"

    # The fixed log1p range assumes nonnegative data.
    # Signed sparse fallback needs data-driven auto range to avoid clamping.
    if rg._sparse_negative_fallback and bin_range == "log1p":
        warnings.warn(
            "bin_range='log1p' is invalid for sparse input with negative values "
            "(the fixed [0, 15] range would clamp them); using bin_range='auto'.",
            RuntimeWarning,
            stacklevel=4,
        )
        bin_range = "auto"

    # Prepare GPU arrays and bin arithmetic
    if bin_range == "auto":
        bin_low, bin_high = _data_range(X)
    else:
        bin_low, bin_high = _LOG1P_RANGE
    n_bins_total = n_bins + 1
    bin_width = bin_high - bin_low
    if bin_width <= 0:
        bin_width = 1.0
    inv_bin_width = float(n_bins / bin_width)

    group_codes = cp.asarray(group_codes_np, dtype=cp.int32)
    n_cells_per_group = cp.asarray(group_sizes, dtype=cp.int64)
    n_cells_per_group_hist_gpu = cp.asarray(n_cells_per_group_hist, dtype=cp.int64)

    batch_kwargs = {
        "group_codes": group_codes,
        "n_groups": n_groups,
        "n_bins": n_bins,
        "bin_low": bin_low,
        "inv_bin_width": inv_bin_width,
        "n_bins_total": n_bins_total,
        "n_cells_per_group": n_cells_per_group,
        "n_cells_total": n_cells,
        "n_cells_per_group_hist": n_cells_per_group_hist_gpu,
        "total_counts_from_all": has_unselected,
        "tie_correct": tie_correct,
        "use_continuity": use_continuity,
        "ireference": ireference,
        "force_dense": rg._sparse_negative_fallback,
    }

    # Pre-allocate output
    all_z = np.empty((n_groups, n_genes), dtype=np.float64)
    all_p = np.empty((n_groups, n_genes), dtype=np.float64)

    # Fuse exact-mean accumulation into the histogram pass for host input (one
    # transfer, not a separate _basic_stats pass). CSC/dense chunks accumulate
    # their disjoint columns. CSR accumulates every column during the first
    # histogram window only; later windows disable the fused planes.
    is_host = isinstance(X, np.ndarray) or sp.issparse(X)
    fuse_means = (
        is_host
        and not rg._sparse_negative_fallback
        and (
            isinstance(X, np.ndarray) or (sp.issparse(X) and X.format in {"csr", "csc"})
        )
    )
    n_hist_groups = n_cells_per_group_hist.shape[0]
    from ._stream_multi_gpu import resolve_stream_devices

    device_ids = (
        resolve_stream_devices(multi_gpu=rg._multi_gpu)
        if is_host
        else [cp.cuda.Device().id]
    )
    if fuse_means:
        group_sums_ext = cp.zeros((n_groups + 1, n_genes), dtype=cp.float64)
        group_nnz_ext = (
            cp.zeros((n_groups + 1, n_genes), dtype=cp.float64) if rg.comp_pts else None
        )
        batch_kwargs["group_sums"] = group_sums_ext
        batch_kwargs["group_nnz"] = group_nnz_ext
    else:
        rg._basic_stats()

    use_multi_stream = fuse_means and len(device_ids) > 1
    if use_multi_stream:
        from ._stream_multi_gpu import run_binned_hist_multi

    if chunk_size is not None:
        chunk_width = chunk_size
    else:
        chunk_width = _auto_chunk_width(
            n_genes,
            n_groups,
            n_hist_groups,
            n_bins_total,
            device_ids=device_ids,
            ireference=ireference,
            tie_correct=tie_correct,
            gathers_multi_gpu=use_multi_stream,
        )

    is_host_csr = sp.issparse(X) and X.format == "csr"
    is_host_csc = sp.issparse(X) and X.format == "csc"
    for start in range(0, n_genes, chunk_width):
        stop = min(start + chunk_width, n_genes)

        if use_multi_stream:
            # CSR aggregation covers all genes, so fuse it only into the first
            # histogram window. Dense/CSC aggregation follows the gene windows.
            accumulate_means = not is_host_csr or start == 0
            hist, sums_b, nnz_b = run_binned_hist_multi(
                X,
                group_codes_np,
                device_ids,
                n_groups=n_groups,
                n_hist_groups=n_hist_groups,
                n_bins=n_bins,
                bin_low=bin_low,
                inv_bin_width=inv_bin_width,
                comp_pts=rg.comp_pts,
                start=start,
                stop=stop,
                accumulate_means=accumulate_means,
            )
            z_b, p_b = _finish_binned_stats(
                hist,
                is_sparse=sp.issparse(X),
                n_groups=n_groups,
                n_cells_per_group=n_cells_per_group,
                n_cells_total=n_cells,
                n_cells_per_group_hist=n_cells_per_group_hist_gpu,
                total_counts_from_all=has_unselected,
                ireference=ireference,
                tie_correct=tie_correct,
                use_continuity=use_continuity,
            )
            if accumulate_means:
                if is_host_csc:
                    group_sums_ext[:, start:stop] = sums_b
                    if group_nnz_ext is not None:
                        group_nnz_ext[:, start:stop] = nnz_b
                else:
                    group_sums_ext += sums_b
                    if group_nnz_ext is not None:
                        group_nnz_ext += nnz_b
        else:
            z_b, p_b = process_gene_batch(X, start=start, stop=stop, **batch_kwargs)
            if is_host_csr and fuse_means and start == 0:
                # The CSR fused aggregation traverses every stored value, not
                # only the histogram window. Do not add it again next batch.
                batch_kwargs["group_sums"] = None
                batch_kwargs["group_nnz"] = None

        all_z[:, start:stop] = cp.asnumpy(z_b)
        all_p[:, start:stop] = cp.asnumpy(p_b)

    if fuse_means:
        _fill_binned_means(rg, group_sums_ext, group_nnz_ext, n_cells)

    return [
        (group_index, all_z[group_index], all_p[group_index])
        for group_index in range(n_groups)
        if group_index != ireference
    ]


def process_gene_batch(
    X,
    *,
    start: int,
    stop: int,
    group_codes: cp.ndarray,
    n_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    n_bins_total: int,
    n_cells_per_group: cp.ndarray,
    n_cells_total: int,
    n_cells_per_group_hist: cp.ndarray,
    total_counts_from_all: bool,
    tie_correct: bool = False,
    use_continuity: bool = False,
    ireference: int | None = None,
    force_dense: bool = False,
    group_sums: cp.ndarray | None = None,
    group_nnz: cp.ndarray | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Process one gene batch, dispatching on Dask vs in-memory.

    When ``group_sums`` is given, host branches also accumulate exact group sums
    (+ ``group_nnz``) on the streamed window (fused means, no ``_basic_stats``).
    """
    n_hist_groups = n_cells_per_group_hist.shape[0]
    n_genes_batch = stop - start

    is_sparse = False
    if force_dense and (cpsp.issparse(X) or sp.issparse(X)):
        # Negative sparse fallback: bin 0 is only correct for nonnegative data.
        # Densify the column window so dense bins span the full [min, max].
        hist = _launch_dense(
            _get_column_block(X, start, stop),
            group_codes,
            n_hist_groups,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
        )
    elif isinstance(X, np.ndarray):
        # In-memory host dense: stream column windows (no full-matrix copy).
        hist = _launch_dense_host(
            X,
            group_codes,
            n_hist_groups,
            start=start,
            stop=stop,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
            group_sums=group_sums,
            group_nnz=group_nnz,
        )
    elif sp.issparse(X) and X.format == "csc":
        hist = _launch_csc_host(
            X,
            group_codes,
            n_hist_groups,
            start=start,
            stop=stop,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
            group_sums=group_sums,
            group_nnz=group_nnz,
        )
        is_sparse = True
    elif sp.issparse(X) and X.format == "csr":
        hist = _launch_csr_host(
            X,
            group_codes,
            n_hist_groups,
            start=start,
            stop=stop,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
            group_sums=group_sums,
            group_nnz=group_nnz,
        )
        is_sparse = True
    elif isinstance(X, DaskArray):
        hist = _process_dask(
            X,
            start=start,
            stop=stop,
            group_codes=group_codes,
            n_hist_groups=n_hist_groups,
            n_genes_batch=n_genes_batch,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
            n_bins_total=n_bins_total,
        )
        is_sparse = cpsp.issparse(X._meta)
    elif isinstance(X, cpsp.csc_matrix):
        hist = _launch_csc(
            X,
            group_codes,
            n_hist_groups,
            start=start,
            stop=stop,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
        )
        is_sparse = True
    elif isinstance(X, cpsp.csr_matrix):
        hist = _launch_csr(
            X,
            group_codes,
            n_hist_groups,
            start=start,
            stop=stop,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
        )
        is_sparse = True
    else:
        hist = _launch_dense(
            X[:, start:stop],
            group_codes,
            n_hist_groups,
            n_bins=n_bins,
            bin_low=bin_low,
            inv_bin_width=inv_bin_width,
        )

    return _finish_binned_stats(
        hist,
        is_sparse=is_sparse,
        n_groups=n_groups,
        n_cells_per_group=n_cells_per_group,
        n_cells_total=n_cells_total,
        n_cells_per_group_hist=n_cells_per_group_hist,
        total_counts_from_all=total_counts_from_all,
        ireference=ireference,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
    )


def _finish_binned_stats(
    hist: cp.ndarray,
    *,
    is_sparse: bool,
    n_groups: int,
    n_cells_per_group: cp.ndarray,
    n_cells_total: int,
    n_cells_per_group_hist: cp.ndarray,
    total_counts_from_all: bool,
    ireference: int | None,
    tie_correct: bool,
    use_continuity: bool,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Zero-bin fill + z-scores/p-values from a full (n_genes, n_hist_groups, nbt)
    histogram (single-GPU chunk or the gathered multi-GPU histogram)."""
    # Sparse kernels only fill bins 1..n_bins; compute bin 0 (zeros) here.
    if is_sparse:
        _fill_sparse_zero_bin(hist, n_cells_per_group_hist)

    # If there's a dummy group (vs-rest with unselected cells),
    # compute total_counts from all groups before slicing off the dummy.
    tc = None
    if total_counts_from_all:
        tc = hist.sum(axis=1)
        hist = hist[:, :n_groups, :]

    if ireference is not None:
        return _compute_stats_vs_ref(
            hist,
            ireference,
            n_cells_per_group,
            tie_correct=tie_correct,
            use_continuity=use_continuity,
        )
    return _compute_stats(
        hist,
        n_cells_per_group,
        n_cells_total,
        total_counts=tc,
        tie_correct=tie_correct,
        use_continuity=use_continuity,
    )


def _compute_stats(
    hist: cp.ndarray,
    n_cells_per_group: cp.ndarray,
    n_cells_total: int,
    *,
    total_counts: cp.ndarray | None = None,
    tie_correct: bool = False,
    use_continuity: bool = False,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute Wilcoxon z-scores from histograms."""
    n = cp.int64(n_cells_total)

    if total_counts is None:
        total_counts = hist.sum(axis=1)
    cum_before = cp.cumsum(total_counts, axis=1) - total_counts
    midranks = cum_before + (total_counts + 1) / 2.0

    hist_f = hist.astype(cp.float64)
    rank_sums = cp.einsum("igb,ib->ig", hist_f, midranks).T

    n_g = n_cells_per_group[:, None].astype(cp.float64)
    n_rest = n - n_g
    expected = n_g * (n + 1) / 2.0
    variance = n_g * n_rest * (n + 1) / 12.0

    if tie_correct:
        # Each bin is a tie group; t = total_counts per bin per gene
        t = total_counts.astype(cp.float64)
        tie_term = (t * t * t - t).sum(axis=1)  # (n_genes,)
        tc = 1.0 - tie_term / (float(n) ** 3 - float(n))
        variance = variance * tc[None, :]

    diff = rank_sums - expected
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z_scores = diff / cp.sqrt(variance)
    cp.nan_to_num(z_scores, copy=False)
    pvals = cupyx_special.erfc(cp.abs(z_scores) * cp.float64(cp.sqrt(0.5)))

    return z_scores, pvals


def _compute_stats_vs_ref(
    hist: cp.ndarray,
    ireference: int,
    n_cells_per_group: cp.ndarray,
    *,
    tie_correct: bool = False,
    use_continuity: bool = False,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Compute Wilcoxon z-scores for each group vs a specific reference."""
    # hist shape: (n_genes, n_groups, n_bins_total)
    ref_hist = hist[:, ireference : ireference + 1, :]  # (n_genes, 1, n_bins_total)

    # Pairwise total per group: counts from group_i + reference
    pair_total = hist + ref_hist  # broadcasts over group axis

    # Midranks from pairwise cumulative counts
    cum_before = cp.cumsum(pair_total, axis=2) - pair_total
    midranks = cum_before + (pair_total + 1) / 2.0

    # Rank sum: for each group sum hist[gene, g, bin] * midranks[gene, g, bin]
    hist_f = hist.astype(cp.float64)
    rank_sums = cp.einsum("igb,igb->ig", hist_f, midranks).T  # (n_groups, n_genes)

    n_g = n_cells_per_group[:, None].astype(cp.float64)
    n_r = cp.float64(n_cells_per_group[ireference])
    n_combined = n_g + n_r

    expected = n_g * (n_combined + 1) / 2.0
    variance = n_g * n_r * (n_combined + 1) / 12.0

    if tie_correct:
        # Each bin is a tie group; t = pair_total per bin
        t = pair_total.astype(cp.float64)
        tie_term = (t * t * t - t).sum(axis=2)  # (n_genes, n_groups)
        tc = 1.0 - tie_term.T / (n_combined**3 - n_combined)  # (n_groups, n_genes)
        variance = variance * tc

    diff = rank_sums - expected
    if use_continuity:
        diff = cp.sign(diff) * cp.maximum(cp.abs(diff) - 0.5, 0.0)
    z_scores = diff / cp.sqrt(variance)
    cp.nan_to_num(z_scores, copy=False)
    pvals = cupyx_special.erfc(cp.abs(z_scores) * cp.float64(cp.sqrt(0.5)))

    return z_scores, pvals


def _launch_dense(
    chunk: cp.ndarray,
    group_codes: cp.ndarray,
    n_groups: int,
    *,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
) -> cp.ndarray:
    n_cells, n_genes = chunk.shape
    chunk_f = cp.asfortranarray(chunk)
    hist = cp.zeros((n_genes, n_groups, n_bins + 1), dtype=cp.uint32)

    _wb.dense_hist(
        chunk_f,
        group_codes,
        hist,
        n_cells=n_cells,
        n_genes=n_genes,
        n_groups=n_groups,
        n_bins=n_bins,
        bin_low=float(bin_low),
        inv_bin_width=float(inv_bin_width),
        stream=cp.cuda.get_current_stream().ptr,
    )
    return hist


def _launch_csc(
    X: cpsp.csc_matrix,
    group_codes: cp.ndarray,
    n_groups: int,
    *,
    start: int,
    stop: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
) -> cp.ndarray:
    """Read directly from CSC indptr via gene_start — no column slicing."""
    n_cells = X.shape[0]
    n_genes = stop - start
    hist = cp.zeros((n_genes, n_groups, n_bins + 1), dtype=cp.uint32)

    _wb.csc_hist(
        X.data,
        X.indices,
        X.indptr,
        group_codes,
        hist,
        n_cells=n_cells,
        n_genes=n_genes,
        n_groups=n_groups,
        n_bins=n_bins,
        bin_low=float(bin_low),
        inv_bin_width=float(inv_bin_width),
        gene_start=start,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return hist


def _launch_csr(
    X: cpsp.csr_matrix,
    group_codes: cp.ndarray,
    n_groups: int,
    *,
    start: int,
    stop: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
) -> cp.ndarray:
    """Read directly from CSR via gene_start — no column slicing."""
    n_cells = X.shape[0]
    n_genes = stop - start
    hist = cp.zeros((n_genes, n_groups, n_bins + 1), dtype=cp.uint32)

    _wb.csr_hist(
        X.data,
        X.indices,
        X.indptr,
        group_codes,
        hist,
        n_cells=n_cells,
        n_genes=n_genes,
        n_groups=n_groups,
        n_bins=n_bins,
        bin_low=float(bin_low),
        inv_bin_width=float(inv_bin_width),
        gene_start=start,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return hist


def _host_hist_data(data):
    """Cast host sparse values to a kernel dtype (float32/float64) as needed."""
    if data.dtype in (np.float32, np.float64):
        return data
    return data.astype(np.float32)


def _fill_binned_means(
    rg: _RankGenes,
    group_sums_ext: cp.ndarray,
    group_nnz_ext: cp.ndarray | None,
    n_cells: int,
) -> None:
    """Fill exact means/pts from fused ``(n_groups+1, n_genes)`` sums (row
    ``n_groups`` = unselected cells, so total = sum across rows)."""
    n_groups = len(rg.groups_order)
    group_sums = group_sums_ext[:n_groups]
    sizes = cp.asarray(rg.group_sizes, dtype=cp.float64)[:, None]
    rg.means = cp.asnumpy(group_sums / sizes)
    rg.vars = None
    rg.pts = (
        cp.asnumpy(group_nnz_ext[:n_groups] / sizes)
        if group_nnz_ext is not None
        else None
    )
    if rg.ireference is None:
        n_rest = cp.float64(n_cells) - sizes
        total_sums = group_sums_ext.sum(axis=0, keepdims=True)
        rg.means_rest = cp.asnumpy((total_sums - group_sums) / n_rest)
        rg.vars_rest = None
        if group_nnz_ext is not None:
            total_nnz = group_nnz_ext.sum(axis=0, keepdims=True)
            rg.pts_rest = cp.asnumpy((total_nnz - group_nnz_ext[:n_groups]) / n_rest)
        else:
            rg.pts_rest = None
    else:
        rg.means_rest = None
        rg.vars_rest = None
        rg.pts_rest = None


def _launch_csr_host(
    X,
    group_codes: cp.ndarray,
    n_groups: int,
    *,
    start: int,
    stop: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    group_sums: cp.ndarray | None = None,
    group_nnz: cp.ndarray | None = None,
) -> cp.ndarray:
    """Host CSR histogram: stream row-blocks, filling the [start, stop) window."""
    n_cells, n_genes = X.shape
    hist = cp.zeros((stop - start, n_groups, n_bins + 1), dtype=cp.uint32)
    _rss.hist_csr_host(
        _host_hist_data(X.data),
        X.indices,
        X.indptr,
        group_codes,
        hist,
        group_sums=group_sums,
        group_nnz=group_nnz,
        n_cells=n_cells,
        n_genes=n_genes,
        n_groups=n_groups,
        n_bins=n_bins,
        bin_low=float(bin_low),
        inv_bin_width=float(inv_bin_width),
        col_start=start,
        col_stop=stop,
    )
    return hist


def _launch_csc_host(
    X,
    group_codes: cp.ndarray,
    n_groups: int,
    *,
    start: int,
    stop: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    group_sums: cp.ndarray | None = None,
    group_nnz: cp.ndarray | None = None,
) -> cp.ndarray:
    """Host CSC histogram: stream the [start, stop) column window."""
    n_cells, n_genes = X.shape
    hist = cp.zeros((stop - start, n_groups, n_bins + 1), dtype=cp.uint32)
    _rss.hist_csc_host(
        _host_hist_data(X.data),
        X.indices,
        X.indptr,
        group_codes,
        hist,
        group_sums=group_sums,
        group_nnz=group_nnz,
        n_cells=n_cells,
        n_genes=n_genes,
        n_groups=n_groups,
        n_bins=n_bins,
        bin_low=float(bin_low),
        inv_bin_width=float(inv_bin_width),
        col_start=start,
        col_stop=stop,
    )
    return hist


def _launch_dense_host(
    X: np.ndarray,
    group_codes: cp.ndarray,
    n_groups: int,
    *,
    start: int,
    stop: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    group_sums: cp.ndarray | None = None,
    group_nnz: cp.ndarray | None = None,
) -> cp.ndarray:
    """Host dense histogram: stream the [start, stop) column window."""
    hist = cp.zeros((stop - start, n_groups, n_bins + 1), dtype=cp.uint32)
    Xh = X if X.dtype in (np.float32, np.float64) else X.astype(np.float32)
    if not (Xh.flags.c_contiguous or Xh.flags.f_contiguous):
        Xh = np.ascontiguousarray(Xh)
    _rss.hist_dense_host(
        Xh,
        group_codes,
        hist,
        group_sums=group_sums,
        group_nnz=group_nnz,
        n_groups=n_groups,
        n_bins=n_bins,
        bin_low=float(bin_low),
        inv_bin_width=float(inv_bin_width),
        col_start=start,
        col_stop=stop,
    )
    return hist


def _process_dask(
    X,
    *,
    start: int,
    stop: int,
    group_codes: cp.ndarray,
    n_hist_groups: int,
    n_genes_batch: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    n_bins_total: int,
) -> cp.ndarray:
    """Build a column-range histogram from an unsliced Dask array."""
    import dask.array as da

    if cpsp.isspmatrix_csr(X._meta):

        def _hist_block(block, block_info=None):
            if block_info is None or block_info == []:
                return cp.zeros(
                    (1, n_genes_batch, n_hist_groups, n_bins_total), dtype=cp.uint32
                )
            row_start = block_info[0]["array-location"][0][0]
            row_stop = block_info[0]["array-location"][0][1]
            codes_chunk = group_codes[row_start:row_stop]
            hist = cp.zeros(
                (n_genes_batch, n_hist_groups, n_bins_total), dtype=cp.uint32
            )
            _wb.csr_hist(
                block.data,
                block.indices,
                block.indptr,
                codes_chunk,
                hist,
                n_cells=block.shape[0],
                n_genes=n_genes_batch,
                n_groups=n_hist_groups,
                n_bins=n_bins,
                bin_low=float(bin_low),
                inv_bin_width=float(inv_bin_width),
                gene_start=start,
                stream=cp.cuda.get_current_stream().ptr,
            )
            return hist[None, ...]

    elif isinstance(X._meta, cp.ndarray):

        def _hist_block(block, block_info=None):
            if block_info is None or block_info == []:
                return cp.zeros(
                    (1, n_genes_batch, n_hist_groups, n_bins_total), dtype=cp.uint32
                )
            row_start = block_info[0]["array-location"][0][0]
            row_stop = block_info[0]["array-location"][0][1]
            codes_chunk = group_codes[row_start:row_stop]

            blk = cp.asfortranarray(cp.asarray(block[:, start:stop]))
            hist = cp.zeros(
                (n_genes_batch, n_hist_groups, n_bins_total), dtype=cp.uint32
            )
            _wb.dense_hist(
                blk,
                codes_chunk,
                hist,
                n_cells=blk.shape[0],
                n_genes=n_genes_batch,
                n_groups=n_hist_groups,
                n_bins=n_bins,
                bin_low=float(bin_low),
                inv_bin_width=float(inv_bin_width),
                stream=cp.cuda.get_current_stream().ptr,
            )
            return hist[None, ...]

    partial_hists = da.map_blocks(
        _hist_block,
        X,
        dtype=cp.uint32,
        meta=cp.empty((), dtype=cp.uint32),
        drop_axis=1,
        new_axis=[1, 2, 3],
        chunks=(
            tuple(1 for _ in X.chunks[0]),
            (n_genes_batch,),
            (n_hist_groups,),
            (n_bins_total,),
        ),
    )
    return partial_hists.sum(axis=0).compute()
