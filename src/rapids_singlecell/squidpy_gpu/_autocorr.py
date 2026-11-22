from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from itertools import batched
from typing import (
    TYPE_CHECKING,
    Literal,
)

import cupy as cp
import numpy as np
import pandas as pd
from anndata.abc import CSCDataset, CSRDataset
from cupyx.scipy import sparse as sparse_gpu
from dask.array import Array as DaskArray
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from rapids_singlecell._cuda import _autocorr_cuda as _ac
from rapids_singlecell._utils import _copy_to_device, parse_device_ids

from ._spatial_data import _extract_adata
from ._utils import _check_precision_issues, _g_moments, _p_value_calc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from spatialdata import SpatialData


# Device memory for a block's dense genes and their lags.
_BLOCK_FRACTION = 0.5


# Bytes of the source matrix uploaded at a time while selecting genes.
_CHUNK_BYTES = 128 << 20


def _csr_chunks(matrix, size):
    """Row views limited by both entry and row-pointer bytes, apart from a single oversized row."""
    rows = max(1, size // matrix.indptr.itemsize)
    entries = max(1, size // (matrix.data.itemsize + matrix.indices.itemsize))
    start = 0
    while start < matrix.shape[0]:
        lo = int(matrix.indptr[start])
        stop = int(matrix.indptr.searchsorted(lo + entries, side="right")) - 1
        stop = min(start + rows, max(start + 1, stop), matrix.shape[0])
        hi = matrix.indptr[stop]
        # SciPy prunes small NumPy views into large base arrays by copying them.
        chunk = sparse.csr_matrix(
            (memoryview(matrix.data)[lo:hi], memoryview(matrix.indices)[lo:hi], matrix.indptr[start : stop + 1] - lo),
            shape=(stop - start, matrix.shape[1]),
        )
        if getattr(matrix, "_has_canonical_format", False):
            chunk.has_canonical_format = True
        yield start, chunk
        start = stop


def _to_gpu_csr(matrix, dtype):
    device = cp.cuda.Device().id
    if sparse.issparse(matrix):
        matrix = matrix.tocsr()
        arrays = (cp.asarray(matrix.data, dtype=dtype), cp.asarray(matrix.indices), cp.asarray(matrix.indptr))
    elif sparse_gpu.issparse(matrix) and matrix.data.device.id != device:
        with cp.cuda.Device(matrix.data.device.id):
            matrix = matrix.tocsr()
        arrays = tuple(_copy_to_device(a, device) for a in (matrix.data, matrix.indices, matrix.indptr))
    else:
        if isinstance(matrix, cp.ndarray):
            matrix = _copy_to_device(matrix, device)
        elif isinstance(matrix, np.ndarray):
            matrix = cp.asarray(matrix, dtype=dtype)
        return sparse_gpu.csr_matrix(matrix, dtype=dtype)
    result = sparse_gpu.csr_matrix(arrays, shape=matrix.shape, dtype=dtype)
    if getattr(matrix, "_has_canonical_format", False):
        result.has_canonical_format = True
    return result


def _select(chunk, lut, *, start, width, n_genes, n_cells, dtype):
    """Pack selected entries by gene block, keeping one flat index and value per entry."""
    if sparse.issparse(chunk) or sparse_gpu.issparse(chunk):
        chunk = _to_gpu_csr(chunk, dtype)
        if not chunk.has_canonical_format:
            chunk = chunk.copy()
            chunk.sum_duplicates()
        slots = lut[cp.asarray(chunk.indices)]
        kept = cp.flatnonzero(slots >= 0)
        rows = cp.searchsorted(cp.asarray(chunk.indptr)[1:], kept, side="right")
        slots = slots[kept]
        values = cp.asarray(chunk.data)[kept]
        del chunk, kept
    else:
        columns = cp.flatnonzero(lut >= 0)
        dense = cp.asarray(chunk)[:, columns]
        rows, local_cols = cp.nonzero(dense)
        slots = lut[columns][local_cols]
        values = dense[rows, local_cols]
    block = slots // width
    stride = cp.minimum(width, n_genes - block * width)
    index_dtype = cp.int64 if n_cells * width > np.iinfo(np.int32).max else cp.int32
    indices = (rows.astype(index_dtype) + start) * stride + (slots % width)
    del rows, slots, stride
    values = values.astype(dtype, copy=False)
    if n_genes <= width:
        return indices, values, (0, len(values))
    order = cp.argsort(block)
    bounds = cp.searchsorted(block[order], cp.arange((n_genes + width - 1) // width + 1)).get()
    return indices[order], values[order], bounds


def _gene_source(source, cols, width, dtype):
    """Index sparse input once. Keep selected entries on the GPU while space permits."""
    if not len(cols) or isinstance(source, (np.ndarray, cp.ndarray)):
        return source
    n, n_vars = source.shape
    lut = cp.full(n_vars, -1, dtype=cp.int32)
    lut[cp.asarray(cols)] = cp.arange(len(cols), dtype=cp.int32)
    blocks = [[] for _ in range(0, len(cols), width)]
    free = cp.cuda.Device().mem_info[0]
    budget = free // 4
    chunk_bytes = min(_CHUNK_BYTES, max(1, free // 32))

    def pack(chunk, start, lookup):
        nonlocal budget
        indices, values, bounds = _select(
            chunk,
            lookup,
            start=start,
            width=width,
            n_genes=len(cols),
            n_cells=n,
            dtype=dtype,
        )
        size = indices.nbytes + values.nbytes
        if size > budget:
            indices, values = indices.get(), values.get()
        else:
            budget -= size
        for block, lo, hi in zip(blocks, bounds[:-1], bounds[1:], strict=True):
            if lo < hi:
                block.append((indices[lo:hi], values[lo:hi]))

    def pack_rows(matrix, row_offset, col_start):
        lookup = lut[col_start : col_start + matrix.shape[1]]
        if sparse.issparse(matrix):
            first, last = cols.searchsorted([col_start, col_start + matrix.shape[1]])
            local_cols = cols[first:last] - col_start
            if 8 * len(local_cols) < matrix.shape[1]:
                # Select small panels before upload. SciPy releases the GIL in its CSR column scan.
                lookup = cp.arange(first, last, dtype=cp.int32)
                if matrix.format == "csc":
                    for start, chunk in _csr_chunks(matrix[:, local_cols].tocsr(), chunk_bytes):
                        pack(chunk, row_offset + start, lookup)
                    return
                chunks = _csr_chunks(matrix.tocsr(), min(chunk_bytes, 32 << 20))
                with ThreadPoolExecutor(max_workers=16) as pool:
                    for batch in batched(chunks, 16, strict=False):
                        selected = pool.map(lambda piece: piece[1][:, local_cols], batch)
                        joined = sparse.vstack(list(selected), format="csr")
                        for start, chunk in _csr_chunks(joined, chunk_bytes):
                            pack(chunk, row_offset + batch[0][0] + start, lookup)
            else:
                for start, chunk in _csr_chunks(matrix.tocsr(), chunk_bytes):
                    pack(chunk, row_offset + start, lookup)
            return
        if sparse_gpu.issparse(matrix):
            matrix = matrix.tocsr()
            nbytes = matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
        else:
            nbytes = matrix.shape[0] * matrix.shape[1] * matrix.dtype.itemsize
        rows = max(1, int(chunk_bytes * matrix.shape[0] // max(nbytes, 1)))
        for start in range(0, matrix.shape[0], rows):
            chunk = matrix if sparse_gpu.issparse(matrix) and rows >= matrix.shape[0] else matrix[start : start + rows]
            pack(chunk, row_offset + start, lookup)

    if isinstance(source, DaskArray):
        col_start = 0
        for j, chunk_width in enumerate(source.chunks[1]):
            col_stop = col_start + chunk_width
            if np.any((cols >= col_start) & (cols < col_stop)):
                row_start = 0
                for i, height in enumerate(source.chunks[0]):
                    pack_rows(source.blocks[i, j].compute(), row_start, col_start)
                    row_start += height
            col_start = col_stop
    elif isinstance(source, (CSRDataset, CSCDataset)):
        # Slice the compressed axis on disk. A row slice of backed CSC reads all columns.
        by_row = isinstance(source, CSRDataset)
        minor = n_vars if by_row else n
        step = max(1, chunk_bytes // (minor * (source.dtype.itemsize + 8) + 8))
        if by_row:
            for start in range(0, n, step):
                pack_rows(source[start : start + step], start, 0)
        else:
            for first in range(0, len(cols), step):
                selected = cols[first : first + step]
                lookup = cp.arange(first, first + len(selected), dtype=cp.int32)
                for start, chunk in _csr_chunks(source[:, selected].tocsr(), chunk_bytes):
                    pack(chunk, start, lookup)
    else:
        pack_rows(source, 0, 0)
    return blocks


def _gene_block(source, cols, block_id, *, dtype, n, device_id):
    """Materialize only this device's selected columns in C order."""
    if isinstance(source, (np.ndarray, cp.ndarray)):
        selection = slice(cols[0], cols[-1] + 1) if cols[-1] - cols[0] + 1 == len(cols) else cols
        if isinstance(source, cp.ndarray):
            with cp.cuda.Device(source.device.id):
                x = cp.ascontiguousarray(source[:, selection], dtype=dtype)
            return _copy_to_device(x, device_id)
        return cp.asarray(source[:, selection], order="C", dtype=dtype)
    block = cp.zeros((n, len(cols)), dtype=dtype)
    for indices, values in source[block_id]:
        if isinstance(indices, cp.ndarray):
            indices = _copy_to_device(indices, device_id)
            values = _copy_to_device(values, device_id)
        else:
            indices, values = cp.asarray(indices), cp.asarray(values)
        block.ravel()[indices] = values
    return block


def _csr_gene_blocks(matrix, cols, width, dtype):
    """Expand contiguous genes in bounded tiles laid out for the scoring kernels."""
    local = _to_gpu_csr(matrix, dtype)
    if not local.has_canonical_format:
        local = local.copy()
        local.sum_duplicates()
    first, count = int(cols[0]), len(cols)
    n = local.shape[0]
    tile_blocks = max(1, _CHUNK_BYTES // (n * width * local.dtype.itemsize))
    indices = local.indices.astype(cp.int32, copy=False)
    for start in range(0, count, tile_blocks * width):
        remaining = min(count - start, tile_blocks * width)
        tile = cp.zeros(((remaining + width - 1) // width, n, width), dtype=local.dtype)
        _ac.csr_blocks(
            local.indptr, indices, local.data, first=first + start, out=tile, stream=cp.cuda.get_current_stream().ptr
        )
        for offset, block in enumerate(tile):
            yield cp.ascontiguousarray(block[:, : remaining - offset * width])


def _block_scores(x, graph, row, col, keys, *, total, uniform, geary, permutation=None):
    """Return the observed scores and the change under each permutation of graph rows."""
    n, n_genes = x.shape
    stream = cp.cuda.get_current_stream().ptr
    # Keep the accumulated lag in float64 so rounding does not break permutation ties.
    z = None if keys is None else cp.empty(x.shape, dtype=cp.float64)
    stats = cp.zeros((4, n_genes))
    # Preserve structural zeros in sparse genes. Fully nonzero genes use a data
    # value as origin to retain small variations on large expression offsets.
    offsets = cp.where(cp.all(x, axis=0), x[0], 0)
    _ac.observed(*graph, x, row, col, offsets=offsets, lag=z, stats=stats, geary=geary, stream=stream)
    sum_x, sum_x2, weighted_x, numerator = stats
    mean = sum_x / n
    ss = sum_x2 - mean * sum_x
    if geary:
        scale = (n - 1) / (2 * total * ss)
        observed = numerator * scale
    else:
        scale = (n / total) / ss
        observed = (numerator + mean * (mean * total - weighted_x)) * scale
    if keys is None:
        return observed, None
    out = cp.zeros((len(keys), 1 if uniform else 2, n_genes))
    _ac.sums(
        x, z, row, keys, offsets=offsets, out=out, geary=geary, uniform=uniform, stream=stream, permutation=permutation
    )
    a = out[:, 0]
    b = 0 if uniform else out[:, 1]
    change = (b - 2 * a if geary else a - mean * b) * scale
    return observed, change


def _permutation_counts(percentages: Sequence[float], n_perms: int | None, n_devices: int) -> list[int]:
    if n_perms is None or n_perms < 1:
        raise ValueError("permutation_percentages requires a positive n_perms.")
    weights = np.asarray(percentages, dtype=np.float64)
    if weights.shape != (n_devices,):
        raise ValueError("permutation_percentages must have one percentage per selected GPU.")
    total = weights.sum()
    if not np.all(np.isfinite(weights) & (weights >= 0)) or not np.isclose(total, 100, rtol=0, atol=1e-6):
        raise ValueError("permutation_percentages must contain finite, nonnegative percentages that sum to 100.")
    quotas = n_perms * (weights / total)
    counts = np.floor(quotas).astype(np.int64)
    # Assign the remaining draws by fractional remainder, breaking ties in device order.
    order = np.argsort(counts - quotas, kind="stable")
    counts[order[: n_perms - counts.sum()]] += 1
    return counts.tolist()


def spatial_autocorr(
    adata: AnnData | SpatialData,
    *,
    table_key: str | None = None,
    connectivity_key: str = "spatial_connectivities",
    genes: str | Sequence[str] | None = None,
    mode: Literal["moran", "geary"] = "moran",
    transformation: bool = True,
    n_perms: int | None = None,
    two_tailed: bool = False,
    corr_method: str | None = "fdr_bh",
    layer: str | None = None,
    use_raw: bool = False,
    use_sparse: bool = True,
    dtype: np.dtype | None = None,
    multi_gpu: bool | list[int] | str | None = None,
    permutation_percentages: Sequence[float] | None = None,
    seed: int | None = None,
    copy: bool = False,
) -> pd.DataFrame | None:
    """
    Calculate spatial autocorrelation for genes in an AnnData object.

    This function computes spatial autocorrelation scores (Moran's I or Geary's C) for each gene in an AnnData object.
    The function also calculates p-values and corrected p-values for multiple testing.

    Note:
        The precision (float32 or float64) is determined by the input data dtype.
        For best numerical stability, especially with genes having low variance,
        use float64 input data.

    Parameters
    ----------
        adata
            Annotated data matrix or a SpatialData object containing the selected table.
        table_key
            Key in ``SpatialData.tables``; required for SpatialData input.
            All reads and writes use this table; ignored for AnnData input.
        connectivity_key
            Key of the connectivity matrix in `adata.obsp`, by default "spatial_connectivities".
        genes
            Genes for which to compute the autocorrelation scores. If None, all genes or highly variable genes will be used.
        mode
            Spatial autocorrelation method to use, either "moran" or "geary", by default "moran".
        transformation
            If True, row-normalize the connectivity matrix, by default True.
            Moran's I includes the factor `n_obs / sum(weights)` after this optional normalization.
        n_perms
            Number of permutations for calculating p-values, by default None.
        two_tailed
            If True, calculate two-tailed p-values, by default False.
        corr_method
            Multiple testing correction method, by default "fdr_bh". Each p-value column
            is corrected over the genes with a finite p-value in that column.
        layer
            Layer in the AnnData object to use, by default None.
        use_raw
            If True, use the raw data in the AnnData object, by default False.
        use_sparse
            Retained for compatibility. Sparse input is processed in gene blocks.
        dtype
            Data type for computation. If None, uses input data dtype.
            For best numerical stability, especially with genes having low variance, use `np.float64`.
        multi_gpu
            GPUs that share the calculation. By default, each GPU scores a separate block of genes.
            - None: Use all GPUs if available (default)
            - True: Use all available GPUs
            - False: Use only the current GPU
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")
        permutation_percentages
            Split permutations across GPUs in the order selected by `multi_gpu`.
            For example, `multi_gpu=[0, 1], permutation_percentages=[60, 40]` assigns
            60% of permutations to GPU 0 and 40% to GPU 1. Each active GPU scores all
            selected genes. Percentages must be nonnegative and sum to 100, and `n_perms`
            must be positive. Extra draws go to the largest fractional remainders, with
            ties resolved in device order. A GPU assigned zero draws skips scoring.
        seed
            Seed for the keyed permutations of the graph rows, shared across genes and GPUs.
        copy
            If True, return the results as a DataFrame instead of storing them in `adata.uns`, by default False.

    Returns
    -------
            DataFrame containing the autocorrelation scores, p-values, and corrected p-values for each gene. \
            If `copy` is False, the results are stored in `adata.uns` and None is returned.
    """
    adata = _extract_adata(adata, table_key=table_key)
    if genes is None:
        if "highly_variable" in adata.var:
            genes = adata[:, adata.var["highly_variable"]].var_names.values
        else:
            genes = adata.var_names.values
    if isinstance(genes, str):
        genes = [genes]
    if use_raw:
        if adata.raw is None:
            raise AttributeError("No `.raw` attribute found. Try specifying `use_raw=False`.")
        genes = [gene for gene in genes if gene in adata.raw.var_names]
        source, var_names = adata.raw.X, adata.raw.var_names
    else:
        source = adata.layers[layer] if layer is not None else adata.X
        var_names = adata.var_names
    cols = pd.Index(var_names).get_indexer(genes)
    if (cols < 0).any():
        raise KeyError(f"Genes not found: {list(np.asarray(genes)[cols < 0])[:5]}")

    if mode not in ("moran", "geary"):
        raise ValueError(f"Invalid mode: {mode}")
    # Determine dtype from parameter or input data
    compute_dtype = np.dtype(dtype) if dtype is not None else np.dtype(source.dtype)
    if compute_dtype not in (np.float32, np.float64):
        compute_dtype = np.dtype(np.float32)

    n = adata.n_obs
    if not 2 <= n <= np.iinfo(np.int32).max:
        raise ValueError("Spatial autocorrelation needs between 2 and 2**31 - 1 observations.")
    devices = parse_device_ids(multi_gpu=multi_gpu)
    perm_counts = None
    if permutation_percentages is not None:
        counts = _permutation_counts(permutation_percentages, n_perms, len(devices))
        devices = [d for d, count in zip(devices, counts, strict=True) if count and len(cols)]
        perm_counts = [count for count in counts if count and len(cols)]
    adj = _to_gpu_csr(adata.obsp[connectivity_key], compute_dtype)
    if transformation:
        normalized = cp.empty_like(adj.data)
        _ac.normalize(adj.indptr, adj.data, out=normalized, stream=cp.cuda.get_current_stream().ptr)
        # The wrapper owns its data attribute even when the original CSR storage is shared.
        adj.data = normalized
    row, col, moments = _g_moments(adj)
    if not np.isfinite(moments[0]) or moments[0] == 0:
        raise ValueError("Spatial autocorrelation requires a finite, nonzero sum of connectivity weights.")
    uniform = bool(n_perms) and bool(cp.all(row == row[0]))
    graph = (adj.indptr, adj.indices.astype(cp.int32, copy=False), adj.data)

    # Six Feistel round keys per permutation; a permutation moves the rows of the weights.
    keys = None
    if n_perms:
        keys = np.random.default_rng(seed).integers(0, 2**32, size=(n_perms, _ac.ROUNDS), dtype=np.uint32)

    cols, gene_order = np.unique(cols, return_inverse=True)
    n_genes = len(cols)
    itemsize = compute_dtype.itemsize
    free = min((cp.cuda.Device(d).mem_info[0] for d in devices), default=0)
    bytes_per_cell = itemsize + (np.dtype(np.float64).itemsize if keys is not None else 0)
    width = int(max(1, min(32, _BLOCK_FRACTION * free // (n * bytes_per_cell))))
    blocks = range((n_genes + width - 1) // width)
    if perm_counts is None:
        devices = devices[: len(blocks)]
    owner = cp.cuda.Device().id
    if isinstance(source, cp.ndarray):
        owner = source.device.id
    elif sparse_gpu.issparse(source):
        owner = source.data.device.id
    # Each device can expand its assigned contiguous genes from a local copy of the CSR.
    tiled_csr = (
        keys is not None
        and n_genes > width
        and 8 * n_genes >= source.shape[1]
        and cols[-1] - cols[0] + 1 == n_genes
        and (sparse.issparse(source) or sparse_gpu.issparse(source))
        and getattr(source, "format", None) == "csr"
        and source.shape[1] < 2**31
        and source.nnz * (itemsize + source.indices.itemsize) + source.indptr.nbytes + 2 * _CHUNK_BYTES < free // 4
    )
    # For wide, resident sparse panels the analytic score can walk only expressed cells.
    # CSC conversion and symmetrizing the weights need scratch proportional to their nonzeros.
    wide_sparse = (
        keys is None
        and n_genes >= 256
        and 4 * n_genes >= source.shape[1]
        and (sparse.issparse(source) or sparse_gpu.issparse(source))
        and source.format in ("csr", "csc")
        and source.nnz <= 0.2 * n * source.shape[1]
        and max(source.nnz, source.shape[1], 2 * adj.nnz) < 2**31
        and 64 * (max(source.shape) + source.nnz + adj.nnz)
        < min(cp.cuda.Device(d).mem_info[0] for d in {*devices, owner, cp.cuda.Device().id}) // 2
    )
    if wide_sparse:
        weights = adj.astype(cp.float64, copy=False)
        symmetric = (weights + weights.T).tocsr()
        graph = (
            symmetric.indptr.astype(cp.int32, copy=False),
            symmetric.indices.astype(cp.int32, copy=False),
            symmetric.data,
        )
        row = row + col
        with cp.cuda.Device(owner):
            # Compressed rows of the transpose are contiguous genes, ready to shard.
            vals = source.astype(compute_dtype, copy=False).tocsc().T
            if n_genes != source.shape[1]:
                vals = vals[cols]
            if not vals.has_canonical_format:
                vals = vals.copy()
                vals.sum_duplicates()
        del weights, symmetric
    elif tiled_csr or sparse.issparse(source):
        vals = source
    else:
        with cp.cuda.Device(owner):
            vals = _gene_source(source, cols, width, compute_dtype)
    del adj

    results = np.empty((1 if keys is None else 4, n_genes), dtype=np.float64)

    def run(index: int, device_id: int):
        draws = keys
        partial = results
        if perm_counts is None:
            mine = blocks[index * len(blocks) // len(devices) : (index + 1) * len(blocks) // len(devices)]
        else:
            assert keys is not None
            start = sum(perm_counts[:index])
            draws = keys[start : start + perm_counts[index]]
            mine = blocks
            if index:
                partial = np.empty_like(results)
        with cp.cuda.Device(device_id):
            w = tuple(_copy_to_device(a, device_id) for a in graph)
            r = _copy_to_device(row, device_id)
            if wide_sparse:
                slots = slice(mine.start * width, min(mine.stop * width, n_genes))
                with cp.cuda.Device(owner if sparse_gpu.issparse(vals) else device_id):
                    local = vals if slots.stop - slots.start == n_genes else vals[slots]
                    local = _to_gpu_csr(local, compute_dtype)
                    arrays = (
                        local.indptr.astype(cp.int32, copy=False),
                        local.indices.astype(cp.int32, copy=False),
                        local.data,
                    )
                out = cp.empty(slots.stop - slots.start, dtype=cp.float64)
                arrays = tuple(_copy_to_device(a, device_id) for a in arrays)
                _ac.sparse_observed(
                    *arrays,
                    *w,
                    r,
                    total=moments[0],
                    geary=mode == "geary",
                    out=out,
                    stream=cp.cuda.get_current_stream().ptr,
                )
                partial[0, slots] = out.get()
                return partial
            c = _copy_to_device(col, device_id)
            k = None if draws is None else cp.asarray(draws)
            single = k is not None and len(k) == 1
            permutation = None
            if single and n * np.dtype(np.int32).itemsize < cp.cuda.Device(device_id).mem_info[0] // 4:
                permutation = cp.empty(n, dtype=cp.int32)
                _ac.permutation(k[0], out=permutation, stream=cp.cuda.get_current_stream().ptr)
            mine_cols = cols[mine.start * width : mine.stop * width]
            if tiled_csr:
                tiles = _csr_gene_blocks(source, mine_cols, width, compute_dtype)
            else:
                # Pack host sparse input directly on the consuming GPU, even when it needs chunks.
                local = _gene_source(source, mine_cols, width, compute_dtype) if sparse.issparse(source) else vals
                first_block = mine.start if sparse.issparse(source) else 0
            for block_id in mine:
                slots = slice(block_id * width, (block_id + 1) * width)
                x = (
                    next(tiles)
                    if tiled_csr
                    else _gene_block(
                        local, cols[slots], block_id - first_block, dtype=compute_dtype, n=n, device_id=device_id
                    )
                )
                observed, change = _block_scores(
                    x,
                    w,
                    r,
                    c,
                    k,
                    total=moments[0],
                    uniform=uniform,
                    geary=mode == "geary",
                    permutation=permutation,
                )
                del x  # Release this dense block before allocating the next one.
                if change is None:
                    partial[0, slots] = observed.get()
                elif single:
                    partial[[0, 2], slots] = cp.stack((observed, change[0])).get()
                else:
                    partial[:, slots] = cp.stack(
                        (
                            observed,
                            ((observed + change) >= observed).sum(axis=0),
                            change.mean(axis=0),
                            change.var(axis=0),
                        )
                    ).get()
            if single:
                slots = slice(mine.start * width, mine.stop * width)
                observed, change = partial[0, slots], partial[2, slots]
                partial[1, slots] = observed + change >= observed
                with np.errstate(invalid="ignore"):
                    partial[3, slots] = change - change
        return partial

    if len(devices) > 1:
        # Worker threads have their own current streams. Finish preparing the
        # shared graph and expression before either device starts reading them.
        cp.cuda.get_current_stream().synchronize()
        if owner != cp.cuda.Device().id:
            with cp.cuda.Device(owner):
                cp.cuda.get_current_stream().synchronize()
        with ThreadPoolExecutor(len(devices)) as pool:
            futures = [pool.submit(run, i, device) for i, device in enumerate(devices)]
            for i, future in enumerate(futures):
                part = future.result()
                if perm_counts is not None and i:
                    # Pool the population variances including the between-shard difference in means.
                    previous, size = sum(perm_counts[:i]), perm_counts[i]
                    fraction = size / (previous + size)
                    delta = part[2] - results[2]
                    results[1] += part[1]
                    results[2] += fraction * delta
                    results[3] = (previous * results[3] + size * part[3]) / (previous + size)
                    results[3] += delta**2 * (previous / (previous + size)) * fraction
    elif devices:
        run(0, devices[0])
    score = results[0]
    _check_precision_issues(score, compute_dtype)

    params = {"two_tailed": two_tailed}
    # Set mode-specific params
    if mode == "moran":
        params["stat"] = "I"
        params["expected"] = -1.0 / (adata.shape[0] - 1)
        params["ascending"] = False
        params["mode"] = "moranI"
    else:  # geary
        params["stat"] = "C"
        params["expected"] = 1.0
        params["ascending"] = True
        params["mode"] = "gearyC"

    with np.errstate(divide="ignore"):
        pval_results = _p_value_calc(
            score,
            sim_stats=None if keys is None else results[1:],
            n_perms=n_perms,
            n_obs=n,
            moments=moments,
            params=params,
        )

    df = pd.DataFrame({params["stat"]: score, **pval_results}).iloc[gene_order]
    df.index = genes

    if corr_method is not None and len(df):
        for pv in filter(lambda x: "pval" in x, df.columns):
            pvalues = df[pv].to_numpy()
            finite = np.isfinite(pvalues)
            pvals_adj = np.full(len(pvalues), np.nan)
            if finite.any():
                pvals_adj[finite] = multipletests(pvalues[finite], alpha=0.05, method=corr_method)[1]
            df[f"{pv}_{corr_method}"] = pvals_adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)
    if copy:
        return df
    adata.uns[params["mode"]] = df
    return None
