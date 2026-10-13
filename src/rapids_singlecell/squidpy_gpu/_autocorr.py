from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import (
    TYPE_CHECKING,
    Literal,
)

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as sparse_gpu
from dask.array import Array as DaskArray
from scipy import sparse
from statsmodels.stats.multitest import multipletests

from rapids_singlecell._cuda import _autocorr_cuda as _ac
from rapids_singlecell._utils import _array_to_device, _copy_to_device, parse_device_ids

from ._spatial_data import _extract_adata
from ._utils import _check_precision_issues, _p_value_calc

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from spatialdata import SpatialData


# Device memory for a block's dense genes and their lags.
_BLOCK_FRACTION = 0.5


# Bytes of the source matrix uploaded at a time while selecting genes.
_CHUNK_BYTES = 1 << 30


def _select(chunk, lut: cp.ndarray, start: int, dtype):
    """Entries of a chunk of observations in the selected genes: (rows, slots, values)."""
    if not (sparse.issparse(chunk) or sparse_gpu.issparse(chunk)):
        chunk = sparse_gpu.csr_matrix(cp.asarray(chunk))
    chunk = chunk.tocsr()
    slots = lut[cp.asarray(chunk.indices)]
    kept = cp.flatnonzero(slots >= 0)
    rows = cp.searchsorted(cp.asarray(chunk.indptr)[1:], kept, side="right")
    return (
        (rows + start).astype(cp.int32),
        slots[kept],
        cp.asarray(chunk.data)[kept].astype(dtype, copy=False),
    )


def _gene_source(source, cols: np.ndarray, dtype):
    """The genes ``cols`` of every observation, selected on the GPU.

    Sparse, host and Dask matrices (e.g. from :func:`rapids_singlecell.io.read_lazy`)
    are read in chunks of observations; only the selected genes' entries stay,
    as (rows, slots, values) on the GPU, or on the host if they would take
    more than half the GPU's free memory. Dense GPU matrices are kept as they are.
    """
    if isinstance(source, cp.ndarray):
        return source
    lut = cp.full(source.shape[1], -1, dtype=cp.int32)
    lut[cp.asarray(cols)] = cp.arange(len(cols), dtype=cp.int32)
    n = source.shape[0]
    parts = []

    def chunk_rows(nbytes):  # observations per upload of about _CHUNK_BYTES
        return max(1, int(_CHUNK_BYTES * n // max(nbytes, 1)))

    if isinstance(source, DaskArray):
        start = 0
        for block in source.blocks.ravel():
            chunk = block.compute()
            parts.append(_select(chunk, lut, start, dtype))
            start += chunk.shape[0]
    elif sparse.issparse(source):
        source = source.tocsr()
        rows = chunk_rows(source.data.nbytes + source.indices.nbytes)
        for start in range(0, n, rows):
            stop = min(start + rows, n)
            # upload views of the rows' entries; rebase the offsets on the GPU
            lo, hi = int(source.indptr[start]), int(source.indptr[stop])
            chunk = sparse_gpu.csr_matrix(
                (
                    _array_to_device(source.data[lo:hi]),
                    _array_to_device(source.indices[lo:hi]),
                    cp.asarray(source.indptr[start : stop + 1] - lo),
                ),
                shape=(stop - start, source.shape[1]),
            )
            parts.append(_select(chunk, lut, start, dtype))
    else:  # GPU sparse or host dense
        rows = chunk_rows(
            source.data.nbytes if sparse_gpu.issparse(source) else source.nbytes
        )
        for start in range(0, n, rows):
            parts.append(_select(source[start : start + rows], lut, start, dtype))
    entries = tuple(cp.concatenate(p) for p in zip(*parts, strict=True))
    if sum(e.nbytes for e in entries) > cp.cuda.Device().mem_info[0] // 2:
        return tuple(e.get() for e in entries)
    return entries


def _gene_block(vals, cols: np.ndarray, dtype, n: int) -> cp.ndarray:
    """Dense C-ordered ``(n_cells, len(cols))`` block of the selected genes ``cols``."""
    if isinstance(vals, cp.ndarray):
        return cp.ascontiguousarray(vals[:, cp.asarray(cols)], dtype=dtype)
    rows, slots, values = vals
    lo, hi = int(cols[0]), int(cols[-1]) + 1  # blocks are contiguous slots
    keep = (slots >= lo) & (slots < hi)
    rows, slots, values = (cp.asarray(e[keep]) for e in (rows, slots, values))
    block = cp.zeros((n, hi - lo), dtype=dtype)
    block[rows, slots - lo] = values
    return block


def _block_scores(x, adj, row_sums, col_sums, keys, *, geary: bool):
    """Observed and permuted scores of one dense gene block, ``(n_genes,)`` and ``(n_perms, n_genes)``."""
    n, n_genes = x.shape
    stream = cp.cuda.get_current_stream().ptr
    z = cp.empty_like(x)
    _ac.lag(adj.indptr, adj.indices, adj.data, x, z=z, stream=stream)
    stats = cp.zeros((4, n_genes))
    _ac.colstats(x, col_sums, out=stats, stream=stream)
    sum_x, sum_x2, cx, cx2 = stats
    mean = sum_x / n
    ss = sum_x2 - n * mean * mean  # sum of squared deviations
    total = float(row_sums.sum(dtype=cp.float64))

    # row-normalized weights without isolated cells: B does not depend on pi
    uniform = float(row_sums.min()) == float(row_sums.max())

    def score(batch_keys):
        out = cp.zeros((1 if batch_keys is None else len(batch_keys), 2, n_genes))
        _ac.sums(
            x,
            z,
            row_sums,
            batch_keys,
            out=out,
            geary=geary,
            uniform=uniform,
            stream=stream,
        )
        a, b = out[:, 0], out[:, 1]
        if uniform:
            b = float(row_sums[0]) * (sum_x2 if geary else sum_x)
        if geary:  # sum_ij w_pi(i)j (x_i - x_j)^2, over the weights' sum
            return (n - 1) * (b - 2 * a + cx2) / (2 * total * ss)
        # sum_i (x_i - m) (W (x - m))_pi(i), over the squared deviations
        return (a - mean * b - mean * cx + mean * mean * total) / ss

    observed = score(None)[0]
    if keys is None:
        return observed, None
    perms = [score(keys[i : i + _ac.BATCH]) for i in range(0, len(keys), _ac.BATCH)]
    return observed, cp.concatenate(perms)


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
        n_perms
            Number of permutations for calculating p-values, by default None.
        two_tailed
            If True, calculate two-tailed p-values, by default False.
        corr_method
            Multiple testing correction method to use, by default "fdr_bh".
        layer
            Layer in the AnnData object to use, by default None.
        use_raw
            If True, use the raw data in the AnnData object, by default False.
        use_sparse
            Ignored; kept for compatibility. Genes are processed in dense blocks of up to 32.
        dtype
            Data type for computation. If None, uses input data dtype.
            For best numerical stability, especially with genes having low variance, use `np.float64`.
        multi_gpu
            GPUs that share the blocks of genes:
            - None: Use all GPUs if available (default)
            - True: Use all available GPUs
            - False: Use only the current GPU
            - list[int]: Use specific GPU IDs (e.g., [0, 2])
            - str: Comma-separated GPU IDs (e.g., "0,2")
        seed
            Random seed for the permutations. Results do not depend on the GPUs used.
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
        genes = list(set(genes) & set(adata.raw.var_names))
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

    # create Adj-Matrix
    adj = adata.obsp[connectivity_key]
    if sparse.issparse(adj):  # page-locked uploads of the large arrays
        adj = adj.tocsr()
        adj = sparse_gpu.csr_matrix(
            (
                _array_to_device(adj.data),
                _array_to_device(adj.indices),
                cp.asarray(adj.indptr),
            ),
            shape=adj.shape,
        )
    adj = sparse_gpu.csr_matrix(adj, dtype=compute_dtype)
    if transformation:  # row-normalize
        row_sums = adj.sum(axis=1).reshape(-1, 1)
        non_zero_rows = row_sums != 0
        row_sums[non_zero_rows] = 1.0 / row_sums[non_zero_rows]
        adj = adj.multiply(sparse_gpu.csr_matrix(row_sums)).tocsr()
    n = adj.shape[0]
    # int32 columns; int64 row offsets only past 2^31 edges
    adj = sparse_gpu.csr_matrix(
        (
            adj.data,
            adj.indices.astype(cp.int32, copy=False),
            adj.indptr.astype(cp.int64 if adj.nnz >= 2**31 else cp.int32, copy=False),
        ),
        shape=adj.shape,
    )
    if n < 2:
        raise ValueError("Spatial autocorrelation needs at least two observations.")

    # Six Feistel round keys per permutation; a permutation moves the rows of the weights.
    keys = None
    if n_perms:
        keys = np.random.default_rng(seed).integers(
            0, 2**32, size=(n_perms, _ac.ROUNDS), dtype=np.uint32
        )

    vals = _gene_source(source, cols, compute_dtype)
    n_genes = len(genes)
    itemsize = compute_dtype.itemsize
    free = cp.cuda.Device().mem_info[0]
    width = int(max(1, min(32, _BLOCK_FRACTION * free // (2 * n * itemsize))))
    blocks = [np.arange(i, min(i + width, n_genes)) for i in range(0, n_genes, width)]
    devices = parse_device_ids(multi_gpu=multi_gpu)[: len(blocks)]

    def run(device_id: int, mine: list[np.ndarray]):
        """Scores of this device's gene blocks, on the host."""
        with cp.cuda.Device(device_id):
            w = adj
            if device_id != adj.data.device.id:
                parts = (adj.data, adj.indices, adj.indptr)
                w = sparse_gpu.csr_matrix(
                    tuple(_copy_to_device(a, device_id) for a in parts), shape=adj.shape
                )
            row = cp.ascontiguousarray(w.sum(axis=1).ravel(), dtype=compute_dtype)
            col = cp.ascontiguousarray(w.sum(axis=0).ravel(), dtype=cp.float64)
            k = None if keys is None else cp.asarray(keys)
            local = vals
            if isinstance(vals, cp.ndarray) and vals.device.id != device_id:
                local = _copy_to_device(vals, device_id)
            elif isinstance(vals, tuple) and isinstance(vals[0], cp.ndarray):
                if vals[0].device.id != device_id:
                    local = tuple(_copy_to_device(e, device_id) for e in vals)
            out = []
            for cols in mine:
                x = _gene_block(local, cols, compute_dtype, n)
                observed, perms = _block_scores(
                    x, w, row, col, k, geary=mode == "geary"
                )
                out.append(
                    (cols, observed.get(), None if perms is None else perms.get())
                )
            return out

    if len(devices) > 1:
        with ThreadPoolExecutor(len(devices)) as pool:
            parts = pool.map(
                lambda d: run(d, blocks[devices.index(d) :: len(devices)]), devices
            )
            results = [r for part in parts for r in part]
    else:
        results = run(devices[0] if devices else cp.cuda.Device().id, blocks)
    score = np.empty(n_genes)
    score_perms = None if keys is None else np.empty((len(keys), n_genes))
    for cols, observed, perms in results:
        score[cols] = observed
        if perms is not None:
            score_perms[:, cols] = perms
    _check_precision_issues(cp.asarray(score), compute_dtype)

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
            score, sims=score_perms, weights=adj, params=params
        )

    df = pd.DataFrame({params["stat"]: score, **pval_results}, index=genes)

    if corr_method is not None:
        for pv in filter(lambda x: "pval" in x, df.columns):
            _, pvals_adj, _, _ = multipletests(df[pv].values, alpha=0.05, method=corr_method)
            df[f"{pv}_{corr_method}"] = pvals_adj

    df.sort_values(by=params["stat"], ascending=params["ascending"], inplace=True)
    if copy:
        return df
    adata.uns[params["mode"]] = df
    return None
