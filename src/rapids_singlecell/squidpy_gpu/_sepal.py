from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as sparse_gpu
from scanpy import logging as logg
from scipy import sparse

from rapids_singlecell._cuda import _sepal_cuda as _sp

from ._spatial_data import _extract_adata
from ._utils import _assert_connectivity_key, _assert_spatial_basis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData
    from spatialdata import SpatialData

# Fraction of free device memory used for the input batch.
_MEM_FRACTION = 0.5
_BLOCK = 1024
# where the simulation state lives (SepalStore)
_SMEM2, _SMEM1, _GLOBAL = 0, 1, 2


def sepal(  # noqa: PLR0917 (squidpy's positional signature)
    adata: AnnData | SpatialData,
    max_neighs: Literal[4, 6],
    genes: str | Sequence[str] | None = None,
    n_iter: int | None = 30000,
    dt: float = 0.001,
    thresh: float = 1e-8,
    connectivity_key: str = "spatial_connectivities",
    spatial_key: str = "spatial",
    layer: str | None = None,
    use_raw: bool = False,  # noqa: FBT001, FBT002
    copy: bool = False,  # noqa: FBT001, FBT002
    *,
    table_key: str | None = None,
    dtype: np.dtype | type = np.float32,
) -> pd.DataFrame | None:
    """\
    Identify spatially variable genes with *Sepal*.

    *Sepal* simulates a diffusion process to quantify spatial structure in
    tissue :cite:`andersson2021`. This is a GPU port of :func:`squidpy.gr.sepal`
    that simulates all genes in parallel.

    Parameters
    ----------
    adata
        Annotated data matrix or a SpatialData object containing the selected table.
    max_neighs
        Maximum number of neighbors of a node in the graph. Valid options are:

            - `4` - for a square-grid (ST, Dbit-seq).
            - `6` - for a hexagonal-grid (Visium).
    genes
        List of gene names, as stored in :attr:`~anndata.AnnData.var_names`,
        used to compute sepal score.

        If `None`, it's computed :attr:`~anndata.AnnData.var` ``['highly_variable']``,
        if present. Otherwise, it's computed for all genes.
    n_iter
        Maximum number of iterations for the diffusion simulation.
        If ``n_iter`` iterations are reached, the simulation will terminate
        even though convergence has not been achieved.
    dt
        Time step in diffusion simulation.
    thresh
        Entropy threshold for convergence of diffusion simulation.
    connectivity_key
        Key in :attr:`~anndata.AnnData.obsp` where the spatial connectivities are stored.
    spatial_key
        Key in :attr:`~anndata.AnnData.obsm` where the spatial coordinates are stored.
    layer
        Layer in :attr:`~anndata.AnnData.layers` to use. If `None`, use :attr:`~anndata.AnnData.X`.
    use_raw
        Whether to access :attr:`~anndata.AnnData.raw`.
    copy
        If `True`, return the result, otherwise save it to the ``adata`` object.
    table_key
        Key in ``SpatialData.tables``; required for SpatialData input.
        All reads and writes use this table; ignored for AnnData input.
    dtype
        Precision of the diffusion. `np.float32` (default) carries each value
        as a compensated pair of floats and is fastest on GPUs with low
        double-precision throughput; `np.float64` diffuses in double
        precision, like squidpy. Both accumulate the entropy changes in a
        cancellation-free float form; on all data we tested, both reproduce
        squidpy's scores exactly.

    Returns
    -------
    If ``copy = True``, returns a :class:`pandas.DataFrame` with the sepal scores.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['sepal_score']`` - the sepal scores.

    Notes
    -----
    If some genes in :attr:`anndata.AnnData.uns` ``['sepal_score']`` are `NaN`,
    consider re-running the function with increased ``n_iter``.
    """
    adata = _extract_adata(adata, table_key=table_key)
    _assert_connectivity_key(adata, connectivity_key)
    _assert_spatial_basis(adata, key=spatial_key)
    if max_neighs not in (4, 6):
        raise ValueError(
            f"Expected `max_neighs` to be either `4` or `6`, found `{max_neighs}`."
        )
    dtype = np.dtype(dtype)
    if dtype not in (np.float32, np.float64):
        raise ValueError(f"`dtype` must be float32 or float64, found `{dtype}`.")

    if genes is None:
        genes = adata.var_names.values
        if "highly_variable" in adata.var.columns:
            genes = genes[adata.var["highly_variable"].values]
    genes = _assert_non_empty_sequence(genes, name="genes")

    g = adata.obsp[connectivity_key]
    g = sparse_gpu.csr_matrix(g) if not sparse_gpu.isspmatrix_csr(g) else g.copy()
    g.eliminate_zeros()

    degrees = cp.diff(g.indptr)
    max_n = int(degrees.max())
    if max_n != max_neighs:
        raise ValueError(
            f"Expected `max_neighs={max_neighs}`, found node with `{max_n}` neighbors."
        )

    spatial = cp.ascontiguousarray(
        cp.asarray(np.asarray(adata.obsm[spatial_key])[:, :2], dtype=cp.float64)
    )
    nbrs, ctr, n_sat = _compute_idxs(g, degrees, spatial, max_neighs)

    vals, genes = _extract_expression(adata, genes=genes, use_raw=use_raw, layer=layer)
    start = logg.info(f"Calculating sepal score for `{len(genes)}` genes on the GPU")
    iters = _simulate(
        vals,
        nbrs=nbrs,
        ctr=ctr,
        n_sat=n_sat,
        max_neighs=max_neighs,
        n_iter=0 if n_iter is None else int(n_iter),
        dt=float(dt),
        thresh=float(thresh),
        dtype=dtype,
    )
    score = np.where(iters >= 0, dt * iters.astype(np.float64), np.nan)

    key_added = "sepal_score"
    sepal_score = pd.DataFrame(score, index=genes, columns=[key_added])

    if sepal_score[key_added].isna().any():
        logg.warning(
            "Found `NaN` in sepal scores, consider increasing `n_iter` to a higher value"
        )
    sepal_score = sepal_score.sort_values(by=key_added, ascending=False)

    if copy:
        logg.info("Finish", time=start)
        return sepal_score

    adata.uns[key_added] = sepal_score
    logg.info(
        "Finish",
        time=start,
        deep=f"Adding `adata.uns[{key_added!r}]`",
    )


def _assert_non_empty_sequence(seq, *, name: str) -> list:
    if isinstance(seq, str) or not hasattr(seq, "__iter__"):
        seq = (seq,)
    res = list(dict.fromkeys(seq))
    if len(res) == 0:
        raise ValueError(f"No {name} have been selected.")
    return res


def _extract_expression(
    adata: AnnData, *, genes: Sequence[str], use_raw: bool, layer: str | None
):
    if use_raw and adata.raw is None:
        logg.warning("AnnData object has no attribute `raw`. Setting `use_raw=False`")
        use_raw = False
    if use_raw:
        raw_names = set(adata.raw.var_names)
        genes = _assert_non_empty_sequence(
            [gene for gene in genes if gene in raw_names], name="genes"
        )
        return adata.raw.X[:, adata.raw.var_names.get_indexer(genes)], genes
    idx = adata.var_names.get_indexer(genes)
    if (idx < 0).any():
        missing = [gene for gene, i in zip(genes, idx, strict=True) if i < 0]
        raise KeyError(f"Genes {missing[:5]} not found in `adata.var_names`.")
    if layer is not None and layer not in adata.layers:
        raise KeyError(f"Layer `{layer}` not found in `adata.layers`.")
    X = adata.X if layer is None else adata.layers[layer]
    return X[:, idx], genes


def _simulate(vals, *, nbrs, ctr, dtype, max_neighs: int, **params) -> np.ndarray:
    """Convergence iteration of every gene column of ``vals``, -1 if none."""
    n_cells, n_genes = vals.shape
    store, cluster_size, n_groups = _plan(n_cells, max_neighs, dtype)
    # (genes, cells) rows of a gene-major copy
    if sparse.issparse(vals) or sparse_gpu.issparse(vals):
        rows = sparse_gpu.csr_matrix(vals, dtype=dtype).T.tocsr()
    else:
        rows = cp.ascontiguousarray(cp.asarray(vals, dtype=dtype).T)
    buf = cp.empty((n_groups, 2, n_cells if store != _SMEM2 else 0), dtype=dtype)
    lo = cp.empty((n_groups, n_cells if dtype == np.float32 else 0), dtype=dtype)
    counter = cp.empty(1, dtype=cp.int32)
    iters = cp.empty(n_genes, dtype=cp.int32)
    free = cp.cuda.Device().mem_info[0]
    batch = max(1, int(free * _MEM_FRACTION) // (dtype.itemsize * n_cells))
    for start in range(0, n_genes, batch):
        stop = min(start + batch, n_genes)
        conc = rows[start:stop]
        conc = conc.toarray() if sparse_gpu.issparse(conc) else conc
        counter[...] = 0
        _sp.diffusion(
            cp.ascontiguousarray(conc),
            buf=buf,
            lo=lo,
            nbrs=nbrs,
            ctr=ctr,
            results=iters[start:stop],
            counter=counter,
            max_neighs=max_neighs,
            store=store,
            cluster_size=cluster_size,
            block_size=_BLOCK,
            stream=cp.cuda.get_current_stream().ptr,
            **params,
        )
    return iters.get()


def _plan(n_cells: int, max_neighs: int, dtype) -> tuple[int, int, int]:
    """Where the state lives, blocks per gene and genes in flight.

    The state goes into the shared memory of one block, else into the
    distributed shared memory of the smallest cluster that holds it (sm_90+),
    else into global memory.
    """
    for store in (_SMEM2, _SMEM1):
        for cluster_size in (1, 2, 4, 6, 8, 12):
            n = _sp.occupancy(
                f32=dtype == np.float32,
                max_neighs=max_neighs,
                store=store,
                cluster_size=cluster_size,
                n_cells=n_cells,
                block_size=_BLOCK,
            )
            if n > 0:
                return store, cluster_size, n
    n = _sp.occupancy(
        f32=dtype == np.float32,
        max_neighs=max_neighs,
        store=_GLOBAL,
        cluster_size=1,
        n_cells=n_cells,
        block_size=_BLOCK,
    )
    return _GLOBAL, 1, n


def _compute_idxs(
    g: sparse_gpu.csr_matrix,
    degrees: cp.ndarray,
    spatial: cp.ndarray,
    sat_thresh: int,
) -> tuple[cp.ndarray, cp.ndarray, int]:
    """Get the saturated source cell of every cell and its neighbors.

    Saturated cells (``max_neighs`` neighbors) are their own source. Each
    unsaturated cell takes the derivative of its first saturated neighbor or,
    if it has none, of the closest saturated cell in L1 distance.

    Returns the neighbors of each cell's source ``(max_neighs, n_cells)``, the
    source ``(n_cells,)`` and the number of saturated cells.
    """
    indptr = g.indptr.astype(cp.int32, copy=False)
    indices = g.indices.astype(cp.int32, copy=False)
    sat_mask = degrees == sat_thresh
    sat = cp.flatnonzero(sat_mask).astype(cp.int32)
    unsat = cp.flatnonzero(degrees < sat_thresh).astype(cp.int32)

    stream = cp.cuda.get_current_stream().ptr
    nearest = cp.empty(unsat.shape[0], dtype=cp.int32)
    _sp.first_sat_neighbor(
        unsat,
        indptr,
        indices,
        cp.ascontiguousarray(sat_mask),
        nearest=nearest,
        stream=stream,
    )
    missing = cp.flatnonzero(nearest < 0)
    if missing.size:
        closest = cp.empty(missing.shape[0], dtype=cp.int32)
        _sp.nearest_sat_l1(
            spatial,
            cp.ascontiguousarray(unsat[missing]),
            sat,
            nearest=closest,
            stream=stream,
        )
        nearest[missing] = closest

    ctr = cp.arange(degrees.shape[0], dtype=cp.int32)
    ctr[unsat] = nearest
    # (max_neighs, n_cells): coalesced loads of the m-th neighbour
    nbrs = cp.ascontiguousarray(
        indices[indptr[ctr][None, :] + cp.arange(sat_thresh, dtype=cp.int32)[:, None]]
    )
    return nbrs, ctr, int(sat.shape[0])
