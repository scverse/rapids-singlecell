from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell._cuda import _nhood_cuda as _nh

from ._spatial_data import _extract_adata
from ._utils import _assert_categorical_obs

if TYPE_CHECKING:
    import numpy  # noqa: ICN001 (full name resolves in the docs)
    from anndata import AnnData
    from spatialdata import SpatialData

# Upper bound on n_batch * n_cells; the shuffle sorts 12-byte keys per label.
_MAX_BATCH_LABELS = 1 << 23
_MAX_BATCH = 128


class NhoodEnrichmentResult(NamedTuple):
    """Result of :func:`~rapids_singlecell.gr.nhood_enrichment`."""

    #: Enrichment z-scores, with shape ``(n_clusters, n_clusters)``.
    zscore: numpy.ndarray
    #: Edge counts between cluster pairs, with shape ``(n_clusters, n_clusters)``.
    counts: numpy.ndarray


def _cluster_edges(
    adata: AnnData, cluster_key: str, connectivity_key: str, *, weights: bool
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray | None, cp.ndarray, cp.ndarray, int]:
    """Edges of the connectivity graph restricted to cells with a cluster label.

    Returns ``(rows, cols, data, labels, valid, n_cats)``; ``rows`` and ``cols``
    index into the labelled cells and ``labels`` holds their cluster codes.
    ``data`` holds the float64 edge weights if ``weights`` is set, else ``None``.
    """
    _assert_categorical_obs(adata, cluster_key)
    if connectivity_key not in adata.obsp:
        raise KeyError(
            f"Spatial connectivity key `{connectivity_key}` not found in `adata.obsp`. "
            "Please run `rapids_singlecell.gr.spatial_neighbors_*` first."
        )
    cats = adata.obs[cluster_key]
    n_cats = len(cats.cat.categories)
    codes = cp.asarray(cats.cat.codes.to_numpy(), dtype=cp.int32)
    valid = codes >= 0
    n_valid = int(valid.sum())
    if n_valid == 0:
        raise RuntimeError(
            f"After removing NaNs in `adata.obs[{cluster_key!r}]`, none remain."
        )

    g = adata.obsp[connectivity_key]
    if g.format != "csr":
        g = g.tocsr()
    # Only the sparsity pattern moves to the GPU; weights only when summed.
    indptr = cp.asarray(g.indptr)
    cols = cp.asarray(g.indices, dtype=cp.int32)
    rows = cp.searchsorted(indptr, cp.arange(cols.size), side="right") - 1
    rows = rows.astype(cp.int32)
    data = cp.asarray(g.data, dtype=cp.float64) if weights else None
    if n_valid < codes.shape[0]:
        keep = valid[rows] & valid[cols]
        remap = cp.cumsum(valid, dtype=cp.int32) - 1
        rows, cols = remap[rows[keep]], remap[cols[keep]]
        data = None if data is None else data[keep]
        codes = codes[valid]
    return rows, cols, data, codes, valid, n_cats


def _pair_counts(rows, cols, labels, k, weights=None) -> cp.ndarray:
    """Sum of edges (or their weights) between every pair of clusters."""
    bins = labels[rows].astype(cp.int64) * k + labels[cols]
    return cp.bincount(bins, weights=weights, minlength=k * k).reshape(k, k)


def interaction_matrix(
    adata: AnnData | SpatialData,
    cluster_key: str,
    *,
    table_key: str | None = None,
    connectivity_key: str = "spatial_connectivities",
    normalized: bool = False,
    weights: bool = False,
    copy: bool = False,
) -> np.ndarray | None:
    """
    Compute the interaction matrix for clusters.

    Counts the spatial graph edges between every pair of clusters. Cells
    whose cluster label is missing are removed together with their edges.

    Parameters
    ----------
    adata
        Annotated data matrix or a SpatialData object containing the selected table.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` with categorical cluster labels.
    table_key
        Key in ``SpatialData.tables``; required for SpatialData input.
        Ignored for AnnData input.
    connectivity_key
        Key in :attr:`anndata.AnnData.obsp` with the spatial connectivity graph.
    normalized
        If ``True``, each row is normalized to sum to 1.
    weights
        If ``True``, sum the edge weights instead of counting edges.
    copy
        If ``True``, return the interaction matrix instead of storing it.

    Returns
    -------
    If ``copy = True``, returns the interaction matrix of shape ``(n_clusters, n_clusters)``.

    Otherwise, modifies the ``adata`` with the following key:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_interactions']`` - the interaction matrix.
    """
    adata = _extract_adata(adata, table_key=table_key)
    rows, cols, data, labels, _, k = _cluster_edges(
        adata, cluster_key, connectivity_key, weights=weights
    )
    graph_dtype = adata.obsp[connectivity_key].dtype
    dtype = (
        int
        if pd.api.types.is_bool_dtype(graph_dtype)
        or pd.api.types.is_integer_dtype(graph_dtype)
        else float
    )
    out = _pair_counts(rows, cols, labels, k, data).get().astype(dtype)

    if normalized:
        out = out / out.sum(axis=1).reshape((-1, 1))

    if copy:
        return out
    adata.uns[f"{cluster_key}_interactions"] = out


def nhood_enrichment(
    adata: AnnData | SpatialData,
    cluster_key: str,
    *,
    table_key: str | None = None,
    library_key: str | None = None,
    connectivity_key: str = "spatial_connectivities",
    n_perms: int = 1000,
    seed: int | None = None,
    copy: bool = False,
) -> NhoodEnrichmentResult | None:
    """
    Compute neighborhood enrichment by permutation test.

    Counts the spatial graph edges between every pair of clusters and compares
    the counts with ``n_perms`` random permutations of the cluster labels.
    All permutations are counted on the GPU in batches. Cells whose cluster
    label is missing are removed together with their edges.

    Parameters
    ----------
    adata
        Annotated data matrix or a SpatialData object containing the selected table.
    cluster_key
        Key in :attr:`anndata.AnnData.obs` with categorical cluster labels.
    table_key
        Key in ``SpatialData.tables``; required for SpatialData input.
        Ignored for AnnData input.
    library_key
        Key in :attr:`anndata.AnnData.obs` with categorical library labels.
        If given, cluster labels are only permuted within each library.
    connectivity_key
        Key in :attr:`anndata.AnnData.obsp` with the spatial connectivity graph.
    n_perms
        Number of permutations for the permutation test.
    seed
        Random seed for the permutations.
    copy
        If ``True``, return the result instead of storing it.

    Returns
    -------
    If ``copy = True``, returns a :class:`~rapids_singlecell.gr.NhoodEnrichmentResult`
    with the z-score and the enrichment count.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['zscore']`` - the enrichment z-score.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_nhood_enrichment']['count']`` - the enrichment count.
    """
    adata = _extract_adata(adata, table_key=table_key)
    if n_perms <= 0:
        raise ValueError(f"Expected `n_perms` to be positive, found `{n_perms}`.")
    rows, cols, _, labels, valid, k = _cluster_edges(
        adata, cluster_key, connectivity_key, weights=False
    )
    if k <= 1:
        raise ValueError(f"Expected at least `2` clusters, found `{k}`.")
    n = labels.shape[0]

    if library_key is None:
        lib = cp.zeros(n, dtype=cp.int32)
    else:
        _assert_categorical_obs(adata, library_key)
        lib = cp.asarray(adata.obs[library_key].cat.codes.to_numpy())[valid]
    # Cells grouped by library code (-1 for NaN); labels move only within a group.
    pos = cp.argsort(lib).astype(cp.int32)
    group_off = cp.searchsorted(lib[pos], cp.arange(-1, int(lib.max()) + 2))
    group_off = group_off.astype(cp.int32)

    count = _pair_counts(rows, cols, labels, k).astype(cp.float64)
    # One SplitMix64 seed per permutation; the GPU shuffles by sorting its stream.
    seeds = cp.asarray(
        np.random.default_rng(seed).integers(0, 2**64, size=n_perms, dtype=np.uint64)
    )
    perms = cp.zeros((n_perms, k, k), dtype=cp.uint64)
    _nh.permuted_counts(
        rows,
        cols,
        labels,
        pos,
        group_off,
        seeds,
        out=perms,
        k=k,
        batch=int(max(1, min(n_perms, _MAX_BATCH, _MAX_BATCH_LABELS // n))),
        stream=cp.cuda.get_current_stream().ptr,
    )
    perms = perms.astype(cp.float64)
    zscore = ((count - perms.mean(axis=0)) / perms.std(axis=0)).get()
    count = count.get().astype(np.int64)

    if copy:
        return NhoodEnrichmentResult(zscore=zscore, counts=count)
    adata.uns[f"{cluster_key}_nhood_enrichment"] = {"zscore": zscore, "count": count}
