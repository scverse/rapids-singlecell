from __future__ import annotations

from contextlib import ExitStack
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cuml.metrics import pairwise_distances

from rapids_singlecell._cuda import _cooc_cuda as _co
from rapids_singlecell._utils import _copies_on_device, _sum_on_device, parse_device_ids

from ._spatial_data import _extract_adata
from ._utils import _assert_categorical_obs, _assert_spatial_basis

if TYPE_CHECKING:
    from anndata import AnnData
    from spatialdata import SpatialData


def co_occurrence(
    adata: AnnData | SpatialData,
    cluster_key: str,
    *,
    table_key: str | None = None,
    spatial_key: str = "spatial",
    interval: int | np.ndarray | cp.ndarray = 50,
    multi_gpu: bool | list[int] | str | None = None,
    copy: bool = False,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute co-occurrence probability of clusters.

    Parameters
    ----------
    adata
        Annotated data matrix or a SpatialData object containing the selected table.
    table_key
        Key in ``SpatialData.tables``; required for SpatialData input.
        All reads and writes use this table; ignored for AnnData input.
    cluster_key
        Key for the cluster labels.
    spatial_key
        Key for the spatial coordinate matrix. Distances use all coordinate columns.
    interval
        Distances interval at which co-occurrence is computed. If :class:`int`, uniformly spaced interval
        of the given size will be used.
    multi_gpu
        GPU selection:
        - None: Use all GPUs if available (default)
        - True: Use all available GPUs
        - False: Use only the current GPU
        - list[int]: Use specific GPU IDs (e.g., [0, 2])
        - str: Comma-separated GPU IDs (e.g., "0,2")
    copy
        If ``True``, return the co-occurrence probability and the distance thresholds intervals.

    Returns
    -------
    If ``copy = True``, returns the co-occurrence probability and the distance thresholds intervals.

    Otherwise, modifies the ``adata`` with the following keys:

        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['occ']`` - the co-occurrence probabilities
          across interval thresholds.
        - :attr:`anndata.AnnData.uns` ``['{cluster_key}_co_occurrence']['interval']`` - the distance thresholds
          computed at ``interval``.
    """
    adata = _extract_adata(adata, table_key=table_key)

    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)
    spatial = cp.array(adata.obsm[spatial_key], dtype=cp.float32, order="C")
    if spatial.ndim != 2 or 0 in spatial.shape:
        raise ValueError("Expected a spatial coordinate matrix with at least one cell and one coordinate.")
    clusters = adata.obs[cluster_key]
    if clusters.hasnans:
        raise KeyError(f"Missing cluster label in adata.obs[{cluster_key!r}]")
    if isinstance(interval, (int, np.integer)):
        thresh_min, thresh_max = _find_min_max(spatial)
        interval = cp.linspace(thresh_min, thresh_max, num=int(interval), dtype=np.float32)
    else:
        interval = cp.asarray(interval, dtype=cp.float32)
        if interval.ndim != 1:
            raise ValueError("Expected a one-dimensional array of distance thresholds.")
        interval = cp.sort(interval)
    if len(interval) <= 1:
        raise ValueError(f"Expected interval to be of length `>= 2`, found `{len(interval)}`.")

    codes = clusters.cat.codes.to_numpy()
    k = int(codes.max()) + 1
    labs = cp.asarray(codes, dtype=cp.int32)
    device_ids = parse_device_ids(multi_gpu=multi_gpu)
    thresholds = cp.square(interval[1:])
    # Squaring can reverse the radius order. NaN radii match no distances.
    thresholds = cp.where(cp.isnan(thresholds), cp.float32(-1), thresholds)
    order = cp.argsort(thresholds)
    counts = _count_spatial_tiles(spatial, labs, thresholds[order], k, device_ids)
    out = cp.empty(counts.shape, dtype=cp.float32)
    marginals = cp.empty((len(order), k), dtype=cp.float32)
    _co.normalize(counts, order, marginals, out, cp.cuda.get_current_stream().ptr)
    out, interval = out.get(), interval.get()
    if copy:
        return out, interval

    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": out, "interval": interval}
    return None


def _find_min_max(spatial: cp.ndarray) -> tuple[float, float]:
    if len(spatial) == 1:
        return 0.0, 0.0
    coord_sum = cp.sum(spatial, axis=1)
    min_idx, min_idx2 = cp.argpartition(coord_sum, min(2, len(spatial) - 1))[:2]
    max_idx = cp.argmax(coord_sum)
    thres_max = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1))[0, 0] / 2.0
    thres_min = pairwise_distances(spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1))[0, 0]
    return thres_min.astype(np.float32), thres_max.astype(np.float32)


def _count_spatial_tiles(
    spatial: cp.ndarray,
    labs: cp.ndarray,
    thresholds: cp.ndarray,
    k: int,
    device_ids: list[int],
    *,
    within_categories: bool = False,
) -> cp.ndarray:
    """Count all unordered cell pairs by processing one triangle of spatially sorted tiles."""
    shape = (k, len(thresholds)) if within_categories else (k, k, len(thresholds))
    if not len(labs) or not len(thresholds):
        return cp.zeros(shape, dtype=cp.uint64)
    n = len(labs)
    keys = cp.empty(n, dtype=cp.uint64)
    stream = cp.cuda.get_current_stream().ptr
    _co.spatial_keys(spatial, labs, spatial.min(axis=0), spatial.max(axis=0), keys, stream)
    order = cp.argsort(keys)
    points = spatial[order]
    offsets = cp.searchsorted(keys[order], cp.arange(k + 1, dtype=cp.uint64) << 32).astype(cp.int32)
    tile_offsets = cp.zeros(k + 1, dtype=cp.int32)
    cp.cumsum((cp.diff(offsets).astype(cp.int64) + 31) // 32, out=tile_offsets[1:])
    n_tiles = int(tile_offsets[-1])
    tiles = cp.empty((n_tiles, _co.TILE_BYTES), dtype=cp.uint8)
    bounds = cp.empty((n_tiles, spatial.shape[1], 2), dtype=cp.float32)
    _co.build_tiles(points, offsets, tile_offsets, tiles, bounds, stream)

    device_ids = device_ids[:n_tiles]
    parts = []
    with ExitStack() as stack:
        inputs = (points, tiles, bounds, thresholds)
        replicas = [stack.enter_context(_copies_on_device(inputs, dev)) for dev in device_ids]
        for rank, (device, replica) in enumerate(zip(device_ids, replicas, strict=True)):
            with cp.cuda.Device(device):
                counts = cp.zeros(shape, dtype=cp.uint64)
                _co.count_tiles(*replica, counts, rank, len(device_ids), cp.cuda.get_current_stream().ptr)
                parts.append(counts)
    counts = _sum_on_device(parts, spatial.device.id)
    return cp.cumsum(counts, axis=-1, out=counts)
