from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from cuml.metrics import pairwise_distances

from rapids_singlecell._cuda import _cooc_cuda as _co
from rapids_singlecell._utils import (
    _calculate_blocks_per_pair,
    _create_category_index_mapping,
    _split_pairs,
    parse_device_ids,
)

from ._utils import _assert_categorical_obs, _assert_spatial_basis

if TYPE_CHECKING:
    from anndata import AnnData


def co_occurrence(
    adata: AnnData,
    cluster_key: str,
    *,
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
        Annotated data object.
    cluster_key
        Key for the cluster labels.
    spatial_key
        Key for the spatial coordinates.
    interval
        Distances interval at which co-occurrence is computed. If :class:`int`, uniformly spaced interval
        of the given size will be used.
    multi_gpu
        GPU selection:
        - None: Use all GPUs if available (default)
        - True: Use all available GPUs
        - False: Use only GPU 0
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

    _assert_categorical_obs(adata, key=cluster_key)
    _assert_spatial_basis(adata, key=spatial_key)
    spatial = cp.array(adata.obsm[spatial_key]).astype(np.float32)
    original_clust = adata.obs[cluster_key]
    clust_map = {v: i for i, v in enumerate(original_clust.cat.categories.values)}
    labs = cp.array([clust_map[c] for c in original_clust], dtype=np.int32)
    # create intervals thresholds
    if isinstance(interval, int):
        thresh_min, thresh_max = _find_min_max(spatial)
        interval = cp.linspace(thresh_min, thresh_max, num=interval, dtype=np.float32)
    else:
        interval = cp.array(sorted(interval), dtype=np.float32, copy=True)
    if len(interval) <= 1:
        raise ValueError(
            f"Expected interval to be of length `>= 2`, found `{len(interval)}`."
        )

    device_ids = parse_device_ids(multi_gpu=multi_gpu)
    out = _co_occurrence_helper(
        spatial, interval, labs, fast=True, device_ids=device_ids
    )
    out, interval = out.get(), interval.get()
    if copy:
        return out, interval

    adata.uns[f"{cluster_key}_co_occurrence"] = {"occ": out, "interval": interval}


def _find_min_max(spatial: cp.ndarray) -> tuple[float, float]:
    coord_sum = cp.sum(spatial, axis=1)
    min_idx, min_idx2 = cp.argpartition(coord_sum, 2)[:2]
    max_idx = cp.argmax(coord_sum)
    thres_max = (
        pairwise_distances(
            spatial[min_idx, :].reshape(1, -1), spatial[max_idx, :].reshape(1, -1)
        )[0, 0]
        / 2.0
    )
    thres_min = pairwise_distances(
        spatial[min_idx, :].reshape(1, -1), spatial[min_idx2, :].reshape(1, -1)
    )[0, 0]
    return thres_min.astype(np.float32), thres_max.astype(np.float32)


def _co_occurrence_helper(
    spatial: cp.ndarray,
    v_radium: cp.ndarray,
    labs: cp.ndarray,
    *,
    fast: bool = True,
    device_ids: list[int] | None = None,
) -> cp.ndarray:
    """
    Fast co-occurrence probability computation using cuda kernels.

    Parameters
    ----------
    spatial
        Spatial coordinates.
    v_radium
        Distance thresholds (in ascending order).
    labs
        Cluster labels (as integers).
    fast
        Whether to use the fast CSR-based kernel.
    device_ids
        List of GPU device IDs to use. If None, uses GPU 0.

    Returns
    -------
    occ_prob
        A 3D array of shape (k, k, len(v_radium)-1) containing the co-occurrence probabilities.

    """
    if device_ids is None:
        device_ids = [0]

    n = spatial.shape[0]
    # labels are dense [0, k)
    k = int(cp.asnumpy(labs.max())) + 1
    l_val = len(v_radium) - 1
    thresholds = (v_radium[1:]) ** 2
    use_fast_kernel = False  # Flag to track which kernel path was taken

    if fast:
        # Early check: can we use the fast kernel with available shared memory?
        kernel_config = _co.get_kernel_config(l_val, n, k)
        if kernel_config is None:
            # Shared memory insufficient, skip CSR prep and use pairwise kernel
            fast = False

    if fast:
        # Prepare category CSR structures
        cat_offsets, cell_indices = _create_category_index_mapping(labs, k)

        # Build pair list (upper triangle including diagonal)
        pair_left = []
        pair_right = []
        for a in range(k):
            for b in range(a, k):
                pair_left.append(a)
                pair_right.append(b)
        pair_left = cp.asarray(pair_left, dtype=cp.int32)
        pair_right = cp.asarray(pair_right, dtype=cp.int32)

        # Use single GPU for small workloads
        MIN_CELLS_FOR_MULTI_GPU = 100_000
        n_devices = len(device_ids)
        if n_devices > 1 and n < MIN_CELLS_FOR_MULTI_GPU:
            device_ids = [device_ids[0]]

        counts, use_fast_kernel = _co_occurrence_gpu(
            spatial=spatial,
            thresholds=thresholds,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            pair_left=pair_left,
            pair_right=pair_right,
            n_cells=n,
            k=k,
            l_val=l_val,
            device_ids=device_ids,
        )
        if use_fast_kernel:
            reader = 1

    # Fallback to the standard kernel if fast=False or shared memory was insufficient
    if not use_fast_kernel:
        counts = cp.zeros((k, k, l_val * 2), dtype=cp.uint64)
        _co.count_pairwise(
            spatial,
            thresholds=thresholds,
            labels=labs,
            result=counts,
            n=n,
            k=k,
            l_val=l_val,
            stream=cp.cuda.get_current_stream().ptr,
        )
        reader = 0

    occ_prob = cp.empty((k, k, l_val), dtype=np.float32)
    ok = False
    if use_fast_kernel:
        ok = _co.reduce_shared(
            counts,
            out=occ_prob,
            k=k,
            l_val=l_val,
            format=reader,
            stream=cp.cuda.get_current_stream().ptr,
        )
    if not ok:
        inter_out = cp.zeros((l_val, k, k), dtype=np.float32)
        _co.reduce_global(
            counts,
            inter_out=inter_out,
            out=occ_prob,
            k=k,
            l_val=l_val,
            format=reader,
            stream=cp.cuda.get_current_stream().ptr,
        )

    return occ_prob


def _co_occurrence_gpu(
    spatial: cp.ndarray,
    thresholds: cp.ndarray,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    *,
    pair_left: cp.ndarray,
    pair_right: cp.ndarray,
    n_cells: int,
    k: int,
    l_val: int,
    device_ids: list[int],
) -> tuple[cp.ndarray, bool]:
    """GPU co-occurrence computation using 4-phase pattern.

    Handles both single-GPU and multi-GPU cases. For single GPU, pairs are not
    split and all computation happens on one device.

    Parameters
    ----------
    spatial
        Spatial coordinates on GPU 0
    thresholds
        Squared distance thresholds on GPU 0
    cat_offsets
        Category offsets for CSR structure on GPU 0
    cell_indices
        Cell indices for CSR structure on GPU 0
    pair_left
        Left category indices for all pairs on GPU 0
    pair_right
        Right category indices for all pairs on GPU 0
    n_cells
        Total number of cells (for block size heuristic)
    k
        Number of categories
    l_val
        Number of threshold bins
    device_ids
        List of GPU device IDs to use

    Returns
    -------
    tuple
        (counts, use_fast_kernel) where counts is the aggregated count array
        of shape (k, k, l_val), and use_fast_kernel indicates if the optimized
        kernel was used (False means shared memory was insufficient).
    """
    kernel_configs = {}
    valid_device_ids = []
    for device_id in device_ids:
        with cp.cuda.Device(device_id):
            kernel_config = _co.get_kernel_config(l_val, n_cells, k)
        if kernel_config is not None:
            kernel_configs[device_id] = kernel_config
            valid_device_ids.append(device_id)

    if not valid_device_ids:
        return cp.zeros((k, k, l_val), dtype=cp.uint64), False

    device_ids = valid_device_ids
    n_devices = len(device_ids)
    source_device_id = spatial.device.id

    # Split pairs across devices with load balancing
    group_sizes = cp.diff(cat_offsets).astype(cp.int64)
    pair_chunks = _split_pairs(pair_left, pair_right, n_devices, group_sizes)

    # Phase 1: Create streams and start async data transfer to all devices
    streams: dict[int, cp.cuda.Stream] = {}
    device_data: list[dict | None] = []

    for i, device_id in enumerate(device_ids):
        chunk_left, chunk_right = pair_chunks[i]
        if len(chunk_left) == 0:
            device_data.append(None)
            continue

        with cp.cuda.Device(device_id):
            # Create non-blocking stream for this device
            streams[device_id] = cp.cuda.Stream(non_blocking=True)

            with streams[device_id]:
                # Replicate data to this device (async on stream)
                if device_id == source_device_id:
                    dev_spatial = spatial
                    dev_thresholds = thresholds
                    dev_cat_offsets = cat_offsets
                    dev_cell_indices = cell_indices
                else:
                    dev_spatial = cp.asarray(spatial)
                    dev_thresholds = cp.asarray(thresholds)
                    dev_cat_offsets = cp.asarray(cat_offsets)
                    dev_cell_indices = cp.asarray(cell_indices)

                # Copy pair indices to this device
                dev_pair_left = cp.asarray(chunk_left)
                dev_pair_right = cp.asarray(chunk_right)

                # Initialize local counts array
                dev_counts = cp.zeros((k, k, l_val), dtype=cp.uint64)

                device_data.append(
                    {
                        "spatial": dev_spatial,
                        "thresholds": dev_thresholds,
                        "cat_offsets": dev_cat_offsets,
                        "cell_indices": dev_cell_indices,
                        "pair_left": dev_pair_left,
                        "pair_right": dev_pair_right,
                        "counts": dev_counts,
                        "n_pairs": len(dev_pair_left),
                        "device_id": device_id,
                    }
                )

    # Phase 2: Synchronize data transfers, then launch kernels
    for data in device_data:
        if data is None:
            continue

        device_id = data["device_id"]
        with cp.cuda.Device(device_id):
            # Wait for data transfer to complete on this device
            streams[device_id].synchronize()

            kernel_config = kernel_configs[device_id]
            cell_tile, _l_pad, block_size, shared_mem = kernel_config
            blocks_per_pair = _calculate_blocks_per_pair(data["n_pairs"])

            # Launch kernel with computed configuration
            _co.count_csr_catpairs(
                data["spatial"],
                thresholds=data["thresholds"],
                cat_offsets=data["cat_offsets"],
                cell_indices=data["cell_indices"],
                pair_left=data["pair_left"],
                pair_right=data["pair_right"],
                counts=data["counts"],
                num_pairs=data["n_pairs"],
                k=k,
                l_val=l_val,
                blocks_per_pair=blocks_per_pair,
                cell_tile=cell_tile,
                block_size=block_size,
                shared_mem=shared_mem,
                stream=cp.cuda.get_current_stream().ptr,
            )

    # Phase 3: Synchronize all devices (wait for kernels to complete)
    for data in device_data:
        if data is not None:
            with cp.cuda.Device(data["device_id"]):
                streams[data["device_id"]].synchronize()

    # Phase 4: Aggregate counts on first device
    with cp.cuda.Device(device_ids[0]):
        counts = cp.zeros((k, k, l_val), dtype=cp.uint64)
        for data in device_data:
            if data is not None:
                dev0_counts = cp.asarray(data["counts"])
                counts += dev0_counts

    return counts, True
