"""Shared multi-GPU utilities for parallel computation across devices.

This module provides common utilities for distributing work across multiple GPUs,
following a 4-phase pattern:
1. Split - divide work (pairs) across devices
2. Transfer - async copy shared data to each device
3. Launch - run kernel on each device
4. Gather - aggregate results back

Used by: co_occurrence, edistance, and future multi-GPU functions.
"""

from __future__ import annotations

import cupy as cp

# Cache for device attributes per device (lazy initialization)
_DEVICE_ATTRS_CACHE: dict[int, dict] = {}


def parse_device_ids(*, multi_gpu: bool | list[int] | str | None) -> list[int]:
    """Parse multi_gpu parameter into a list of device IDs.

    Parameters
    ----------
    multi_gpu
        GPU selection:
        - None or True: Use all available GPUs
        - False: Use only GPU 0
        - list[int]: Use specific GPU IDs (e.g., [0, 2])
        - str: Comma-separated GPU IDs (e.g., "0,2")

    Returns
    -------
    list[int]
        List of device IDs to use

    Raises
    ------
    ValueError
        If any specified device ID is invalid or out of range
    """
    n_available = cp.cuda.runtime.getDeviceCount()

    if multi_gpu is None or multi_gpu is True:
        return list(range(n_available))
    elif multi_gpu is False:
        return [0]
    elif isinstance(multi_gpu, str):
        device_ids = [int(x.strip()) for x in multi_gpu.split(",")]
    elif isinstance(multi_gpu, list):
        device_ids = multi_gpu
    else:
        raise ValueError(
            f"multi_gpu must be bool, list[int], or str, got {type(multi_gpu)}"
        )

    # Validate device IDs
    invalid_ids = [d for d in device_ids if d < 0 or d >= n_available]
    if invalid_ids:
        raise ValueError(
            f"Invalid GPU device ID(s): {invalid_ids}. "
            f"Available devices: {list(range(n_available))}"
        )

    if len(device_ids) == 0:
        raise ValueError("multi_gpu must specify at least one device")

    return device_ids


def _get_device_attrs(device_id: int | None = None) -> dict:
    """Get device attributes for a specific device (cached per device).

    Parameters
    ----------
    device_id
        CUDA device ID. If None, uses current device.

    Returns
    -------
    dict
        Dictionary containing 'max_shared_mem' and 'cc_major' for the device.
    """
    if device_id is None:
        device_id = cp.cuda.Device().id

    if device_id not in _DEVICE_ATTRS_CACHE:
        with cp.cuda.Device(device_id):
            device = cp.cuda.Device()
            # compute_capability is a string like "120" for CC 12.0, or "86" for CC 8.6
            cc_str = str(device.compute_capability)
            cc_major = int(cc_str[:-1]) if len(cc_str) > 1 else int(cc_str)
            _DEVICE_ATTRS_CACHE[device_id] = {
                "max_shared_mem": device.attributes["MaxSharedMemoryPerBlock"],
                "cc_major": cc_major,
            }
    return _DEVICE_ATTRS_CACHE[device_id]


def _split_pairs(
    pair_left: cp.ndarray,
    pair_right: cp.ndarray,
    n_devices: int,
    group_sizes: cp.ndarray | None = None,
) -> list[tuple[cp.ndarray, cp.ndarray]]:
    """Split pairs across devices with load balancing.

    When group_sizes is provided, pairs are assigned to balance computational
    work (proportional to group_sizes[left] * group_sizes[right]) across devices.
    Without group_sizes, falls back to simple even splitting by count.

    Parameters
    ----------
    pair_left
        Left indices of pairs
    pair_right
        Right indices of pairs
    n_devices
        Number of devices to split across
    group_sizes
        Size of each group. If provided, enables work-based load balancing.

    Returns
    -------
    list
        List of (pair_left, pair_right) tuples for each device
    """
    n_pairs = len(pair_left)

    if n_pairs == 0:
        return [
            (cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32))
            for _ in range(n_devices)
        ]

    # Simple even split if no group sizes provided or single device
    if group_sizes is None or n_devices == 1:
        pairs_per_device = (n_pairs + n_devices - 1) // n_devices
        chunks = []
        for i in range(n_devices):
            start = i * pairs_per_device
            end = min(start + pairs_per_device, n_pairs)
            if start < n_pairs:
                chunks.append((pair_left[start:end], pair_right[start:end]))
            else:
                chunks.append(
                    (cp.array([], dtype=cp.int32), cp.array([], dtype=cp.int32))
                )
        return chunks

    # Load-balanced split based on work per pair
    # Off-diagonal (i != j): work = n_i * n_j
    # Diagonal (i == i): work = n_i * (n_i - 1) / 2  (within-group, no self-pairs)
    group_sizes = group_sizes.astype(cp.int64, copy=False)
    sizes_left = group_sizes[pair_left]
    sizes_right = group_sizes[pair_right]
    is_diagonal = pair_left == pair_right
    work = cp.where(
        is_diagonal,
        sizes_left * (sizes_left - 1) // 2,
        sizes_left * sizes_right,
    )
    cumulative_work = cp.cumsum(work)
    total_work = cumulative_work[-1]

    # Find split points at 1/n_devices, 2/n_devices, ... of total work
    targets = total_work * cp.arange(1, n_devices, dtype=cp.int64) // n_devices
    split_indices = cp.searchsorted(cumulative_work, targets, side="right").get()

    # Split arrays at those indices
    left_splits = cp.split(pair_left, split_indices)
    right_splits = cp.split(pair_right, split_indices)

    return list(zip(left_splits, right_splits, strict=False))


def _calculate_blocks_per_pair(num_pairs: int) -> int:
    """Calculate optimal blocks_per_pair based on workload.

    Targets ~300K total blocks for good GPU utilization.

    Parameters
    ----------
    num_pairs
        Number of pairs to process

    Returns
    -------
    int
        Optimal number of blocks per pair
    """
    TARGET_TOTAL_BLOCKS = 300_000
    MAX_BLOCKS_PER_PAIR = 32

    blocks_per_pair = max(1, (TARGET_TOTAL_BLOCKS + num_pairs - 1) // num_pairs)
    blocks_per_pair = min(blocks_per_pair, MAX_BLOCKS_PER_PAIR)

    return blocks_per_pair


def _create_category_index_mapping(
    cats: cp.ndarray, n_batches: int
) -> tuple[cp.ndarray, cp.ndarray]:
    """Create a CSR-like data structure mapping categories to cell indices.

    Uses lexicographical sort to group cells by category.

    Parameters
    ----------
    cats
        Category labels for each cell (integers 0 to n_batches-1)
    n_batches
        Number of categories

    Returns
    -------
    cat_offsets
        Array of length n_batches+1 with start/end indices for each category
    cell_indices
        Array of cell indices sorted by category
    """
    cat_counts = cp.zeros(n_batches, dtype=cp.int32)
    cp.add.at(cat_counts, cats, 1)
    cat_offsets = cp.zeros(n_batches + 1, dtype=cp.int32)
    cp.cumsum(cat_counts, out=cat_offsets[1:])

    n_cells = cats.shape[0]
    indices = cp.arange(n_cells, dtype=cp.int32)

    cell_indices = cp.lexsort(cp.stack((indices, cats))).astype(cp.int32)
    return cat_offsets, cell_indices
