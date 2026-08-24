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

import warnings
from functools import cache

import cupy as cp
import numpy as np

# Cache for device attributes per device (lazy initialization)
_DEVICE_ATTRS_CACHE: dict[int, dict] = {}


CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
CUDA_ERROR_PEER_ACCESS_NOT_ENABLED = 705
CUDA_ERROR_TOO_MANY_PEERS = 711

_SAFE_PEER_FALLBACK_ERRORS = {
    CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
    CUDA_ERROR_PEER_ACCESS_NOT_ENABLED,
    CUDA_ERROR_TOO_MANY_PEERS,
}
_P2P_CANARY_EXPECTED = np.arange(1, 9, dtype=np.float64)
_P2P_CANARY_POISON = np.full(8, -7.0, dtype=np.float64)
_WARNED_P2P_FAILURES: set[tuple[int, tuple[tuple[int, int], ...]]] = set()


class MultiGPUFallbackWarning(RuntimeWarning):
    """Warning emitted when unsafe P2P makes an operation use one GPU."""


def _copy_to_device_via_host(array: cp.ndarray, device_id: int) -> cp.ndarray:
    """Copy a device array without touching a peer link.

    This is intentionally reserved for small control arrays whose owner may
    differ from the main input's device. Capturing the source before changing
    devices is essential: calling ``cp.asarray`` first would already attempt
    the peer transfer this helper exists to avoid.
    """
    source_device = array.device.id
    if source_device == device_id:
        return array

    with cp.cuda.Device(source_device):
        host = array.get()
    with cp.cuda.Device(device_id):
        copied = cp.asarray(host)
        cp.cuda.get_current_stream().synchronize()
        return copied


def _run_peer_copy_canary(destination: int, source: int) -> bool:
    """Return whether a small peer copy arrives intact.

    The explicit synchronizations make the canary independent of whichever
    CuPy stream was current when the multi-GPU operation was entered.
    """
    with cp.cuda.Device(source):
        expected = cp.asarray(_P2P_CANARY_EXPECTED)
        cp.cuda.get_current_stream().synchronize()

    with cp.cuda.Device(destination):
        probe = cp.asarray(_P2P_CANARY_POISON)
        cp.cuda.get_current_stream().synchronize()
        try:
            cp.cuda.runtime.memcpyPeer(
                probe.data.ptr,
                destination,
                expected.data.ptr,
                source,
                expected.nbytes,
            )
            cp.cuda.runtime.deviceSynchronize()
            arrived = cp.asnumpy(probe)
        except cp.cuda.runtime.CUDARuntimeError as error:
            if error.status in _SAFE_PEER_FALLBACK_ERRORS:
                return False
            # memcpy/readback can surface an earlier asynchronous failure. In
            # particular, an illegal address poisons the CUDA context and
            # cannot be repaired by switching to one GPU in this process.
            raise

    return bool(np.array_equal(arrived, _P2P_CANARY_EXPECTED))


@cache
def peer_copy_works(destination: int, source: int) -> bool:
    """Return whether ``source -> destination`` P2P transfers are usable.

    ``deviceCanAccessPeer`` is only a capability query. Some affected systems
    report support and return success from ``memcpyPeer`` while leaving the
    destination unchanged, so this function verifies the transferred bytes.
    The result is cached per ordered device pair for the lifetime of the
    process.
    """
    if destination == source:
        return True

    with cp.cuda.Device(destination):
        if not cp.cuda.runtime.deviceCanAccessPeer(destination, source):
            return False
        try:
            cp.cuda.runtime.deviceEnablePeerAccess(source)
        except cp.cuda.runtime.CUDARuntimeError as error:
            if error.status == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
                pass
            elif error.status in _SAFE_PEER_FALLBACK_ERRORS:
                return False
            else:
                raise

    return _run_peer_copy_canary(destination, source)


# Compatibility name used by the container-level P2P diagnostics.
peer_copy_verified = peer_copy_works


def validate_multi_gpu(
    device_ids: list[int],
    *,
    source_device: int | None = None,
    gather_device: int | None = None,
) -> list[int]:
    """Return usable devices, falling back to one GPU when P2P is unsafe.

    Multi-GPU implementations in this package fan input buffers out from a
    source GPU and gather results onto a selected GPU. This validates exactly
    those ordered transfers before sharding or worker threads start.
    If a required link is unavailable or silently corrupts the canary, the
    operation falls back to the source GPU and emits one warning per failed
    link set. Unexpected CUDA errors propagate because they may indicate an
    already-poisoned context.

    Parameters
    ----------
    device_ids
        Requested execution devices.
    source_device
        Device holding shared input buffers. Defaults to the current device.
    gather_device
        Device receiving worker results. Defaults to the first requested
        device.

    Returns
    -------
    list[int]
        The requested devices when all required P2P directions work, otherwise
        a one-element list containing ``source_device``.
    """
    if not device_ids:
        raise ValueError("device_ids must contain at least one device")

    device_ids = list(dict.fromkeys(device_ids))
    if source_device is None:
        source_device = cp.cuda.Device().id

    n_available = cp.cuda.runtime.getDeviceCount()
    invalid_ids = [
        device_id
        for device_id in device_ids
        if device_id < 0 or device_id >= n_available
    ]
    if invalid_ids:
        raise ValueError(
            f"Invalid GPU device ID(s): {invalid_ids}. "
            f"Available devices: {list(range(n_available))}"
        )
    if source_device < 0 or source_device >= n_available:
        raise ValueError(
            f"Invalid source GPU device ID {source_device}. "
            f"Available devices: {list(range(n_available))}"
        )

    if gather_device is None:
        gather_device = device_ids[0]
    if gather_device < 0 or gather_device >= n_available:
        raise ValueError(
            f"Invalid gather GPU device ID {gather_device}. "
            f"Available devices: {list(range(n_available))}"
        )

    required_pairs: set[tuple[int, int]] = set()
    for device_id in device_ids:
        if device_id != source_device:
            required_pairs.add((device_id, source_device))
        if device_id != gather_device:
            required_pairs.add((gather_device, device_id))

    failed_pairs = ()
    for pair in sorted(required_pairs):
        if not peer_copy_works(*pair):
            # Once fallback is required, do not exercise any more suspect
            # links in the caller's CUDA context.
            failed_pairs = (pair,)
            break
    if not failed_pairs:
        return device_ids

    warning_key = (source_device, failed_pairs)
    if warning_key not in _WARNED_P2P_FAILURES:
        transfers = ", ".join(
            f"GPU {source} -> GPU {destination}" for destination, source in failed_pairs
        )
        warnings.warn(
            "Multi-GPU execution was disabled because the required P2P "
            f"transfer(s) failed validation: {transfers}. Falling back to "
            f"GPU {source_device} for this operation. This avoids silent "
            "result corruption but may be slower.",
            MultiGPUFallbackWarning,
            stacklevel=2,
        )
        _WARNED_P2P_FAILURES.add(warning_key)

    return [source_device]


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
