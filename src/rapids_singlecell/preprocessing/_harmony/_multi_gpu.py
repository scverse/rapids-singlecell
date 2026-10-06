"""Cross-GPU helpers of harmonize on several GPUs: copies between the GPUs and
the exchange of exact sums (harmony/comm.cuh)."""

from __future__ import annotations

import cupy as cp

from rapids_singlecell._cuda import _harmony_clustering_cuda as _clustering_cuda
from rapids_singlecell._utils._multi_gpu import (
    _copy_to_device,
    _copy_to_device_via_host,
    _peer_copy_works,
)

# Tests: move every cross-GPU value through host memory.
_FORCE_HOST_COPIES = False


def _peer_copies_work(devices: list[int]) -> bool:
    """Whether validated peer copies work between every pair of ``devices``
    (some systems report peer access whose copies corrupt or hang)."""
    return not _FORCE_HOST_COPIES and all(
        _peer_copy_works(a, b) for a in devices for b in devices if a != b
    )


def _copy(a: cp.ndarray, device: int, *, peer: bool) -> cp.ndarray:
    """``a`` on ``device`` (the array itself when it is already there). The
    copy has landed on return: freeing the source or the copy while a peer
    copy is in flight can fault."""
    if a.device.id == device:
        return a
    out = _copy_to_device(a, device) if peer else _copy_to_device_via_host(a, device)
    with cp.cuda.Device(device):
        cp.cuda.get_current_stream().synchronize()
    return out


def _gather_rows(
    parts: list[cp.ndarray], bounds: list[int], positions: cp.ndarray, *, peer: bool
) -> cp.ndarray:
    """Rows ``positions`` (sorted order) of the shards, on the current GPU."""
    rows = cp.empty((positions.size, parts[0].shape[1]), dtype=parts[0].dtype)
    for part, lo, hi in zip(parts, bounds[:-1], bounds[1:], strict=True):
        # Sources of cross-GPU copies must outlive them (stream-ordered pools
        # reuse freed memory at once): keep them until the synchronize.
        mine = cp.flatnonzero((positions >= lo) & (positions < hi))
        index = positions[mine] - lo
        with cp.cuda.Device(part.device.id):
            local = part[_copy(index, part.device.id, peer=peer)]
        rows[mine] = _copy(local, rows.device.id, peer=peer)
        cp.cuda.get_current_stream().synchronize()
    return rows


def _comm(devices: list[int], capacity: int, *, peer: bool):
    """The exchange of int64 sums across ``devices`` and its buffers, which
    must outlive it: per GPU two send slots (device memory, or pinned host
    memory without peer copies) and two receive slots for all GPUs."""
    buffers = []
    for device in devices:
        with cp.cuda.Device(device):
            for _ in range(2):
                buffers.append(
                    cp.empty(capacity, dtype=cp.int64)
                    if peer
                    else cp.cuda.alloc_pinned_memory(capacity * 8)
                )
            for _ in range(2):
                buffers.append(cp.empty(len(devices) * capacity, dtype=cp.int64))
    pointers = [b.data.ptr if isinstance(b, cp.ndarray) else b.ptr for b in buffers]
    return _clustering_cuda.Comm(devices, pointers, capacity, peer), buffers
