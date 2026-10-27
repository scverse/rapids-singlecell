"""Cross-GPU helpers of harmonize on several GPUs.

The gather of sampled rows and the exchange of exact sums (harmony/comm.cuh).
"""

from __future__ import annotations

import cupy as cp

from rapids_singlecell._cuda import _harmony_clustering_cuda as _clustering_cuda
from rapids_singlecell._utils._multi_gpu import _concat_on_device, _copy_to_device, _peer_copies_work


def _gather_rows(parts: list[cp.ndarray], bounds: list[int], positions: cp.ndarray | None) -> cp.ndarray:
    """Rows ``positions`` (increasing, in sorted order; None: all) of the shards, on the current GPU.

    A shard already there is not copied.
    """
    current = cp.cuda.Device().id
    rows = parts
    if positions is not None:
        rows, cuts = [], cp.searchsorted(positions, cp.asarray(bounds)).get()
        for part, lo, a, b in zip(parts, bounds[:-1], cuts[:-1], cuts[1:], strict=True):
            with cp.cuda.Device(part.device.id):
                rows.append(part[_copy_to_device(positions[a:b], part.device.id) - lo])
    if len(rows) == 1:
        return _copy_to_device(rows[0], current)
    return _concat_on_device(rows, current)


def _comm(devices: list[int], capacity: int, *, groups: list[int]):
    """The exchange of int64 sums across ``devices`` and its buffers, which must outlive it.

    Per GPU two send slots (device memory, or pinned host memory without peer copies) and two receive slots of
    capacity + one value per GPU (comm.cuh sums what would not fit in pieces). ``groups``: first and end group of
    each GPU's cells (sorted by group).
    """
    buffers, peer = [], _peer_copies_work(devices)
    for device in devices:
        with cp.cuda.Device(device):
            for _ in range(2):
                buffers.append(
                    cp.empty(capacity, dtype=cp.int64) if peer else cp.cuda.alloc_pinned_memory(capacity * 8)
                )
            for _ in range(2):
                buffers.append(cp.empty(capacity + len(devices), dtype=cp.int64))
    pointers = [b.data.ptr if isinstance(b, cp.ndarray) else b.ptr for b in buffers]
    return _clustering_cuda.Comm(devices, pointers, capacity, peer, groups), buffers
