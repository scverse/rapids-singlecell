"""Cross-GPU helpers of harmonize on several GPUs: copies between the GPUs and
the exchange of exact sums (harmony/comm.cuh)."""

from __future__ import annotations

import cupy as cp

from rapids_singlecell._cuda import _harmony_clustering_cuda as _clustering_cuda
from rapids_singlecell._utils._multi_gpu import _copy_to_device_p2p, _peer_copy_works

# Tests: move every cross-GPU value through host memory.
_FORCE_HOST_COPIES = False
_RELAY_BYTES = 1 << 23  # pinned chunk of copies through the host
_D2H, _H2D = cp.cuda.runtime.memcpyDeviceToHost, cp.cuda.runtime.memcpyHostToDevice


def _peer_copies_work(devices: list[int]) -> bool:
    """Whether validated peer copies work between every pair of ``devices``
    (some systems report peer access whose copies corrupt or hang)."""
    return not _FORCE_HOST_COPIES and all(
        _peer_copy_works(a, b) for a in devices for b in devices if a != b
    )


def _copy(a: cp.ndarray, device: int, *, out: cp.ndarray | None = None) -> cp.ndarray:
    """``a`` on ``device`` (``a`` itself, ignoring ``out``, when it is already
    there), into ``out`` of the same layout if given: a validated peer copy,
    else through two pinned host buffers, one chunk leaving its GPU while the
    previous one arrives. The copy has landed on return: freeing the source or
    the copy while a peer copy is in flight can fault."""
    source = a.device.id
    if source == device:
        return a
    if _peer_copies_work([source, device]):
        out = _copy_to_device_p2p(a, device, out=out)
    else:
        with cp.cuda.Device(source):
            if not (a.flags.c_contiguous or a.flags.f_contiguous):
                a = cp.ascontiguousarray(a)
            stream = cp.cuda.get_current_stream()
        with cp.cuda.Device(device):
            out = cp.empty_like(a) if out is None else out
        step = max(1, min(a.nbytes, _RELAY_BYTES))
        hosts, free = [cp.cuda.alloc_pinned_memory(step) for _ in range(2)], [None] * 2
        for i, at in enumerate(range(0, a.nbytes, step)):
            size, host = min(step, a.nbytes - at), hosts[i % 2].ptr
            with cp.cuda.Device(source):
                if free[i % 2] is not None:
                    stream.wait_event(free[i % 2])  # the buffer's last chunk arrived
                cp.cuda.runtime.memcpyAsync(
                    host, a.data.ptr + at, size, _D2H, stream.ptr
                )
                ready = stream.record()
            with cp.cuda.Device(device):
                target = cp.cuda.get_current_stream()
                target.wait_event(ready)
                cp.cuda.runtime.memcpyAsync(
                    out.data.ptr + at, host, size, _H2D, target.ptr
                )
                free[i % 2] = target.record()
    with cp.cuda.Device(device):
        cp.cuda.get_current_stream().synchronize()  # also frees the buffers
    return out


def _gather_rows(
    parts: list[cp.ndarray], bounds: list[int], positions: cp.ndarray
) -> cp.ndarray:
    """Rows ``positions`` (increasing, in sorted order) of the shards, on the
    current GPU."""
    current, rows = cp.cuda.Device().id, []
    cuts = cp.searchsorted(positions, cp.asarray(bounds)).get()
    for part, lo, a, b in zip(parts, bounds[:-1], cuts[:-1], cuts[1:], strict=True):
        with cp.cuda.Device(part.device.id):
            local = part[_copy(positions[a:b], part.device.id) - lo]
        rows.append(_copy(local, current))
    return rows[0] if len(rows) == 1 else cp.concatenate(rows)


def _comm(devices: list[int], capacity: int, *, peer: bool, groups: list[int]):
    """The exchange of int64 sums across ``devices`` and its buffers, which
    must outlive it: per GPU two send slots (device memory, or pinned host
    memory without peer copies) and two receive slots of capacity + one value
    per GPU (comm.cuh sums what would not fit in pieces). ``groups``: first
    and end group of each GPU's cells (sorted by group)."""
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
                buffers.append(cp.empty(capacity + len(devices), dtype=cp.int64))
    pointers = [b.data.ptr if isinstance(b, cp.ndarray) else b.ptr for b in buffers]
    return _clustering_cuda.Comm(devices, pointers, capacity, peer, groups), buffers
