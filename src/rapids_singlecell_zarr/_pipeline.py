"""A zarr codec pipeline that reads chunks into GPU memory with KvikIO and decodes them with nvCOMP.

Reads that can take the fast path

1. read the (small) shard indices on the host,
2. read every needed chunk's compressed bytes straight into device memory with KvikIO,
   one read per contiguous byte range (i.e. typically one read per shard),
3. decode all chunks of the request in a single batched nvCOMP call,
4. copy the decoded values into the output buffer
   (fully selected chunks of 1-D arrays are decoded in place, without that copy).

Everything else works with any store: reads into host buffers and writes are delegated to a host
pipeline (the one zarr was configured with before, e.g. zarrs, else zarr's
:class:`~zarr.core.codec_pipeline.FusedCodecPipeline`), and reads into GPU buffers the GPU can't
decode (e.g. blosc-compressed or remote chunks) are decoded by it on the host and then copied.
"""

from __future__ import annotations

import asyncio
import atexit
import importlib
import os
import queue
import sys
import threading
from contextlib import suppress
from dataclasses import dataclass, field
from functools import cached_property
from itertools import product
from math import prod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import google_crc32c
import numpy as np
import zarr
from zarr.abc.codec import GetResult
from zarr.codecs import BytesCodec, Crc32cCodec, ShardingCodec, ZstdCodec
from zarr.core.array_spec import ArraySpec
from zarr.core.buffer import core, cpu
from zarr.core.chunk_utils import fill_value_or_default
from zarr.core.codec_pipeline import BatchedCodecPipeline, FusedCodecPipeline
from zarr.storage import LocalStore, StorePath

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from zarr.abc.codec import CodecPipeline
    from zarr.abc.store import ByteGetter, ByteSetter, Store
    from zarr.core.buffer import NDBuffer
    from zarr.core.indexing import SelectorTuple
    from zarr.core.metadata import ArrayMetadata

    type BatchInfo = tuple[ByteGetter, ArraySpec, SelectorTuple, SelectorTuple, bool]

__all__ = ["KvikioCodecPipeline", "release_memory"]

_MAX_UINT_64 = 2**64 - 1
# Byte ranges of one file closer together than this are fetched with a single read.
_MERGE_GAP = 1 << 20
# zarr config key of the pipeline that reads into host buffers and decodes what the GPU can't:
# the one configured before `rapids_singlecell.io.enable()` (e.g. zarrs)
HOST_PIPELINE_KEY = "rapids_singlecell.host_codec_pipeline"
# zarr's default, which is much slower on the host than its fused pipeline
_ZARR_DEFAULT_PIPELINE = "zarr.core.codec_pipeline.BatchedCodecPipeline"


# ---------------------------------------------------------------------------------------------
# Deciding whether the fast path applies
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Layout:
    """What the fast path needs to know about how chunks are encoded on disk."""

    compressed: bool  # zstd if True, raw bytes if False
    inner_chunk_shape: tuple[int, ...] | None = None  # None if not sharded
    index_at_end: bool = True
    index_has_crc: bool = True


def _is_little_bytes(codec: object) -> bool:
    return isinstance(codec, BytesCodec) and codec.endian in (None, "little")


def _compression(codecs: Sequence[object]) -> bool | None:
    """True for zstd, False for no compression, None if unsupported."""
    if not codecs:
        return False
    if len(codecs) == 1 and isinstance(codecs[0], ZstdCodec) and not codecs[0].checksum:
        return True
    return None


def _layout_of(pipeline: BatchedCodecPipeline) -> _Layout | None:
    if pipeline.array_array_codecs or sys.byteorder != "little":
        return None
    ab = pipeline.array_bytes_codec
    if _is_little_bytes(ab):
        compressed = _compression(pipeline.bytes_bytes_codecs)
        return None if compressed is None else _Layout(compressed=compressed)
    if not isinstance(ab, ShardingCodec) or pipeline.bytes_bytes_codecs:
        return None
    inner, idx = ab.codecs, ab.index_codecs
    if not inner or not _is_little_bytes(inner[0]):
        return None
    if (compressed := _compression(inner[1:])) is None:
        return None
    if not idx or not _is_little_bytes(idx[0]):
        return None
    if len(idx) > 2 or (len(idx) == 2 and not isinstance(idx[1], Crc32cCodec)):
        return None
    return _Layout(
        compressed=compressed,
        inner_chunk_shape=tuple(ab.chunk_shape),
        index_at_end=ab.index_location == "end",
        index_has_crc=len(idx) == 2,
    )


def _local_path(byte_getter: object) -> Path | None:
    if isinstance(byte_getter, StorePath) and type(byte_getter.store) is LocalStore:
        return Path(byte_getter.store.root) / byte_getter.path
    return None


def _is_device_array(arr: object) -> bool:
    return type(arr).__module__.split(".")[0] == "cupy"


def _supported_dtype(spec: ArraySpec) -> bool:
    dtype = spec.dtype.to_native_dtype()
    return dtype.kind in "biufcmM" and not dtype.hasobject and dtype.itemsize > 0


class _HostNDBuffer(cpu.NDBuffer):
    """Host buffer to decode into before copying to the GPU.

    zarr's codec for zarr v2 arrays returns the globally configured buffer class (a GPU one
    after `rapids_singlecell.io.enable()`) instead of the requested one, so accept any.
    """

    def __setitem__(self, key: Any, value: Any) -> None:
        if isinstance(value, core.NDBuffer) and not isinstance(value, cpu.NDBuffer):
            value = value.as_numpy_array()
        super().__setitem__(key, value)


def _on_host(spec: ArraySpec) -> ArraySpec:
    return ArraySpec(
        shape=spec.shape,
        dtype=spec.dtype,
        fill_value=spec.fill_value,
        config=spec.config,
        prototype=cpu.buffer_prototype,
    )


class KvikioCodecPipeline(BatchedCodecPipeline):
    """Codec pipeline that reads chunks from a local file system straight into GPU memory.

    Reads of arrays whose chunks are zstd-compressed (without checksum) or uncompressed,
    optionally sharded, into GPU buffers take the KvikIO/nvCOMP fast path.
    Everything else is handled by the pipeline zarr was configured with before
    (e.g. zarrs), or else by zarr's :class:`~zarr.core.codec_pipeline.FusedCodecPipeline`,
    so the pipeline works with any array.

    zarr finds it (as ``"rapids_singlecell_zarr.KvikioCodecPipeline"``) through an entry point.
    The pipeline is bound when an array is opened, so select it before opening,
    e.g. with :func:`rapids_singlecell.io.enable`.
    """

    # the name zarr's registry and pickle know it by
    __module__ = "rapids_singlecell_zarr"

    @cached_property
    def _layout(self) -> _Layout | None:
        return _layout_of(self)

    @classmethod
    def from_array_metadata_and_store(
        cls, array_metadata: ArrayMetadata, store: Store
    ) -> KvikioCodecPipeline:
        # built from the codecs as zarr does without a store, keeping what host pipelines
        # like zarrs are built from
        from zarr.core.array import create_codec_pipeline

        pipeline = create_codec_pipeline(array_metadata)
        pipeline.__dict__["_array"] = (array_metadata, store)
        return pipeline

    async def read(
        self,
        batch_info: Iterable[BatchInfo],
        out: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> tuple[GetResult, ...]:
        batch = list(batch_info)
        if not batch:
            return ()
        out_arr = out.as_ndarray_like()
        if not _is_device_array(out_arr):
            return await self._host_pipeline(batch[0][1]).read(batch, out, drop_axes)
        if not all(_supported_dtype(spec) for _, spec, *_ in batch):
            return await super().read(batch, out, drop_axes)
        layout = self._layout
        paths = [_local_path(bg) for bg, *_ in batch]
        if layout is None or any(p is None for p in paths):
            return await self._read_via_host(batch, out_arr, drop_axes)
        if out_arr.nbytes < int(os.environ.get(_MIN_BYTES_ENV, _DEFAULT_MIN_BYTES)):
            return await self._read_small(batch, paths, out_arr, drop_axes)
        return await asyncio.to_thread(
            _read_gpu, layout, batch, paths, out_arr, drop_axes
        )

    async def write(
        self,
        batch_info: Iterable[
            tuple[ByteSetter, ArraySpec, SelectorTuple, SelectorTuple, bool]
        ],
        value: NDBuffer,
        drop_axes: tuple[int, ...] = (),
    ) -> None:
        batch = list(batch_info)
        if not batch:
            return
        if _is_device_array(value.as_ndarray_like()):
            await super().write(batch, value, drop_axes)
        else:
            await self._host_pipeline(batch[0][1]).write(batch, value, drop_axes)

    def _host_pipeline(self, spec: ArraySpec) -> CodecPipeline:
        """The host pipeline for this array."""
        if (pipeline := self.__dict__.get("_host")) is None:
            host_cls = _host_pipeline_class()
            if (array := self.__dict__.get("_array")) is not None:
                metadata, store = array
                with suppress(NotImplementedError):  # e.g. zarrs is built from these
                    pipeline = host_cls.from_array_metadata_and_store(
                        array_metadata=metadata, store=store
                    )
            if pipeline is None:
                codecs = (
                    *self.array_array_codecs,
                    self.array_bytes_codec,
                    *self.bytes_bytes_codecs,
                )
                pipeline = host_cls.from_codecs(codecs)
                pipeline = pipeline.evolve_from_array_spec(_on_host(spec))
            # (a frozen dataclass: cache it the way `cached_property` does)
            self.__dict__["_host"] = pipeline
        return pipeline

    async def _read_via_host(
        self, batch: list[BatchInfo], out_arr: Any, drop_axes: tuple[int, ...]
    ) -> tuple[GetResult, ...]:
        """Decode chunks the GPU can't on the host, then copy them into the GPU buffer."""
        import cupy as cp

        host = _HostNDBuffer.from_numpy_array(
            np.empty(out_arr.shape, dtype=out_arr.dtype)
        )
        host_batch = [
            (bg, _on_host(spec), chunk_sel, out_sel, *rest)
            for bg, spec, chunk_sel, out_sel, *rest in batch
        ]
        results = await self._host_pipeline(batch[0][1]).read(
            host_batch, host, drop_axes
        )
        out_arr[...] = cp.asarray(host.as_ndarray_like())
        return results

    async def _read_small(
        self,
        batch: list[BatchInfo],
        paths: list[Path],
        out_arr: Any,
        drop_axes: tuple[int, ...],
    ) -> tuple[GetResult, ...]:
        return await asyncio.to_thread(
            _read_host, self._layout, batch, paths, out_arr, drop_axes
        )


def _host_pipeline_class() -> type[CodecPipeline]:
    path = zarr.config.get(HOST_PIPELINE_KEY, None)
    if path is None or path == _ZARR_DEFAULT_PIPELINE:
        return FusedCodecPipeline
    module, _, name = path.rpartition(".")
    return getattr(importlib.import_module(module), name)


# ---------------------------------------------------------------------------------------------
# The GPU read
# ---------------------------------------------------------------------------------------------

_thread_local = threading.local()

# nvCOMP's batched zstd decode needs about 3.2x the decoded size as scratch space, which the codec
# object keeps (and reuses) for as long as it lives. Decoding therefore happens in calls of at most
# this many decoded bytes, by a small pool of cached codecs per device, bounding the memory held to
# ~3.2x that per codec. Each call also costs a few milliseconds, so larger limits are faster for
# large reads.
_MAX_DECODE_BYTES_ENV = "RSC_IO_MAX_DECODE_BYTES"
_DEFAULT_MAX_DECODE_BYTES = 1 << 30
# Codecs per device. More allow decoding from several threads at once, but reading sparse matrix
# blocks with 10 threads per GPU was not faster with 2-4 of them,
# and each holds its own scratch space.
_DECODERS_ENV = "RSC_IO_DECODERS"
_DEFAULT_DECODERS = 1
# Reads of fewer bytes than this are decoded on the CPU by zarr: every nvCOMP call has a fixed cost
# of a few milliseconds, more than decoding e.g. a small slice of a sparse matrix's `indptr` takes.
_MIN_BYTES_ENV = "RSC_IO_MIN_BYTES"
_DEFAULT_MIN_BYTES = 1 << 20

_pools: dict[int, queue.Queue[tuple[Any, Any]]] = {}
_pools_lock = threading.Lock()


def release_memory() -> None:
    """Release the scratch space held by the cached nvCOMP codecs."""
    with _pools_lock:
        for device_id in list(_pools):
            pool = _pools.pop(device_id)
            for _ in range(pool.maxsize):
                codec, stream = pool.get()  # waits for codecs that are in use
                stream.synchronize()
                # the codec must go before the stream it runs on
                # (a tuple would release its items the other way round)
                del codec
                del stream


# also at exit, so interpreter teardown does not release them in the wrong order
atexit.register(release_memory)


def _pool(device_id: int) -> queue.Queue[tuple[Any, Any]]:
    """The pool of nvCOMP zstd codecs for a device, each with the CUDA stream it runs on."""
    import cupy as cp
    import nvidia.nvcomp as nv

    with _pools_lock:
        if device_id not in _pools:
            n = int(os.environ.get(_DECODERS_ENV, _DEFAULT_DECODERS))
            pool: queue.Queue[tuple[Any, Any]] = queue.Queue(maxsize=n)
            for _ in range(n):
                stream = cp.cuda.Stream(non_blocking=True)
                codec = nv.Codec(
                    algorithm="Zstd",
                    bitstream_kind=nv.BitstreamKind.RAW,
                    device_id=device_id,
                    cuda_stream=stream.ptr,
                )
                pool.put((codec, stream))
            _pools[device_id] = pool
        return _pools[device_id]


def _stream(device_id: int) -> Any:
    """A per-thread, per-device CUDA stream."""
    import cupy as cp

    streams: dict[int, Any] = _thread_local.__dict__.setdefault("streams", {})
    if device_id not in streams:
        streams[device_id] = cp.cuda.Stream(non_blocking=True)
    return streams[device_id]


def _max_decode_bytes() -> int:
    return int(os.environ.get(_MAX_DECODE_BYTES_ENV, _DEFAULT_MAX_DECODE_BYTES))


def _decode(pieces: list[_Piece], stream: Any, device: Any) -> None:
    import nvidia.nvcomp as nv

    limit = _max_decode_bytes()
    groups: list[list[_Piece]] = [[]]
    size = 0
    for p in pieces:
        if size and size + p.target.nbytes > limit:
            groups.append([])
            size = 0
        groups[-1].append(p)
        size += p.target.nbytes
    pool = _pool(device.id)
    codec, codec_stream = pool.get()
    try:
        # the codec's stream waits for the reads/allocations queued on ours, and vice versa
        codec_stream.wait_event(stream.record())
        for group in groups:
            codec.decode(
                nv.as_arrays([p.data for p in group]), out=[p.target for p in group]
            )
        done = codec_stream.record()
    finally:
        # work queued later on the codec's stream runs after ours, so it can be reused right away
        pool.put((codec, codec_stream))
    stream.wait_event(done)
    # the compressed buffers must stay alive until the decode has consumed them
    done.synchronize()


@dataclass(slots=True)
class _Piece:
    """One stored chunk (or inner chunk of a shard) to fetch and decode into ``target``."""

    path: Path
    offset: int
    nbytes: int
    target: Any  # contiguous device uint8 array receiving the decoded bytes
    data: Any = None  # device uint8 array of the stored bytes, set by `_fetch`


@dataclass(slots=True)
class _Entry:
    """How one entry of the batch gets into the output."""

    spec: ArraySpec
    chunk_selection: SelectorTuple
    out_selection: SelectorTuple
    missing: bool = False
    # Region mode: decoded bounding box of the selected part of the chunk, and its offset
    region: Any = None
    origin: tuple[int, ...] = ()
    # (destination, index, source) copies to do once everything is decoded
    copies: list[tuple[Any, Any, Any]] = field(default_factory=list)


def _read_gpu(
    layout: _Layout,
    batch: list[BatchInfo],
    paths: list[Path],
    out: Any,
    drop_axes: tuple[int, ...],
) -> tuple[GetResult, ...]:
    with out.device as device:
        stream = _stream(device.id)
        with stream:
            pieces: list[_Piece] = []
            entries = [
                _plan(layout, entry, path, out, drop_axes=drop_axes, pieces=pieces)
                for entry, path in zip(batch, paths, strict=True)
            ]
            _fetch(pieces)
            if layout.compressed and pieces:
                _decode(pieces, stream, device)
            else:
                for p in pieces:
                    if p.data.size != p.target.size:
                        msg = f"{p.path}: expected {p.target.size} bytes, found {p.data.size}"
                        raise ValueError(msg)
                    p.target[...] = p.data
            results = [_scatter(e, out, drop_axes) for e in entries]
            stream.synchronize()
    return tuple(results)


def _read_host(
    layout: _Layout,
    batch: list[BatchInfo],
    paths: list[Path],
    out: Any,
    drop_axes: tuple[int, ...],
) -> tuple[GetResult, ...]:
    """Read into host memory with plain file reads and CPU decoding, then copy to the GPU at once.

    For small reads, this is faster than a (fixed-cost) nvCOMP call, and avoids zarr's own
    pipeline, which copies every chunk to the GPU separately (and warns about each copy).
    """
    from numcodecs import Zstd

    host = np.empty(out.shape, dtype=out.dtype)
    pieces: list[_Piece] = []
    entries = [
        _plan(layout, entry, path, host, drop_axes=drop_axes, pieces=pieces)
        for entry, path in zip(batch, paths, strict=True)
    ]
    _fetch_host(pieces)
    codec = Zstd() if layout.compressed else None
    for p in pieces:
        if codec is not None:
            codec.decode(p.data, out=p.target)
        elif p.data.size != p.target.size:
            msg = f"{p.path}: expected {p.target.size} bytes, found {p.data.size}"
            raise ValueError(msg)
        else:
            p.target[...] = p.data
    results = tuple(_scatter(e, host, drop_axes) for e in entries)
    out.set(host)
    return results


def _scatter(entry: _Entry, out: Any, drop_axes: tuple[int, ...]) -> GetResult:
    if entry.missing:
        out[_index(entry.out_selection, out)] = fill_value_or_default(entry.spec)
        return GetResult(status="missing")
    for dest, index, src in entry.copies:
        dest[index] = src
    if entry.region is not None:
        value = entry.region[_index(_shift(entry.chunk_selection, entry.origin), out)]
        if drop_axes:
            value = value.squeeze(axis=drop_axes)
        out[_index(entry.out_selection, out)] = value
    return GetResult(status="present")


def _plan(
    layout: _Layout,
    batch_entry: BatchInfo,
    path: Path,
    out: Any,
    *,
    drop_axes: tuple[int, ...],
    pieces: list[_Piece],
) -> _Entry:
    """Register the pieces to read for one batch entry and decide where they are decoded to.

    Contiguous selections of 1-D arrays are handled in *direct mode*: stored chunks that are
    selected as a whole are decoded straight into ``out``, the others into temporaries that are
    copied over. Everything else uses *region mode*: the stored chunks are decoded into a
    bounding box of the selection, which is then indexed into ``out``.
    """
    _, spec, chunk_sel, out_sel, _ = batch_entry
    chunk_shape = tuple(spec.shape)
    chunk_sel = _normalize(chunk_sel, chunk_shape)
    entry = _Entry(spec, chunk_sel, out_sel)
    try:
        file_size = path.stat().st_size
    except FileNotFoundError:
        entry.missing = True
        return entry

    if layout.inner_chunk_shape is None:  # the file is the one stored chunk
        stored = [((0,) * len(chunk_shape), 0, file_size)]
        stored_shape = chunk_shape
    else:
        stored_shape = layout.inner_chunk_shape
        per_shard = tuple(
            s // c for s, c in zip(chunk_shape, stored_shape, strict=True)
        )
        index = _read_shard_index(layout, path, file_size, per_shard)
        touched = [
            _touched_chunks(s, c, n)
            for s, c, n in zip(chunk_sel, stored_shape, per_shard, strict=True)
        ]
        stored = [
            (
                tuple(i * c for i, c in zip(coords, stored_shape, strict=True)),
                *(int(v) for v in index[coords]),
            )
            for coords in product(*touched)
        ]
    direct = _contiguous_1d(out, chunk_sel, out_sel, drop_axes)
    plan = _plan_direct if direct is not None else _plan_region
    plan(entry, path, stored, stored_shape, out=out, direct=direct, pieces=pieces)
    return entry


type _Stored = list[
    tuple[tuple[int, ...], int, int]
]  # (start in chunk, offset, nbytes)


def _plan_direct(
    entry: _Entry,
    path: Path,
    stored: _Stored,
    stored_shape: tuple[int, ...],
    *,
    out: Any,
    direct: tuple[int, int, int],
    pieces: list[_Piece],
) -> None:
    xp = _array_module(out)
    sel_start, sel_stop, out_start = direct
    (n,) = stored_shape
    out_bytes = out.view(np.uint8)  # one byte view to slice targets from
    dtype = entry.spec.dtype.to_native_dtype()
    fill = fill_value_or_default(entry.spec)
    for (s,), offset, nbytes in stored:
        a, b = max(s, sel_start), min(s + n, sel_stop)
        if a >= b:
            continue
        dest = slice(out_start + a - sel_start, out_start + b - sel_start)
        if (offset, nbytes) == (_MAX_UINT_64, _MAX_UINT_64):  # inner chunk not written
            out[dest] = fill
            continue
        if (a, b) == (s, s + n):
            isz = out.itemsize
            target_bytes = out_bytes[dest.start * isz : dest.stop * isz]
        else:
            target = xp.empty(n, dtype)
            entry.copies.append((out, dest, target[a - s : b - s]))
            target_bytes = target.view(np.uint8)
        pieces.append(_Piece(path, offset, nbytes, target_bytes))


def _plan_region(
    entry: _Entry,
    path: Path,
    stored: _Stored,
    stored_shape: tuple[int, ...],
    *,
    out: Any,
    direct: None,
    pieces: list[_Piece],
) -> None:
    xp = _array_module(out)
    dtype = entry.spec.dtype.to_native_dtype()
    fill = fill_value_or_default(entry.spec)
    starts = np.array([s for s, *_ in stored])
    entry.origin = tuple(int(v) for v in starts.min(axis=0))
    region_shape = tuple(
        int(hi) + c - o
        for hi, c, o in zip(starts.max(axis=0), stored_shape, entry.origin, strict=True)
    )
    entry.region = xp.empty(region_shape, dtype)
    for start, offset, nbytes in stored:
        local = tuple(
            slice(s - o, s - o + c)
            for s, o, c in zip(start, entry.origin, stored_shape, strict=True)
        )
        target = entry.region[local]
        if (offset, nbytes) == (_MAX_UINT_64, _MAX_UINT_64):  # inner chunk not written
            target[...] = fill
            continue
        if not target.flags.c_contiguous:
            tmp = xp.empty(stored_shape, dtype)
            entry.copies.append((entry.region, local, tmp))
            target = tmp
        pieces.append(_Piece(path, offset, nbytes, _as_bytes(target)))


def _as_bytes(arr: Any) -> Any:
    return arr.reshape(-1).view(np.uint8)


def _array_module(arr: Any) -> Any:
    if _is_device_array(arr):
        import cupy as cp

        return cp
    return np


def _contiguous_1d(
    out: Any,
    chunk_sel: tuple[Any, ...],
    out_sel: SelectorTuple,
    drop_axes: tuple[int, ...],
) -> tuple[int, int, int] | None:
    """(start, stop, output start) if the entry is a contiguous slice of a 1-D array."""
    if not isinstance(out_sel, tuple):
        out_sel = (out_sel,)
    if (
        len(chunk_sel) != 1
        or out.ndim != 1
        or drop_axes
        or not out.flags.c_contiguous
        or not isinstance(cs := chunk_sel[0], slice)
        or not isinstance(os_ := out_sel[0], slice)
        or cs.step != 1
        or os_.step not in (None, 1)
    ):
        return None
    return cs.start, cs.stop, os_.start or 0


def _normalize(sel: SelectorTuple, shape: tuple[int, ...]) -> tuple[Any, ...]:
    if not isinstance(sel, tuple):
        sel = (sel,)
    # boolean masks become the indices they select, which `_shift` can move into a region
    return tuple(
        slice(*s.indices(n))
        if isinstance(s, slice)
        else np.flatnonzero(s)
        if isinstance(s, np.ndarray) and s.dtype == bool
        else s
        for s, n in zip(sel, shape, strict=True)
    )


def _touched_chunks(sel: Any, chunk_len: int, n_chunks: int) -> list[int]:
    """Sorted indices of the inner chunks a per-dimension selection touches."""
    if isinstance(sel, int | np.integer):
        return [int(sel) // chunk_len]
    if isinstance(sel, slice):
        start, stop, step = sel.start, sel.stop, sel.step
        if start >= stop:
            return [min(start // chunk_len, n_chunks - 1)]
        if step == 1:
            return list(range(start // chunk_len, (stop - 1) // chunk_len + 1))
        idx = np.arange(start, stop, step)
    else:
        idx = np.asarray(sel).ravel()
        if idx.size == 0:
            return [0]
    return np.unique(idx // chunk_len).tolist()


def _shift(sel: tuple[Any, ...], origin: tuple[int, ...]) -> tuple[Any, ...]:
    """Express a selection within a chunk relative to a region starting at ``origin``."""
    shifted = []
    for s, o in zip(sel, origin, strict=True):
        if isinstance(s, slice):
            shifted.append(slice(s.start - o, s.stop - o, s.step))
        elif isinstance(s, int | np.integer):
            shifted.append(int(s) - o)
        else:
            shifted.append(np.asarray(s) - o)
    return tuple(shifted)


def _index(sel: Any, like: Any) -> Any:
    """An index with its arrays on the device (or host) that `like` is on."""
    if not _is_device_array(like):
        return sel
    import cupy as cp

    if isinstance(sel, tuple):
        return tuple(_index(s, like) for s in sel)
    return cp.asarray(sel) if isinstance(sel, np.ndarray) else sel


def _read_shard_index(
    layout: _Layout, path: Path, file_size: int, per_shard: tuple[int, ...]
) -> np.ndarray:
    n_bytes = 16 * prod(per_shard)
    size = n_bytes + (4 if layout.index_has_crc else 0)
    with path.open("rb") as f:
        raw = os.pread(f.fileno(), size, file_size - size if layout.index_at_end else 0)
    if len(raw) != size:
        msg = f"{path}: shard index is truncated"
        raise ValueError(msg)
    payload = raw[:n_bytes]
    if layout.index_has_crc and google_crc32c.value(payload) != int.from_bytes(
        raw[n_bytes:], "little"
    ):
        msg = f"{path}: shard index checksum mismatch"
        raise ValueError(msg)
    return np.frombuffer(payload, dtype="<u8").reshape((*per_shard, 2))


def _byte_ranges(pieces: list[_Piece]) -> Iterator[tuple[Path, int, int, list[_Piece]]]:
    """Group pieces by file and merge nearby ones into (path, start, stop, pieces) ranges."""
    by_path: dict[Path, list[_Piece]] = {}
    for p in pieces:
        by_path.setdefault(p.path, []).append(p)
    for path, group in by_path.items():
        group.sort(key=lambda p: p.offset)
        i = 0
        while i < len(group):
            j, start, stop = i + 1, group[i].offset, group[i].offset + group[i].nbytes
            while j < len(group) and group[j].offset <= stop + _MERGE_GAP:
                stop = max(stop, group[j].offset + group[j].nbytes)
                j += 1
            yield path, start, stop, group[i:j]
            i = j


def _fetch(pieces: list[_Piece]) -> None:
    """Read all pieces into device memory, one KvikIO read per contiguous byte range."""
    import cupy as cp
    import kvikio

    handles: dict[Path, Any] = {}
    futures = []
    try:
        for path, start, stop, group in _byte_ranges(pieces):
            if path not in handles:
                handles[path] = kvikio.CuFile(path, "r")
            buf = cp.empty(stop - start, cp.uint8)
            futures.append(handles[path].pread(buf, stop - start, file_offset=start))
            for p in group:
                p.data = buf[p.offset - start : p.offset - start + p.nbytes]
        for fut in futures:
            fut.get()
    finally:
        for fh in handles.values():
            fh.close()


def _fetch_host(pieces: list[_Piece]) -> None:
    """Read all pieces into host memory, one read per contiguous byte range."""
    for path, start, stop, group in _byte_ranges(pieces):
        with path.open("rb") as fh:
            buf = np.frombuffer(
                os.pread(fh.fileno(), stop - start, start), dtype=np.uint8
            )
        for p in group:
            p.data = buf[p.offset - start : p.offset - start + p.nbytes]
