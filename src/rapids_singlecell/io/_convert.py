from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import anndata as ad
import numpy as np
import zarr

from ._check import _zarr_kvikio

if TYPE_CHECKING:
    from zarr.storage import StoreLike


def convert_zarr(
    src: StoreLike,
    dst: StoreLike,
    *,
    chunk_bytes: int = 256 << 10,
    chunks_per_shard: int = 512,
    level: int = 3,
    n_threads: int = 16,
    verify: bool = True,
) -> None:
    """Rewrite an AnnData zarr store so it can be read fast on the GPU.

    The new store has the same content, with `X` laid out for
    :func:`~rapids_singlecell.io.read_lazy` and :func:`~rapids_singlecell.io.enable`:

    * It is zstd-compressed (which nvCOMP decodes on the GPU), in chunks of `chunk_bytes`:
      nvCOMP decodes every chunk with a single thread block, so large chunks decode slowly.
      A sparse `X` is chunked along `data`, `indices` and `indptr`; a dense `X` by whole rows
      (or row pieces, if a single row is larger than `chunk_bytes`).
    * The chunks are sharded, `chunks_per_shard` per file, so a read opens a few files
      instead of one per chunk.
    * The `indices` of a sparse `X` get the smallest sufficient dtype, e.g. uint16 for < 65,536 genes.

    Other matrices (e.g. in `layers`, `raw`, `obsm` or `obsp`) are streamed the same way,
    so none is read into memory at once; all other elements (`obs`, `var`, `uns`, …)
    are copied unchanged.

    Parameters
    ----------
    src
        The AnnData zarr store to convert. Its `X` must be a CSR or CSC matrix or a dense array.
    dst
        Where to write the new store (must not exist).
    chunk_bytes
        Size of the (decompressed) chunks, in bytes.
    chunks_per_shard
        Chunks per shard (file).
    level
        zstd compression level.
    n_threads
        Threads to convert shards with.
    verify
        Read the converted matrices back and check they are identical to the source.
    """
    zk = _zarr_kvikio()
    with zk.cpu_pipeline():
        src_g = zarr.open_group(src, mode="r")
        fmt = src_g["X"].attrs.get("encoding-type")
        if fmt not in {*_SPARSE, "array"}:
            msg = f"`X` must be a CSR or CSC matrix or a dense array, got {fmt}."
            raise ValueError(msg)
        dst_g = zarr.open_group(dst, mode="w-", zarr_format=3)
        dst_g.attrs.update({"encoding-type": "anndata", "encoding-version": "0.1.0"})
        opts = {
            "chunk_bytes": chunk_bytes,
            "chunks_per_shard": chunks_per_shard,
            "level": level,
            "n_threads": n_threads,
            "verify": verify,
        }
        for key in src_g:
            _copy(zk, src_g, dst_g, key, opts)
        zarr.consolidate_metadata(dst_g.store)


_SPARSE = {"csr_matrix", "csc_matrix"}
# elements whose children are copied one by one, so no matrix is read into memory at once
_MAPPINGS = {"dict", "raw"}


def _copy(zk: Any, src_g: zarr.Group, dst_g: zarr.Group, key: str, opts: dict) -> None:
    elem = src_g[key]
    enc = elem.attrs.get("encoding-type")
    if enc in _SPARSE and isinstance(elem, zarr.Group):
        _convert_sparse(zk, elem, dst_g.create_group(key), opts)
    elif enc == "array" and isinstance(elem, zarr.Array) and elem.ndim == 2 and elem.dtype.kind in "biuf" and elem.size:
        zk.convert_array(elem, dst_g, key, **opts)
    elif enc in _MAPPINGS and isinstance(elem, zarr.Group):
        sub = dst_g.create_group(key)
        sub.attrs.update(dict(elem.attrs))
        for k in elem:
            _copy(zk, elem, sub, k, opts)
    else:
        ad.io.write_elem(dst_g, key, ad.io.read_elem(elem))


def _convert_sparse(zk: Any, sx: zarr.Group, dx: zarr.Group, opts: dict) -> None:
    """Stream a sparse matrix into `dx`, shard by shard, in the layout for GPU reads."""
    fmt = sx.attrs["encoding-type"]
    shape = tuple(cast("list[int]", sx.attrs["shape"]))
    n_minor = shape[1] if fmt == "csr_matrix" else shape[0]
    s_data, s_idx = sx["data"], sx["indices"]
    assert isinstance(s_data, zarr.Array) and isinstance(s_idx, zarr.Array)
    dx.attrs.update({"encoding-type": fmt, "encoding-version": "0.1.0", "shape": list(shape)})
    # the same chunks for all three arrays (`indptr` small ones too: every read of a row range
    # reads a piece of it), so `data` and `indices` of a row range are in the same shards
    chunks, shards = zk.layout(
        s_data.shape,
        s_data.dtype.itemsize,
        chunk_bytes=opts["chunk_bytes"],
        chunks_per_shard=opts["chunks_per_shard"],
    )
    kw = {**opts, "chunks": chunks, "shards": shards}

    def in_range(block: np.ndarray, start: int, stop: int) -> None:
        if block.size and (block.min() < 0 or block.max() >= n_minor):
            msg = f"`{s_idx.path}` out of range in [{start}, {stop})"
            raise ValueError(msg)

    zk.convert_array(sx["indptr"], dx, "indptr", **kw)
    zk.convert_array(s_data, dx, "data", **kw)
    zk.convert_array(s_idx, dx, "indices", dtype=_index_dtype(n_minor), validate=in_range, **kw)


def _index_dtype(n_minor: int) -> np.dtype:
    for dt in (np.uint16, np.int32):
        if n_minor - 1 <= np.iinfo(dt).max:
            return np.dtype(dt)
    return np.dtype(np.int64)
