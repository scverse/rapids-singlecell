from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from importlib.util import find_spec
from math import ceil
from typing import TYPE_CHECKING

import anndata as ad
import numpy as np
import zarr
from tqdm.auto import tqdm
from zarr.codecs import ZstdCodec

if TYPE_CHECKING:
    from collections.abc import Callable

    from donfig.config_obj import ConfigSet
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
    """\
    Rewrite an AnnData zarr store so it can be read fast on the GPU.

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
    with _cpu_pipeline():
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
            _copy(src_g, dst_g, key, opts)
        zarr.consolidate_metadata(dst_g.store)


_SPARSE = {"csr_matrix", "csc_matrix"}
# elements whose children are copied one by one, so no matrix is read into memory at once
_MAPPINGS = {"dict", "raw"}


def _copy(src_g: zarr.Group, dst_g: zarr.Group, key: str, opts: dict) -> None:
    elem = src_g[key]
    enc = elem.attrs.get("encoding-type")
    if enc in _SPARSE:
        _convert_sparse(elem, dst_g.create_group(key), **opts)
    elif enc == "array" and elem.ndim == 2 and elem.dtype.kind in "biuf" and elem.size:
        _convert_dense(elem, dst_g, key, **opts)
    elif enc in _MAPPINGS:
        sub = dst_g.create_group(key)
        sub.attrs.update(dict(elem.attrs))
        for k in elem:
            _copy(elem, sub, k, opts)
    else:
        ad.io.write_elem(dst_g, key, ad.io.read_elem(elem))


def _convert_sparse(
    sx: zarr.Group,
    dx: zarr.Group,
    *,
    chunk_bytes: int,
    chunks_per_shard: int,
    level: int,
    n_threads: int,
    verify: bool,
) -> None:
    """Stream a sparse matrix into `dx`, shard by shard, in the layout for GPU reads."""
    fmt = sx.attrs["encoding-type"]
    shape = tuple(sx.attrs["shape"])
    n_minor = shape[1] if fmt == "csr_matrix" else shape[0]
    s_data, s_idx, s_indptr = sx["data"], sx["indices"], sx["indptr"]
    assert isinstance(s_data, zarr.Array) and isinstance(s_idx, zarr.Array)
    assert isinstance(s_indptr, zarr.Array)
    indptr = np.asarray(s_indptr[...])
    nnz = int(indptr[-1])
    idx_dtype = _index_dtype(n_minor)
    inner = max(1, chunk_bytes // s_data.dtype.itemsize)
    shard = inner * chunks_per_shard
    n_shards = ceil(nnz / shard)

    dx.attrs.update(
        {"encoding-type": fmt, "encoding-version": "0.1.0", "shape": list(shape)}
    )
    compressors = [ZstdCodec(level=level)]
    # `indptr` gets small chunks too: every read of a row range reads a piece of it
    dx.create_array(
        "indptr",
        data=indptr,
        chunks=(inner,),
        shards=(shard,),
        compressors=compressors,
    )
    kw = {"chunks": (inner,), "shards": (shard,), "compressors": compressors}
    d_data = dx.create_array("data", shape=(nnz,), dtype=s_data.dtype, **kw)
    d_idx = dx.create_array("indices", shape=(nnz,), dtype=idx_dtype, **kw)
    name = sx.path

    def convert(k: int) -> int:
        s, e = k * shard, min(nnz, (k + 1) * shard)
        idx = np.asarray(s_idx[s:e])
        if idx.size and (idx.min() < 0 or idx.max() >= n_minor):
            msg = f"`{name}/indices` out of range in [{s}, {e})"
            raise ValueError(msg)
        d_data[s:e] = s_data[s:e]
        d_idx[s:e] = idx.astype(idx_dtype)
        return e - s

    def check(k: int) -> int:
        s, e = k * shard, min(nnz, (k + 1) * shard)
        same = np.array_equal(
            d_data[s:e], s_data[s:e], equal_nan=True
        ) and np.array_equal(d_idx[s:e], s_idx[s:e])
        if not same:
            msg = f"The converted `{name}` differs from the source in [{s}, {e})."
            raise AssertionError(msg)
        return e - s

    _run(
        name,
        nnz,
        n_shards,
        convert,
        check=check if verify else None,
        n_threads=n_threads,
    )
    if verify and not np.array_equal(dx["indptr"][...], indptr):
        msg = f"The converted `{name}/indptr` differs from the source."
        raise AssertionError(msg)


def _convert_dense(
    sx: zarr.Array,
    dst_g: zarr.Group,
    key: str,
    *,
    chunk_bytes: int,
    chunks_per_shard: int,
    level: int,
    n_threads: int,
    verify: bool,
) -> None:
    """Stream a dense matrix into `dst_g[key]`, shard by shard, in the layout for GPU reads."""
    n_rows, n_cols = sx.shape
    itemsize = sx.dtype.itemsize
    if n_cols * itemsize <= chunk_bytes:  # whole rows per chunk
        chunks = (max(1, chunk_bytes // (n_cols * itemsize)), n_cols)
        shards = (chunks[0] * chunks_per_shard, n_cols)
    else:  # every row in pieces of `chunk_bytes`
        chunks = (1, max(1, chunk_bytes // itemsize))
        per_row = ceil(n_cols / chunks[1])
        shards = (max(1, chunks_per_shard // per_row), per_row * chunks[1])
    dx = dst_g.create_array(
        key,
        shape=sx.shape,
        dtype=sx.dtype,
        chunks=chunks,
        shards=shards,
        compressors=[ZstdCodec(level=level)],
    )
    dx.attrs.update(dict(sx.attrs))
    rows = shards[0]
    name = sx.path

    def convert(k: int) -> int:
        s, e = k * rows, min(n_rows, (k + 1) * rows)
        dx[s:e] = sx[s:e]
        return (e - s) * n_cols

    def check(k: int) -> int:
        s, e = k * rows, min(n_rows, (k + 1) * rows)
        if not np.array_equal(dx[s:e], sx[s:e], equal_nan=True):
            msg = f"The converted `{name}` differs from the source in rows [{s}, {e})."
            raise AssertionError(msg)
        return (e - s) * n_cols

    n_blocks = ceil(n_rows / rows)
    _run(
        name,
        n_rows * n_cols,
        n_blocks,
        convert,
        check=check if verify else None,
        n_threads=n_threads,
    )


def _run(
    name: str,
    total: int,
    n_blocks: int,
    convert: Callable[[int], int],
    *,
    check: Callable[[int], int] | None,
    n_threads: int,
) -> None:
    """Convert (and check) all blocks with `n_threads` threads, with progress bars."""
    steps = [(f"Converting {name}", convert)]
    if check is not None:
        steps.append((f"Verifying {name}", check))
    for desc, fn in steps:
        with (
            ThreadPoolExecutor(n_threads) as ex,
            tqdm(total=total, desc=desc, unit_scale=True, unit=" values") as bar,
        ):
            for n in ex.map(fn, range(n_blocks)):
                bar.update(n)


def _index_dtype(n_minor: int) -> np.dtype:
    for dt in (np.uint16, np.int32):
        if n_minor - 1 <= np.iinfo(dt).max:
            return np.dtype(dt)
    return np.dtype(np.int64)


def _cpu_pipeline() -> ConfigSet:
    """Host buffers and a fast CPU codec pipeline, for the arrays opened within."""
    settings = {
        "buffer": "zarr.buffer.cpu.Buffer",
        "ndbuffer": "zarr.buffer.cpu.NDBuffer",
    }
    if find_spec("zarrs") is not None:
        import zarrs  # noqa: F401  (registers the pipeline)

        settings["codec_pipeline.path"] = "zarrs.ZarrsCodecPipeline"
    else:
        settings["codec_pipeline.path"] = "zarr.core.codec_pipeline.FusedCodecPipeline"
    return zarr.config.set(settings)
