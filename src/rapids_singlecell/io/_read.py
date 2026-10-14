from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import anndata as ad
import numpy as np
import zarr
from anndata.experimental import read_elem_lazy

from ._check import check_store
from ._config import PIPELINE_PATH

if TYPE_CHECKING:
    from zarr.storage import StoreLike

# Chunk sizes (in observations) measured best for reading an 11.4M-cell atlas on 2 GPUs:
# GPU reads favour few large chunks (fixed costs per chunk), as long as every GPU gets enough of them;
# CPU reads favour smaller ones.
_GPU_CHUNKS_PER_GPU = 55
_GPU_CHUNK_RANGE = (20_000, 200_000)
_CPU_CHUNK = 20_000


def read_lazy(path: StoreLike, *, chunks: int | Literal["auto"] = "auto") -> ad.AnnData:
    """\
    Lazily read an AnnData zarr store for out-of-core analysis on the GPU.

    `obs` and `var` are read into memory, `X` becomes a Dask array of GPU (CuPy) chunks of
    `chunks` observations each, read from disk only when computed.

    This is :func:`anndata.experimental.read_elem_lazy` for `X` with GPU chunks and a chunk size
    picked for you, so it reads like zarr is configured: after :func:`rapids_singlecell.io.enable`
    (or in a cluster from :func:`rapids_singlecell.dask.start_cluster`), chunks are read straight
    into GPU memory where the store allows it (see :func:`~rapids_singlecell.io.check`),
    otherwise they are decompressed on the CPU and copied to the GPU.

    Parameters
    ----------
    path
        The AnnData zarr store. Its `X` must be a CSR matrix or a dense array.
    chunks
        Observations per chunk. `"auto"` picks a size that worked best in benchmarks:
        ~55 chunks per GPU (Dask worker, else visible GPU; 20,000 to 200,000 observations) for GPU reads,
        20,000 observations otherwise.

    Returns
    -------
    An :class:`~anndata.AnnData` with `obs`, `var` and a lazy `X`.
    """
    from rapids_singlecell.get import X_to_GPU

    g = zarr.open_group(path, mode="r")
    with zarr.config.set(
        {"buffer": "zarr.buffer.cpu.Buffer", "ndbuffer": "zarr.buffer.cpu.NDBuffer"}
    ):
        obs, var = ad.io.read_elem(g["obs"]), ad.io.read_elem(g["var"])
    x = g["X"]
    encoding = x.attrs.get("encoding-type")
    if encoding not in {"csr_matrix", "array"}:
        msg = f"`X` must be a CSR matrix or a dense array to be chunked by observations, got {encoding}."
        raise ValueError(msg)

    gpu_reads = (
        zarr.config.get("codec_pipeline.path") == PIPELINE_PATH
        and check_store(path)["gpu_readable"]
    )
    rows = _chunk_size(len(obs), chunks, gpu_reads=gpu_reads, n_gpus=_n_gpus())
    # chunks read into GPU memory already stay as they are
    X = X_to_GPU(read_elem_lazy(x, chunks=(rows, -1)))
    return ad.AnnData(X=X, obs=obs, var=var)


def _n_gpus() -> int:
    """GPUs computing the chunks: the workers of the current Dask cluster, else the visible GPUs."""
    import cupy as cp

    try:
        from distributed import get_client

        return max(1, len(get_client().nthreads()))
    except (ImportError, ValueError):  # no distributed, or no client
        pass

    return max(1, cp.cuda.runtime.getDeviceCount())


def _chunk_size(
    n_obs: int, chunks: int | Literal["auto"], *, gpu_reads: bool, n_gpus: int
) -> int:
    if chunks != "auto":
        return int(chunks)
    if gpu_reads:
        lo, hi = _GPU_CHUNK_RANGE
        rows = int(np.clip(n_obs / (_GPU_CHUNKS_PER_GPU * n_gpus), lo, hi))
        rows = -(-rows // 1_000) * 1_000  # round up to whole thousands
    else:
        rows = _CPU_CHUNK
    return max(1, min(rows, n_obs))
