from __future__ import annotations

import warnings
from contextlib import nullcontext
from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from distributed import Client

# the codec pipeline of zarr-kvikio (installed by the `io` extras)
PIPELINE_PATH = "zarr_kvikio.KvikioCodecPipeline"
INSTALL_HINT = (
    "Reading zarr into GPU memory needs zarr-kvikio with KvikIO and nvCOMP: "
    "`pip install 'rapids-singlecell[io-cu13]'` (or `[io-cu12]`)."
)


def gpu_io_available() -> bool:
    """Whether zarr arrays can be read into GPU memory (zarr-kvikio, KvikIO and nvCOMP are installed)."""
    if find_spec("zarr_kvikio") is None:
        return False
    import zarr_kvikio

    return zarr_kvikio.available()


def require_gpu_io() -> None:
    if not gpu_io_available():
        raise ImportError(INSTALL_HINT)


def enable(client: Client | None = None, *, num_threads: int = 8) -> AbstractContextManager:
    """Read :mod:`zarr` arrays straight into GPU memory with KvikIO and nvCOMP.

    zarr reads the compressed chunks of an array with KvikIO (using GPUDirect Storage
    if available) and decompresses them on the GPU with nvCOMP, instead of decompressing
    them on the CPU and copying the result to the GPU. The GPU decodes chunks that are
    zstd-compressed (or uncompressed), ideally sharded with small inner chunks,
    see :func:`~rapids_singlecell.io.convert_zarr` and :func:`~rapids_singlecell.io.check`.
    Other chunks (e.g. blosc-compressed) are decompressed on the CPU and then copied to the GPU,
    so any store can be read. The CPU uses the codec pipeline zarr was configured with before
    (e.g. zarrs), or else zarr's fused pipeline.
    The reads are done by the codec pipeline of zarr-kvikio. Without it (or without KvikIO and
    nvCOMP), this warns and changes nothing: zarr keeps decoding on the CPU.

    Parameters
    ----------
    client
        A :class:`~distributed.Client` of a (dask-cuda) cluster.
        If given, every worker of the cluster (including ones started later) reads into
        GPU memory, while this process keeps reading into host memory (e.g. ``obs`` and
        ``var``). Arrays opened here and computed on the workers,
        e.g. with :func:`~anndata.experimental.read_elem_lazy`, are read on the GPU there.
        If not given, this process reads zarr arrays into GPU memory
        (like ``zarr.config.enable_gpu()``).
    num_threads
        Number of threads KvikIO reads with (a global KvikIO setting; its default is 1).

    Returns
    -------
    The change takes effect right away. Used as a context manager, it is undone at the end
    of the ``with`` block, here and on the workers of `client`.
    """
    if not gpu_io_available():
        warnings.warn(
            f"{INSTALL_HINT} Without them, zarr keeps decoding chunks on the CPU.",
            stacklevel=2,
        )
        return nullcontext()
    import zarr_kvikio

    if client is None:
        return zarr_kvikio.enable(num_threads=num_threads)
    import zarr_kvikio.dask

    return zarr_kvikio.dask.enable(client, num_threads=num_threads)


def release_memory() -> None:
    """Release the GPU scratch space held for decoding chunks (see :func:`~rapids_singlecell.io.enable`)."""
    if find_spec("zarr_kvikio") is not None:
        import zarr_kvikio

        zarr_kvikio.release_memory()
