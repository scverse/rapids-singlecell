from __future__ import annotations

import sys
import warnings
from contextlib import nullcontext
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from distributed import Client

# the codec pipeline of zarr-kvikio (installed by the `io` extras)
PIPELINE_PATH = "zarr_kvikio.KvikioCodecPipeline"
INSTALL_HINT = (
    "Reading zarr into GPU memory needs zarr-kvikio with KvikIO and nvCOMP: "
    "`pip install 'zarr-kvikio[cuda13]'` (or `[cuda12]`)."
)
# warnings point at the caller of rapids_singlecell
_SKIP = (str(Path(__file__).parents[1]),)


def zarr_kvikio_or_none() -> Any:
    """zarr-kvikio, or None if it isn't installed or fails to import (e.g. with an untested zarr)."""
    if find_spec("zarr_kvikio") is None:
        return None
    try:
        import zarr_kvikio
    except Exception as e:  # noqa: BLE001  (it subclasses zarr internals, which zarr may move)
        warnings.warn(
            f"zarr-kvikio failed to import, so zarr decodes chunks on the CPU: {e!r}", skip_file_prefixes=_SKIP
        )
        return None
    return zarr_kvikio


def gpu_io_available() -> bool:
    """Whether zarr arrays can be read into GPU memory (zarr-kvikio, KvikIO and nvCOMP are installed)."""
    zk = zarr_kvikio_or_none()
    return zk is not None and zk.available()


def require_gpu_io() -> None:
    if not gpu_io_available():
        raise ImportError(INSTALL_HINT)


def enable(client: Client | None = None) -> AbstractContextManager:
    """Read :mod:`zarr` arrays straight into GPU memory with KvikIO and nvCOMP.

    zarr reads the compressed chunks of an array with KvikIO
    and decompresses them on the GPU with nvCOMP, instead of decompressing
    them on the CPU and copying the result to the GPU. The GPU decodes chunks that are
    zstd-compressed (or uncompressed), ideally sharded with small inner chunks,
    see :func:`~rapids_singlecell.io.convert_zarr` and :func:`~rapids_singlecell.io.check`.
    Other chunks (e.g. blosc-compressed) are decompressed on the CPU and then copied to the GPU,
    so any store can be read. The CPU uses the codec pipeline zarr was configured with before
    (e.g. zarrs), or else zarr's fused pipeline.
    The reads are done by the codec pipeline of `zarr-kvikio <https://github.com/scverse/zarr-kvikio>`_,
    from local files (zarr's :class:`~zarr.storage.LocalStore`; other stores are decoded on the CPU).
    Without it (or without KvikIO and nvCOMP: with `client`, on its workers, which need zarr-kvikio
    installed), this warns and changes nothing: zarr keeps decoding on the CPU.
    Its settings, e.g. KvikIO's thread count, are keys of zarr's config
    (``zarr.config.set({"kvikio.num_threads": 16})``, see zarr-kvikio's documentation).

    Parameters
    ----------
    client
        A :class:`~distributed.Client` of a (dask-cuda) cluster.
        If given, every worker of the cluster (including ones started later) reads into
        GPU memory, while this process keeps reading into host memory (e.g. ``obs`` and
        ``var``; with in-process workers, e.g. ``Client(processes=False)``, it reads into GPU
        memory too, until this is undone, also after the cluster closes). Arrays opened here and
        computed on the workers, e.g. with :func:`~anndata.experimental.read_elem_lazy`, are read on the GPU there,
        with this process's ``kvikio.*`` settings as they are when this is called.
        If not given, this process reads zarr arrays into GPU memory
        (like ``zarr.config.enable_gpu()``).

    Returns
    -------
    The change takes effect right away. Used as a context manager, it is undone at the end
    of the ``with`` block, here and on the workers of `client`.
    """
    zk = zarr_kvikio_or_none()
    # (with a client, only its workers need KvikIO and nvCOMP)
    if zk is None or (client is None and not zk.available()):
        if zk is not None or find_spec("zarr_kvikio") is None:  # (else zarr_kvikio_or_none warned)
            warnings.warn(
                f"{INSTALL_HINT} Without them, zarr keeps decoding chunks on the CPU.", skip_file_prefixes=_SKIP
            )
        return nullcontext()
    if client is None:
        return zk.enable()
    import zarr_kvikio.dask

    try:
        return zarr_kvikio.dask.enable(client)
    except ImportError as e:  # (a worker lacks KvikIO or nvCOMP: undone on all of them)
        warnings.warn(f"{e} Without them, zarr keeps decoding chunks on the CPU.", skip_file_prefixes=_SKIP)
        return nullcontext()


def release_memory() -> None:
    """Release the GPU scratch space held for decoding chunks and the pinned host read buffers.

    In this process (for the workers of a Dask cluster, ``client.run(rapids_singlecell.io.release_memory)``;
    ``client.restart()`` frees it too), see :func:`~rapids_singlecell.io.enable`.
    """
    if (zk := sys.modules.get("zarr_kvikio")) is not None:  # (else it holds nothing)
        zk.release_memory()
