from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import zarr

from rapids_singlecell_zarr._pipeline import HOST_PIPELINE_KEY

if TYPE_CHECKING:
    from contextlib import AbstractContextManager

    from distributed import Client
    from donfig.config_obj import ConfigSet

PIPELINE_PATH = "rapids_singlecell_zarr.KvikioCodecPipeline"
INSTALL_HINT = (
    "Reading zarr into GPU memory needs KvikIO and nvCOMP: "
    "`pip install 'rapids-singlecell[io-cu13]'` (or `[io-cu12]`)."
)


def _missing_gpu_io() -> str | None:
    """Why zarr arrays can't be read into GPU memory here, if they can't."""
    # `find_spec("nvidia.nvcomp")` raises if there is no `nvidia` package at all
    if any(find_spec(m) is None for m in ("kvikio", "nvidia", "nvidia.nvcomp")):
        return INSTALL_HINT
    return None


def gpu_io_available() -> bool:
    """Whether zarr arrays can be read into GPU memory (KvikIO and nvCOMP are installed)."""
    return _missing_gpu_io() is None


def require_gpu_io() -> None:
    if (msg := _missing_gpu_io()) is not None:
        raise ImportError(msg)


def enable(
    client: Client | None = None, *, num_threads: int = 8
) -> AbstractContextManager:
    """\
    Read :mod:`zarr` arrays straight into GPU memory with KvikIO and nvCOMP.

    zarr reads the compressed chunks of an array with KvikIO (using GPUDirect Storage
    if available) and decompresses them on the GPU with nvCOMP, instead of decompressing
    them on the CPU and copying the result to the GPU. The GPU decodes chunks that are
    zstd-compressed (or uncompressed), ideally sharded with small inner chunks,
    see :func:`~rapids_singlecell.io.convert_zarr` and :func:`~rapids_singlecell.io.check`.
    Other chunks (e.g. blosc-compressed) are decompressed on the CPU and then copied to the GPU,
    so any store can be read. The CPU uses the codec pipeline zarr was configured with before
    (e.g. zarrs), or else zarr's fused pipeline.

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
    require_gpu_io()
    host = _host_pipeline()
    if client is None:
        return enable_local(num_threads, host)
    from ._dask_plugin import KvikioPlugin

    client.register_plugin(KvikioPlugin(num_threads=num_threads, host_pipeline=host))
    # zarr fixes an array's codec pipeline when the array is opened, and arrays opened here
    # are shipped to the workers with it.
    local = zarr.config.set(
        {"codec_pipeline.path": PIPELINE_PATH, HOST_PIPELINE_KEY: host}
    )
    return _ClusterConfigSet(local, client, KvikioPlugin.name)


def enable_local(num_threads: int, host_pipeline: str | None) -> ConfigSet:
    """Read zarr arrays into GPU memory in this process."""
    import kvikio.defaults

    kvikio.defaults.set({"num_threads": num_threads})
    return zarr.config.set(
        {
            "codec_pipeline.path": PIPELINE_PATH,
            HOST_PIPELINE_KEY: host_pipeline,
            "buffer": "zarr.buffer.gpu.Buffer",
            "ndbuffer": "zarr.buffer.gpu.NDBuffer",
        }
    )


def _host_pipeline() -> str | None:
    """The pipeline zarr is configured with, unless it is ours (then the one before that)."""
    path = zarr.config.get("codec_pipeline.path")
    return zarr.config.get(HOST_PIPELINE_KEY, None) if path == PIPELINE_PATH else path


class _ClusterConfigSet:
    """Undoes :func:`enable` with a client here and on the workers."""

    def __init__(self, local: ConfigSet, client: Client, plugin: str) -> None:
        self._local, self._client, self._plugin = local, client, plugin

    def __enter__(self) -> _ClusterConfigSet:
        return self

    def __exit__(self, *exc: object) -> None:
        self._client.unregister_worker_plugin(self._plugin)
        self._local.__exit__(*exc)
