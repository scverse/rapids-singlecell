from __future__ import annotations

import multiprocessing
import os
import warnings
from importlib.util import find_spec
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from distributed import Client

# A GPU with more memory than this already in use probably runs something else.
_BUSY_FRACTION = 0.1
# Initial pool of RMM's pool allocator (used with managed memory); the asynchronous one needs none.
_POOL_SIZE = 0.2


def start_cluster(
    gpus: Sequence[int] | str | None = None,
    *,
    threads_per_worker: int = 4,
    protocol: Literal["auto", "ucx", "tcp"] = "auto",
    pool_size: float | str = 0,
    max_pool_size: float | str = 0.75,
    gpu_reads: bool | Literal["auto"] = "auto",
    **cluster_kwargs,
) -> Client:
    """\
    Start a local Dask cluster with one worker per GPU, for out-of-core analysis.

    This only saves the setup: the result is a regular :class:`~distributed.Client`
    for a :class:`~dask_cuda.LocalCUDACluster`, which keeps running until you shut it down::

        client = rsc.dask.start_cluster()
        adata = rsc.io.read_lazy("data.zarr")
        ...
        client.close()
        client.cluster.close()  # stops the workers and the scheduler

    The defaults are chosen to work out of the box:

    * One worker per GPU with `threads_per_worker` threads:
      a few threads overlap reading chunks with computing on them;
      more mainly increase memory use.
    * Each worker allocates GPU memory with RMM's asynchronous allocator (``cudaMallocAsync``),
      using at most `max_pool_size` of the GPU, which leaves room for this process
      (e.g. for the PCA) and for decompressing chunks on the GPU. Workers keep the memory they used
      for reuse; ``client.restart()`` releases it (and drops the data persisted on the workers),
      e.g. before using the GPUs in this process.
      Unlike RMM's pool allocator, it does not fragment with the large chunks of out-of-core work.
      With CUDA managed memory (`rmm_managed_memory=True`), RMM's pool allocator is used.
    * UCX for communication between workers if available (`protocol="auto"`), else TCP.
      With CUDA managed memory (`rmm_managed_memory=True`), which UCX does not support, TCP.
    * If KvikIO and nvCOMP are installed, the workers read zarr chunks straight into
      GPU memory (see :func:`rapids_singlecell.io.enable`).

    Parameters
    ----------
    gpus
        GPUs to use, e.g. `[0, 1]` or `"0,1"`. Defaults to all visible GPUs.
    threads_per_worker
        Threads per worker.
    protocol
        Communication protocol between workers.
    pool_size
        GPU memory each worker reserves up front: a fraction of the GPU memory (e.g. `0.2`),
        a number of bytes, or a string like `"20GB"`. By default nothing with the asynchronous
        allocator, which is as fast without (idle workers then leave the GPU to this process),
        and `0.2` with RMM's pool allocator.
    max_pool_size
        Maximum GPU memory each worker uses, in the same units as `pool_size`.
    gpu_reads
        Read zarr chunks straight into GPU memory on the workers. `"auto"` does if possible.
    **cluster_kwargs
        Passed on to :class:`~dask_cuda.LocalCUDACluster`.

    Returns
    -------
    A :class:`~distributed.Client` connected to the cluster (`client.cluster`).
    """
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        # Dask starts workers by running the main script again (multiprocessing "spawn"),
        # so an unguarded `start_cluster()` in a script would start clusters in its own workers.
        msg = (
            "`rapids_singlecell.dask.start_cluster()` was called while a Dask worker process was starting, "
            "so the workers could not start. In a script, start the cluster under "
            "`if __name__ == '__main__':`."
        )
        raise RuntimeError(msg)
    try:
        from dask_cuda import LocalCUDACluster
        from distributed import Client
    except ImportError as e:
        msg = "`rapids_singlecell.dask.start_cluster` needs `dask-cuda` (`pip install dask-cuda`)."
        raise ImportError(msg) from e
    from rapids_singlecell.io._config import gpu_io_available, require_gpu_io

    if gpu_reads is True:
        require_gpu_io()
    devices = _devices(gpus)
    _warn_if_busy(devices)
    managed = cluster_kwargs.get("rmm_managed_memory", False)
    # RMM's pool allocator fragments with large chunks; managed memory needs it
    use_async = cluster_kwargs.setdefault("rmm_async", not managed)
    if protocol == "auto":
        # UCX does not work with CUDA managed memory
        ucx = find_spec("distributed_ucxx") is not None
        protocol = "ucx" if ucx and not managed else "tcp"
    local_cluster = LocalCUDACluster(
        **(
            {"CUDA_VISIBLE_DEVICES": ",".join(map(str, devices))}
            if devices is not None
            else {}
        ),
        threads_per_worker=threads_per_worker,
        protocol=protocol,
        rmm_pool_size=pool_size or (None if use_async else _POOL_SIZE),
        rmm_maximum_pool_size=max_pool_size,
        rmm_allocator_external_lib_list=["cupy"],
        **cluster_kwargs,
    )
    client = Client(local_cluster)
    try:
        _wait_for_workers(client, len(local_cluster.worker_spec))
        if gpu_reads is True or (gpu_reads == "auto" and gpu_io_available()):
            from rapids_singlecell.io import enable

            enable(client)
    except BaseException:
        client.close()
        local_cluster.close()
        raise
    return client


def _wait_for_workers(client: Client, n: int, timeout: float = 60) -> None:
    """Make sure every worker started: Dask carries on with fewer, which then just waits."""
    try:
        client.wait_for_workers(n, timeout=timeout)
    except TimeoutError:
        started = len(client.scheduler_info()["workers"])
        msg = (
            f"Only {started} of {n} Dask workers started, see the log above for why. "
            "A common cause is starting the cluster in a script without `if __name__ == '__main__':`."
        )
        raise RuntimeError(msg) from None


def _devices(gpus: Sequence[int] | str | None) -> list[int] | None:
    if gpus is None:  # dask-cuda follows CUDA_VISIBLE_DEVICES (which may hold UUIDs)
        return None
    if isinstance(gpus, str):
        return [int(g) for g in gpus.split(",") if g.strip()]
    return [int(g) for g in gpus]


def _warn_if_busy(devices: list[int] | None) -> None:
    try:
        import pynvml
    except ImportError:
        return
    try:
        pynvml.nvmlInit()
        if devices is None:
            visible = os.environ.get("CUDA_VISIBLE_DEVICES")
            devices = (
                range(pynvml.nvmlDeviceGetCount())
                if visible is None
                # GPUs given by UUID are not checked
                else [int(g) for g in visible.split(",") if g.strip().isdigit()]
            )
        for i in devices:
            mem = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
            if mem.used > _BUSY_FRACTION * mem.total:
                warnings.warn(
                    f"GPU {i} already has {mem.used / 2**30:.0f} of {mem.total / 2**30:.0f} GiB in use, "
                    "e.g. by another cluster or notebook. Sharing GPUs can run out of memory.",
                    stacklevel=3,
                )
    except pynvml.NVMLError:
        return
