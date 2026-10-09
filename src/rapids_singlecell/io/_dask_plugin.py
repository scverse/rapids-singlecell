from __future__ import annotations

from distributed.diagnostics.plugin import WorkerPlugin


class KvikioPlugin(WorkerPlugin):
    """Make a Dask worker read zarr arrays into GPU memory with KvikIO and nvCOMP."""

    name = "rapids-singlecell-gpu-io"

    def __init__(
        self, *, num_threads: int = 8, host_pipeline: str | None = None
    ) -> None:
        self.num_threads = num_threads
        # the client's host pipeline (e.g. zarrs), for what the GPU can't decode
        self.host_pipeline = host_pipeline

    def setup(self, worker) -> None:
        from ._config import enable_local

        self._config = enable_local(self.num_threads, self.host_pipeline)

    def teardown(self, worker) -> None:
        from rapids_singlecell_zarr import release_memory

        if (config := getattr(self, "_config", None)) is not None:  # set up
            config.__exit__(None, None, None)
        release_memory()
