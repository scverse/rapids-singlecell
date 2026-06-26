from __future__ import annotations

import dask
import pytest
from dask.distributed import Client
from dask_cuda import LocalCUDACluster
from dask_cuda.utils_test import IncreasedCloseTimeoutNanny


@pytest.fixture(scope="session")
def cluster():
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0",
        protocol="tcp",
        scheduler_port=0,
        worker_class=IncreasedCloseTimeoutNanny,
    )
    yield cluster
    cluster.close()


@pytest.fixture(scope="function")
def dist_client(cluster):
    """Real distributed client backed by a (session-scoped) LocalCUDACluster.

    Only needed by tests that exercise the multi-GPU ``cuml.dask`` /
    ``cugraph.dask`` code paths (dask clustering, dask logreg, dense ``full``
    dask PCA), which raise ``ValueError: No clients found`` without a live
    distributed client. The client itself stays function-scoped so each test
    gets an isolated client (connecting to the shared cluster is cheap).
    """
    client = Client(cluster)
    try:
        yield client
    finally:
        # Always deregister the global-default client, even if the test fails,
        # so it can't leak into later `client`-fixture (synchronous) tests.
        client.close()


@pytest.fixture(scope="function")
def client():
    """Lightweight no-op stand-in for scheduler-agnostic dask tests.

    The vast majority of dask tests only build dask arrays and call
    ``.compute()`` / ``.persist()``, which run on dask's default scheduler and
    never touch the client object. Handing them ``None`` avoids spinning up a
    LocalCUDACluster and skips the distributed serialization round-trips of
    cupy chunks, which are pure overhead on the tiny test arrays.

    Forces the synchronous scheduler so these tests can never be hijacked by a
    distributed client left as the global default by an earlier ``dist_client``
    test (which would route ``.compute()`` through the shared cluster and stall
    on a GIL-holding cupy op -> the random 60s pytest-timeout hangs).
    """
    with dask.config.set(scheduler="synchronous"):
        yield None
