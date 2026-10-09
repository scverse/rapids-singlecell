from __future__ import annotations

import anndata as ad
import cupyx.scipy.sparse as cpx
import numpy as np
import pytest
import zarr
from distributed import futures_of
from scipy import sparse
from zarr.codecs import BloscCodec

import rapids_singlecell as rsc
from rapids_singlecell.dask._cluster import _devices
from rapids_singlecell.io._config import PIPELINE_PATH, gpu_io_available

# small pools: other dask tests may share the GPU
CLUSTER_KW = {
    "gpus": [0],
    "threads_per_worker": 2,
    "protocol": "tcp",
    "pool_size": 0.003,  # fractions of the GPU memory, as by default
    "max_pool_size": "2GB",
}


@pytest.fixture
def client():
    client = rsc.dask.start_cluster(**CLUSTER_KW)
    yield client
    client.close()
    client.cluster.close()


def test_start_cluster(client):
    assert len(client.scheduler_info()["workers"]) == 1
    pipelines = set(
        client.run(
            lambda: __import__("zarr").config.get("codec_pipeline.path")
        ).values()
    )
    default = zarr.config.get("codec_pipeline.path")
    assert pipelines == ({PIPELINE_PATH} if gpu_io_available() else {default})


def test_read_lazy_and_persist(tmp_path, client):
    rng = np.random.default_rng(0)
    X = sparse.random(1000, 200, density=0.1, format="csr", dtype=np.float32, rng=rng)
    ad.io.write_zarr(tmp_path / "src.zarr", ad.AnnData(X), compressors=[BloscCodec()])
    rsc.io.convert_zarr(tmp_path / "src.zarr", tmp_path / "dst.zarr", chunk_bytes=4096)

    adata = rsc.io.read_lazy(tmp_path / "dst.zarr", chunks=250)
    assert isinstance(adata.X._meta, cpx.csr_matrix)
    adata.X = adata.X.persist()
    assert futures_of(adata.X)
    assert (adata.X.compute().get() != X).nnz == 0


def test_read_lazy_without_gpu_reads_on_workers(tmp_path):
    X = sparse.random(500, 100, density=0.1, format="csr", dtype=np.float32)
    ad.io.write_zarr(tmp_path / "src.zarr", ad.AnnData(X), compressors=[BloscCodec()])
    rsc.io.convert_zarr(tmp_path / "src.zarr", tmp_path / "dst.zarr", chunk_bytes=4096)

    client = rsc.dask.start_cluster(**CLUSTER_KW, gpu_reads=False)
    try:
        # chunks are decoded on the CPU and copied to the GPU
        adata = rsc.io.read_lazy(tmp_path / "dst.zarr", chunks=100)
        assert (adata.X.compute().get() != X).nnz == 0
    finally:
        client.close()
        client.cluster.close()


@pytest.mark.skipif(not gpu_io_available(), reason="needs KvikIO and nvCOMP")
@pytest.mark.parametrize("store", ["converted", "blosc"])
def test_plain_dask_and_anndata(tmp_path, store):
    """Without rsc's helpers: dask's cluster, anndata's reader, and `rsc.io.enable(client)`."""
    from anndata.experimental import read_elem_lazy
    from dask_cuda import LocalCUDACluster
    from distributed import Client

    X = sparse.random(500, 100, density=0.1, format="csr", dtype=np.float32)
    ad.io.write_zarr(tmp_path / "src.zarr", ad.AnnData(X), compressors=[BloscCodec()])
    path = tmp_path / "src.zarr"
    if store == "converted":
        rsc.io.convert_zarr(path, tmp_path / "dst.zarr", chunk_bytes=4096)
        path = tmp_path / "dst.zarr"

    before = zarr.config.get("codec_pipeline.path")
    cluster = LocalCUDACluster(
        CUDA_VISIBLE_DEVICES="0", n_workers=1, protocol="tcp", rmm_pool_size="256MB"
    )
    client = Client(cluster)
    try:
        rsc.io.enable(client)
        X_lazy = read_elem_lazy(zarr.open_group(path, mode="r")["X"], chunks=(100, -1))
        # the workers read every chunk into GPU memory, whatever the store
        assert isinstance(X_lazy.blocks[0].compute(), cpx.csr_matrix)
        assert (rsc.get.X_to_GPU(X_lazy).compute().get() != X).nnz == 0
    finally:
        client.close()
        cluster.close()
        zarr.config.set({"codec_pipeline.path": before})


def test_cluster_in_spawned_process(monkeypatch):
    # what an unguarded `start_cluster()` in a script does in the workers it starts
    import multiprocessing

    monkeypatch.setattr(
        multiprocessing.current_process(), "_inheriting", True, raising=False
    )
    with pytest.raises(RuntimeError, match=r"if __name__ == '__main__':"):
        rsc.dask.start_cluster(**CLUSTER_KW)


def test_cluster_missing_workers(monkeypatch):
    from distributed import Client

    def no_workers(self, n_workers, timeout=None):
        raise TimeoutError

    monkeypatch.setattr(Client, "wait_for_workers", no_workers)
    with pytest.raises(RuntimeError, match=r"of 1 Dask workers started"):
        rsc.dask.start_cluster(**CLUSTER_KW)


@pytest.mark.parametrize(
    ("gpus", "expected"), [("0,1", [0, 1]), ([2, 3], [2, 3]), ((1,), [1])]
)
def test_devices(gpus, expected):
    assert _devices(gpus) == expected
