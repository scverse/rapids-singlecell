from __future__ import annotations

import pickle
import warnings
from contextlib import contextmanager
from typing import TYPE_CHECKING

import cupy as cp
import numcodecs
import numpy as np
import pytest
import zarr
from zarr.codecs import (
    BloscCodec,
    BytesCodec,
    Crc32cCodec,
    GzipCodec,
    ShardingCodec,
    ZstdCodec,
)
from zarr.core.codec_pipeline import FusedCodecPipeline

import rapids_singlecell.io as rio
from rapids_singlecell.io._config import gpu_io_available
from rapids_singlecell_zarr import _pipeline

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

pytestmark = pytest.mark.skipif(
    not gpu_io_available(), reason="needs KvikIO and nvCOMP"
)

PIPELINE = "rapids_singlecell_zarr.KvikioCodecPipeline"
ZSTD = (ZstdCodec(level=3),)


@contextmanager
def gpu_pipeline() -> Generator[None]:
    with rio.enable():
        yield


@pytest.fixture(autouse=True, params=["gpu", "host"])
def read_path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Run every test through both read paths: KvikIO + nvCOMP, and host reads + CPU decoding
    (which is used for reads smaller than `RSC_IO_MIN_BYTES`)."""
    monkeypatch.setenv(
        "RSC_IO_MIN_BYTES", "0" if request.param == "gpu" else str(1 << 40)
    )
    return request.param


@pytest.fixture
def fused_reads(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records reads delegated to zarr's fused pipeline, by where their output is."""
    calls: list[str] = []
    orig = FusedCodecPipeline.read

    async def spy(self, batch_info, out, drop_axes=()):
        calls.append("host" if isinstance(out.as_ndarray_like(), np.ndarray) else "gpu")
        return await orig(self, batch_info, out, drop_axes)

    monkeypatch.setattr(FusedCodecPipeline, "read", spy)
    return calls


@pytest.fixture
def fast_path_calls(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Records reads that zarr-kvikio handles itself (instead of zarr's own pipeline)."""
    calls: list[str] = []
    for name in ["_read_gpu", "_read_host"]:
        orig = getattr(_pipeline, name)

        def spy(*args, _orig=orig, _name=name, **kwargs):
            calls.append(_name)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(_pipeline, name, spy)
    return calls


def write(
    path: Path,
    data: np.ndarray,
    chunks: tuple[int, ...],
    *,
    shards: tuple[int, ...] | None = None,
    compressors=ZSTD,
    fill_value=0,
    **kwargs,
) -> zarr.Array:
    arr = zarr.create_array(
        path,
        shape=data.shape,
        dtype=data.dtype,
        chunks=chunks,
        shards=shards,
        compressors=compressors,
        fill_value=fill_value,
        **kwargs,
    )
    arr[...] = data
    return arr


def read_gpu(path: Path, selection, *, how: str = "getitem") -> np.ndarray:
    with gpu_pipeline():
        arr = zarr.open_array(path, mode="r")
        assert type(arr._async_array.codec_pipeline).__name__ == "KvikioCodecPipeline"
        res = getattr(arr, how)[selection] if how != "getitem" else arr[selection]
    assert isinstance(res, cp.ndarray)
    return cp.asnumpy(res)


def read_cpu(path: Path, selection, *, how: str = "getitem") -> np.ndarray:
    arr = zarr.open_array(path, mode="r")
    return getattr(arr, how)[selection] if how != "getitem" else arr[selection]


def check(path: Path, selection, *, how: str = "getitem") -> None:
    np.testing.assert_array_equal(
        read_gpu(path, selection, how=how), read_cpu(path, selection, how=how)
    )


rng = np.random.default_rng(0)

SELECTIONS_1D = [
    slice(None),
    slice(3, 997),
    slice(250, 750),  # whole inner chunks in the middle
    slice(5, 900, 7),
    17,
    -1,
]


@pytest.mark.parametrize("dtype", ["float32", "int64", "uint8", "float64"])
@pytest.mark.parametrize(
    "compressors", [(ZstdCodec(level=3),), ()], ids=["zstd", "raw"]
)
@pytest.mark.parametrize("shards", [None, (500,)], ids=["unsharded", "sharded"])
@pytest.mark.parametrize("selection", SELECTIONS_1D, ids=repr)
def test_1d(tmp_path, fast_path_calls, *, dtype, compressors, shards, selection):
    data = (rng.random(1003) * 100).astype(dtype)
    write(tmp_path / "a.zarr", data, (50,), shards=shards, compressors=compressors)
    check(tmp_path / "a.zarr", selection)
    assert fast_path_calls


@pytest.mark.parametrize("shards", [None, (20, 30, 8)], ids=["unsharded", "sharded"])
@pytest.mark.parametrize(
    ("selection", "how"),
    [
        (slice(None), "getitem"),
        ((slice(3, 37), slice(None), 5), "getitem"),
        ((slice(None, None, 3), slice(4, 50, 5), slice(None)), "getitem"),
        ((7, 11, slice(None)), "getitem"),
        (([0, 3, 21, 39], slice(None), [1, 6]), "oindex"),
        ((np.arange(40) % 3 == 0, slice(2, 9), slice(None)), "oindex"),
        pytest.param(
            ([0, 3, 21, 39], [1, 59, 2, 30], [7, 0, 3, 5]),
            "vindex",
            marks=pytest.mark.xfail(
                reason="zarr's vindex calls np.array on the output, so it fails for GPU buffers "
                "with any pipeline",
                raises=TypeError,
                strict=True,
            ),
        ),
        ((slice(1, 2), slice(0, 1), slice(None)), "blocks"),
    ],
    ids=["all", "slices", "steps", "ints", "orthogonal", "mask", "coords", "blocks"],
)
def test_nd(tmp_path, fast_path_calls, shards, selection, how):
    data = rng.random((40, 60, 8)).astype("float32")
    write(tmp_path / "a.zarr", data, (10, 15, 4), shards=shards)
    check(tmp_path / "a.zarr", selection, how=how)
    assert fast_path_calls


@pytest.mark.parametrize("index_location", ["start", "end"])
@pytest.mark.parametrize(
    "index_codecs",
    [(BytesCodec(),), (BytesCodec(), Crc32cCodec())],
    ids=["plain", "crc"],
)
def test_shard_index_variants(tmp_path, fast_path_calls, index_location, index_codecs):
    data = rng.random((64, 64)).astype("float32")
    arr = zarr.create_array(
        tmp_path / "a.zarr",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(32, 32),
        serializer=ShardingCodec(
            chunk_shape=(8, 8),
            codecs=[BytesCodec(), ZstdCodec(level=1)],
            index_codecs=index_codecs,
            index_location=index_location,
        ),
        compressors=None,
    )
    arr[...] = data
    check(tmp_path / "a.zarr", (slice(5, 60), slice(None)))
    assert fast_path_calls


@pytest.mark.parametrize("shards", [None, (100,)], ids=["unsharded", "sharded"])
def test_missing_chunks(tmp_path, fast_path_calls, shards):
    data = np.full(400, 7, dtype="float32")
    data[:25] = np.arange(
        25
    )  # all other (inner) chunks equal the fill value and are not written
    write(tmp_path / "a.zarr", data, (20,), shards=shards, fill_value=7)
    check(tmp_path / "a.zarr", slice(None))
    check(tmp_path / "a.zarr", slice(150, 390))  # only missing chunks / an empty shard
    assert fast_path_calls


def test_empty_array(tmp_path):
    write(tmp_path / "a.zarr", np.zeros(0, "float32"), (10,))
    check(tmp_path / "a.zarr", slice(None))


def test_corrupt_index(tmp_path):
    data = rng.random(100).astype("float32")
    write(tmp_path / "a.zarr", data, (10,), shards=(50,))
    shard = tmp_path / "a.zarr" / "c" / "0"
    raw = bytearray(shard.read_bytes())
    raw[-10] ^= 0xFF
    shard.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="checksum"):
        read_gpu(tmp_path / "a.zarr", slice(None))


@pytest.mark.parametrize(
    "compressors",
    [(GzipCodec(),), (BloscCodec(),), (ZstdCodec(level=3, checksum=True),)],
    ids=["gzip", "blosc", "zstd-checksum"],
)
def test_unsupported_codecs_fall_back(
    tmp_path, fast_path_calls, fused_reads, compressors
):
    data = rng.random(300).astype("float32")
    write(tmp_path / "a.zarr", data, (50,), compressors=compressors)
    check(tmp_path / "a.zarr", slice(10, 290))
    assert not fast_path_calls
    # decoded on the host by zarr's fused pipeline, then copied to the GPU
    assert fused_reads
    assert set(fused_reads) == {"host"}


def test_zarr_v2_falls_back(tmp_path, fast_path_calls, fused_reads):
    """Like most existing AnnData stores: zarr v2, blosc-compressed."""
    data = rng.random((40, 30)).astype("float32")
    arr = zarr.create_array(
        tmp_path / "a.zarr",
        shape=data.shape,
        dtype=data.dtype,
        chunks=(10, 30),
        zarr_format=2,
        compressors=numcodecs.Blosc(),
    )
    arr[...] = data
    check(tmp_path / "a.zarr", (slice(5, 37), slice(None)))
    assert not fast_path_calls
    assert fused_reads


def test_host_buffers_fall_back(tmp_path, fast_path_calls, fused_reads):
    data = rng.random(300).astype("float32")
    write(tmp_path / "a.zarr", data, (50,), shards=(150,))
    with zarr.config.set({"codec_pipeline.path": PIPELINE}):
        res = zarr.open_array(tmp_path / "a.zarr", mode="r")[:]
    assert isinstance(res, np.ndarray)
    np.testing.assert_array_equal(res, data)
    assert not fast_path_calls
    assert fused_reads


def test_memory_store_falls_back(fast_path_calls):
    data = rng.random(300).astype("float32")
    store = zarr.storage.MemoryStore()
    with gpu_pipeline():
        arr = zarr.create_array(store, shape=data.shape, dtype=data.dtype, chunks=(50,))
        arr[...] = cp.asarray(data)
        res = arr[:]
    np.testing.assert_array_equal(cp.asnumpy(res), data)
    assert not fast_path_calls


def test_write_and_read_roundtrip(tmp_path, fast_path_calls):
    data = rng.random((30, 30)).astype("float32")
    with gpu_pipeline():
        arr = zarr.create_array(
            tmp_path / "a.zarr",
            shape=data.shape,
            dtype=data.dtype,
            chunks=(10, 10),
            shards=(30, 30),
        )
        arr[...] = cp.asarray(data)
        res = zarr.open_array(tmp_path / "a.zarr", mode="r")[:]
    np.testing.assert_array_equal(cp.asnumpy(res), data)
    assert fast_path_calls


def test_pickle_keeps_pipeline(tmp_path):
    data = rng.random(300).astype("float32")
    write(tmp_path / "a.zarr", data, (50,), shards=(150,))
    with gpu_pipeline():
        arr = pickle.loads(pickle.dumps(zarr.open_array(tmp_path / "a.zarr", mode="r")))
        assert type(arr._async_array.codec_pipeline).__name__ == "KvikioCodecPipeline"
        np.testing.assert_array_equal(cp.asnumpy(arr[:]), data)


def test_found_by_zarr():
    """Through the entry point, without importing rapids_singlecell first."""
    import subprocess
    import sys

    code = (
        "import sys, zarr; from zarr.registry import get_pipeline_class\n"
        f"zarr.config.set({{'codec_pipeline.path': {PIPELINE!r}}})\n"
        "assert get_pipeline_class().__name__ == 'KvikioCodecPipeline'\n"
        "assert 'rapids_singlecell' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize("shards", [None, (200,)], ids=["unsharded", "sharded"])
def test_decode_in_several_calls(tmp_path, monkeypatch, fast_path_calls, shards):
    monkeypatch.setenv(
        "RSC_IO_MAX_DECODE_BYTES", "300"
    )  # < 2 chunks of 40 float32 per call
    data = rng.random(1000).astype("float32")
    write(tmp_path / "a.zarr", data, (40,), shards=shards)
    check(tmp_path / "a.zarr", slice(7, 993))
    assert fast_path_calls


def test_release_memory(tmp_path):
    data = rng.random(10_000).astype("float32")
    write(tmp_path / "a.zarr", data, (1000,), shards=(5000,))
    check(tmp_path / "a.zarr", slice(None))
    assert _pipeline._pools
    rio.release_memory()
    assert not _pipeline._pools
    check(tmp_path / "a.zarr", slice(None))  # a new codec is created on demand


def test_enable(tmp_path, fast_path_calls):
    data = rng.random(1000).astype("float32")
    write(tmp_path / "a.zarr", data, (100,), shards=(500,))
    before = zarr.config.get("codec_pipeline.path"), zarr.config.get("ndbuffer")
    with rio.enable():
        res = zarr.open_array(tmp_path / "a.zarr", mode="r")[:]
    assert isinstance(res, cp.ndarray)
    np.testing.assert_array_equal(cp.asnumpy(res), data)
    assert fast_path_calls
    assert (
        zarr.config.get("codec_pipeline.path"),
        zarr.config.get("ndbuffer"),
    ) == before


def test_dask_enable(tmp_path):
    distributed = pytest.importorskip("distributed")
    da = pytest.importorskip("dask.array")

    data = rng.random(1000).astype("float32")
    write(tmp_path / "a.zarr", data, (100,), shards=(500,))
    before = zarr.config.get("codec_pipeline.path")
    with (
        distributed.LocalCluster(
            n_workers=1, threads_per_worker=2, processes=True
        ) as cluster,
        distributed.Client(cluster) as client,
    ):
        try:
            rio.enable(client, num_threads=2)
            # arrays opened here are read on the workers, into GPU memory
            x = da.from_zarr(zarr.open_array(tmp_path / "a.zarr", mode="r"), chunks=250)
            # check on the workers (moving GPU arrays between processes needs extra dependencies)
            on_gpu = x.map_blocks(
                lambda b: np.array([isinstance(b, cp.ndarray)]), chunks=(1,)
            )
            assert on_gpu.compute().all()
            np.testing.assert_array_equal(x.map_blocks(cp.asnumpy).compute(), data)
            # while this process keeps reading into host memory
            assert isinstance(
                zarr.open_array(tmp_path / "a.zarr", mode="r")[:], np.ndarray
            )
            workers = client.run(lambda: zarr.config.get("codec_pipeline.path"))
            assert set(workers.values()) == {PIPELINE}
        finally:
            zarr.config.set({"codec_pipeline.path": before})


def test_small_reads_decoded_on_host(tmp_path, monkeypatch, fast_path_calls):
    monkeypatch.setenv("RSC_IO_MIN_BYTES", str(1 << 20))
    data = rng.random(1 << 20).astype("float32")  # 4 MiB
    write(tmp_path / "a.zarr", data, (1 << 16,), shards=(1 << 18,))
    with warnings.catch_warnings():
        # not through zarr's own pipeline, which warns about copying each chunk to the GPU
        warnings.simplefilter("error")
        check(tmp_path / "a.zarr", slice(1000, 5000))  # 16 KB
        check(tmp_path / "a.zarr", (np.arange(0, 1 << 20, 4099),), how="oindex")
    assert set(fast_path_calls) == {"_read_host"}
    fast_path_calls.clear()
    check(tmp_path / "a.zarr", slice(None))  # 4 MiB
    assert fast_path_calls == ["_read_gpu"]


@pytest.mark.parametrize("decoders", ["1", "3"])
def test_concurrent_reads(tmp_path, monkeypatch, decoders):
    from concurrent.futures import ThreadPoolExecutor

    monkeypatch.setenv("RSC_IO_DECODERS", decoders)
    rio.release_memory()
    data = rng.random(200_000).astype("float32")
    write(tmp_path / "a.zarr", data, (4096,), shards=(4096 * 8,))
    with gpu_pipeline():
        arr = zarr.open_array(tmp_path / "a.zarr", mode="r")
        sels = [slice(i * 7919, i * 7919 + 20_000) for i in range(16)]
        with ThreadPoolExecutor(8) as ex:
            results = list(ex.map(lambda s: cp.asnumpy(arr[s]), sels))
    for s, r in zip(sels, results, strict=True):
        np.testing.assert_array_equal(r, data[s])
    rio.release_memory()
