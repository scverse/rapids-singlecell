from __future__ import annotations

from typing import TYPE_CHECKING

import anndata as ad
import cupyx.scipy.sparse as cpx
import numpy as np
import pandas as pd
import pytest
import zarr
from dask.array import Array as DaskArray
from scipy import sparse
from zarr.codecs import BloscCodec

import rapids_singlecell.io as rio
from rapids_singlecell.io._check import check_store
from rapids_singlecell.io._config import gpu_io_available
from rapids_singlecell.io._read import _chunk_size

if TYPE_CHECKING:
    from pathlib import Path

gpu_io = pytest.mark.skipif(not gpu_io_available(), reason="needs KvikIO and nvCOMP")


def make_adata(fmt: str = "csr", n_obs: int = 500, n_vars: int = 300) -> ad.AnnData:
    rng = np.random.default_rng(0)
    X = sparse.random(
        n_obs,
        n_vars,
        density=0.1,
        format="csr" if fmt == "dense" else fmt,
        dtype=np.float32,
        rng=rng,
    )
    X = X.toarray() if fmt == "dense" else X
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical(rng.choice(["a", "b", "c"], n_obs))},
        index=[f"cell{i}" for i in range(n_obs)],
    )
    var = pd.DataFrame(index=[f"gene{i}" for i in range(n_vars)])
    return ad.AnnData(X=X, obs=obs, var=var)


def write_blosc(adata: ad.AnnData, path: Path) -> Path:
    """A store like most existing ones: blosc-compressed, which the GPU cannot decode."""
    ad.io.write_zarr(path, adata, compressors=[BloscCodec()])
    return path


def convert(adata: ad.AnnData, tmp_path: Path) -> Path:
    src = write_blosc(adata, tmp_path / "src.zarr")
    dst = tmp_path / "dst.zarr"
    rio.convert_zarr(src, dst, chunk_bytes=1024, chunks_per_shard=4)
    return dst


@pytest.mark.parametrize("fmt", ["csr", "csc", "dense"])
def test_convert_zarr(tmp_path, fmt):
    adata = make_adata(fmt)
    dst = convert(adata, tmp_path)

    converted = ad.io.read_zarr(dst)
    pd.testing.assert_frame_equal(converted.obs, adata.obs)
    pd.testing.assert_frame_equal(converted.var, adata.var)

    x = zarr.open_group(dst, mode="r")["X"]
    if fmt == "dense":
        np.testing.assert_array_equal(converted.X, adata.X)
        # a row (300 float32) is larger than a chunk (1024 bytes): rows in pieces of 256 values
        assert x.chunks == (1, 256)
    else:
        assert (converted.X != adata.X).nnz == 0
        assert x["indices"].dtype == np.uint16
        assert x["data"].chunks == (256,)  # 1024 bytes of float32
        assert x["data"].shards == (256 * 4,)

    store = check_store(dst)
    assert store["gpu_readable"]
    assert store["problems"] == []
    original = check_store(tmp_path / "src.zarr")
    assert not original["gpu_readable"]
    assert any("cannot be decoded on the GPU" in p for p in original["problems"])


@gpu_io
@pytest.mark.parametrize("enabled", [True, False], ids=["enabled", "not_enabled"])
def test_read_lazy_gpu(tmp_path, enabled):
    adata = make_adata()
    dst = convert(adata, tmp_path)
    if enabled:
        rio.enable()
    pipeline = zarr.config.get("codec_pipeline.path")

    lazy = rio.read_lazy(dst, chunks=100)
    assert isinstance(lazy.X, DaskArray)
    assert isinstance(lazy.X._meta, cpx.csr_matrix)
    assert lazy.X.chunks[0] == (100,) * 5
    # without `enable()`, chunks are decoded on the CPU and copied
    assert (lazy.X.compute().get() != adata.X).nnz == 0
    pd.testing.assert_frame_equal(lazy.obs, adata.obs)
    assert zarr.config.get("codec_pipeline.path") == pipeline  # no global changes


@pytest.mark.parametrize(
    "enabled",
    [pytest.param(True, marks=gpu_io), False],
    ids=["enabled", "not_enabled"],
)
def test_read_lazy_blosc(tmp_path, enabled):
    adata = make_adata()
    src = write_blosc(adata, tmp_path / "src.zarr")
    if enabled:
        rio.enable()

    lazy = rio.read_lazy(src, chunks=100)  # blosc: decoded on the CPU, then copied
    assert isinstance(lazy.X._meta, cpx.csr_matrix)
    assert (lazy.X.compute().get() != adata.X).nnz == 0


def test_read_lazy_needs_rows(tmp_path):
    make_adata("csc").write_zarr(tmp_path / "csc.zarr")
    with pytest.raises(ValueError, match=r"CSR matrix or a dense array"):
        rio.read_lazy(tmp_path / "csc.zarr")


@pytest.mark.parametrize(
    ("n_obs", "gpu_reads", "n_gpus", "expected"),
    [
        (11_441_407, True, 2, 105_000),  # ~55 chunks per GPU
        (11_441_407, True, 1, 200_000),  # capped
        (500_000, True, 2, 20_000),  # at least 20k
        (11_441_407, False, 2, 20_000),  # CPU reads
        (5_000, True, 1, 5_000),  # small data: one chunk
    ],
)
def test_chunk_size(n_obs, gpu_reads, n_gpus, expected):
    assert _chunk_size(n_obs, "auto", gpu_reads=gpu_reads, n_gpus=n_gpus) == expected
    assert _chunk_size(n_obs, 1234, gpu_reads=gpu_reads, n_gpus=n_gpus) == 1234


def test_check(tmp_path, capsys):
    dst = convert(make_adata(), tmp_path)
    report = rio.check(dst)
    assert set(report) == {"gpu_io_available", "versions", "gds", "store"}
    assert report["store"]["gpu_readable"]
    assert "laid out for fast GPU reads" in capsys.readouterr().out

    rio.check(tmp_path / "src.zarr")
    assert "convert_zarr" in capsys.readouterr().out
