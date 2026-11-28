from __future__ import annotations

from contextlib import nullcontext

import cupy as cp
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData
from cupyx.scipy import sparse

import rapids_singlecell as rsc
from rapids_singlecell.io._config import gpu_io_available

A = pytest.importorskip("anndata.acc").A


def _batch_inputs(*, interleaved):
    rng = np.random.default_rng(42)
    counts = rng.poisson(rng.uniform(0.05, 1.0, size=256), size=(120, 256)).astype(np.float32)
    counts[13] = 0  # filtered by Scrublet
    counts = cp.asarray(counts)
    if interleaved:
        counts = sparse.csr_matrix(counts)
        labels = pd.Categorical(np.tile(["b", "a"], 60), categories=["b", "unused", "a"])
    else:
        labels = np.repeat(np.array([2, 1], dtype="timedelta64[ns]"), 60)
    obs = pd.DataFrame({"capture": labels}, index=[f"cell-{i}" for i in range(len(labels))])
    eager = AnnData(counts, obs=obs.copy())
    lazy_counts = da.from_array(counts, chunks=(23, -1), asarray=False)
    lazy = AnnData(obs=obs.copy(), var=eager.var.copy(), layers={"counts": lazy_counts})
    return eager, lazy


@pytest.mark.parametrize("interleaved", [False, True], ids=["dense-grouped", "csr-interleaved"])
def test_scrublet_batched(client, interleaved):
    eager, lazy = _batch_inputs(interleaved=interleaved)
    reads = []

    def read_chunk(chunk, block_id):
        reads.append(block_id[0])
        return chunk

    counts = lazy.layers["counts"]
    lazy.layers["counts"] = counts.map_blocks(read_chunk, meta=counts._meta)
    kwargs = {
        "batch": "obs.capture",
        "n_prin_comps": 5,
        "threshold": 0.04,
        "get_doublet_neighbor_parents": True,
        "verbose": False,
    }
    if interleaved:
        kwargs["rng"] = 17
    rsc.pp.scrublet(eager, **kwargs)
    rsc.pp.scrublet(lazy, use="layers.counts", **kwargs)

    assert sorted(reads) == list(range(counts.numblocks[0]))
    assert lazy.obs.loc["cell-13", ["doublet_score", "predicted_doublet"]].isna().all()
    pd.testing.assert_frame_equal(lazy.obs, eager.obs)
    np.testing.assert_equal(lazy.uns["scrublet"], eager.uns["scrublet"])


def test_scrublet_distributed_legacy_random_state(dist_client):
    eager, lazy = _batch_inputs(interleaved=True)
    eager_rng, lazy_rng = np.random.RandomState(17), np.random.RandomState(17)
    kwargs = {
        "batch": "obs.capture",
        "n_prin_comps": 5,
        "threshold": 0.04,
        "synthetic_doublet_umi_subsampling": 0.9,
        "out": True,
    }
    expected = rsc.pp.scrublet(eager, random_state=eager_rng, **kwargs)
    actual = rsc.pp.scrublet(lazy, use=A.layers["counts"], random_state=lazy_rng, **kwargs)
    np.testing.assert_allclose(actual.doublet_score, expected.doublet_score)
    pd.testing.assert_series_equal(pd.Series(actual.predicted_doublet), pd.Series(expected.predicted_doublet))
    np.testing.assert_equal(lazy_rng.get_state(), eager_rng.get_state())


@pytest.mark.parametrize("kvikio", [False, True], ids=["cpu-decoder", "kvikio"])
def test_scrublet_read_lazy(client, tmp_path, kvikio):
    if kvikio and not gpu_io_available():
        pytest.skip("needs KvikIO and nvCOMP")
    eager, _ = _batch_inputs(interleaved=True)
    path = tmp_path / "counts.zarr"
    AnnData(eager.X.get(), obs=eager.obs.copy()).write_zarr(path)
    if kvikio:
        converted = tmp_path / "converted.zarr"
        rsc.io.convert_zarr(path, converted, chunk_bytes=1024, chunks_per_shard=4)
        path = converted

    kwargs = {"batch": "obs.capture", "rng": 17, "n_prin_comps": 5, "threshold": 0.04}
    rsc.pp.scrublet(eager, **kwargs)
    with zarr.config.set({"kvikio.fetch": "kvikio"}), rsc.io.enable() if kvikio else nullcontext():
        lazy = rsc.io.read_lazy(path, chunks=23)
        rsc.pp.scrublet(lazy, **kwargs)
    pd.testing.assert_frame_equal(lazy.obs, eager.obs)
    np.testing.assert_equal(lazy.uns["scrublet"], eager.uns["scrublet"])


def test_scrublet_dask_unsupported_inputs(client):
    adata = AnnData(da.ones((4, 3), chunks=(2, 3)))
    adata.obs["capture"] = ["a", "a", "b", "b"]
    with pytest.raises(ValueError, match="batch"):
        rsc.pp.scrublet(adata)
    with pytest.raises(ValueError, match="adata_sim"):
        rsc.pp.scrublet(adata, adata_sim=adata, batch="obs.capture")
    adata.obs.loc["0", "capture"] = None
    with pytest.raises(ValueError, match="batch"):
        rsc.pp.scrublet(adata, batch="obs.capture")
