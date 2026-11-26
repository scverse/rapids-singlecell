from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData, io
from anndata.experimental import read_elem_lazy
from dask.callbacks import Callback
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu

import rapids_singlecell as rsc
from rapids_singlecell.tools._rank_genes_groups import _wilcoxon


@pytest.mark.parametrize("reference", ["rest", "b"])
@pytest.mark.parametrize("mode", ["counts", "lognorm", "fallback_pts"])
def test_dask_wilcoxon_csr_zarr(request, monkeypatch, tmp_path, mode, reference):
    streamed = mode != "fallback_pts"
    request.getfixturevalue("dist_client" if streamed else "client")
    rng = np.random.default_rng(42)
    levels = [0, 0, 1, 32, 67] if mode == "counts" else [-0.125, 0, 0, 0.5, 1, 2]
    X = rng.choice(np.array(levels, dtype=np.float32), size=(117, 13))
    X[:, 0] = 0
    labels = pd.Categorical(["a", "b", "c"] * 38 + [None] * 3)
    X[:, 1] = 1
    X[labels == "a", 1] = 2 if mode == "counts" else np.nextafter(np.float32(1), np.float32(2))
    adata = AnnData(csr_matrix(X))
    adata.obs["group"] = labels
    kwargs = {
        "method": "wilcoxon",
        "use": "X",
        "groups": ["c", "a"],
        "reference": reference,
        "tie_correct": True,
        "pts": not streamed,
        "use_continuity": True,
        "return_u_values": reference == "b",
        "mean_in_log_space": reference == "rest",
        "out": True,
    }
    expected = rsc.tl.rank_genes_groups(adata, "group", multi_gpu=False, **kwargs)
    store = zarr.open_group(tmp_path / "data.zarr", mode="w")
    io.write_elem(store, "X", adata.X)
    adata.X = read_elem_lazy(store["X"], chunks=(23, -1))
    if streamed:
        adata.X = rsc.get.X_to_GPU(adata.X)
        monkeypatch.setattr(_wilcoxon, "_rank_dask_batch", lambda *a: pytest.fail("unexpected fallback"))

    def check_panel(key, value, dsk, state, worker_id):
        if isinstance(value, type(adata.X._meta)) and value.ndim == 2 and value.shape[0] == len(X):
            assert value.shape[1] <= 3

    with dask.config.set({"array.chunk-size": 3 * len(X) * X.itemsize}), Callback(posttask=check_panel):
        actual = rsc.tl.rank_genes_groups(adata, "group", **kwargs)
    np.testing.assert_array_equal(actual["names"], expected["names"])
    for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        for group in expected[field].dtype.names:
            np.testing.assert_allclose(actual[field][group], expected[field][group], rtol=1e-6, atol=1e-12)
    for field in ("pts", "pts_rest"):
        if field in expected:
            pd.testing.assert_frame_equal(actual[field], expected[field])
    if reference == "b":
        gene = np.flatnonzero(actual["names"]["a"] == adata.var_names[1]).item()
        assert actual["scores"]["a"][gene] == mannwhitneyu(X[labels == "a", 1], X[labels == "b", 1]).statistic
