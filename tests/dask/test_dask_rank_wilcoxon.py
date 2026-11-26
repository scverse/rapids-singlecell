from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData, io
from anndata.experimental import read_elem_lazy
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu

import rapids_singlecell as rsc
from rapids_singlecell.tools._rank_genes_groups import _wilcoxon


@pytest.mark.parametrize("reference", ["rest", "b"])
@pytest.mark.parametrize(
    "kind,mode",
    [
        ("dense", "host"),
        ("csr", "host"),
        ("dense", "gpu-lognorm"),
        ("csr", "gpu-lognorm"),
        ("csr", "gpu"),
        ("csr", "gpu-infinite"),
        ("csr", "gpu-float64"),
    ],
)
def test_dask_wilcoxon_zarr(request, monkeypatch, tmp_path, *, kind, mode, reference):
    lognorm = mode == "gpu-lognorm"
    request.getfixturevalue("client" if mode == "host" else "dist_client")
    rng = np.random.default_rng(42)
    X = rng.choice(np.array([0, 0, 1, 2, 32, 67], dtype=np.float32), size=(117, 13))
    labels = pd.Categorical(["a", "b", "c"] * 38 + [None] * 3)
    X[:, 0] = 0
    X[:, 1] = 1
    X[labels == "a", 1] = 2 if lognorm else np.nextafter(np.float32(1), np.float32(2))
    X[0, :] = 0
    if not lognorm:
        X[1, 0], X[2, 2] = -0.0, -0.125
    if mode == "gpu-infinite":
        X[4, 3] = np.inf
    if mode == "gpu-float64":
        X = X.astype(np.float64)
    data = csr_matrix(X) if kind == "csr" else X
    if kind == "csr" and not lognorm:
        # Preserve unsorted duplicate entries and an explicit signed zero in the physical CSR store.
        data = csr_matrix(
            (
                np.r_[[1, -0.0, -1], data.data].astype(X.dtype),
                np.r_[[2, 0, 2], data.indices],
                np.r_[0, data.indptr[1:] + 3],
            ),
            shape=X.shape,
        )
    store = zarr.open_group(tmp_path / "data.zarr", mode="w")
    io.write_elem(store, "X", data)
    lazy = AnnData(read_elem_lazy(store["X"], chunks=(23, -1)))
    eager = AnnData(csr_matrix(X) if kind == "csr" else X.copy())
    for adata in (eager, lazy):
        adata.obs["group"] = labels
        if mode != "host":
            adata.X = rsc.get.X_to_GPU(adata.X)
        if lognorm:
            rsc.pp.normalize_total(adata, target_sum=10_000)
            rsc.pp.log1p(adata)
    kwargs = {
        "method": "wilcoxon",
        "use": "X",
        "groups": ["c", "a"],
        "reference": reference,
        "tie_correct": True,
        "pts": True,
        "use_continuity": True,
        "return_u_values": True,
        "mean_in_log_space": reference == "rest",
        "out": True,
    }
    expected = rsc.tl.rank_genes_groups(eager, "group", multi_gpu=False, **kwargs)
    width = 3 if reference == "rest" else 100
    if mode == "host":
        rank = _wilcoxon._rank_dask_batch

        def check_rank(rows, *args):
            assert all(row.shape[1] <= min(width, (X.shape[1] + 1) // 2) for row in rows)
            assert sum(row.shape[0] for row in rows) == len(X)
            if kind == "dense":
                assert all(row.base is None for row in rows)
            return rank(rows, *args)

        monkeypatch.setattr(_wilcoxon, "_rank_dask_batch", check_rank)
    entry_bytes = X.itemsize * (2 if kind == "csr" else 1)
    budget = width * len(X) * entry_bytes
    with dask.config.set({"array.chunk-size": budget, "rapids-singlecell.wilcoxon.batch-size": budget}):
        actual = rsc.tl.rank_genes_groups(lazy, "group", **kwargs)
    np.testing.assert_array_equal(actual["names"], expected["names"])
    for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        for group in expected[field].dtype.names:
            np.testing.assert_allclose(actual[field][group], expected[field][group], rtol=1e-6, atol=1e-12)
    for field in ("pts", "pts_rest"):
        assert (field in actual) == (field in expected)
        if field in expected:
            pd.testing.assert_frame_equal(actual[field], expected[field])
    for group in actual["pts"]:
        mask = labels == group
        np.testing.assert_array_equal(actual["pts"][group], np.count_nonzero(X[mask], axis=0) / mask.sum())
        if reference == "rest":
            np.testing.assert_array_equal(actual["pts_rest"][group], np.count_nonzero(X[~mask], axis=0) / (~mask).sum())
    column = eager.X[:, 1:2]
    if hasattr(column, "get"):
        column = column.get()
    if hasattr(column, "toarray"):
        column = column.toarray()
    group = labels == "a"
    control = ~group if reference == "rest" else labels == reference
    pos = np.flatnonzero(actual["names"]["a"] == eager.var_names[1]).item()
    u = mannwhitneyu(column[group].astype(np.float64), column[control].astype(np.float64)).statistic.item()
    assert actual["scores"]["a"][pos] == u
