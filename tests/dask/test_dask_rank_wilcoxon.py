from __future__ import annotations

from types import SimpleNamespace

import cupy as cp
import cupyx.scipy.sparse as cpsp
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
from rapids_singlecell.tools._rank_genes_groups._wilcoxon_dask import (
    _project_window,
    _sample_gene_sizes,
    rank_large_groups,
)
from rapids_singlecell.tools._rank_genes_groups._wilcoxon_records import (
    _gather_keys,
    pack_records,
    range_base,
    rank_records,
)


def test_range_packing_preserves_float_bits_and_group_codes():
    low = 0x3E000000
    patterns = np.array([[low, low + 1], [low + 33, low + 2**26 - 1]], dtype=np.uint32)
    x = cp.asarray(patterns).view(cp.float32)
    groups = np.array([63, 17], dtype=np.uint16)
    (part,) = pack_records(cpsp.csr_matrix(x), groups, np.array([0, 2]))
    assert part[1].nbytes == x.nbytes
    assert part[2].size == 1 and part[2].item() == low
    assert part[3] == 6
    _, keys = _gather_keys([part], 0, 2, 64)
    unpacked = keys.get()
    np.testing.assert_array_equal((unpacked >> 6).astype(np.uint32) ^ 0x80000000, patterns.T.ravel())
    np.testing.assert_array_equal(unpacked & 63, np.tile(groups, 2))
    assert range_base(cp.asarray([low, low + 2**26], cp.uint32).view(cp.float32), 6) is None


@pytest.mark.parametrize("reference", [None, 1])
def test_large_group_keeps_half_integer_u(reference):
    # One positive in each group, all other values are implicit zeros.
    size = 100_000_001
    rg = SimpleNamespace(
        group_sizes=np.array([size, 3]),
        n_obs=size + 3,
        ireference=reference,
        comp_pts=True,
        mean_in_log_space=True,
        _log1p_base=None,
    )
    block = (np.array([0, 2], np.int32), cp.ones(2, cp.float32).view(cp.uint32), cp.arange(2, dtype=cp.uint16), 0)
    opts = {"tie_correct": True, "use_continuity": False, "return_u_values": True}
    _, scores, _, _, counts = rank_records([block], 0, 1, rg, opts)
    # Each target zero ties two reference zeros. Its positive wins twice and ties once.
    assert scores[0, 0].item() == size - 1 + 2.5
    np.testing.assert_array_equal(counts.get(), [[1], [1]])


@pytest.mark.parametrize("reference", [None, 1])
def test_dense_integer_rank_rows(reference):
    sizes = np.array([8, 3, 4, 2])
    labels = np.repeat(np.arange(5), [*sizes, 3])
    x = np.random.default_rng(18).integers(-2, 3, (len(labels), 4)).astype(np.float32)
    rg = SimpleNamespace(
        group_codes=labels,
        group_sizes=sizes,
        ireference=reference,
        comp_pts=True,
        mean_in_log_space=True,
        _log1p_base=None,
        pts_rest=None,
    )
    opts = {"tie_correct": True, "use_continuity": False, "return_u_values": True}
    stats = rank_large_groups([x[:8], x[8:]], rg, opts)
    ref = labels != 0 if reference is None else labels == reference
    expected = mannwhitneyu(
        x[labels == 0].astype(np.float64), x[ref].astype(np.float64), method="asymptotic", use_continuity=False
    )
    np.testing.assert_array_equal(stats["scores"][0], expected.statistic)
    np.testing.assert_allclose(stats["pvals"][0], expected.pvalue)
    np.testing.assert_array_equal(stats["pts"][0], (x[labels == 0] != 0).mean(axis=0))
    compact, positions = _wilcoxon._compact_rank_groups(rg, np.array([0, 2]))
    np.testing.assert_array_equal(positions, [0, 2] if reference is None else [0, 1])
    np.testing.assert_array_equal(compact.group_codes[labels >= 3], len(compact.group_sizes))
    partial = rank_large_groups([x[:8], x[8:]], compact, opts)
    for name in ("scores", "pvals"):
        np.testing.assert_array_equal(partial[name], stats[name][positions])


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
        monkeypatch.setattr(_wilcoxon, "_HALF_INTEGER_LIMIT", 10)
        monkeypatch.setattr(_wilcoxon, "_MAX_RECORD_GROUPS", 1)
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


def test_project_window_retries_with_all_rows(dist_client):
    row_sizes = [64, 65, 66, 67]
    width = 64
    budget = 32 * 1024
    rows = [dask.delayed(rsc.get.X_to_GPU)(csr_matrix(np.ones((n, width), np.float32))) for n in row_sizes]
    codes = [np.zeros(n, np.uint16) for n in row_sizes]
    options = {"client": dist_client, "workers": list(dist_client.nthreads())[:1], "concurrency": 2, "budget": budget}
    columns, retry_width = _project_window(rows, codes, np.array([0, width]), **options)
    assert columns == []
    assert 1 <= retry_width < width

    columns, _ = _project_window(rows, codes, np.array([0, retry_width]), **options)
    (column,) = columns
    parts = dist_client.gather(column)
    assert sum(ptr.nbytes + words.nbytes + aux.nbytes for ptr, words, aux, _ in parts) <= budget
    # Each source block has a distinct row count and contributes a value to every gene.
    assert sorted(int(ptr[-1]) for ptr, *_ in parts) == [n * retry_width for n in row_sizes]


def test_sample_gene_sizes_counts_stored_entries(dist_client):
    # Every block has an empty gene and a gene expressed in half the cells.
    X = csr_matrix(np.array([[0, 1, 2], [0, 0, 2]] * 8, np.float32))
    rows = [dask.delayed(rsc.get.X_to_GPU)(X) for _ in range(4)]
    codes = [np.zeros(X.shape[0], np.uint16)] * len(rows)
    prefix = _sample_gene_sizes(dist_client, rows, codes, concurrency=2)
    expected = 4 * len(rows) * (np.array([0, 8, 16]) + 1)
    np.testing.assert_array_equal(np.diff(prefix), expected)
