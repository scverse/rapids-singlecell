from __future__ import annotations

from types import SimpleNamespace

import cupy as cp
import cupyx.scipy.sparse as cpsp
import dask
import dask.array as da
import numpy as np
import pandas as pd
import pytest
import zarr
from anndata import AnnData, io
from anndata.experimental import read_elem_lazy
from anndata.utils import asarray
from scipy.sparse import csr_matrix
from scipy.stats import mannwhitneyu

import rapids_singlecell as rsc
from rapids_singlecell.tools._rank_genes_groups import _wilcoxon, _wilcoxon_dask, _wilcoxon_records


def test_rank_budgets_respect_worker_and_shared_memory():
    gib = 2**30
    # Two workers on one node with 96 GiB each and 10 GB RMM pools.
    workers = [(96 * gib, 10**10, "node", 24 * gib, 256 * gib)] * 2
    assert _wilcoxon._rank_budgets(workers, 10**13) == (3 * 10**9, 24 * gib)
    assert _wilcoxon._rank_budgets(workers, 10**6) == (3 * 10**9, 0)
    workers[1] = (*workers[1][:4], 64 * gib)
    assert _wilcoxon._rank_budgets(workers, 10**13)[1] == 8 * gib
    workers[1] = (96 * gib, 10**10, "other", 24 * gib, 64 * gib)
    assert _wilcoxon._rank_budgets(workers, 10**13)[1] == 16 * gib
    with dask.config.set({"rapids-singlecell.wilcoxon.batch-size": "2 GB"}):
        assert _wilcoxon._rank_budgets(workers, 10**13)[0] == 2 * 10**9
    for pool in (82 * gib, None):
        assert _wilcoxon._rank_budgets([(96 * gib, pool, "node", 32 * gib, 256 * gib)], 10**13) == (24 * gib, 0)
    small_gpu = (8 * gib, None, "node")
    assert _wilcoxon._rank_budgets([(*small_gpu, 24 * gib, 128 * gib)], 10**13) == (2 * gib, 24 * gib)
    assert _wilcoxon._rank_budgets([(*small_gpu, 3 * gib, 128 * gib)], 10**13) == (2 * gib, 0)


def test_worker_memory_without_dask_limits(monkeypatch):
    import psutil
    from distributed import system

    gib = 2**30
    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (4 * gib, 8 * gib))
    monkeypatch.setattr(system, "MEMORY_LIMIT", 64 * gib)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=40 * gib))
    worker = SimpleNamespace(
        plugins={"rmm": type("RMMSetup", (), {})()},  # Older plugin without a configured pool.
        memory_manager=SimpleNamespace(memory_limit=None),
        monitor=SimpleNamespace(get_process_memory=lambda: gib),
        host="node",
    )
    assert _wilcoxon._worker_rank_memory(worker) == (8 * gib, None, "node", 16 * gib, 40 * gib)
    worker.memory_manager.memory_limit = gib
    assert _wilcoxon._worker_rank_memory(worker)[3] == 0


def test_range_packing_preserves_float_bits_and_group_codes():
    low = 0x3E000000
    patterns = np.array([[low, low + 1], [low + 33, low + 2**26 - 1]], dtype=np.uint32)
    x = cp.asarray(patterns).view(cp.float32)
    groups = np.array([63, 17], dtype=np.uint16)
    (part,) = _wilcoxon_records.pack_records(cpsp.csr_matrix(x), groups, np.array([0, 2]))
    assert part[1].nbytes == x.nbytes
    assert part[2].size == 1 and part[2].item() == low
    assert part[3] == 6
    _, keys = _wilcoxon_records._gather_keys([part], 0, 2, 64)
    unpacked = keys.get()
    np.testing.assert_array_equal((unpacked >> 6).astype(np.uint32) ^ 0x80000000, patterns.T.ravel())
    np.testing.assert_array_equal(unpacked & 63, np.tile(groups, 2))
    assert _wilcoxon_records.range_base(cp.asarray([low, low + 2**26], cp.uint32).view(cp.float32), 6) is None


@pytest.fixture(params=[None, 1])
def rank_context(request):
    return SimpleNamespace(ireference=request.param, comp_pts=True, mean_in_log_space=True, _log1p_base=None)


def test_large_group_keeps_half_integer_u(rank_context):
    # One positive in each group, all other values are implicit zeros.
    size = 100_000_001
    rg = rank_context
    rg.group_sizes = np.array([size, 3])
    rg.n_obs = size + 3
    block = (np.array([0, 2], np.int32), cp.ones(2, cp.float32).view(cp.uint32), cp.arange(2, dtype=cp.uint16), 0)
    opts = {"tie_correct": True, "use_continuity": False, "return_u_values": True}
    scores, _, _, counts = _wilcoxon_records.rank_records([block], 0, 1, rg, opts)
    # Each target zero ties two reference zeros. Its positive wins twice and ties once.
    assert scores[0, 0].item() == size - 1 + 2.5
    np.testing.assert_array_equal(counts.get(), [[1], [1]])


def test_dense_integer_rank_rows(rank_context):
    sizes = np.array([8, 3, 4, 2])
    labels = np.repeat(np.arange(5), [*sizes, 3])
    x = np.random.default_rng(18).integers(-2, 3, (len(labels), 4)).astype(np.float32)
    rg = rank_context
    rg.group_codes, rg.group_sizes, rg.pts_rest = labels, sizes, None
    reference = rg.ireference
    opts = {"tie_correct": True, "use_continuity": False, "return_u_values": True}
    stats = _wilcoxon_dask.rank_large_groups([x[:8], x[8:]], rg, opts)
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
    partial = _wilcoxon_dask.rank_large_groups([x[:8], x[8:]], compact, opts)
    for name in ("scores", "pvals"):
        np.testing.assert_array_equal(partial[name], stats[name][positions])


@pytest.mark.parametrize("reference", ["rest", "b"])
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["host", "gpu-lognorm", "gpu", "gpu-host-window", "gpu-infinite", "gpu-float64"])
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
    if mode == "gpu-host-window":
        # The second row block fits a value dictionary; the first needs raw values and group codes.
        X[23:46, 1:] = 1
        X[23, [1, 7]] = -1
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
    lazy = AnnData(read_elem_lazy(store["X"], chunks=(25 if mode == "gpu" else 23, -1)))
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
    budget = width * len(X) * X.itemsize * (2 if kind == "csr" else 1)
    gpu_batch = 16 * 2**30
    explicit_large_batch = lognorm and reference == "b"
    config = {"array.chunk-size": budget}
    if mode != "gpu":
        config["rapids-singlecell.wilcoxon.batch-size"] = gpu_batch if explicit_large_batch else budget
    if mode == "gpu" or explicit_large_batch:
        config["rapids-singlecell.wilcoxon.host-batch-size"] = 0
        monkeypatch.setattr(_wilcoxon, "_rank_budgets", lambda *args: (gpu_batch, 4096))
        reserve = 2 * gpu_batch // 3
        budgets = [reserve, gpu_batch - reserve, gpu_batch - reserve, gpu_batch] if mode == "gpu" else [gpu_batch] * 2
        project = _wilcoxon_dask._project_window
        projections = []

        def check_project(matrix, codes, edges, *, selected=None, **opts):
            assert not opts.get("on_host", False)
            assert codes.chunks == (matrix.chunks[0], (1,))
            assert (edges[0] == 0 and edges[-1] == X.shape[1]) == (mode == "gpu" and not projections)
            assert opts["concurrency"] <= (lazy.X.numblocks[0] + 1) // 2
            assert opts["budget"] == budgets[len(projections)]
            projections.append(set(range(matrix.numblocks[0]) if selected is None else selected))
            if len(projections) == 3 and reference == "b":
                opts["budget"] = 1  # Reject the remaining rows of the second window.
            return project(matrix, codes, edges, selected=selected, **opts)

        monkeypatch.setattr(_wilcoxon_dask, "_project_window", check_project)
    if mode == "gpu-host-window":
        config["rapids-singlecell.wilcoxon.host-batch-size"] = 4096
        config["rapids-singlecell.wilcoxon.batch-size"] = 512
        rank_columns = _wilcoxon_dask._rank_columns

        def check_host_blocks(blocks, *args):
            assert all(isinstance(a, np.ndarray) for ptr, words, aux, _ in blocks for a in (ptr, words, aux))
            assert any(bits and aux.size > 1 for _, _, aux, bits in blocks) == (kind == "csr")
            assert any(not bits for *_, bits in blocks)
            return rank_columns(blocks, *args)

        monkeypatch.setattr(_wilcoxon_dask, "_rank_columns", check_host_blocks)
        monkeypatch.setattr(_wilcoxon, "_rank_dask_window", lambda *a, **kw: pytest.fail("expected record rank path"))
    if mode == "gpu-float64":
        monkeypatch.setattr(
            _wilcoxon_dask, "_project_window", lambda *a, **kw: pytest.fail("expected float64 fallback")
        )
    with dask.config.set(config):
        actual = rsc.tl.rank_genes_groups(lazy, "group", **kwargs)
    if mode == "gpu":
        first, rest, *later = projections
        assert rest == set(range(0, lazy.X.numblocks[0], 3)) and first.isdisjoint(rest)
        assert first | rest == set(range(lazy.X.numblocks[0]))
        assert later == ([rest, first | rest] if reference == "b" else [rest])
    if explicit_large_batch:
        assert len(projections) == 2 and projections[0] == projections[1]
        assert len(projections[0]) == lazy.X.numblocks[0]
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
    column = asarray(eager.X[:, 1:2])
    group = labels == "a"
    control = ~group if reference == "rest" else labels == reference
    pos = np.flatnonzero(actual["names"]["a"] == eager.var_names[1]).item()
    u = mannwhitneyu(column[group].astype(np.float64), column[control].astype(np.float64)).statistic.item()
    assert actual["scores"]["a"][pos] == u


@pytest.mark.parametrize("on_host", [False, True])
def test_project_window_retries_with_all_rows(dist_client, *, on_host):
    row_sizes = (64, 65, 66, 67)
    width = 64
    budget = 32 * 1024
    matrix = csr_matrix(np.ones((sum(row_sizes), width), np.float32))
    X = rsc.get.X_to_GPU(da.from_array(matrix, chunks=(row_sizes, -1)))
    codes = da.zeros((X.shape[0], 1), dtype=np.uint16, chunks=(row_sizes, 1))
    options = {"client": dist_client, "workers": list(dist_client.nthreads())[:1], "concurrency": 2, "budget": budget}
    columns, retry_width = _wilcoxon_dask._project_window(X, codes, np.array([0, width]), on_host=on_host, **options)
    assert columns == []
    assert 1 <= retry_width < width

    columns, _ = _wilcoxon_dask._project_window(X, codes, np.array([0, retry_width]), on_host=on_host, **options)
    (parts,) = dist_client.gather(columns)
    assert sum(ptr.nbytes + words.nbytes + aux.nbytes for ptr, words, aux, _ in parts) <= budget
    # Each source block has a distinct row count and contributes a value to every gene.
    assert sorted(int(ptr[-1]) for ptr, *_ in parts) == [n * retry_width for n in row_sizes]


@pytest.mark.parametrize("sparse", [False, True])
def test_sample_gene_sizes_counts_stored_entries(dist_client, *, sparse):
    matrix = np.array([[0, 1, 2], [0, 0, 2]] * 32, np.float32)
    X = rsc.get.X_to_GPU(da.from_array(csr_matrix(matrix) if sparse else matrix, chunks=(16, -1)))
    codes = da.zeros((X.shape[0], 1), dtype=np.uint16, chunks=(16, 1))
    prefix = _wilcoxon_dask._sample_gene_sizes(dist_client, X, codes, concurrency=2)
    expected = X.numblocks[0] * ((4 if sparse else 6) * np.array([0, 8, 16]) + 4)
    np.testing.assert_array_equal(np.diff(prefix), expected)
