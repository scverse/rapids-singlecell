from __future__ import annotations

from functools import partial
from types import SimpleNamespace
from unittest.mock import Mock

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
from scipy.sparse import csr_matrix, vstack
from scipy.stats import mannwhitneyu

import rapids_singlecell as rsc
from rapids_singlecell.tools._rank_genes_groups import _wilcoxon, _wilcoxon_dask, _wilcoxon_records

GIB = 2**30
U_OPTIONS = {"tie_correct": True, "return_u_values": True, "use_continuity": False}


def test_rank_budgets_respect_worker_and_shared_memory():
    # Two workers on one node with 96 GiB each and 10 GB RMM pools.
    workers = [(96 * GIB, 10**10, "node", 24 * GIB, 256 * GIB)] * 2
    assert _wilcoxon._rank_budgets(workers, 10**13) == (3 * 10**9, 24 * GIB)
    assert _wilcoxon._rank_budgets(workers, 10**6) == (3 * 10**9, 0)
    for host, host_gib in [("node", 8), ("other", 16)]:
        workers[1] = (96 * GIB, 10**10, host, 24 * GIB, 64 * GIB)
        assert _wilcoxon._rank_budgets(workers, 10**13)[1] == host_gib * GIB
    with dask.config.set({"rapids-singlecell.wilcoxon.batch-size": "2 GB"}):
        assert _wilcoxon._rank_budgets(workers, 10**13)[0] == 2 * 10**9
    for pool in (82 * GIB, None):
        assert _wilcoxon._rank_budgets([(96 * GIB, pool, "node", 32 * GIB, 256 * GIB)], 10**13) == (24 * GIB, 0)
    small_gpu = (8 * GIB, None, "node")
    assert _wilcoxon._rank_budgets([(*small_gpu, 24 * GIB, 128 * GIB)], 10**13) == (2 * GIB, 24 * GIB)
    assert _wilcoxon._rank_budgets([(*small_gpu, 3 * GIB, 128 * GIB)], 10**13) == (2 * GIB, 0)


def test_worker_memory_without_dask_limits(monkeypatch):
    import psutil
    from distributed import system

    monkeypatch.setattr(cp.cuda.runtime, "memGetInfo", lambda: (4 * GIB, 8 * GIB))
    monkeypatch.setattr(system, "MEMORY_LIMIT", 64 * GIB)
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=40 * GIB))
    worker = Mock(host="node", plugins={"rmm": type("RMMSetup", (), {})()})  # Older plugin without a pool.
    worker.monitor.get_process_memory.return_value = GIB
    for limit, headroom in [(None, 16 * GIB), (GIB, 0)]:
        worker.memory_manager.memory_limit = limit
        assert _wilcoxon._worker_rank_memory(worker) == (8 * GIB, None, "node", headroom, 40 * GIB)


def test_range_packing_preserves_float_bits_and_group_codes():
    low = 0x3E000000
    patterns = np.array([[low, low + 1], [low + 33, low + 2**26 - 1]], dtype=np.uint32)
    x = cpsp.csr_matrix(cp.asarray(patterns).view(cp.float32))
    groups = np.array([63, 17], dtype=np.uint16)
    parts = _wilcoxon_dask._pack_row(x, groups[:, None], edges=[0, 1, 2], on_host=True, short_from=None).item()
    table = parts[0][2]
    assert isinstance(table, np.ndarray) and table is parts[1][2]
    assert table.shape == (1,) and table.item() == low
    assert all(bits == 6 and words.nbytes == patterns[:, 0].nbytes for _, words, _, bits in parts)
    unpacked = np.concatenate([_wilcoxon_records._gather_keys([part], 0, 1, 64)[1].get() for part in parts])
    np.testing.assert_array_equal((unpacked >> 6).astype(np.uint32) ^ 0x80000000, patterns.T.ravel())
    np.testing.assert_array_equal(unpacked & 63, np.tile(groups, 2))
    assert _wilcoxon_records.range_base(cp.asarray([low, low + 2**26], cp.uint32).view(cp.float32), 6) is None


@pytest.fixture(params=[None, 1])
def rg(request):
    return SimpleNamespace(ireference=request.param, comp_pts=True, mean_in_log_space=True, _log1p_base=None)


def test_large_group_keeps_half_integer_u(rg):
    # One positive in each group, all other values are implicit zeros.
    rg.group_sizes = np.array([100_000_001, 3])
    rg.n_obs = int(rg.group_sizes.sum())
    block = (np.array([0, 2], np.int32), cp.ones(2, cp.float32).view(cp.uint32), cp.arange(2, dtype=cp.uint16), 0)
    scores, _, _, counts = _wilcoxon_records.rank_records([block], 0, 1, rg, U_OPTIONS)
    # Each target zero ties two reference zeros. Its positive wins twice and ties once.
    assert scores[0, 0].item() == rg.group_sizes[0] - 1 + 2.5
    np.testing.assert_array_equal(counts.get(), [[1], [1]])


def test_dense_integer_rank_rows(rg):
    sizes = np.array([8, 3, 4, 2])
    labels = np.repeat(np.arange(5), [*sizes, 3])
    x = np.random.default_rng(18).integers(-2, 3, (len(labels), 4)).astype(np.float64)
    rg.group_codes, rg.group_sizes, rg.pts_rest = labels, sizes, None
    rank = partial(_wilcoxon_dask.rank_large_groups, np.split(x.astype(np.float32), [8]), options=U_OPTIONS)
    stats = rank(rg)
    ref = labels != 0 if rg.ireference is None else labels == rg.ireference
    expected = mannwhitneyu(x[labels == 0], x[ref], method="asymptotic", use_continuity=False)
    np.testing.assert_array_equal(stats["scores"][0], expected.statistic)
    np.testing.assert_allclose(stats["pvals"][0], expected.pvalue)
    np.testing.assert_array_equal(stats["pts"][0], (x[labels == 0] != 0).mean(axis=0))
    compact, positions = _wilcoxon._compact_rank_groups(rg, np.array([0, 2]))
    np.testing.assert_array_equal(positions, [0, 2] if rg.ireference is None else [0, 1])
    np.testing.assert_array_equal(compact.group_codes[labels >= 3], len(compact.group_sizes))
    subset = rank(compact)
    for name in ("scores", "pvals"):
        np.testing.assert_array_equal(subset[name], stats[name][positions])


@pytest.mark.parametrize("reference", ["rest", "b"])
@pytest.mark.parametrize("kind", ["dense", "csr"])
@pytest.mark.parametrize("mode", ["host", "gpu-lognorm", "gpu", "gpu-host-window", "gpu-infinite", "gpu-float64"])
def test_dask_wilcoxon_zarr(request, monkeypatch, tmp_path, *, kind, mode, reference):
    lognorm = mode == "gpu-lognorm"
    request.getfixturevalue("client" if mode == "host" else "dist_client")
    dtype = np.float64 if mode == "gpu-float64" else np.float32
    X = np.random.default_rng(42).choice([0, 0, 1, 2, 32, 67], size=(117, 13)).astype(dtype)
    labels = pd.Categorical(["a", "b", "c"] * 38 + [None] * 3)
    X[:, :2] = [0, 1]
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
    data = csr_matrix(X) if kind == "csr" else X
    eager = AnnData(data.copy())
    if kind == "csr" and not lognorm:
        # Preserve unsorted duplicate entries and an explicit signed zero in the physical CSR store.
        row = csr_matrix((np.array([1, -0.0, -1], X.dtype), [2, 0, 2], [0, 3]), shape=(1, X.shape[1]))
        data = vstack([row, data[1:]], format="csr")
    store = zarr.open_group(tmp_path / "data.zarr", mode="w")
    io.write_elem(store, "X", data)
    lazy = AnnData(read_elem_lazy(store["X"], chunks=(25 if mode == "gpu" else 23, -1)))
    for adata in (eager, lazy):
        adata.obs["group"] = labels
        if mode != "host":
            adata.X = rsc.get.X_to_GPU(adata.X)
        if lognorm:
            rsc.pp.normalize_total(adata, target_sum=10_000)
            rsc.pp.log1p(adata)
    contrast = {"groups": ["c", "a"], "reference": reference, "mean_in_log_space": reference == "rest"}
    kwargs = U_OPTIONS | contrast | {"method": "wilcoxon", "use": "X", "pts": True, "out": True, "use_continuity": True}
    expected = rsc.tl.rank_genes_groups(eager, "group", multi_gpu=False, **kwargs)
    width = 3 if reference == "rest" else 100
    if mode == "host":
        monkeypatch.setattr(_wilcoxon, "_HALF_INTEGER_LIMIT", 10)
        monkeypatch.setattr(_wilcoxon, "_MAX_RECORD_GROUPS", 1)
        rank_batch = Mock(wraps=_wilcoxon._rank_dask_batch)
        monkeypatch.setattr(_wilcoxon, "_rank_dask_batch", rank_batch)
    budget = width * len(X) * X.itemsize * (2 if kind == "csr" else 1)
    gpu_batch = 16 * GIB
    host_reuse = lognorm and reference == "rest"
    explicit_large_batch = lognorm and reference == "b"
    config = {"array.chunk-size": budget}
    if mode != "gpu" and not host_reuse:
        config["rapids-singlecell.wilcoxon.batch-size"] = gpu_batch if explicit_large_batch else budget
    projections = iter(())
    if mode == "gpu" or lognorm:
        if not host_reuse:
            config["rapids-singlecell.wilcoxon.host-batch-size"] = 0
        budgets = (512, 12 * GIB) if host_reuse else (gpu_batch, 4096)
        monkeypatch.setattr(_wilcoxon, "_rank_budgets", lambda *args: budgets)
        rows = set(range(lazy.X.numblocks[0]))
        if host_reuse:
            tail = len(rows) - max(1, len(rows) // 8)
            split = (X.shape[1] + 3) // 4
            plans = [(rows, budgets[1] // 2), (set(range(tail, len(rows))), budgets[1] // 2)]
        elif explicit_large_batch:
            plans = [(rows, gpu_batch)] * 2
        else:
            reserve = 2 * gpu_batch // 3
            reread = set(range(0, len(rows), 3))
            plans = [(rows - reread, reserve)] + [(reread, gpu_batch - reserve)] * 2
            if reference == "b":
                plans.append((rows, gpu_batch))
        projections = iter(enumerate(plans))
        project = _wilcoxon_dask._project_window

        def check_project(matrix, codes, edges, *, selected=None, **opts):
            assert opts.get("on_host", False) == host_reuse
            assert codes.chunks == (matrix.chunks[0], (1,))
            i, expected_call = next(projections)
            assert (rows if selected is None else set(selected), opts["budget"]) == expected_call
            assert opts["concurrency"] <= (len(rows) + 1) // 2
            assert (edges[0] == 0 and edges[-1] == X.shape[1]) == (i == 0 and not explicit_large_batch)
            if host_reuse:
                assert edges[len(edges) // 2 if i == 0 else 0] == split
                assert opts.get("short_from") == (tail if i == 0 else None)
            if i == 2 and reference == "b":
                opts["budget"] = 1  # Reject the remaining rows of the second window.
            return project(matrix, codes, edges, selected=selected, **opts)

        monkeypatch.setattr(_wilcoxon_dask, "_project_window", check_project)
    if mode == "gpu-host-window":
        config |= {"rapids-singlecell.wilcoxon.host-batch-size": 4096, "rapids-singlecell.wilcoxon.batch-size": 512}
        rank_columns = _wilcoxon_dask._rank_columns

        def check_host_blocks(blocks, *args):
            assert all(isinstance(a, np.ndarray) for ptr, words, aux, _ in blocks for a in (ptr, words, aux))
            assert any(bits and aux.size > 1 for _, _, aux, bits in blocks) == (kind == "csr")
            assert any(not bits for *_, bits in blocks)
            return rank_columns(blocks, *args)

        monkeypatch.setattr(_wilcoxon_dask, "_rank_columns", check_host_blocks)
        monkeypatch.setattr(_wilcoxon, "_rank_dask_window", lambda *a, **kw: pytest.fail("expected record rank path"))
    if mode == "gpu-float64":
        monkeypatch.setattr(_wilcoxon_dask, "_project_window", lambda *a, **kw: pytest.fail("float64 used packing"))
    with dask.config.set(config):
        actual = rsc.tl.rank_genes_groups(lazy, "group", **kwargs)
    if mode == "host":
        assert rank_batch.called
        for call in rank_batch.call_args_list:
            rows = call.args[0]
            assert sum(row.shape[0] for row in rows) == len(X)
            assert all(row.shape[1] <= min(width, (X.shape[1] + 1) // 2) for row in rows)
            assert kind == "csr" or all(row.base is None for row in rows)
    assert next(projections, None) is None
    np.testing.assert_array_equal(actual["names"], expected["names"])
    for field in ("scores", "pvals", "pvals_adj", "logfoldchanges"):
        pd.testing.assert_frame_equal(pd.DataFrame(actual[field]), pd.DataFrame(expected[field]), rtol=1e-6, atol=1e-12)
    assert actual.keys() == expected.keys()
    for field in expected.keys() & {"pts", "pts_rest"}:
        pd.testing.assert_frame_equal(actual[field], expected[field])
        for group in actual[field]:
            mask = labels == group if field == "pts" else labels != group
            np.testing.assert_array_equal(actual[field][group], (X[mask] != 0).mean(axis=0))
    column = asarray(eager.X[:, 1:2]).astype(np.float64)
    group = labels == "a"
    control = ~group if reference == "rest" else labels == reference
    pos = np.flatnonzero(actual["names"]["a"] == eager.var_names[1]).item()
    assert actual["scores"]["a"][pos] == mannwhitneyu(column[group], column[control]).statistic.item()


@pytest.mark.parametrize("on_host", [False, True])
@pytest.mark.parametrize("sparse", [False, True])
def test_project_window_retries_with_all_rows(dist_client, *, on_host, sparse):
    row_sizes = (64, 65, 66, 67)
    width = 64
    budget = 32 * 1024
    matrix = np.ones((sum(row_sizes), width), np.float32)
    matrix[::2, -2], matrix[:, -1] = 0, 0
    X = rsc.get.X_to_GPU(da.from_array(csr_matrix(matrix) if sparse else matrix, chunks=(row_sizes, -1)))
    codes = da.zeros((X.shape[0], 1), dtype=np.uint16, chunks=(row_sizes, 1))
    prefix = _wilcoxon_dask._sample_gene_sizes(dist_client, X, codes, concurrency=2)
    entry_bytes = 4 if sparse else 6
    expected = 4 * X.numblocks[0] + entry_bytes * np.array([len(matrix), len(matrix) // 2, 0])
    np.testing.assert_array_equal(np.diff(prefix)[-3:], expected)

    options = {"client": dist_client, "workers": list(dist_client.nthreads())[:1], "concurrency": 2, "budget": budget}
    project = partial(_wilcoxon_dask._project_window, X, codes, on_host=on_host, **options)
    columns, retry_width = project(np.array([0, width]))
    assert columns == [] and 1 <= retry_width < width - 2
    columns, _ = project(np.array([0, retry_width]))
    (parts,) = dist_client.gather(columns)
    assert sum(ptr.nbytes + words.nbytes + aux.nbytes for ptr, words, aux, _ in parts) <= budget
    assert sorted(int(ptr[-1]) for ptr, *_ in parts) == [n * retry_width for n in row_sizes]
