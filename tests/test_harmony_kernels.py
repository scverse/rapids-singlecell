"""Unit tests for harmony CUDA kernels against CuPy references."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell._cuda import (
    _harmony_clustering_cuda as _cl,
)
from rapids_singlecell._cuda import (
    _harmony_correction_batched_cuda as _cbat,
)
from rapids_singlecell._cuda import (
    _harmony_correction_cuda as _corr,
)
from rapids_singlecell.preprocessing._harmony import _segments
from rapids_singlecell.preprocessing._harmony._kmeans import _fused_distances
from rapids_singlecell.preprocessing._harmony._multi_gpu import _comm

pytestmark = pytest.mark.skipif(
    _cl is None or _corr is None,
    reason="Harmony CUDA modules not available",
)

DTYPES = [np.float32, np.float64]


# ---------- l2_row_normalize ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows,n_cols", [(100, 50), (1, 20), (500, 3)])
@pytest.mark.parametrize("in_place", [False, True])
def test_l2_row_normalize(dtype, n_rows, n_cols, in_place):
    rng = cp.random.default_rng(42)
    src = rng.standard_normal((n_rows, n_cols), dtype=dtype)
    expected = src.copy()
    dst = src if in_place else cp.empty_like(src)

    _cl.l2_row_normalize(src, dst=dst)
    cp.cuda.Device().synchronize()

    # Reference: L2 row normalize
    norms = cp.linalg.norm(expected, axis=1, keepdims=True)
    norms = cp.maximum(norms, 1e-12)
    expected /= norms

    atol = 1e-6 if dtype == np.float32 else 1e-12
    cp.testing.assert_allclose(dst, expected, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("dtype", DTYPES)
def test_l2_row_normalize_zero_row(dtype):
    """Zero rows should not produce NaN (clamped to 1e-12)."""
    src = cp.zeros((3, 10), dtype=dtype)
    src[1, :] = 1.0  # only middle row is non-zero
    dst = cp.empty_like(src)

    _cl.l2_row_normalize(src, dst=dst)
    cp.cuda.Device().synchronize()

    assert not cp.any(cp.isnan(dst))
    # Zero rows should stay zero (0 / clamp(0, 1e-12) = 0)
    cp.testing.assert_array_equal(dst[0], cp.zeros(10, dtype=dtype))
    cp.testing.assert_array_equal(dst[2], cp.zeros(10, dtype=dtype))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_rows,n_categories", [(2305, 13), (2049, 9000)])
def test_scatter_add(dtype, n_rows, n_categories):
    n_cols = 17
    rng = np.random.default_rng(734)
    values_host = rng.normal(size=(n_rows, n_cols)).astype(dtype)
    categories_host = np.arange(n_rows, dtype=np.int32) % (n_categories - 1)
    categories_host[:2100] = 0
    expected = np.zeros((n_categories, n_cols), dtype=np.float64)
    np.add.at(expected, categories_host, values_host.astype(np.float64))
    values, categories = cp.asarray(values_host), cp.asarray(categories_host)
    workspace_bytes = _cl.get_scatter_temp_bytes(
        n_rows=n_rows,
        n_cols=n_cols,
        n_categories=n_categories,
        itemsize=np.dtype(dtype).itemsize,
    )
    workspace = cp.empty(workspace_bytes, dtype=cp.uint8)
    outputs = []
    for _ in range(2):
        out = cp.zeros((n_categories, n_cols), dtype=dtype)
        scatter_add = partial(_cl.scatter_add, values, categories=categories, out=out, workspace=workspace)
        scatter_add()
        outputs.append(out.copy())
    with pytest.raises(ValueError, match="scatter workspace is too small"):
        scatter_add(workspace=workspace[:-1])
    assert outputs[0].get().tobytes() == outputs[1].get().tobytes()
    atol = 2e-4 if dtype == np.float32 else 1e-10
    cp.testing.assert_allclose(outputs[0], cp.asarray(expected), rtol=1e-5, atol=atol)
    assert not cp.any(outputs[0][-1])


@pytest.mark.parametrize("dtype", DTYPES)
def test_select_kmeans_center_totals_workspace(dtype):
    n_rows = _cl.KMEANS_WEIGHT_TILE_ROWS + 1
    X = cp.arange(n_rows * 2, dtype=dtype).reshape(n_rows, 2)
    weights = cp.zeros(n_rows, dtype=dtype)
    weights[-1] = 1
    centers = cp.empty((1, 2), dtype=dtype)
    totals = cp.empty(2, dtype=cp.float64)
    n_draws = cp.zeros(1, dtype=cp.int32)
    kwargs = {
        "weights": weights,
        "uniforms": cp.asarray([0.5], dtype=cp.float64),
        "centers": centers,
        "n_draws": n_draws,
        "cluster": 0,
    }
    with pytest.raises(ValueError, match="k-means totals workspace is too small"):
        _cl.select_kmeans_center(X, totals=totals[:-1], **kwargs)
    _cl.select_kmeans_center(X, totals=totals, **kwargs)
    cp.testing.assert_array_equal(centers[0], X[-1])
    assert int(n_draws[0]) == 1


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_clusters,n_cols", [(1, 3), (37, 50), (128, 64)])
def test_kmeans_assign(dtype, n_clusters, n_cols):
    rng = cp.random.default_rng(3)
    X = rng.standard_normal((1000, n_cols)).astype(dtype)
    if not _fused_distances(X, n_clusters):
        pytest.skip("centers exceed this GPU's shared memory (k-means falls back)")
    centers = rng.standard_normal((n_clusters, n_cols)).astype(dtype)
    labels = cp.empty(1000, dtype=cp.int32)
    minimum = cp.empty(1000, dtype=dtype)
    _cl.kmeans_assign(X, centers=centers, labels=labels, minimum=minimum)
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    cp.testing.assert_array_equal(labels, distances.argmin(1))
    cp.testing.assert_allclose(minimum, distances.min(1), rtol=1e-5)


def test_failed_launch_setup_leaves_no_stale_error():
    # A shape beyond the GPU's shared memory raises, and the next kernel launch
    # (checked with cudaGetLastError) does not inherit that error.
    X = cp.ones((10, 600), cp.float64)
    with pytest.raises(RuntimeError, match="shared memory"):
        _cl.kmeans_assign(
            X,
            centers=cp.ones((128, 600), cp.float64),
            labels=cp.empty(10, cp.int32),
            minimum=cp.empty(10, cp.float64),
        )
    out = cp.empty_like(X)
    _cl.l2_row_normalize(X, dst=out)
    cp.testing.assert_allclose(out, X / cp.sqrt(600.0))


@pytest.mark.parametrize("dtype", DTYPES)
def test_kmeans_closest(dtype):
    rng = cp.random.default_rng(4)
    X = rng.standard_normal((777, 21)).astype(dtype)
    center = rng.standard_normal(21).astype(dtype)
    closest = rng.random(777).astype(dtype) * 40
    expected = cp.minimum(closest, ((X - center) ** 2).sum(1))
    _cl.kmeans_closest(X, center=center, closest=closest)
    cp.testing.assert_allclose(closest, expected, rtol=1e-5)


# ---------- update blocks ----------


def test_draw_blocks_split_invariant():
    # A cell's block depends only on its global position: drawing two shards
    # (split at a unit boundary) gives the blocks of drawing all cells, and
    # every full unit deals the same number of cells to every block.
    rng = cp.random.default_rng(3)
    n, n_groups, n_blocks, chunk = 5000, 4, 7, 8
    unit = n_blocks * chunk * 5
    groups = cp.sort(rng.integers(0, n_groups, n)).astype(cp.int32)

    def blocks(first, last):
        m = last - first
        idx = cp.empty(m, cp.int32)
        offsets = cp.empty(n_blocks * n_groups + 1, cp.int32)
        _cl.draw_blocks(
            groups[first:last],
            idx_list=idx,
            block_cat_offsets=offsets,
            idx_list_alt=cp.empty(m, cp.int32),
            sort_keys=cp.empty(m, cp.uint32),
            sort_keys_alt=cp.empty(m, cp.uint32),
            cub_temp=cp.empty(_cl.get_cub_sort_temp_bytes(n_cells=m), cp.uint8),
            n_groups=n_groups,
            n_blocks=n_blocks,
            unit=unit,
            seed=11,
            shuffle_chunk=chunk,
            first=first,
        )
        key = cp.repeat(cp.arange(n_blocks * n_groups), cp.diff(offsets).tolist())
        cp.testing.assert_array_equal(key % n_groups, groups[first:last][idx])
        out = cp.empty(m, cp.int32)
        out[idx] = key // n_groups
        return out

    full = blocks(0, n)
    cp.testing.assert_array_equal(full, cp.concatenate([blocks(0, 2 * unit), blocks(2 * unit, n)]))
    whole = n // unit * unit
    per_unit = cp.bincount(cp.arange(whole) // unit * n_blocks + full[:whole])
    assert (per_unit == unit // n_blocks).all()


@pytest.mark.parametrize("dtype", DTYPES)
def test_group_absmax_is_exact_and_split_invariant(dtype):
    # Per-group maxima of |X| (bit patterns of the right-hand side bounds):
    # exact, an empty group stays 0, a stale output is cleared, and the
    # shards' maxima give the whole's.
    rng = np.random.default_rng(0)
    offsets = np.array([0, 3000, 3000, 5000, 9000], dtype=np.int32)
    X = rng.standard_normal((9000, 7)).astype(dtype)
    X[4000, 3] = np.finfo(dtype).min
    X[5000:, 5] = -0.0

    def absmax(lo, hi):
        out = cp.full((4, 7), 9, dtype=dtype)
        seg_start, seg_group = _segments(np.clip(offsets, lo, hi) - lo)
        _cbat.group_absmax(cp.asarray(X[lo:hi]), seg_start=seg_start, seg_group=seg_group, out=out)
        return cp.asnumpy(out)

    expected = np.stack(
        [np.abs(X[a:b]).max(0) if b > a else np.zeros(7, dtype) for a, b in zip(offsets[:-1], offsets[1:], strict=True)]
    )
    whole = absmax(0, 9000)
    np.testing.assert_array_equal(whole, expected)
    np.testing.assert_array_equal(np.maximum(absmax(0, 4096), absmax(4096, 9000)), whole)


# ---------- fused initialization (assignment kernels) ----------


@pytest.mark.parametrize(
    "n_clusters,force_blocks,large",
    [(7, False, False), (7, True, False), (300, True, False), (300, True, True)],
)
@pytest.mark.parametrize("n_pcs", [50, 1023])
@pytest.mark.parametrize("n_keys", [1, 2])
@pytest.mark.parametrize("stabilized", [True, False])
@pytest.mark.parametrize("dtype", DTYPES)
def test_fused_initialize_matches_reference(n_clusters, force_blocks, large, n_pcs, n_keys, *, stabilized, dtype):
    # One unpenalized assignment pass over the update blocks: R is the
    # softmax of -2/sigma (1 - z.y), O its column sums per batch level,
    # E = Pr_b x column sums, plus the objective. Large: one block of 384
    # cells per tile slot, so tiles of six row groups, three cluster blocks.
    # Two keys: 3 x 400 levels, so runs of one joint category are about 2.5
    # (small) or 240 cells (large). Then one penalized pass (below).
    if large and (n_pcs > 50 or not stabilized):
        pytest.skip("the float64 reference is slow")
    rng = cp.random.default_rng(7)
    n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
    levels, sigma = [3, 400][:n_keys], 0.1
    n_batches = sum(levels)
    n_cells, block_size = (384 * 4 * n_sm,) * 2 if large else (3000, 700)
    Z = rng.standard_normal((n_cells, n_pcs), dtype=dtype)
    Z /= cp.linalg.norm(Z, axis=1, keepdims=True)
    Y = Z[:n_clusters] + 0.1
    Y = (Y / cp.linalg.norm(Y, axis=1, keepdims=True)).astype(dtype)
    # Marginal batch levels per key, cells sorted by joint category.
    codes = cp.sort(rng.integers(0, 3 * 400, n_cells))
    joint_ids, joints = cp.unique(codes, return_inverse=True)
    joints = joints.astype(cp.int32)
    joint_cats = cp.stack([joint_ids // 400, 3 + joint_ids % 400], axis=1)
    codes = cp.stack([codes // 400, 3 + codes % 400], axis=1)[:, :n_keys]
    Pr_b = (cp.bincount(codes.ravel(), minlength=n_batches) / n_cells).astype(dtype)
    theta = 0.5 + 2 * rng.random(n_batches, dtype=dtype)  # per level
    n_blocks = -(-n_cells // block_size)
    n_groups = n_batches if n_keys == 1 else 1  # the tiles' groups
    n_tiles = 4 * n_sm + n_groups + 1
    idx_list = cp.empty(n_cells, cp.int32)
    block_cat_offsets = cp.empty(n_blocks * (2 * n_groups + 1) + 1, cp.int32)
    joint_args = {}
    if n_keys > 1:
        joint_args = {
            "joint_cats": joint_cats.astype(cp.int32),
            "joints": joints,
            "n_covariates": n_keys,
            "n_first": levels[0],
        }
    _cl.draw_blocks(
        codes[:, 0].astype(cp.int32) if n_keys == 1 else cp.zeros(n_cells, cp.int32),
        idx_list=idx_list,
        block_cat_offsets=block_cat_offsets,
        idx_list_alt=cp.empty(n_cells, cp.int32),
        sort_keys=cp.empty(n_cells, cp.uint32),
        sort_keys_alt=cp.empty(n_cells, cp.uint32),
        cub_temp=cp.empty(_cl.get_cub_sort_temp_bytes(n_cells=n_cells), cp.uint8),
        n_groups=n_groups,
        n_blocks=n_blocks,
        unit=n_cells,
        seed=3,
    )
    R = cp.empty((n_cells, n_clusters), dtype)
    O = cp.empty((n_batches, n_clusters), dtype)
    E = cp.empty_like(O)
    stride = -(-n_clusters // _cl.FUSED_CLUSTER_STRIDE) * _cl.FUSED_CLUSTER_STRIDE
    loop = partial(
        _cl.clustering_loop,
        Z,
        R=R,
        E=E,
        O=O,
        Pr_b=Pr_b,
        theta=theta,
        Y_norm=Y,
        idx_list=idx_list,
        penalty=cp.empty_like(O),
        objective_partials=cp.empty(n_cells + 8, dtype),
        block_cat_offsets=block_cat_offsets,
        block_starts=block_cat_offsets[: n_blocks * n_groups + 1 : n_groups].get(),
        block_counts=cp.empty((n_blocks + 3) * n_batches * n_clusters, cp.int64),
        seg_start=cp.asarray([0, n_cells], dtype=cp.int32),
        y_bound=cp.full(n_pcs, np.frexp(n_cells)[1], cp.int16),
        y_acc=cp.empty(3 * n_clusters * n_pcs, cp.int64),
        y_t=cp.empty((_cl.fused_padded_pcs(n_pcs), stride), dtype),
        col_workspace=cp.empty(n_tiles * _cl.FUSED_WARPS * n_clusters, cp.int64),
        dots_workspace=cp.empty(n_tiles * _cl.FUSED_GROUP * stride, dtype),
        force_blocks=force_blocks,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_blocks=n_blocks,
        sigma=sigma,
        stabilized=stabilized,
        **joint_args,
    )

    def counts(R, codes):  # column sums of R per batch level
        O = cp.zeros((n_batches, n_clusters))
        cp.add.at(O, codes, R[:, None])
        return O

    def penalty(O):  # E, theta log((E + 1) / (O + E + 1)); plain: O + 1
        E = Pr_b[:, None] * O[: levels[0]].sum(0)
        return E, theta[:, None] * cp.log((E + 1) / (O + stabilized * E + 1))

    rtol, atol = (1e-4, 1e-6) if dtype == np.float32 else (1e-9, 1e-12)

    def check(objective, R_ref, sim):
        cp.testing.assert_allclose(R, R_ref, rtol=rtol, atol=atol)
        O_ref = counts(R_ref, codes)
        cp.testing.assert_allclose(O, O_ref, rtol=rtol, atol=atol)
        E_ref, log_pen = penalty(O_ref)
        cp.testing.assert_allclose(E, E_ref, rtol=rtol)
        # Distance + sigma entropy; the diversity is -sigma O log penalty. The
        # objective is an exact sum, rounded to the dtype once.
        terms = R_ref * (2 * (1 - sim) + sigma * cp.log(R_ref))
        expected = float(terms.sum() - sigma * (O_ref * log_pen).sum())
        assert objective == pytest.approx(expected, rel=1e-6 if dtype == np.float32 else 1e-12)

    objective = loop(max_iter=0, initialize=True)
    Z64 = Z.astype(cp.float64)
    sim = Z64 @ Y.T.astype(cp.float64)
    w = cp.exp(-2 / sigma * (1 - sim))
    check(objective, w / w.sum(1, keepdims=True), sim)
    if n_blocks == 1:
        return  # no other block's counts, so no penalty
    # One penalized pass from the centroids of R (seed 0: blocks in order):
    # each block against the counts without its own, a block's new counts
    # entering them before the block after next. A cell's log penalty sums
    # its levels' (two keys: its joint category's).
    R0 = R.astype(cp.float64)
    objective = loop(max_iter=1)
    Y = R0.T @ Z64
    sim = Z64 @ (Y / cp.linalg.norm(Y, axis=1, keepdims=True)).T
    starts = block_cat_offsets[: n_blocks * n_groups + 1 : n_groups].tolist()
    blocks = [idx_list[a:b] for a, b in zip(starts, starts[1:], strict=False)]
    old = [counts(R0[c], codes[c]) for c in blocks]
    total, new, R_ref = sum(old), [], cp.empty_like(R0)
    for b, c in enumerate(blocks):
        if b >= 2:
            total += new[b - 2] - old[b - 2]
        log_pen = penalty(total - old[b])[1][codes[c]].sum(1)
        w = cp.exp(-2 / sigma * (1 - sim[c]) + log_pen)
        R_ref[c] = w / w.sum(1, keepdims=True)
        new.append(counts(R_ref[c], codes[c]))
    check(objective, R_ref, sim)


# ---------- several GPUs (ranks on one GPU) ----------


def test_clustering_objective_is_rank_uniform():
    # Every rank returns the objective of rank 0 (the exact sum over all
    # cells, plus rank 0's diversity, whose bits travel in the exchange), so
    # GPUs whose arithmetic differs still take the same trip count. Here
    # rank 1's theta, which only the diversity reads in the unpenalized pass,
    # is scaled, and a one-rank run with that theta shows the scale matters.
    n_cells, unit, n_pcs, n_clusters, n_batches = 4000, 2000, 8, 7, 3
    rng = cp.random.default_rng(5)
    Z = rng.standard_normal((n_cells, n_pcs), dtype=cp.float32)
    Z /= cp.linalg.norm(Z, axis=1, keepdims=True)
    Y = Z[:n_clusters] + 0.1
    Y /= cp.linalg.norm(Y, axis=1, keepdims=True)
    codes = cp.sort(rng.integers(0, n_batches, n_cells)).astype(cp.int32)
    Pr_b = (cp.bincount(codes, minlength=n_batches) / n_cells).astype(cp.float32)
    theta = 0.5 + 2 * rng.random(n_batches, dtype=cp.float32)

    def objective(lo, hi, theta, comm=0, rank=0):
        m, n_blocks = hi - lo, 2
        idx_list = cp.empty(m, cp.int32)
        offsets = cp.empty(n_blocks * (2 * n_batches + 1) + 1, cp.int32)
        _cl.draw_blocks(
            codes[lo:hi],
            idx_list=idx_list,
            block_cat_offsets=offsets,
            idx_list_alt=cp.empty(m, cp.int32),
            sort_keys=cp.empty(m, cp.uint32),
            sort_keys_alt=cp.empty(m, cp.uint32),
            cub_temp=cp.empty(_cl.get_cub_sort_temp_bytes(n_cells=m), cp.uint8),
            n_groups=n_batches,
            n_blocks=n_blocks,
            unit=unit,
            seed=3,
            first=lo,
        )
        O = cp.empty((n_batches, n_clusters), cp.float32)
        return _cl.clustering_loop(
            Z[lo:hi],
            R=cp.empty((m, n_clusters), cp.float32),
            E=cp.empty_like(O),
            O=O,
            Pr_b=Pr_b,
            theta=theta,
            Y_norm=Y.copy(),
            idx_list=idx_list,
            penalty=cp.empty_like(O),
            objective_partials=cp.empty(m + 8, cp.float32),
            block_cat_offsets=offsets,
            block_starts=offsets[: n_blocks * n_batches + 1 : n_batches].get(),
            block_counts=cp.empty((n_blocks + 3) * n_batches * n_clusters, cp.int64),
            seg_start=cp.asarray([0, m], dtype=cp.int32),
            y_bound=cp.full(n_pcs, np.frexp(n_cells)[1], cp.int16),
            y_acc=cp.empty(3 * n_clusters * n_pcs, cp.int64),
            y_t=cp.empty((n_pcs, 128), cp.float32),
            n_cells=m,
            n_total=n_cells,
            n_pcs=n_pcs,
            n_clusters=n_clusters,
            n_batches=n_batches,
            n_blocks=n_blocks,
            sigma=0.1,
            stabilized=True,
            max_iter=0,
            initialize=True,
            comm=comm,
            rank=rank,
        )

    one = objective(0, n_cells, theta)
    assert objective(0, n_cells, 3 * theta) != one
    comm, buffers = _comm([0, 0], 2 * n_batches * n_clusters, groups=[0, n_batches, 0, n_batches])
    cuts, thetas = [0, unit, n_cells], [theta, 3 * theta]
    with ThreadPoolExecutor(2) as pool:
        ranks = [pool.submit(objective, cuts[r], cuts[r + 1], thetas[r], comm.handle, r) for r in range(2)]
        assert [rank.result() for rank in ranks] == [one, one]


def test_comm_gives_up_on_a_stuck_gpu():
    # A GPU whose stream never completes (here: blocked by a host function)
    # holds the other ranks' streams through the events they wait on. Once a
    # rank aborts, their host synchronizations raise instead of waiting
    # forever, and the comm's final wait for in-flight exchanges is bounded.
    comm, buffers = _comm([0, 0], 16, groups=[0, 1, 0, 1])
    gate, n = threading.Event(), 8
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(2)]
    data = [cp.arange(n, dtype=cp.int64) for _ in range(2)]
    cp.cuda.Device().synchronize()
    streams[1].launch_host_func(lambda _: gate.wait(), None)

    def rank(r):
        comm.allreduce(r, data[r].data.ptr, n, streams[r].ptr)
        comm.wait(streams[r].ptr)

    try:
        with ThreadPoolExecutor(2) as pool:
            ranks = [pool.submit(rank, r) for r in range(2)]
            time.sleep(0.2)
            assert not any(rank.done() for rank in ranks)
            comm.abort()
            for rank in ranks:
                with pytest.raises(RuntimeError, match="another GPU failed"):
                    rank.result(timeout=10)
        start = time.monotonic()
        comm.synchronize(0.3)  # rank 1's staging is still behind the gate
        assert 0.25 < time.monotonic() - start < 3
    finally:
        gate.set()
        for stream in streams:
            stream.synchronize()
    for d in data:  # the exchange completes once released
        cp.testing.assert_array_equal(d, 2 * cp.arange(n))
