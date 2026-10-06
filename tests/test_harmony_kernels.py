"""Unit tests for harmony CUDA kernels against CuPy references."""

from __future__ import annotations

from functools import partial

import cupy as cp
import numpy as np
import pytest

from rapids_singlecell._cuda import (
    _harmony_clustering_cuda as _cl,
)
from rapids_singlecell._cuda import (
    _harmony_correction_cuda as _corr,
)

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
        scatter_add = partial(
            _cl.scatter_add, values, categories=categories, out=out, workspace=workspace
        )
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


def _inv_mat_reference(O_col, lambda_col, dtype):
    """Pure CuPy reference for the algebraic fast-inverse."""
    n_batches = len(O_col)
    nb1 = n_batches + 1
    O_col = O_col.astype(dtype)
    lambda_col = lambda_col.astype(dtype)

    factor = dtype(1) / (O_col + lambda_col)
    P_row0 = -factor * O_col
    N_k = O_col.sum()
    c = N_k - (factor * O_col * O_col).sum()
    c_inv = dtype(1) / c

    inv = cp.empty((nb1, nb1), dtype=dtype)
    inv[0, 0] = c_inv
    inv[0, 1:] = c_inv * P_row0
    inv[1:, 0] = P_row0 * c_inv
    inv[1:, 1:] = cp.outer(P_row0, P_row0) * c_inv + cp.diag(factor)
    return inv


@pytest.mark.parametrize("n_batches", [5, 50, 200])
@pytest.mark.parametrize("n_clusters", [10, 50])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_compute_inv_mat(n_batches, n_clusters, dtype):
    """Test that compute_inv_mat matches CuPy reference."""
    rng = np.random.default_rng(42)
    O = cp.array(rng.random((n_batches, n_clusters)) * 100, dtype=dtype)
    # Non-uniform lambda_kb to catch stride/transpose bugs
    lambda_kb = cp.array(rng.random((n_batches, n_clusters)) * 5 + 0.1, dtype=dtype)
    lambda_kb[0, 0] = dtype(1e30)  # sentinel: exercises the pruning path
    nb1 = n_batches + 1
    stream = cp.cuda.get_current_stream().ptr

    g_factor = cp.empty(n_batches, dtype=dtype)
    g_P_row0 = cp.empty(n_batches, dtype=dtype)

    for k in range(min(n_clusters, 3)):  # test a few clusters
        inv_mat = cp.empty((nb1, nb1), dtype=dtype)

        _corr.compute_inv_mat(
            O,
            lambda_kb=lambda_kb,
            n_batches=n_batches,
            n_clusters=n_clusters,
            cluster_k=k,
            inv_mat=inv_mat,
            g_factor=g_factor,
            g_P_row0=g_P_row0,
            stream=stream,
        )
        cp.cuda.Device().synchronize()

        expected = _inv_mat_reference(O[:, k], lambda_kb[:, k], dtype)
        atol = 1e-6 if dtype == cp.float32 else 1e-12
        cp.testing.assert_allclose(inv_mat, expected, atol=atol, rtol=1e-5)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_compute_inv_mat_absent_batch(dtype):
    """compute_inv_mat stays finite when O=0 and lambda_kb is large (pruned)."""
    n_batches, n_clusters = 5, 10
    rng = np.random.default_rng(99)
    O = cp.array(rng.random((n_batches, n_clusters)) * 100, dtype=dtype)
    lambda_kb = cp.array(rng.random((n_batches, n_clusters)) * 2 + 0.1, dtype=dtype)

    O[0, 0] = 0
    lambda_kb[0, 0] = dtype(1e30)
    O[3, 2] = 0
    lambda_kb[3, 2] = dtype(1e30)

    nb1 = n_batches + 1
    stream = cp.cuda.get_current_stream().ptr
    g_factor = cp.empty(n_batches, dtype=dtype)
    g_P_row0 = cp.empty(n_batches, dtype=dtype)

    for k in [0, 2]:
        inv_mat = cp.empty((nb1, nb1), dtype=dtype)
        _corr.compute_inv_mat(
            O,
            lambda_kb=lambda_kb,
            n_batches=n_batches,
            n_clusters=n_clusters,
            cluster_k=k,
            inv_mat=inv_mat,
            g_factor=g_factor,
            g_P_row0=g_P_row0,
            stream=stream,
        )
        cp.cuda.Device().synchronize()

        assert cp.all(cp.isfinite(inv_mat)), f"inv_mat non-finite for cluster {k}"
        expected = _inv_mat_reference(O[:, k], lambda_kb[:, k], dtype)
        atol = 1e-6 if dtype == cp.float32 else 1e-12
        cp.testing.assert_allclose(inv_mat, expected, atol=atol, rtol=1e-5)


# ---------- k-means initialization kernels ----------


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("n_clusters,n_cols", [(1, 3), (37, 50), (128, 64)])
def test_kmeans_assign(dtype, n_clusters, n_cols):
    rng = cp.random.default_rng(3)
    X = rng.standard_normal((1000, n_cols)).astype(dtype)
    centers = rng.standard_normal((n_clusters, n_cols)).astype(dtype)
    labels = cp.empty(1000, dtype=cp.int32)
    minimum = cp.empty(1000, dtype=dtype)
    _cl.kmeans_assign(X, centers=centers, labels=labels, minimum=minimum)
    distances = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
    cp.testing.assert_array_equal(labels, distances.argmin(1))
    cp.testing.assert_allclose(minimum, distances.min(1), rtol=1e-5)


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
    cp.testing.assert_array_equal(
        full, cp.concatenate([blocks(0, 2 * unit), blocks(2 * unit, n)])
    )
    whole = n // unit * unit
    per_unit = cp.bincount(cp.arange(whole) // unit * n_blocks + full[:whole])
    assert (per_unit == unit // n_blocks).all()


# ---------- fused initialization (assignment kernels) ----------


@pytest.mark.parametrize(
    "n_clusters,force_general", [(7, False), (7, True), (300, True)]
)
def test_fused_initialize_matches_reference(n_clusters, force_general):
    # One unpenalized assignment pass over the update blocks: R is the
    # softmax of -2/sigma (1 - z.y), O its per-batch column sums,
    # E = Pr_b x column sums, plus the objective.
    rng = cp.random.default_rng(7)
    n_cells, n_pcs, n_batches, sigma, block_size = 3000, 50, 3, 0.1, 700
    Z = rng.standard_normal((n_cells, n_pcs), dtype=cp.float32)
    Z /= cp.linalg.norm(Z, axis=1, keepdims=True)
    Y = Z[:n_clusters] + 0.1
    Y = (Y / cp.linalg.norm(Y, axis=1, keepdims=True)).astype(cp.float32)
    cats = cp.sort(rng.integers(0, n_batches, n_cells)).astype(cp.int32)
    Pr_b = (cp.bincount(cats, minlength=n_batches) / n_cells).astype(cp.float32)
    theta = cp.full(n_batches, 2.0, dtype=cp.float32)
    n_blocks = -(-n_cells // block_size)
    n_tiles = 4 * cp.cuda.Device().attributes["MultiProcessorCount"] + n_batches + 1
    idx_list = cp.empty(n_cells, cp.int32)
    block_cat_offsets = cp.empty(n_blocks * n_batches + n_batches + 2, cp.int32)
    _cl.draw_blocks(
        cats,
        idx_list=idx_list,
        block_cat_offsets=block_cat_offsets,
        idx_list_alt=cp.empty(n_cells, cp.int32),
        sort_keys=cp.empty(n_cells, cp.uint32),
        sort_keys_alt=cp.empty(n_cells, cp.uint32),
        cub_temp=cp.empty(_cl.get_cub_sort_temp_bytes(n_cells=n_cells), cp.uint8),
        n_groups=n_batches,
        n_blocks=n_blocks,
        unit=n_cells,
        seed=3,
    )
    R = cp.empty((n_cells, n_clusters), cp.float32)
    O = cp.empty((n_batches, n_clusters), cp.float32)
    E = cp.empty_like(O)
    stride = -(-n_clusters // 128) * 128
    objective = _cl.clustering_loop(
        Z,
        R=R,
        E=E,
        O=O,
        Pr_b=Pr_b,
        theta=theta,
        Y_norm=Y,
        idx_list=idx_list,
        penalty=cp.empty_like(O),
        objective_partials=cp.empty(n_cells + 8, cp.float32),
        block_cat_offsets=block_cat_offsets,
        block_counts=cp.empty((n_blocks + 3) * n_batches * n_clusters, cp.int64),
        seg_start=cp.asarray([0, n_cells], dtype=cp.int32),
        y_scale=cp.empty(1, cp.float64),
        y_acc=cp.empty(3 * n_clusters * n_pcs, cp.int64),
        y_t_general=cp.empty((52, stride), cp.float32),
        col_workspace=cp.empty(n_tiles * 8 * n_clusters, cp.int64),
        force_general=force_general,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_blocks=n_blocks,
        sigma=sigma,
        max_iter=0,
        stabilized=True,
        initialize=True,
    )
    sim = Z.astype(cp.float64) @ Y.T.astype(cp.float64)
    w = cp.exp(-2 / sigma * (1 - sim))
    R_ref = w / w.sum(1, keepdims=True)
    cp.testing.assert_allclose(R, R_ref, rtol=1e-4, atol=1e-6)
    O_ref = cp.stack([R_ref[cats == b].sum(0) for b in range(n_batches)])
    cp.testing.assert_allclose(O, O_ref, rtol=1e-4)
    E_ref = Pr_b[:, None].astype(cp.float64) * R_ref.sum(0)[None, :]
    cp.testing.assert_allclose(E, E_ref, rtol=1e-4)
    distance = (R_ref * 2 * (1 - sim)).sum()
    entropy = (R_ref * cp.log(R_ref)).sum()
    diversity = sigma * 2.0 * (O_ref * cp.log((O_ref + E_ref + 1) / (E_ref + 1))).sum()
    expected = float(distance + sigma * entropy + diversity)
    assert objective == pytest.approx(expected, rel=1e-4)
