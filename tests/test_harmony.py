from __future__ import annotations

import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import anndata as ad
import cupy as cp
import numpy as np
import pandas as pd
import pooch
import pytest
import scanpy as sc
from scipy.stats import pearsonr

import rapids_singlecell as rsc
import rapids_singlecell.preprocessing._harmony as harmony_module
import rapids_singlecell.preprocessing._harmony._chunks_to_device as chunks_module
from rapids_singlecell.preprocessing._harmony import (
    _SUPPRESS_PENALTY,
    _compute_lambda_kb,
    _correction,
    _correction_multi,
    _rhs_bounds,
    _segments,
)
from rapids_singlecell.preprocessing._harmony._helper import (
    _factorize_joint_codes,
    _get_batch_codes,
    _get_theta_array,
    _stratified_sample_indices,
)


def _solve_spd_batched(gram, rhs, *, one_at_a_time=False):
    # The cluster-system solve of the multi-key correction (C++).
    x = cp.empty_like(rhs)
    harmony_module._correction_batched_cuda.solve_spd(
        gram,
        rhs,
        x,
        one_at_a_time=one_at_a_time,
        stream=cp.cuda.get_current_stream().ptr,
        blas=cp.cuda.device.get_cublas_handle(),
        solver=cp.cuda.device.get_cusolver_handle(),
    )
    return x


def _get_measure(x, base, norm):
    assert norm in ["r", "L2"]

    if norm == "r":
        corr, _ = pearsonr(x, base)
        return corr
    else:
        return np.linalg.norm(x - base) / np.linalg.norm(base)


_HARMONY_DATA_BASE = "https://scverse-exampledata.s3.amazonaws.com/rapids-singlecell/harmony_data"
_IRCOLITIS_HARMONYPY2_H5AD = (
    "ircolitis_blood_cd8_2048_harmonypy2_2_0_0.h5ad",
    "sha256:c52c4a916fc6b811134dbfb1dc105d83f53195ea3092f277f30c8dbb987d641a",
)
_HARMONYPY2_MULTIKEY_H5AD = (
    "harmonypy2_two_covariates_2_0_0.h5ad",
    "sha256:1ac1542ee31b0660175ed307c29077c5621225af062a0e14d251322abcc1ac46",
)


def test_harmony_multikey_marginal_codes_and_theta():
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(
                ["b0", "b1", "b2", "b3", "b4", "b0"],
                categories=["b0", "b1", "b2", "b3", "b4"],
            ),
            "sex": pd.Categorical(["f", "m", "f", "m", "f", "m"], categories=["f", "m"]),
        }
    )

    codes, n_levels = _get_batch_codes(obs, ["batch", "sex"])
    codes = cp.asnumpy(codes)

    np.testing.assert_array_equal(n_levels, np.array([5, 2], dtype=np.int32))
    np.testing.assert_array_equal(codes[:, 0], np.array([0, 1, 2, 3, 4, 0]))
    np.testing.assert_array_equal(codes[:, 1], np.array([5, 6, 5, 6, 5, 6]))
    counts = np.bincount(codes.ravel(), minlength=int(n_levels.sum()))
    assert counts[:5].sum() == len(obs)
    assert counts[5:].sum() == len(obs)
    cp.testing.assert_array_equal(
        _get_theta_array([2.0, 0.1], n_levels, cp.float32),
        cp.array([2, 2, 2, 2, 2, 0.1, 0.1], dtype=cp.float32),
    )
    cp.testing.assert_array_equal(
        _get_theta_array(2.0, n_levels, cp.float32),
        cp.full(7, 2.0, dtype=cp.float32),
    )
    expanded_theta = cp.arange(7, dtype=cp.float32)
    cp.testing.assert_array_equal(_get_theta_array(expanded_theta, n_levels, cp.float32), expanded_theta)


@pytest.mark.parametrize("size", [1, 6])
def test_harmony_multikey_theta_rejects_invalid_length(size):
    with pytest.raises(
        ValueError,
        match=r"batch variables \(2\) or categorical levels \(7\)",
    ):
        _get_theta_array([2.0] * size, np.array([5, 2]), cp.float32)


def test_harmony_theta_rejects_unsupported_type():
    with pytest.raises(ValueError, match="Theta must be a scalar or an array-like"):
        _get_theta_array({"batch": 2.0}, np.array([5, 2]), cp.float32)


def test_harmony_batch_keys_are_nonempty_and_complete():
    obs = pd.DataFrame({"batch": ["a", None]})
    with pytest.raises(ValueError, match="contains missing values"):
        _get_batch_codes(obs, "batch")
    with pytest.raises(ValueError, match="at least one column"):
        _get_batch_codes(obs, [])


@pytest.mark.parametrize("layout_seed", [0, 42, 734])
def test_harmony_stratified_sample_random_offsets_and_groups(layout_seed):
    rng = np.random.default_rng(layout_seed)
    n_groups = 128
    group_sizes = rng.integers(0, 257, size=n_groups, dtype=np.int64)
    group_sizes[::11] = 0
    group_sizes[1:3] = [1, 256]
    offsets = np.concatenate(([0], np.cumsum(group_sizes)))
    n_cells = int(offsets[-1])

    group_by_cell = np.repeat(np.arange(n_groups, dtype=np.int32), group_sizes)
    rng.shuffle(group_by_cell)
    cell_indices = np.argsort(group_by_cell, kind="stable").astype(np.int32)

    nonempty = np.flatnonzero(group_sizes)
    gpu_offsets = cp.asarray(offsets, dtype=cp.int32)
    gpu_indices = cp.asarray(cell_indices)

    targets = [nonempty.size, max(nonempty.size, n_cells // 3), n_cells]
    for n_target in targets:
        # a fresh generator per call: same seed, same sample
        sampled = cp.asnumpy(gpu_indices[_stratified_sample_indices(gpu_offsets, n_target, np.random.default_rng(17))])
        repeated = cp.asnumpy(gpu_indices[_stratified_sample_indices(gpu_offsets, n_target, np.random.default_rng(17))])

        assert sampled.size == n_target
        assert np.unique(sampled).size == n_target
        assert np.all((0 <= sampled) & (sampled < n_cells))
        np.testing.assert_array_equal(sampled, repeated)

        counts = np.bincount(group_by_cell[sampled], minlength=n_groups)
        np.testing.assert_array_equal(np.flatnonzero(counts), nonempty)
        assert np.all(counts <= group_sizes)

        if n_target == nonempty.size:
            np.testing.assert_array_equal(counts[nonempty], 1)
        elif n_target == n_cells:
            np.testing.assert_array_equal(counts, group_sizes)
            np.testing.assert_array_equal(np.sort(sampled), np.sort(cell_indices))


def test_harmony_stratified_sample_known_quotas():
    offsets = cp.asarray([0, 0, 1, 3, 8, 8, 16], dtype=cp.int32)
    cell_indices = cp.asarray([12, 7, 15, 1, 10, 4, 13, 2, 14, 9, 0, 11, 5, 8, 3, 6])
    cell_groups = np.array([5, 3, 3, 5, 3, 5, 5, 2, 5, 5, 3, 5, 1, 3, 5, 2])

    sampled = cp.asnumpy(cell_indices[_stratified_sample_indices(offsets, 9, np.random.default_rng(0))])
    counts = np.bincount(cell_groups[sampled], minlength=6)

    np.testing.assert_array_equal(counts, [0, 1, 1, 3, 0, 4])


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("batch_key", ["stratum", ["row", "column"]])
def test_harmony_initialization_sample_covers_strata(batch_key, monkeypatch):
    rng = np.random.default_rng(0)
    strata = np.tile(np.arange(9), 2)
    adata = ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {
                "stratum": pd.Categorical(strata, categories=np.arange(12)),
                "row": strata // 3,
                "column": strata % 3,
            },
            index=[f"cell_{i}" for i in range(strata.size)],
        ),
        obsm={"X_pca": rng.normal(size=(strata.size, 4)).astype(np.float32)},
    )
    monkeypatch.setattr(harmony_module, "_KMEANS_INIT_CELLS_PER_CLUSTER", 2)
    rsc.pp.harmony_integrate(
        adata,
        batch_key,
        n_clusters=2,
        max_iter_harmony=1,
        max_iter_clustering=2,
        block_proportion=1.0,
        random_state=0,
    )

    assert adata.obsm["X_pca_harmony"].shape == (strata.size, 4)
    assert np.isfinite(adata.obsm["X_pca_harmony"]).all()


def test_harmony_joint_code_overflow_fallback_is_one_dimensional():
    n_covariates = 64
    batch_codes = np.stack(
        (
            np.zeros(n_covariates, dtype=np.int32),
            np.ones(n_covariates, dtype=np.int32),
            np.zeros(n_covariates, dtype=np.int32),
        )
    )

    joint_cats, joint_codes = _factorize_joint_codes(batch_codes, np.full(n_covariates, 2, dtype=np.int32))

    assert joint_cats.shape == (2, n_covariates)
    assert joint_codes.shape == (3,)
    np.testing.assert_array_equal(joint_codes, np.array([0, 1, 0]))


@pytest.mark.parametrize("one_at_a_time", [False, True])
def test_harmony_large_multikey_systems_solve_one_at_a_time(one_at_a_time):
    # Few large systems take the blocked per-matrix factorization, others the
    # batched one; a singular system still falls back to least squares.
    rng = np.random.default_rng(0)
    n, d = 300, 7
    M = rng.standard_normal((2, n, n))
    B = rng.standard_normal((n, 50))
    gram = np.stack([M[0] @ M[0].T + n * np.eye(n), M[1] @ M[1].T + n * np.eye(n), B @ B.T])
    rhs = gram @ rng.standard_normal((3, n, d))
    result = _solve_spd_batched(cp.asarray(gram), cp.asarray(rhs), one_at_a_time=one_at_a_time)
    result = cp.asnumpy(result)
    np.testing.assert_allclose(result[:2], np.linalg.solve(gram[:2], rhs[:2]), rtol=1e-9)
    np.testing.assert_allclose(gram[2] @ result[2], rhs[2], rtol=1e-7, atol=1e-7)


def test_harmony_interrupt_during_the_solve_is_raised_after_it():
    # Ctrl-C while the main thread waits for the GPU in the C++ solve: its
    # scratch frees run no Python, so KeyboardInterrupt follows the call (a
    # Python free raised it in a destructor, which aborted the process).
    script = """
import _thread, time
import cupy as cp
from rapids_singlecell._cuda import _harmony_correction_batched_cuda as m
gram, rhs = 2 * cp.eye(2)[None], cp.ones((1, 2, 3))
stream = cp.cuda.get_current_stream()
kw = {"stream": stream.ptr, "blas": cp.cuda.device.get_cublas_handle(),
      "solver": cp.cuda.device.get_cusolver_handle()}
m.solve_spd(gram, rhs, cp.empty_like(rhs), **kw)
# The stream waits for this until the solve has queued its work behind it.
stream.launch_host_func(lambda _: time.sleep(0.2) or _thread.interrupt_main(), 0)
try:
    m.solve_spd(gram, rhs, cp.empty_like(rhs), **kw)
except KeyboardInterrupt:
    print("interrupted")
"""
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert (run.returncode, run.stdout) == (0, "interrupted\n"), run.stderr


def test_harmony_multikey_singular_gram_uses_least_squares():
    dtype = cp.float64
    gram = cp.asarray(
        [
            [[4.0, 1.0], [1.0, 3.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=dtype,
    )
    rhs = cp.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[2.0, 4.0], [2.0, 4.0]],
        ],
        dtype=dtype,
    )

    result = _solve_spd_batched(gram, rhs)
    expected = cp.asarray(
        [
            [[0.0, 2.0 / 11.0], [1.0, 14.0 / 11.0]],
            [[1.0, 2.0], [1.0, 2.0]],
        ],
        dtype=dtype,
    )

    assert bool(cp.isfinite(result).all())
    cp.testing.assert_allclose(result, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("ridge_lambda", [0.0, -0.1, float("inf"), float("nan")])
@pytest.mark.parametrize("multikey", [False, True])
def test_harmony1_rejects_bad_ridge_lambda(ridge_lambda, *, multikey):
    rng = np.random.default_rng(734)
    batch = np.resize(["a", "b", "c"], 60)
    adata = ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {"batch": batch, "duplicate_batch": batch},
            index=[f"cell_{index}" for index in range(60)],
        ),
        obsm={"X_pca": rng.normal(size=(60, 6)).astype(np.float32)},
    )
    key = ["batch", "duplicate_batch"] if multikey else "batch"

    with pytest.raises(ValueError, match="ridge_lambda must be a finite positive"):
        rsc.pp.harmony_integrate(
            adata,
            key,
            flavor="harmony1",
            ridge_lambda=ridge_lambda,
            n_clusters=3,
            max_iter_harmony=1,
            max_iter_clustering=2,
            block_proportion=1.0,
            random_state=734,
            dtype=cp.float32,
        )


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sigma": 0}, "sigma"),
        ({"sigma": -0.0}, "sigma"),
        ({"sigma": float("nan")}, "sigma"),
        ({"theta": float("nan")}, "theta"),
        ({"theta": float("inf")}, "theta"),
        ({"block_proportion": 0}, "block_proportion"),
        ({"block_proportion": 1.5}, "block_proportion"),
        ({"block_proportion": float("nan")}, "block_proportion"),
        ({"shuffle_chunk_size": 0}, "shuffle_chunk_size"),
        ({"shuffle_chunk_size": -2}, "shuffle_chunk_size"),
        ({"shuffle_chunk_size": 1.5}, "shuffle_chunk_size"),
        ({"shuffle_chunk_size": "8"}, "shuffle_chunk_size"),
        ({"shuffle_chunk_size": float("inf")}, "shuffle_chunk_size"),
        ({"shuffle_chunk_size": float("nan")}, "shuffle_chunk_size"),
        ({"colsum_algo": "bogus"}, "colsum_algo"),
        ({"max_iter_harmony": 0}, "max_iter_harmony"),
        ({"n_clusters": 0}, "n_clusters"),
    ],
)
def test_harmony_rejects_invalid_parameters(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _integrate(**kwargs)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("key", ["batch", ["batch", "second"]])
def test_harmony_accepts_numpy_integers(key, dtype):
    # NumPy integers act as Python integers; in float64 they once made the
    # bindings run their float32 overload on converted copies.
    kwargs = {"max_iter_harmony": 2, "max_iter_clustering": 3, "n_clusters": 7}
    expected = _integrate(dtype, key=key, **kwargs)
    for name, value in kwargs.items():
        for kind in (np.int64, np.uint8):
            result = _integrate(dtype, key=key, **{**kwargs, name: kind(value)})
            assert result.tobytes() == expected.tobytes()


def test_harmony_dtype_none_keeps_the_input_dtype():
    expected = _integrate(np.float32, max_iter_harmony=2)
    assert _integrate(None, max_iter_harmony=2).tobytes() == expected.tobytes()
    assert _integrate(">f4", max_iter_harmony=2).tobytes() == expected.tobytes()
    X = _repeatability_adata(np.float64).obsm["X_pca"]
    expected = _integrate(np.float64, X_pca=X)
    for data in (pd.DataFrame(X, index=np.arange(len(X)).astype(str)), X.astype(">f8")):
        assert _integrate(None, X_pca=data).tobytes() == expected.tobytes()


def test_harmony_accepts_huge_finite_values():
    # A bound beyond float64 (n_b max|X| > 1.8e308) is still finite input.
    X = _repeatability_adata(np.float64).obsm["X_pca"]
    X[0, 0] = 1e307
    assert np.isfinite(_integrate(np.float64, X_pca=X, max_iter_harmony=2)).all()


def test_harmony_multikey_rejects_huge_designs_early():
    # Several keys solve a dense system over all their levels; about one level
    # per cell cannot fit and fails before any GPU work.
    n = 9000
    adata = ad.AnnData(
        obs=pd.DataFrame(
            {"batch": np.arange(n) % 5, "cell": np.arange(n)},
            index=np.arange(n).astype(str),
        ).astype("category"),
        obsm={"X_pca": np.random.default_rng(0).standard_normal((n, 5), np.float32)},
    )
    with pytest.raises(MemoryError, match="levels"):
        rsc.pp.harmony_integrate(adata, ["batch", "cell"])


def test_harmony_did_not_converge_points_at_the_caller():
    with pytest.warns(UserWarning, match="did not converge") as record:
        _integrate(max_iter_harmony=1, tol_harmony=-np.inf)
    assert record[0].filename == __file__


@pytest.mark.parametrize("dtype", [np.float16, np.int32, "bf16"])
def test_harmony_rejects_unsupported_dtype(dtype):
    adata = _repeatability_adata(np.float32)
    with pytest.raises(ValueError, match="dtype must be float32, float64"):
        rsc.pp.harmony_integrate(adata, "batch", dtype=dtype)


def _dense_design_correction(X, R, cats, lambda_kb, n_batches):
    """Harmony correction with an explicit intercept + one-hot design matrix,
    in float64; pruned levels (lambda at the sentinel) are dropped from each
    solve."""
    X, R, lambda_kb = (np.asarray(a, dtype=np.float64) for a in (X, R, lambda_kb))
    n_cells = X.shape[0]
    design = np.zeros((n_cells, n_batches + 1), dtype=X.dtype)
    design[:, 0] = 1
    for column in cats.reshape(n_cells, -1).T:
        design[np.arange(n_cells), column + 1] = 1
    expected = X.copy()
    for cluster in range(R.shape[1]):
        active = lambda_kb[:, cluster] < X.dtype.type(_SUPPRESS_PENALTY)
        retained = np.flatnonzero(np.concatenate(([True], active)))
        weighted_design = R[:, cluster, None] * design
        gram = design.T @ weighted_design
        gram[1:, 1:] += np.diag(np.where(active, lambda_kb[:, cluster], 0))
        W = np.zeros((n_batches + 1, X.shape[1]), dtype=X.dtype)
        W[retained] = np.linalg.solve(gram[np.ix_(retained, retained)], (weighted_design.T @ X)[retained])
        W[0] = 0
        expected -= R[:, cluster, None] * (design @ W)
    return expected


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize(
    ("method", "normalize"),
    [("batched", False), ("batched", True), ("fast", False), ("chunks", True)],
)
@pytest.mark.parametrize(
    ("n_cells", "n_pcs", "n_clusters", "n_batches"),
    # The large case has counts ~2e5 against lambda < 1 (harmony1-like): an
    # explicit inverse loses the solution to cancellation there. 300 batches:
    # the intercept sums run over several tiles of batches.
    [
        (40, 5, 3, 4),
        (3000, 50, 100, 4),
        (3000, 300, 300, 4),
        (1_500_000, 8, 2, 4),
        (1500, 300, 3, 300),
    ],
)
def test_harmony_correction_matches_dense_design(
    dtype, *, method, normalize, n_cells, n_pcs, n_clusters, n_batches, monkeypatch
):
    if method == "chunks":  # batched, one cluster per pass (odd offsets too)
        monkeypatch.setattr(harmony_module, "_batched_cluster_chunk", lambda *_: 1)
        method = "batched"
    rng = np.random.default_rng(734)
    np_dtype = np.dtype(dtype)
    # Cells sorted by batch; batch 2 is empty.
    levels = np.delete(np.arange(n_batches), 2)
    cats_np = np.sort(rng.choice(levels, size=n_cells)).astype(np.int32)
    X_np = rng.normal(size=(n_cells, n_pcs)).astype(np_dtype)
    R_np = rng.random(size=(n_cells, n_clusters)).astype(np_dtype)
    R_np /= R_np.sum(axis=1, keepdims=True)
    O_np = np.zeros((n_batches, n_clusters), dtype=np_dtype)
    np.add.at(O_np, cats_np, R_np)
    lambda_np = rng.uniform(0.2, 1.0, size=O_np.shape).astype(np_dtype)
    lambda_np[2] = np_dtype.type(_SUPPRESS_PENALTY)
    lambda_np[1, 1] = np_dtype.type(_SUPPRESS_PENALTY)
    offsets = np.searchsorted(cats_np, np.arange(n_batches + 1)).astype(np.int32)

    result = _correction(
        cp.asarray(X_np),
        R=cp.asarray(R_np),
        O=cp.asarray(O_np),
        lambda_kb=cp.asarray(lambda_np),
        correction_method=method,
        cats=cp.asarray(cats_np),
        n_batches=n_batches,
        cat_offsets=cp.asarray(offsets),
        normalize=normalize,
        segments=(
            *_segments(offsets),
            cp.asarray(_rhs_bounds([cp.asarray(X_np)], offsets, [0, n_cells])),
        ),
    )

    expected = _dense_design_correction(X_np, R_np, cats_np, lambda_np, n_batches)
    if normalize:
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    atol = 2e-5 if dtype == cp.float32 else 1e-11
    cp.testing.assert_allclose(result, cp.asarray(expected), atol=atol, rtol=atol)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("method", ["batched", "fast"])
def test_harmony_correction_keeps_a_large_single_batch_cluster(dtype, method):
    # One batch, one cluster of 2**25 cells, ridge_lambda 1: the coefficients
    # are zero and the embedding stays (an explicit inverse's pivot
    # N - O^2 / (O + 1) cancels to 0 in float32).
    n = 2**25
    X = cp.tile(cp.asarray([3, 4], dtype=dtype), (n, 1))
    offsets = np.array([0, n], dtype=np.int32)
    result = _correction(
        X,
        R=cp.ones((n, 1), dtype=dtype),
        O=cp.full((1, 1), n, dtype=dtype),
        lambda_kb=cp.ones((1, 1), dtype=dtype),
        correction_method=method,
        cats=cp.zeros(n, dtype=cp.int32),
        n_batches=1,
        cat_offsets=cp.asarray(offsets),
        cell_indices=cp.arange(n, dtype=cp.int32),
        segments=(*_segments(offsets), cp.asarray(_rhs_bounds([X], offsets, [0, n]))),
    )
    cp.testing.assert_allclose(result, X, rtol=1e-12 if dtype == cp.float64 else 0)


@pytest.mark.parametrize("big", [1e19, 1e18])
@pytest.mark.parametrize("mixed", [False, True])
def test_harmony_extreme_batch_keeps_the_others(big, mixed):
    # The exact right-hand sides are scaled per batch and PC, so a batch of
    # 1e19 does not round away the sums of the others: batched = fast. Mixed:
    # the extreme batch's other cells sum in the other PC (per-PC bounds).
    n = 300
    X = np.repeat(np.array([[big, 0], [0, 1], [0, 3]], np.float32), n, axis=0)
    if mixed:
        X[: n // 2] = [0, 2]
    obs = pd.DataFrame(
        {"batch": pd.Categorical(np.repeat(["b0", "b1", "b2"], n))},
        index=np.arange(3 * n).astype(str),
    )

    def run(method):
        adata = ad.AnnData(obs=obs, obsm={"X_pca": X})
        rsc.pp.harmony_integrate(adata, "batch", n_clusters=2, sigma=0.01, correction_method=method, rng=0)
        return adata.obsm["X_pca_harmony"][X[:, 0] != big]

    np.testing.assert_allclose(run("batched"), run("fast"), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize(("n_covariates", "n_cells"), [(2, 31), (3, 31), (4, 31), (2, 1_000_000)])
def test_harmony_multikey_correction_matches_dense_design(dtype, n_covariates, n_cells, monkeypatch):
    rng = np.random.default_rng(734)
    levels = np.arange(2, 2 + n_covariates, dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(levels)[:-1])).astype(np.int32)
    n_batches = int(levels.sum())
    n_pcs, n_clusters = 5, 3
    np_dtype = np.dtype(dtype)

    local_codes = np.column_stack([rng.integers(0, level, size=n_cells) for level in levels]).astype(np.int32)
    cats_np = local_codes + offsets
    X_np = rng.normal(size=(n_cells, n_pcs)).astype(np_dtype)
    R_np = rng.random(size=(n_cells, n_clusters)).astype(np_dtype)
    R_np /= R_np.sum(axis=1, keepdims=True)

    O_np = np.zeros((n_batches, n_clusters), dtype=np_dtype)
    for covariate in range(n_covariates):
        np.add.at(O_np, cats_np[:, covariate], R_np)
    lambda_np = rng.uniform(0.2, 1.0, size=O_np.shape).astype(np_dtype)
    lambda_np[1, 1] = np_dtype.type(_SUPPRESS_PENALTY)

    # The correction expects cells sorted by joint category.
    joint_cats_np, joint_codes_np = np.unique(cats_np, axis=0, return_inverse=True)
    order = np.argsort(joint_codes_np, kind="stable")
    cats_np, X_np, R_np = cats_np[order], X_np[order], R_np[order]
    joint_codes_np = joint_codes_np[order]
    joint_offsets = np.searchsorted(joint_codes_np, np.arange(len(joint_cats_np) + 1)).astype(np.int32)
    marginal = sorted((level, joint) for joint, levels in enumerate(joint_cats_np) for level in levels)
    marginal_offsets = np.searchsorted([level for level, _ in marginal], np.arange(n_batches + 1)).astype(np.int32)

    def correct(normalize=False):
        return _correction_multi(
            cp.asarray(X_np),
            cp.asarray(R_np),
            lambda_kb=cp.asarray(lambda_np),
            n_first=int(levels[0]),
            joint_cats=cp.asarray(joint_cats_np, dtype=cp.int32),
            segments=(
                *_segments(joint_offsets),
                cp.asarray(
                    _rhs_bounds([cp.asarray(X_np)], joint_offsets, [0, n_cells])
                ),
            ),
            marginal_joint_offsets=cp.asarray(marginal_offsets),
            marginal_joint_indices=cp.asarray([joint for _, joint in marginal], dtype=cp.int32),
            normalize=normalize,
        )

    result = correct()

    expected = _dense_design_correction(X_np, R_np, cats_np, lambda_np, n_batches)
    # float64: the NumPy reference rounds differently per CPU (up to 2.4e-11)
    atol = 2e-5 if dtype == cp.float32 else 1e-10
    cp.testing.assert_allclose(result, cp.asarray(expected), atol=atol, rtol=atol)

    monkeypatch.setattr(
        harmony_module,
        "_multi_correction_cluster_chunk_size",
        lambda **_kwargs: 1,
    )

    chunked = correct()  # chunks continue the sums in the output
    cp.testing.assert_allclose(chunked, cp.asarray(expected), atol=atol, rtol=atol)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    cp.testing.assert_allclose(correct(normalize=True), cp.asarray(expected), atol=atol, rtol=atol)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("case", "theta", "n_clusters", "max_iter_harmony"),
    [
        ("nclust1", [2.0, 0.1], 1, 1),
        ("nclust4", [2.0, 0.1], 4, 10),
        ("nclust4_batch_only", [2.0, 0.0], 4, 10),
        ("nclust4_sex_only", [0.0, 2.0], 4, 10),
    ],
)
def test_harmony2_multikey_reference(
    adata_harmonypy2_multikey,
    case,
    theta,
    n_clusters,
    max_iter_harmony,
):
    reference = adata_harmonypy2_multikey.obsm[f"harmony2_ref_{case}"]

    def run(seed):
        adata = adata_harmonypy2_multikey.copy()
        rsc.pp.harmony_integrate(
            adata,
            ["batch", "sex"],
            theta=theta,
            flavor="harmony2",
            dtype=cp.float64,
            sigma=0.1,
            n_clusters=n_clusters,
            max_iter_harmony=max_iter_harmony,
            max_iter_clustering=4,
            tol_clustering=1e-3,
            tol_harmony=1e-2,
            block_proportion=0.05,
            random_state=seed,
            alpha=0.2,
            batch_prune_threshold=1e-5,
        )
        return adata.obsm["X_pca_harmony"]

    result = run(734)
    assert _get_measure(reference, result, "r").min() > 0.95
    assert _get_measure(reference, result, "L2").max() < 0.1
    # 80 cells have several optima that the reference also reaches depending
    # on its seed (seed 2: L2 up to 0.19); every seed stays close.
    for seed in range(5):
        result = run(seed)
        assert _get_measure(reference, result, "r").min() > 0.95
        assert _get_measure(reference, result, "L2").max() < 0.25


@pytest.fixture(scope="module")
def adata_harmonypy2_multikey():
    filename, known_hash = _HARMONYPY2_MULTIKEY_H5AD
    reference_file = pooch.retrieve(f"{_HARMONY_DATA_BASE}/{filename}", known_hash=known_hash)
    return ad.read_h5ad(reference_file)


@pytest.fixture(scope="module")
def adata_reference():
    X_pca_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/pbmc_3500_pcs.tsv.gz",
        known_hash="md5:27e319b3ddcc0c00d98e70aa8e677b10",
    )
    X_pca = pd.read_csv(X_pca_file, delimiter="\t")
    X_pca_harmony_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/pbmc_3500_pcs_harmonized.tsv.gz",
        known_hash="md5:a7c4ce4b98c390997c66d63d48e09221",
    )
    X_pca_harmony = pd.read_csv(X_pca_harmony_file, delimiter="\t")
    meta_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/pbmc_3500_meta.tsv.gz",
        known_hash="md5:8c7ca20e926513da7cf0def1211baecb",
    )
    meta = pd.read_csv(meta_file, delimiter="\t")
    return ad.AnnData(
        X=None,
        obs=meta,
        obsm={"X_pca": X_pca.values, "harmony_org": X_pca_harmony.values},
    )


@pytest.fixture(scope="module")
def adata_ircolitis_harmonypy2():
    """Stratified 2,048-cell IRcolitis harmonypy 2.0.0 reference."""
    filename, known_hash = _IRCOLITIS_HARMONYPY2_H5AD
    reference_file = pooch.retrieve(f"{_HARMONY_DATA_BASE}/{filename}", known_hash=known_hash)
    return ad.read_h5ad(reference_file)


@pytest.mark.parametrize("bad_alpha", [-0.1, 0.0, float("inf"), float("nan")])
def test_harmony_integrate_bad_alpha(bad_alpha):
    """Non-positive or non-finite alpha with flavor='harmony2' raises ValueError."""
    adata = sc.datasets.pbmc68k_reduced()
    with pytest.raises(ValueError, match="alpha must be a finite positive"):
        rsc.pp.harmony_integrate(adata, "bulk_labels", alpha=bad_alpha)


@pytest.mark.parametrize("bad_threshold", [-0.1, 1.5, 2.0])
def test_harmony_integrate_bad_prune_threshold(bad_threshold):
    """batch_prune_threshold outside [0, 1] raises ValueError."""
    adata = sc.datasets.pbmc68k_reduced()
    with pytest.raises(ValueError, match="batch_prune_threshold must be in"):
        rsc.pp.harmony_integrate(adata, "bulk_labels", batch_prune_threshold=bad_threshold)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_integrate_warns_for_original_correction(monkeypatch):
    adata = sc.datasets.pbmc68k_reduced()
    correction_fast = harmony_module._correction_fast
    calls = 0

    def traced_correction_fast(*args, **kwargs):
        nonlocal calls
        calls += 1
        return correction_fast(*args, **kwargs)

    monkeypatch.setattr(harmony_module, "_correction_fast", traced_correction_fast)
    with pytest.warns(
        FutureWarning,
        match="correction_method='original' is deprecated",
    ):
        rsc.pp.harmony_integrate(
            adata,
            "bulk_labels",
            correction_method="original",
            dtype=cp.float32,
            max_iter_harmony=1,
        )
    assert calls == 1
    assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("correction_method", ["fast", "batched"])
def test_harmony_integrate(correction_method):
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.

    This is a pure shape/contract check: the output shape is independent of
    dtype and iteration count, so we run float32 with a single harmony
    iteration to exercise both correction-method paths cheaply.
    """
    adata = sc.datasets.pbmc68k_reduced()
    rsc.pp.harmony_integrate(
        adata,
        "bulk_labels",
        correction_method=correction_method,
        dtype=cp.float32,
        max_iter_harmony=1,
    )
    assert adata.obsm["X_pca_harmony"].shape == adata.obsm["X_pca"].shape


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("correction_method", ["fast", "batched"])
def test_harmony_integrate_reference(adata_reference, *, dtype, correction_method):
    """
    Test that Harmony integrate works.
    """
    adata = adata_reference.copy()
    rsc.pp.harmony_integrate(
        adata,
        "donor",
        correction_method=correction_method,
        dtype=dtype,
        colsum_algo="atomics",  # no effect, accepted for compatibility
        max_iter_harmony=20,
        flavor="harmony1",
    )

    assert (
        _get_measure(
            adata.obsm["harmony_org"],
            adata.obsm["X_pca_harmony"],
            "L2",
        ).max()
        < 0.05
    )
    assert (
        _get_measure(
            adata.obsm["harmony_org"],
            adata.obsm["X_pca_harmony"],
            "r",
        ).min()
        > 0.95
    )


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_compute_lambda_kb_pruning(dtype):
    """_compute_lambda_kb suppresses correction for N_b==0 and below-threshold pairs."""
    n_batches, n_clusters = 4, 3
    alpha = 0.2
    threshold = 1e-5
    sentinel = dtype(_SUPPRESS_PENALTY)

    # batch 0 has zero cells (N_b==0), batch 2 has very few (below threshold)
    N_b = cp.array([0, 100, 1, 50], dtype=dtype)
    O = cp.array(
        [
            [0, 0, 0],  # batch 0: no cells
            [30, 40, 30],  # batch 1: well-represented
            [0, 0, 1],  # batch 2: 1 cell total, only in cluster 2
            [20, 15, 15],
        ],  # batch 3: well-represented
        dtype=dtype,
    )
    E = cp.ones((n_batches, n_clusters), dtype=dtype) * 10

    result = _compute_lambda_kb(
        E,
        O=O,
        N_b=N_b,
        alpha=alpha,
        threshold=threshold,
        ridge_lambda=1.0,
        dynamic_lambda=True,
    )

    # batch 0 (N_b==0): all clusters must be sentinel
    assert cp.all(result[0] == sentinel)
    # batch 1 (well-represented): should be alpha * E = 2.0
    cp.testing.assert_allclose(result[1], cp.full(n_clusters, alpha * 10, dtype=dtype))
    # batch 2, clusters 0,1 (O/N_b = 0/1 < threshold): sentinel
    assert result[2, 0] == sentinel
    assert result[2, 1] == sentinel
    # batch 2, cluster 2 (O/N_b = 1/1 = 1.0 >= threshold): alpha * E
    cp.testing.assert_allclose(result[2, 2], dtype(alpha * 10))
    # batch 3: all alpha * E
    cp.testing.assert_allclose(result[3], cp.full(n_clusters, alpha * 10, dtype=dtype))


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_compute_lambda_kb_dynamic_false(dtype):
    """_compute_lambda_kb returns uniform ridge_lambda when dynamic_lambda=False."""
    n_batches, n_clusters = 3, 5
    E = cp.ones((n_batches, n_clusters), dtype=dtype)
    O = cp.ones((n_batches, n_clusters), dtype=dtype)
    N_b = cp.ones(n_batches, dtype=dtype)

    result = _compute_lambda_kb(
        E,
        O=O,
        N_b=N_b,
        alpha=0.5,
        threshold=1e-5,
        ridge_lambda=1.0,
        dynamic_lambda=False,
    )
    cp.testing.assert_array_equal(result, cp.full_like(E, 1.0))


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_compute_lambda_kb_zero_denom(dtype):
    """_compute_lambda_kb guards against O==0 and E==0 (zero-denominator)."""
    sentinel = dtype(_SUPPRESS_PENALTY)
    # E==0 means lambda_kb = alpha*0 = 0; combined with O==0 triggers zero-denom guard
    E = cp.array([[0.0, 5.0]], dtype=dtype)
    O = cp.array([[0.0, 10.0]], dtype=dtype)
    N_b = cp.array([100.0], dtype=dtype)

    result = _compute_lambda_kb(
        E,
        O=O,
        N_b=N_b,
        alpha=0.2,
        threshold=None,
        ridge_lambda=1.0,
        dynamic_lambda=True,
    )
    # (0,0): O+lambda_kb = 0+0 = 0 → sentinel
    assert result[0, 0] == sentinel
    # (0,1): normal → alpha * E = 1.0
    cp.testing.assert_allclose(result[0, 1], dtype(1.0))


@pytest.mark.parametrize(
    ("dtype", "correction_method"),
    [
        (cp.float32, "fast"),
        (cp.float32, "batched"),
        (cp.float64, "fast"),
        (cp.float64, "batched"),
    ],
)
def test_harmony2_ircolitis_reference(adata_ircolitis_harmonypy2, correction_method, dtype):
    """Harmony2 on a real 11-batch CI subset matches harmonypy 2.0.0."""
    adata = adata_ircolitis_harmonypy2.copy()
    rsc.pp.harmony_integrate(
        adata,
        "batch",
        theta=2.0,
        flavor="harmony2",
        correction_method=correction_method,
        dtype=dtype,
        sigma=0.1,
        n_clusters=2,
        max_iter_harmony=10,
        max_iter_clustering=4,
        tol_clustering=1e-3,
        tol_harmony=1e-2,
        block_proportion=0.05,
        random_state=734,
        alpha=0.2,
        batch_prune_threshold=1e-5,
    )

    ref = adata.obsm["harmony2_ref"]
    result = adata.obsm["X_pca_harmony"]

    assert _get_measure(ref, result, "r").min() > 0.95
    assert _get_measure(ref, result, "L2").max() < 0.1


def test_harmony_unseeded_random_state():
    """``random_state=None`` means unseeded, not a crash."""
    rng = np.random.default_rng(734)
    batch = np.resize(["a", "b", "c"], 60)
    adata = ad.AnnData(
        X=None,
        obs=pd.DataFrame({"batch": batch}, index=[f"cell_{index}" for index in range(60)]),
        obsm={"X_pca": rng.normal(size=(60, 6)).astype(np.float32)},
    )

    rsc.pp.harmony_integrate(adata, "batch", n_clusters=3, max_iter_harmony=1, random_state=None)

    assert np.isfinite(adata.obsm["X_pca_harmony"]).all()


def _repeatability_adata(dtype, n_pcs=17):
    rng = np.random.default_rng(5102)
    n_cells = 480
    ordinal = np.arange(n_cells)
    return ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {
                "batch": ordinal % 5,
                "second": (ordinal // 5) % 3,
                "third": (ordinal // 15) % 2,
                "fine": ordinal % 37,  # with batch: runs of ~2.6 cells
                "many": ordinal % 300,  # with batch: 306 levels
                # One level on every shard of multi_gpu, and an empty one.
                "skewed": pd.Categorical(
                    np.where(ordinal % 10, "a", "c"), categories=["a", "b", "c"]
                ),
            },
            index=ordinal.astype(str),
        ),
        obsm={"X_pca": rng.standard_normal((n_cells, n_pcs)).astype(dtype)},
    )


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("key", ["batch", ["batch", "second"], ["batch", "second", "third"]])
@pytest.mark.parametrize("kmeans_cells_per_cluster", [5000, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("n_pcs", [17, 300])
def test_harmony_integrate_repeats_bitwise(monkeypatch, key, kmeans_cells_per_cluster, dtype, n_pcs):
    monkeypatch.setattr(harmony_module, "_KMEANS_INIT_CELLS_PER_CLUSTER", kmeans_cells_per_cluster)
    outputs = []
    for _ in range(2):
        adata = _repeatability_adata(dtype, n_pcs)
        rsc.pp.harmony_integrate(
            adata,
            key,
            dtype=dtype,
            rng=734,
            n_clusters=7,
            max_iter_harmony=2,
        )
        outputs.append(adata.obsm["X_pca_harmony"])
    assert outputs[0].dtype == dtype
    assert np.isfinite(outputs[0]).all()
    assert outputs[0].tobytes() == outputs[1].tobytes()


def _integrate(
    dtype=np.float32,
    *,
    X_pca=None,
    key="batch",
    n_clusters=7,
    max_iter_harmony=2,
    **kwargs,
):
    adata = _repeatability_adata(np.float32)
    if X_pca is not None:
        adata.obsm["X_pca"] = X_pca
    rsc.pp.harmony_integrate(
        adata,
        key,
        dtype=dtype,
        rng=734,
        n_clusters=n_clusters,
        max_iter_harmony=max_iter_harmony,
        **kwargs,
    )
    return adata.obsm["X_pca_harmony"]


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("n_clusters", [7, 8])  # 8: four-value stores
def test_harmony_bfloat16_assignments(n_clusters):
    first = _integrate("bfloat16", n_clusters=n_clusters)
    assert first.dtype == np.float32
    assert first.tobytes() == _integrate("bfloat16", n_clusters=n_clusters).tobytes()
    float32 = _integrate(n_clusters=n_clusters)
    assert 0 < _get_measure(first, float32, "L2") < 2e-2


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("flavor", "theta", "n_clusters", "sigma", "kwargs"),
    [
        ("harmony2", 200.0, 7, 0.1, {}),
        ("harmony2", 200.0, 7, 0.02, {}),
        ("harmony2", 200.0, 300, 0.1, {}),
        ("harmony1", 50.0, 20, 0.1, {}),
        ("harmony1", 50.0, 20, 0.1, {"dtype": np.float64}),
        ("harmony1", 300.0, 20, 0.1, {"correction_method": "fast"}),
        ("harmony1", 50.0, 20, 0.1, {"key": ["batch", "second"]}),
    ],
)
def test_harmony_large_theta_stays_finite(monkeypatch, *, flavor, theta, n_clusters, sigma, kwargs):
    # theta=200 takes float32 penalties of evenly mixed batches to about
    # 0.5**200, far below the smallest float: log-space penalties keep the
    # assignments and the objective finite (sigma=0.02: shifted per row).
    # harmony1 empties clusters entirely; they get no correction (several
    # keys: a unit pivot, no least-squares fallback).
    kwargs = dict(kwargs)
    key = kwargs.pop("key", "batch")
    multi = harmony_module._correction_batched_cuda.correction_multi
    monkeypatch.setattr(  # no least-squares fallback
        harmony_module._correction_batched_cuda,
        "correction_multi",
        lambda *a, **k: multi(*a, **k)[1] and pytest.fail("singular cluster system"),
    )
    objectives = []
    convergent = harmony_module._is_convergent_harmony
    monkeypatch.setattr(
        harmony_module,
        "_is_convergent_harmony",
        lambda objs, tol: objectives.append(list(objs)) or convergent(objs, tol),
    )
    result = _integrate(key=key, n_clusters=n_clusters, flavor=flavor, theta=theta, sigma=sigma, **kwargs)
    assert np.isfinite(result).all()
    assert all(0 < o < 1e6 for o in objectives[-1])


def test_harmony_verbose_reports_convergence(capsys):
    _integrate(verbose=True, tol_harmony=0.5, max_iter_harmony=10)
    assert capsys.readouterr().out == "Harmony converged in 1 iterations\n"


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("key", ["batch", ["batch", "second"]])
def test_harmony_unused_category_level(key):
    # A level without cells (an empty group of every update block) gives a
    # finite result close to the one without it; the sums over batches are
    # tiled differently, so not the bits.
    def run(batch):
        adata = _repeatability_adata(np.float32)
        adata.obs["batch"] = batch
        rsc.pp.harmony_integrate(adata, key, rng=734, n_clusters=7, max_iter_harmony=2)
        return adata.obsm["X_pca_harmony"]

    batch = pd.Categorical(_repeatability_adata(np.float32).obs["batch"], categories=[0, 1, 9, 2, 3, 4])
    result = run(batch)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(result, run(batch.remove_unused_categories()), atol=1e-5)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_tau_discounts_theta():
    discounted = _integrate(tau=5)
    assert np.isfinite(discounted).all()
    assert discounted.tobytes() != _integrate().tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_shuffle_chunk_size():
    exact = _integrate(shuffle_chunk_size=1)
    assert exact.tobytes() == _integrate(shuffle_chunk_size=1).tobytes()
    assert exact.tobytes() != _integrate(shuffle_chunk_size=8).tobytes()
    assert _integrate(shuffle_chunk_size=8.0).tobytes() == _integrate(shuffle_chunk_size=8).tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_host_input_matches_device_input(monkeypatch):
    host = _repeatability_adata(np.float32).obsm["X_pca"]
    device = _integrate(X_pca=cp.asarray(host))
    assert _integrate(X_pca=host).tobytes() == device.tobytes()
    # Small chunks, the last one short: the upload and the download reuse
    # their staging slots.
    monkeypatch.setattr(chunks_module, "_UPLOAD_BYTES", 13 * host[0].nbytes)
    assert 480 % chunks_module._staging_rows(480, host[0].nbytes) == 12
    assert _integrate(X_pca=host).tobytes() == device.tobytes()
    for value in (np.nan, np.inf, -np.inf):
        bad = host.copy()
        bad[-1, 2] = value  # in the last chunk
        for X in (bad, cp.asarray(bad)):
            with pytest.raises(ValueError, match="NaN or infinite"):
                _integrate(X_pca=X)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_host_input_is_converted_in_the_staging(monkeypatch):
    # float64, F-ordered or big-endian host input is cast and made contiguous
    # in the pinned staging chunks, so no converted host copy is made.
    X64 = np.random.default_rng(1).standard_normal((480, 17))
    expected = _integrate(X_pca=X64.astype(np.float32))
    uploaded = []
    sorted_copy = harmony_module._sorted_device_copy
    monkeypatch.setattr(harmony_module, "_sorted_device_copy", lambda Z, *a: uploaded.append(Z) or sorted_copy(Z, *a))
    for X in (X64, np.asfortranarray(X64), X64.astype(">f8")):
        assert _integrate(X_pca=X).tobytes() == expected.tobytes()
        assert uploaded[-1] is X


@pytest.mark.parametrize("n", [480, 481])  # 481: a short final chunk
def test_harmony_chunked_transfers(monkeypatch, n):
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((n, 17)).astype(np.float32)
    monkeypatch.setattr(chunks_module, "_UPLOAD_BYTES", 40 * Z[0].nbytes)
    assert chunks_module._staging_rows(n, Z[0].nbytes) == 15
    order = rng.permutation(n)
    sorted_copy = chunks_module._sorted_device_copy(Z, cp.asarray(order))
    np.testing.assert_array_equal(cp.asnumpy(sorted_copy), Z[order])
    out = np.empty_like(Z)
    chunks_module._download(sorted_copy, out)
    np.testing.assert_array_equal(out, Z[order])


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_small_host_input_stages_its_rows(monkeypatch):
    # Pinned and GPU staging chunks hold at most the input's rows.
    rows = []
    pinned = chunks_module._pinned
    monkeypatch.setattr(
        chunks_module,
        "_pinned",
        lambda n, d, dtype: rows.append(n) or pinned(n, d, dtype),
    )
    _integrate(X_pca=_repeatability_adata(np.float32).obsm["X_pca"])
    assert rows and max(rows) <= 480


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("dtype", [np.float32, "bfloat16"])
def test_harmony_auto_batched_in_cluster_chunks(monkeypatch, dtype):
    # The batched workspace of 3 of the 7 clusters (37 batches x 17 PCs x 16
    # bytes each) fits the limit: no fast correction, chunks of 3 clusters.
    monkeypatch.setattr(harmony_module, "_CORRECTION_WORKSPACE_LIMIT_BYTES", 3 * 37 * 17 * 16)
    monkeypatch.setattr(harmony_module, "_correction_fast", None)
    passes = []
    batched = harmony_module._correction_batched_cuda.correction_batched
    monkeypatch.setattr(
        harmony_module._correction_batched_cuda,
        "correction_batched",
        lambda *a, **k: passes.append(k["chunk"]) or batched(*a, **k),
    )
    auto = _integrate(dtype, key="fine")
    assert passes[0] == 3  # the workspace of one pass of 3 clusters
    forced = _integrate(dtype, key="fine", correction_method="batched")
    assert auto.tobytes() == forced.tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_bfloat16_falls_back_to_float32():
    key = ["batch", "second"]  # bfloat16 assignments need one batch key
    with pytest.warns(UserWarning, match="dtype='bfloat16'") as record:
        fallback = _integrate("bfloat16", key=key)
    assert record[0].filename == __file__  # the warning points at the caller
    assert fallback.tobytes() == _integrate(key=key).tobytes()
    adata = _repeatability_adata(np.float64)  # harmonize alone: float64 assignments
    with pytest.warns(UserWarning, match="using float64 assignments"):
        harmony_module.harmonize(cp.asarray(adata.obsm["X_pca"]), adata.obs, "batch", max_iter_harmony=1, bfloat16=True)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("key", "n_pcs", "n_clusters", "kwargs"),
    [
        ("batch", 17, 7, {}),
        (["batch", "second"], 17, 7, {}),
        ("batch", 3, 128, {}),
        ("batch", 128, 100, {}),
        ("batch", 300, 100, {}),
        (["batch", "second"], 300, 7, {}),
        (["batch", "fine"], 17, 7, {}),
        ("batch", 1023, 20, {}),
        ("batch", 50, 100, {"sigma": 0.02}),  # each row shifted by its maximum
        ("batch", 50, 64, {"dtype": np.float64}),
    ],
)
def test_harmony_cluster_blocks_match_fused(monkeypatch, *, key, n_pcs, n_clusters, kwargs):
    # The cluster-block path (more than 128 clusters) gives the bits of the
    # single-block one.
    X = np.random.default_rng(3).standard_normal((480, n_pcs)).astype(np.float32)

    def run():
        adata = _repeatability_adata(np.float32)
        adata.obsm["X_pca"] = X
        rsc.pp.harmony_integrate(adata, key, rng=734, n_clusters=n_clusters, max_iter_harmony=2, **kwargs)
        return adata.obsm["X_pca_harmony"]

    fused = run()
    monkeypatch.setattr(harmony_module, "_FORCE_CLUSTER_BLOCKS", True)
    assert run().tobytes() == fused.tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("cells_per_sm", [160, 480])
@pytest.mark.parametrize(
    ("dtype", "n_pcs", "key"),
    [
        ("float32", 50, "batch"),
        ("float32", 100, "batch"),
        ("float64", 32, "batch"),
        ("float64", 50, "batch"),
        ("bfloat16", 50, "batch"),
        ("float32", 50, ["batch", "fine"]),
    ],
)
def test_harmony_fused_tiles_of_several_groups(monkeypatch, cells_per_sm, dtype, n_pcs, key):
    # One block of 160 cells per SM: tiles of two or three groups of 64 rows
    # with one or two CTAs per SM (T4, float64: one); 480 per SM: the tiles of
    # large blocks (three groups, or one with one CTA per SM). Centroids are
    # resident up to 64 float32 (32 float64) PCs, streamed beyond. Two keys:
    # runs of 1-2 cells of a joint category.
    n_cells = cells_per_sm * cp.cuda.Device().attributes["MultiProcessorCount"]
    obs = pd.DataFrame(
        {
            "batch": pd.Categorical(np.arange(n_cells) % 2),
            "fine": pd.Categorical(np.arange(n_cells) % 997),
        },
        index=np.arange(n_cells).astype(str),
    )
    X = np.random.default_rng(7).standard_normal((n_cells, n_pcs))

    def run():
        adata = ad.AnnData(obs=obs, obsm={"X_pca": X.astype(np.float32)})
        rsc.pp.harmony_integrate(
            adata,
            key,
            dtype=dtype,
            rng=734,
            block_proportion=1.0,
            max_iter_harmony=1,
        )
        return adata.obsm["X_pca_harmony"]

    fused = run()
    assert run().tobytes() == fused.tobytes()
    monkeypatch.setattr(harmony_module, "_FORCE_CLUSTER_BLOCKS", True)
    assert run().tobytes() == fused.tobytes()


def _workspace_shapes(monkeypatch) -> list:
    """The shapes of each clustering workspace allocated from now on."""
    allocate, shapes = harmony_module._allocate_clustering_workspace, []

    def record(*args, **kwargs):
        workspace = allocate(*args, **kwargs)
        shapes.append({k: v.shape for k, v in workspace.items() if hasattr(v, "shape")})
        return workspace

    monkeypatch.setattr(harmony_module, "_allocate_clustering_workspace", record)
    return shapes


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_multikey_workspace_is_per_level(monkeypatch):
    # Several keys keep counts per batch level, and tiles take any cells: no
    # clustering buffer grows with the joint categories (here 185).
    shapes = _workspace_shapes(monkeypatch)
    monkeypatch.setattr(harmony_module, "_KEEP_BLOCK_COUNTS", True)
    adata = _repeatability_adata(np.float32)
    rsc.pp.harmony_integrate(adata, ["batch", "fine"], rng=734, n_clusters=300, max_iter_harmony=1)
    cl, n_blocks = harmony_module._clustering_cuda, 20
    (shape,) = shapes
    assert shape["block_counts"] == (n_blocks + 3, 5 + 37, 300)
    assert shape["block_cat_offsets"] == (3 * n_blocks + 1,)
    assert shape["dots_workspace"] == (cl.assign_slots(itemsize=4) * cl.FUSED_GROUP * 3 * cl.FUSED_CLUSTER_STRIDE,)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_clustering_workspace_is_bounded(monkeypatch):
    # One-cell blocks (480): block counts are recounted from R rather than
    # each block's kept, and the cluster-block scratch holds the resident
    # CTAs, however many batch levels (5 or 37).
    shapes = _workspace_shapes(monkeypatch)
    for key in ("batch", "fine"):
        _integrate(key=key, n_clusters=300, max_iter_harmony=1, block_proportion=0.002)
    few, many = shapes
    assert (few["block_counts"], many["block_counts"]) == ((6, 5, 300), (6, 37, 300))
    assert few["col_workspace"] == many["col_workspace"]
    assert few["dots_workspace"] == many["dots_workspace"]
    cl = harmony_module._clustering_cuda
    assert few["dots_workspace"] == (cl.assign_slots(itemsize=4) * cl.FUSED_GROUP * 3 * cl.FUSED_CLUSTER_STRIDE,)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("key", "dtype", "n_clusters", "block_proportion"),
    [
        ("batch", np.float32, 7, 0.05),
        ("batch", "bfloat16", 300, 0.05),
        ("batch", np.float64, 7, 0.002),  # one-cell blocks
        (["batch", "second"], np.float32, 7, 0.05),
        (["batch", "fine"], np.float32, 300, 0.01),
    ],
)
def test_harmony_recounted_block_counts_are_exact(monkeypatch, key, dtype, n_clusters, block_proportion):
    # Block counts recounted from R give the bits of the kept ones.
    outputs = []
    for keep in (True, False):
        monkeypatch.setattr(harmony_module, "_KEEP_BLOCK_COUNTS", keep)
        outputs.append(_integrate(dtype, key=key, n_clusters=n_clusters, block_proportion=block_proportion))
    assert outputs[0].tobytes() == outputs[1].tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("n_slots", [1, 3])
@pytest.mark.parametrize("key", ["batch", ["batch", "fine"]])
def test_harmony_cluster_blocks_on_few_slots(monkeypatch, key, n_slots):
    # A scratch of one or three CTAs: they claim every tile, with the same bits.
    expected = _integrate(key=key, n_clusters=300)
    monkeypatch.setattr(harmony_module._clustering_cuda, "assign_slots", lambda itemsize: n_slots)
    assert _integrate(key=key, n_clusters=300).tobytes() == expected.tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("key", "dtype"),
    [("batch", "float32"), ("batch", "bfloat16"), (["batch", "second"], "float32")],
)
def test_harmony_many_clusters(key, dtype):
    # More than 128 clusters go in cluster blocks.
    outputs = [_integrate(dtype, key=key, n_clusters=300) for _ in range(2)]
    assert np.isfinite(outputs[0]).all()
    assert outputs[0].tobytes() == outputs[1].tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("key", "dtype", "n_pcs", "chunk", "recount"),
    [
        ("batch", "float32", 17, None, False),
        ("batch", "bfloat16", 17, None, False),
        (["batch", "second"], "float64", 17, None, False),
        ("batch", "float32", 300, None, False),
        ("batch", "bfloat16", 17, 7, False),  # batched correction, 7 clusters per pass
        (["batch", "many"], "float32", 17, None, False),  # systems split between GPUs
        ("batch", "bfloat16", 17, None, True),  # recounted block counts
        (["batch", "second"], "float32", 17, None, True),
    ],
)
def test_harmony_multi_gpu_matches_one_gpu(
    monkeypatch, key, dtype, n_pcs, *, chunk, recount
):
    # Every sum is exact and the blocks and segments follow the global cell
    # order, so two GPUs give the bits of one (recount: block counts are
    # recounted from R on each GPU).
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two GPUs")
    if chunk:
        monkeypatch.setattr(harmony_module, "_batched_cluster_chunk", lambda *_: chunk)
    if recount:
        monkeypatch.setattr(harmony_module, "_KEEP_BLOCK_COUNTS", False)
    import rapids_singlecell._utils._multi_gpu as utils_multi_gpu
    import rapids_singlecell.preprocessing._harmony._multi_gpu as multi_gpu_module

    rng = np.random.default_rng(3)
    n_cells = 20_000
    obs = pd.DataFrame(
        {"batch": rng.integers(0, 5, n_cells), "second": rng.integers(0, 3, n_cells)},
        index=np.arange(n_cells).astype(str),
    )
    X = rng.standard_normal((n_cells, n_pcs)) + obs["batch"].to_numpy()[:, None]
    obs["many"] = rng.integers(0, 300, n_cells)

    def run(device_input=None, **kwargs):
        adata = ad.AnnData(obs=obs.astype("category"), obsm={"X_pca": X.copy()})
        if device_input is not None:  # on that GPU
            with cp.cuda.Device(device_input):
                adata.obsm["X_pca"] = cp.asarray(X)
        rsc.pp.harmony_integrate(
            adata,
            key,
            dtype=dtype,
            rng=734,
            n_clusters=20,
            max_iter_harmony=3,
            **kwargs,
        )
        return adata.obsm["X_pca_harmony"]

    ranks = _record_ranks(monkeypatch)
    one = run()
    assert run(multi_gpu=[0, 1]).tobytes() == one.tobytes()
    assert ranks == [[0, 1]]  # two GPUs really ran
    for device in (0, 1):
        assert run(multi_gpu=[0, 1], device_input=device).tobytes() == one.tobytes()
    # Three shards on two GPUs: sums reduce-scattered over peer copies.
    monkeypatch.setattr(
        utils_multi_gpu, "parse_device_ids", lambda multi_gpu: multi_gpu
    )
    assert run(multi_gpu=[0, 1, 0]).tobytes() == one.tobytes()
    # Without validated peer copies everything moves through the host (in
    # pinned chunks of a few rows).
    monkeypatch.setattr(multi_gpu_module, "_FORCE_HOST_COPIES", True)
    monkeypatch.setattr(multi_gpu_module, "_RELAY_BYTES", 1000)
    assert run(multi_gpu=[0, 1]).tobytes() == one.tobytes()
    assert run(multi_gpu=[0, 1], device_input=1).tobytes() == one.tobytes()
    assert run(multi_gpu=[0, 1, 0]).tobytes() == one.tobytes()


@pytest.mark.parametrize("peer", [False, True])
@pytest.mark.parametrize("source", [0, 1])
def test_harmony_device_input_is_gathered_on_its_gpu(monkeypatch, source, peer):
    # Device input is gathered on its GPU and only shards move: with pools
    # limited to one 4 MiB shard (and its index) per GPU, 8 MiB of input on
    # either GPU reaches GPUs 0 and 1.
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two GPUs")
    if peer and not harmony_module._peer_copies_work([0, 1]):
        pytest.skip("no validated peer copies")
    import rapids_singlecell.preprocessing._harmony._multi_gpu as multi_gpu_module

    monkeypatch.setattr(multi_gpu_module, "_FORCE_HOST_COPIES", not peer)
    with cp.cuda.Device(source):
        X = cp.random.default_rng(0).standard_normal((4096, 512), dtype=cp.float32)
    order = cp.asarray(np.random.default_rng(1).permutation(4096))
    allocator, pool = cp.cuda.get_allocator(), cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    try:
        for device in (0, 1):
            with cp.cuda.Device(device):
                pool.set_limit(size=X.nbytes * 9 // 16)
        parts = harmony_module._upload_sorted(X, order, [0, 2048, 4096], [0, 1])
        assert [part.device.id for part in parts] == [0, 1]
        expected = cp.asnumpy(X)[cp.asnumpy(order)]
        np.testing.assert_array_equal(
            np.concatenate(list(map(cp.asnumpy, parts))), expected
        )
    finally:
        cp.cuda.set_allocator(allocator)


def _record_ranks(monkeypatch) -> list:
    """The devices of every multi-GPU exchange harmonize sets up."""
    ranks, comm = [], harmony_module._comm
    monkeypatch.setattr(
        harmony_module,
        "_comm",
        lambda devices, *a, **k: ranks.append(list(devices)) or comm(devices, *a, **k),
    )
    return ranks


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("key", "n_pcs", "dtype", "n_clusters"),
    [
        ("batch", 17, np.float32, 7),
        (["batch", "second"], 17, np.float32, 7),
        (["batch", "second"], 17, np.float64, 7),  # wider exact sums
        ("batch", 300, np.float32, 7),
        (["batch", "second"], 17, np.float32, 300),  # cluster blocks
        (["batch", "fine"], 17, np.float32, 7),  # joint runs of ~2.6 cells
        (["batch", "many"], 17, np.float32, 7),  # systems split between GPUs
        ("skewed", 17, np.float32, 7),  # a batch on every shard, an empty one
    ],
)
@pytest.mark.parametrize("n_ranks", [2, 3])
def test_harmony_two_ranks_on_one_gpu(
    monkeypatch, *, key, n_pcs, dtype, n_clusters, n_ranks
):
    # Two or three shards on one GPU run the exchanges (three: larger sums
    # reduce-scattered and all-gathered) and the shard I/O of multi_gpu
    # (one-GPU CI covers them) and give the bits of one.
    import rapids_singlecell.preprocessing._harmony._multi_gpu as multi_gpu_module

    adata = _repeatability_adata(dtype, n_pcs)
    ranks = _record_ranks(monkeypatch)

    def run(devices, data=adata, **kwargs):
        out = np.empty_like(data.obsm["X_pca"])
        harmony_module.harmonize(
            data.obsm["X_pca"],
            data.obs,
            key,
            rng=734,
            n_clusters=n_clusters,
            max_iter_harmony=2,
            devices=devices,
            out=out,
            **kwargs,
        )
        return out

    one, several = run([0]), [0] * n_ranks
    assert run(several).tobytes() == one.tobytes()
    assert ranks == [several]
    # Exchanges larger than a slot go in pieces.
    monkeypatch.setattr(harmony_module, "_COMM_VALUES", 100)
    assert run(several).tobytes() == one.tobytes()
    if isinstance(key, str):
        with pytest.warns(UserWarning, match="fast correction runs on one GPU"):
            run(several, correction_method="fast")
        assert len(ranks) == 2  # no third exchange
    else:  # 'fast' is ignored with several keys
        with pytest.warns(UserWarning, match="is ignored"):
            run(several, correction_method="fast")
        # Cluster chunks of the correction, each with its own exchange.
        monkeypatch.setattr(
            harmony_module, "_multi_correction_cluster_chunk_size", lambda **_: 2
        )
        solved, solve = [], harmony_module._solve_spd_batched
        monkeypatch.setattr(
            harmony_module,
            "_solve_spd_batched",
            lambda gram, rhs: solved.append(len(gram)) or solve(gram, rhs),
        )
        one = run([0])
        on_one, solved[:] = sum(solved), []
        assert run(several).tobytes() == one.tobytes()
        if key[1] == "many":  # each GPU solves its share of the systems
            assert sum(solved) == on_one
    # Without peer copies (pinned host send slots), in several upload chunks
    # per shard: scattered rows, and runs of rows from input sorted by batch.
    monkeypatch.setattr(multi_gpu_module, "_FORCE_HOST_COPIES", True)
    monkeypatch.setattr(
        harmony_module, "_UPLOAD_BYTES", 40 * adata.obsm["X_pca"][0].nbytes
    )
    assert run(several).tobytes() == run([0]).tobytes()
    by_batch = adata[np.argsort(adata.obs["batch"].to_numpy(), kind="stable")].copy()
    assert run(several, by_batch).tobytes() == run([0], by_batch).tobytes()
    # The last batch sorts into the last shard.
    X = adata.obsm["X_pca"]
    X[adata.obs["batch"].to_numpy() == 4, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinite"):
        run(several)


@pytest.mark.parametrize("layout", ["C", "F"])
def test_harmony_host_input_shards_stage_only_their_rows(monkeypatch, layout):
    # Several GPUs stage only their own input rows (runs and scattered rows),
    # so every row crosses the host once.
    X = np.random.default_rng(0).standard_normal((1000, 8)).astype(np.float32)
    order = np.concatenate(
        [np.arange(500, 1000), np.random.default_rng(1).permutation(500)]
    )
    staged, copy = [], harmony_module._parallel_copy
    monkeypatch.setattr(
        harmony_module,
        "_parallel_copy",
        lambda pool, dst, *a: staged.append(len(dst)) or copy(pool, dst, *a),
    )
    monkeypatch.setattr(harmony_module, "_UPLOAD_BYTES", 40 * X[0].nbytes)
    bounds = [0, 300, 650, 1000]
    parts = harmony_module._upload_sorted(
        np.asarray(X, order=layout), cp.asarray(order), bounds, [0, 0, 0]
    )
    assert sum(staged) == len(X)
    for part, lo, hi in zip(parts, bounds[:-1], bounds[1:], strict=True):
        np.testing.assert_array_equal(cp.asnumpy(part), X[order[lo:hi]])


def test_harmonize_checks_out_first():
    # A wrong output, or none with several GPUs, fails before any GPU work.
    adata = _repeatability_adata(np.float32)
    X = adata.obsm["X_pca"]
    for out in (
        np.empty((len(X) - 1, X.shape[1]), X.dtype),
        np.empty_like(X, np.float64),
    ):
        with pytest.raises(ValueError, match="out must be"):
            harmony_module.harmonize(X, adata.obs, "batch", out=out)
    with pytest.raises(ValueError, match="host array"):
        harmony_module.harmonize(X, adata.obs, "batch", devices=[0, 0])


def test_harmonize_failure_joins_the_output_prefault(monkeypatch):
    # The thread zeroing `out` is joined when harmonize raises too, so it
    # never writes the caller's array afterwards.
    raised = threading.Event()

    class Late(np.ndarray):
        def fill(self, value):
            raised.wait(10)  # from the failure on, however long setup took
            time.sleep(0.2)
            super().fill(value)

    def fail(*args, **kwargs):
        raised.set()
        raise RuntimeError("upload")

    adata = _repeatability_adata(np.float32)
    X = adata.obsm["X_pca"]
    out = np.ones_like(X).view(Late)
    monkeypatch.setattr(harmony_module, "_upload_sorted", fail)
    with pytest.raises(RuntimeError, match="upload"):
        harmony_module.harmonize(X, adata.obs, "batch", out=out)
    assert not out.any()


@pytest.mark.parametrize("devices", [[0, 0], [0, 0, 0]])
@pytest.mark.parametrize("where", ["draw", "thread"])
def test_harmony_multi_gpu_reports_the_failing_gpu(monkeypatch, devices, where):
    # An error on the other GPUs reaches the caller, not the first GPU's
    # 'another GPU failed'. A GPU thread that cannot start stops the others.
    class Failure(Exception):
        pass

    draw = harmony_module._draw_blocks

    def draw_or_fail(*args, first, **kwargs):
        if first:
            raise Failure
        return draw(*args, first=first, **kwargs)

    class Pool(ThreadPoolExecutor):
        def submit(self, fn, *args):
            if fn.__name__ == "guarded" and args[0] == len(devices) - 1:
                raise Failure
            return super().submit(fn, *args)

    if where == "draw":
        monkeypatch.setattr(harmony_module, "_draw_blocks", draw_or_fail)
    else:
        monkeypatch.setattr(harmony_module, "ThreadPoolExecutor", Pool)
    threads = threading.active_count()
    X = _repeatability_adata(np.float32).obsm["X_pca"]
    with pytest.raises(Failure):
        harmony_module.harmonize(
            X,
            _repeatability_adata(np.float32).obs,
            "batch",
            rng=734,
            n_clusters=7,
            devices=devices,
            out=np.empty_like(X),
        )
    assert threading.active_count() == threads


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmonize_into_its_input():
    adata = _repeatability_adata(np.float32)
    X, kwargs = adata.obsm["X_pca"], {"rng": 734, "n_clusters": 20}
    ref = cp.asnumpy(harmony_module.harmonize(X, adata.obs, "batch", **kwargs))
    Z = X.copy()
    harmony_module.harmonize(Z, adata.obs, "batch", out=Z, **kwargs)
    assert Z.tobytes() == ref.tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("correction_method", ["batched", "fast"])
def test_harmonize_on_another_gpu(correction_method):
    # devices=[1] called from GPU 0 returns the result on GPU 1.
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two GPUs")
    adata = _repeatability_adata(np.float32)
    X = adata.obsm["X_pca"]
    kwargs = {"rng": 734, "n_clusters": 20, "correction_method": correction_method}
    ref = cp.asnumpy(harmony_module.harmonize(X, adata.obs, "batch", **kwargs))
    with cp.cuda.Device(0):
        other = harmony_module.harmonize(X, adata.obs, "batch", devices=[1], **kwargs)
    assert other.device.id == 1
    assert cp.asnumpy(other).tobytes() == ref.tobytes()


def test_harmony_device_input_shards_hold_only_their_rows():
    X = cp.random.default_rng(0).standard_normal((1000, 8), dtype=cp.float32)
    order = cp.asarray(np.random.default_rng(1).permutation(1000))
    parts = harmony_module._upload_sorted(X, order, [0, 400, 1000], [0, 0])
    assert all(p.data.mem.size < X.nbytes for p in parts)
    assert (cp.concatenate(parts) == X[order]).all()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("peer", [False, True])
def test_harmony_one_gpu_takes_device_input_from_another(monkeypatch, peer):
    # One GPU moves CuPy input from another GPU in chunks, by peer copy where
    # validated: the input's GPU holds a chunk, not a sorted copy.
    if cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two GPUs")
    if peer and not harmony_module._peer_copies_work([0, 1]):
        pytest.skip("no validated peer copies")
    import rapids_singlecell.preprocessing._harmony._multi_gpu as multi_gpu_module

    adata = _repeatability_adata(np.float32, n_pcs=300)
    X, kwargs = adata.obsm["X_pca"], {"rng": 734, "n_clusters": 7}
    rsc.pp.harmony_integrate(adata, "batch", **kwargs)
    expected = adata.obsm["X_pca_harmony"]
    with cp.cuda.Device(1):
        adata.obsm["X_pca"] = cp.asarray(X)
    peer_copies, p2p = [], cp.cuda.runtime.memcpyPeerAsync
    monkeypatch.setattr(
        cp.cuda.runtime,
        "memcpyPeerAsync",
        lambda *a: peer_copies.append(a[4]) or p2p(*a),
    )
    monkeypatch.setattr(multi_gpu_module, "_FORCE_HOST_COPIES", not peer)
    monkeypatch.setattr(multi_gpu_module, "_RELAY_BYTES", 1000)
    monkeypatch.setattr(harmony_module, "_UPLOAD_BYTES", X.nbytes // 4)
    allocator, pool = cp.cuda.get_allocator(), cp.cuda.MemoryPool()
    cp.cuda.set_allocator(pool.malloc)
    try:
        with cp.cuda.Device(1):
            pool.set_limit(size=X.nbytes // 2)
        with cp.cuda.Device(0):
            rsc.pp.harmony_integrate(adata, "batch", **kwargs)
    finally:
        cp.cuda.set_allocator(allocator)
    assert adata.obsm["X_pca_harmony"].tobytes() == expected.tobytes()
    assert sum(peer_copies) >= X.nbytes if peer else not peer_copies


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_shards_write_their_own_rows(monkeypatch):
    # Every GPU copies its own result rows (in input order) into the host
    # output: no rows move between the GPUs, so every row crosses PCIe once,
    # also without peer copies.
    adata = _repeatability_adata(np.float32)
    X, options = adata.obsm["X_pca"], {"rng": 734, "n_clusters": 7}
    one = cp.asnumpy(harmony_module.harmonize(X, adata.obs, "batch", **options))
    write = harmony_module._write_shards

    def write_without_copies(*args, **kwargs):
        monkeypatch.setattr(
            harmony_module, "_copy", lambda *_, **__: pytest.fail("rows moved")
        )
        write(*args, **kwargs)

    monkeypatch.setattr(harmony_module, "_write_shards", write_without_copies)
    out = np.empty_like(X)
    harmony_module.harmonize(X, adata.obs, "batch", devices=[0, 0], out=out, **options)
    assert out.tobytes() == one.tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_releases_shards_before_the_output(monkeypatch):
    # The input shard and the result (replaced by its rows in input order) are
    # released before the output is copied to the host.
    refs, upload = [], harmony_module._upload_sorted
    write, download = harmony_module._write_shards, harmony_module._download

    def upload_tracked(*args, **kwargs):
        parts = upload(*args, **kwargs)
        refs.extend(map(weakref.ref, parts))
        return parts

    def write_tracked(shards, *args, **kwargs):
        refs.extend(map(weakref.ref, shards))
        write(shards, *args, **kwargs)

    def download_checked(*args, **kwargs):
        assert all(ref() is None for ref in refs)
        download(*args, **kwargs)

    monkeypatch.setattr(harmony_module, "_upload_sorted", upload_tracked)
    monkeypatch.setattr(harmony_module, "_write_shards", write_tracked)
    monkeypatch.setattr(harmony_module, "_download", download_checked)
    adata = _repeatability_adata(np.float32)
    X = adata.obsm["X_pca"]
    harmony_module.harmonize(X, adata.obs, "batch", rng=734, out=np.empty_like(X))
    assert len(refs) == 2


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_kmeans_beside_the_clustering_buffers(monkeypatch):
    # The clustering buffers exist before the k-means initialization, whose
    # many temporaries are then allocated faster (non-pooled allocators).
    calls = []
    for name in ("_allocate_clustering_workspace", "_kmeans"):
        monkeypatch.setattr(
            harmony_module,
            name,
            lambda *a, f=getattr(harmony_module, name), name=name, **k: (
                calls.append(name) or f(*a, **k)
            ),
        )
    adata = _repeatability_adata(np.float32)
    rsc.pp.harmony_integrate(adata, "batch", rng=734, max_iter_harmony=1)
    assert calls == ["_allocate_clustering_workspace", "_kmeans"]


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("other_gpu", [False, True])
def test_harmony_threads_wait_for_the_callers_stream(monkeypatch, other_gpu):
    # The caller works on a non-blocking stream that a sleeping host callback
    # holds up before the cell order and the centroids are written: the
    # upload and GPU threads, on their own streams, must still read them. A
    # memory pool keeps cudaFree from synchronizing the device, which would
    # hide a missing wait. With ``other_gpu`` both GPU threads run on GPU 1
    # for a caller on GPU 0, which holds the centroids.
    if other_gpu and cp.cuda.runtime.getDeviceCount() < 2:
        pytest.skip("requires two GPUs")
    devices = list(range(min(2, cp.cuda.runtime.getDeviceCount())))
    rng = np.random.default_rng(11)
    n_cells = 6000
    obs = pd.DataFrame(
        {"batch": rng.integers(0, 5, n_cells).astype(str)},
        index=np.arange(n_cells).astype(str),
    ).astype("category")
    X = rng.standard_normal((n_cells, 17)).astype(np.float32)
    options = {"rng": 734, "n_clusters": 20, "max_iter_harmony": 2}

    def run(data):
        if other_gpu:
            out = np.empty_like(data)
            harmony_module.harmonize(
                data, obs, "batch", devices=[1, 1], out=out, **options
            )
            return out
        adata = ad.AnnData(obs=obs, obsm={"X_pca": data.copy()})
        rsc.pp.harmony_integrate(adata, "batch", multi_gpu=devices, **options)
        return adata.obsm["X_pca_harmony"]

    def late(x):
        held = cp.full_like(x, -1)  # what other streams see until the callback returns
        cp.cuda.get_current_stream().launch_host_func(lambda _: time.sleep(0.2), None)
        held[...] = x
        return held

    allocator = cp.cuda.get_allocator()
    cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
    try:
        run(rng.standard_normal(X.shape).astype(np.float32))  # pools hold other rows
        shard_rows, centroids = (
            harmony_module._shard_rows,
            harmony_module._kmeans_centroids,
        )
        with monkeypatch.context() as m:
            m.setattr(
                harmony_module,
                "_shard_rows",
                lambda order, *a, **k: shard_rows(late(order), *a, **k),
            )
            m.setattr(
                harmony_module,
                "_kmeans_centroids",
                lambda *a, **k: late(centroids(*a, **k)),
            )
            with cp.cuda.Stream(non_blocking=True):
                result = run(X)
        assert result.tobytes() == run(X).tobytes()
    finally:
        cp.cuda.set_allocator(allocator)
