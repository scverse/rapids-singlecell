from __future__ import annotations

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
from rapids_singlecell.preprocessing._harmony import (
    _SUPPRESS_PENALTY,
    _compute_lambda_kb,
    _correction_multi,
    _solve_spd_batched,
)
from rapids_singlecell.preprocessing._harmony._helper import (
    _choose_colsum_algo_benchmark,
    _choose_colsum_algo_heuristic,
    _colsum_heuristic,
    _factorize_joint_codes,
    _get_batch_codes,
    _get_theta_array,
    _scatter_add_cp,
)


def _get_measure(x, base, norm):
    assert norm in ["r", "L2"]

    if norm == "r":
        corr, _ = pearsonr(x, base)
        return corr
    else:
        return np.linalg.norm(x - base) / np.linalg.norm(base)


_HARMONY_DATA_BASE = (
    "https://scverse-exampledata.s3.amazonaws.com/rapids-singlecell/harmony_data"
)
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
            "sex": pd.Categorical(
                ["f", "m", "f", "m", "f", "m"], categories=["f", "m"]
            ),
        }
    )

    codes, n_levels = _get_batch_codes(obs, ["batch", "sex"])

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
    cp.testing.assert_array_equal(
        _get_theta_array(expanded_theta, n_levels, cp.float32), expanded_theta
    )


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


def test_harmony_joint_code_overflow_fallback_is_one_dimensional():
    n_covariates = 64
    batch_codes = np.stack(
        (
            np.zeros(n_covariates, dtype=np.int32),
            np.ones(n_covariates, dtype=np.int32),
            np.zeros(n_covariates, dtype=np.int32),
        )
    )

    joint_cats, joint_codes = _factorize_joint_codes(
        batch_codes, np.full(n_covariates, 2, dtype=np.int32)
    )

    assert joint_cats.shape == (2, n_covariates)
    assert joint_codes.shape == (3,)
    np.testing.assert_array_equal(joint_codes, np.array([0, 1, 0]))


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_harmony_multikey_singular_gram_uses_least_squares(dtype):
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
    atol = 1e-5 if dtype == cp.float32 else 1e-12
    cp.testing.assert_allclose(result, expected, atol=atol, rtol=atol)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony1_multikey_zero_ridge_is_finite():
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

    rsc.pp.harmony_integrate(
        adata,
        ["batch", "duplicate_batch"],
        flavor="harmony1",
        ridge_lambda=0.0,
        n_clusters=3,
        max_iter_harmony=1,
        max_iter_clustering=2,
        block_proportion=1.0,
        random_state=734,
        dtype=cp.float32,
    )

    assert np.isfinite(adata.obsm["X_pca_harmony"]).all()


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("n_covariates", [2, 3, 4])
def test_harmony_multikey_correction_matches_dense_design(
    dtype, n_covariates, monkeypatch
):
    rng = np.random.default_rng(734)
    levels = np.arange(2, 2 + n_covariates, dtype=np.int32)
    offsets = np.concatenate(([0], np.cumsum(levels)[:-1])).astype(np.int32)
    n_batches = int(levels.sum())
    n_cells, n_pcs, n_clusters = 31, 5, 3
    np_dtype = np.dtype(dtype)

    local_codes = np.column_stack(
        [rng.integers(0, level, size=n_cells) for level in levels]
    ).astype(np.int32)
    cats_np = local_codes + offsets
    X_np = rng.normal(size=(n_cells, n_pcs)).astype(np_dtype)
    R_np = rng.random(size=(n_cells, n_clusters)).astype(np_dtype)
    R_np /= R_np.sum(axis=1, keepdims=True)

    O_np = np.zeros((n_batches, n_clusters), dtype=np_dtype)
    for covariate in range(n_covariates):
        np.add.at(O_np, cats_np[:, covariate], R_np)
    lambda_np = rng.uniform(0.2, 1.0, size=O_np.shape).astype(np_dtype)
    lambda_np[1, 1] = np_dtype.type(_SUPPRESS_PENALTY)

    joint_cats_np, joint_codes_np = np.unique(cats_np, axis=0, return_inverse=True)
    cats = cp.asarray(cats_np, dtype=cp.int32)

    result = _correction_multi(
        cp.asarray(X_np),
        cp.asarray(R_np),
        O=cp.asarray(O_np),
        lambda_kb=cp.asarray(lambda_np),
        cats=cats,
        n_batches=n_batches,
        n_covariates=n_covariates,
        joint_cats=cp.asarray(joint_cats_np, dtype=cp.int32),
        joint_codes=cp.asarray(joint_codes_np, dtype=cp.int32),
    )

    design = np.zeros((n_cells, n_batches + 1), dtype=np_dtype)
    design[:, 0] = 1
    for covariate in range(n_covariates):
        design[np.arange(n_cells), cats_np[:, covariate] + 1] = 1

    expected = X_np.copy()
    for cluster in range(n_clusters):
        active = lambda_np[:, cluster] < np_dtype.type(_SUPPRESS_PENALTY)
        retained = np.concatenate(([True], active))
        weighted_design = R_np[:, cluster, None] * design
        gram = design.T @ weighted_design
        gram[1:, 1:] += np.diag(
            np.where(active, lambda_np[:, cluster], np_dtype.type(0))
        )
        rhs = weighted_design.T @ X_np

        W = np.zeros((n_batches + 1, n_pcs), dtype=np_dtype)
        retained_indices = np.flatnonzero(retained)
        W[retained] = np.linalg.solve(
            gram[np.ix_(retained_indices, retained_indices)], rhs[retained]
        )
        W[0] = 0
        expected -= R_np[:, cluster, None] * (design @ W)

    atol = 2e-5 if dtype == cp.float32 else 1e-11
    cp.testing.assert_allclose(result, cp.asarray(expected), atol=atol, rtol=atol)

    monkeypatch.setattr(
        harmony_module,
        "_multi_correction_cluster_chunk_size",
        lambda **_kwargs: 1,
    )

    chunked = _correction_multi(
        cp.asarray(X_np),
        cp.asarray(R_np),
        O=cp.asarray(O_np),
        lambda_kb=cp.asarray(lambda_np),
        cats=cats,
        n_batches=n_batches,
        n_covariates=n_covariates,
        joint_cats=cp.asarray(joint_cats_np, dtype=cp.int32),
        joint_codes=cp.asarray(joint_codes_np, dtype=cp.int32),
    )
    cp.testing.assert_allclose(chunked, cp.asarray(expected), atol=atol, rtol=atol)


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
    adata = adata_harmonypy2_multikey.copy()
    reference = adata.obsm[f"harmony2_ref_{case}"].copy()

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
        random_state=734,
        alpha=0.2,
        batch_prune_threshold=1e-5,
    )

    result = adata.obsm["X_pca_harmony"]
    assert _get_measure(reference, result, "r").min() > 0.95
    assert _get_measure(reference, result, "L2").max() < 0.1


@pytest.fixture(scope="module")
def adata_harmonypy2_multikey():
    filename, known_hash = _HARMONYPY2_MULTIKEY_H5AD
    reference_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/{filename}", known_hash=known_hash
    )
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
    reference_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/{filename}", known_hash=known_hash
    )
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
        rsc.pp.harmony_integrate(
            adata, "bulk_labels", batch_prune_threshold=bad_threshold
        )


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


@pytest.mark.parametrize("algo", ["columns", "atomics", "gemm"])
@pytest.mark.parametrize("dtype", [cp.float32, cp.float64, cp.int32])
def test_colsum_algo(algo, dtype):
    # Int32 testing for correctness of the algorithm
    if dtype == cp.int32:
        X = cp.random.randint(0, 10, size=(20, 10), dtype=dtype)
    else:
        X = cp.random.randn(20, 10, dtype=dtype)
    algo_func = _choose_colsum_algo_heuristic(X.shape[0], X.shape[1], algo)
    algo_out = algo_func(X)
    cupy_out = X.sum(axis=0)
    if dtype == cp.int32:
        cp.testing.assert_array_equal(algo_out, cupy_out)
    elif dtype == cp.float32:
        cp.testing.assert_allclose(algo_out, cupy_out, atol=1e-5)
    else:
        cp.testing.assert_allclose(algo_out, cupy_out)


@pytest.mark.parametrize("compute_capability", ["100", "80"])
def test_choose_colsum_algo(compute_capability):
    # Test that the choose_colsum_algo function returns the correct algorithm
    # for the given shape of the matrix
    for rows in np.arange(1000, 300000, 1000):
        for columns in np.arange(10, 5000, 50):
            algo = _colsum_heuristic(rows, columns, compute_capability)
            assert algo in ["columns", "atomics", "gemm"]


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
def test_benchmark_colsum_algorithms(dtype):
    # Test that the benchmark_colsum_algorithms function returns the correct algorithm
    # for the given shape of the matrix
    test_shape = (1000, 100)
    algo_func = _choose_colsum_algo_benchmark(test_shape[0], test_shape[1], dtype)
    assert callable(algo_func)


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize("column", ["gemm", "columns", "atomics"])
@pytest.mark.parametrize("correction_method", ["fast", "batched"])
def test_harmony_integrate_reference(
    adata_reference, *, dtype, column, correction_method
):
    """
    Test that Harmony integrate works.
    """
    adata = adata_reference.copy()
    rsc.pp.harmony_integrate(
        adata,
        "donor",
        correction_method=correction_method,
        dtype=dtype,
        colsum_algo=column,
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


@pytest.mark.parametrize("n_cells", [1000, 60000])
@pytest.mark.parametrize("n_pcs", [20, 50])
@pytest.mark.parametrize("n_batches", [3, 10])
@pytest.mark.parametrize("switcher", [0, 1])
def test_scatter_add_shared_vs_optimized(n_cells, n_pcs, n_batches, switcher):
    """
    Test that shared memory and non-shared scatter add kernels produce identical results.

    Uses small integer values (as float32) for exact verification of correctness.
    """
    rng = np.random.default_rng(42)
    X_np = rng.integers(1, 10, size=(n_cells, n_pcs)).astype(np.float32)
    cats_np = rng.integers(0, n_batches, size=n_cells, dtype=np.int32)

    X = cp.asarray(X_np)
    cats = cp.asarray(cats_np)

    # Compute expected result using numpy
    expected_np = np.zeros((n_batches, n_pcs), dtype=np.float32)
    for i in range(n_cells):
        cat = cats_np[i]
        if switcher == 1:
            expected_np[cat] += X_np[i]
        else:
            expected_np[cat] -= X_np[i]
    expected = cp.asarray(expected_np)

    # Run optimized (non-shared) kernel via _scatter_add_cp
    out_optimized = cp.zeros((n_batches, n_pcs), dtype=cp.float32)
    _scatter_add_cp(X, out_optimized, cats, switcher, n_batches, use_shared=False)

    # Run shared memory kernel via _scatter_add_cp
    out_shared = cp.zeros((n_batches, n_pcs), dtype=cp.float32)
    _scatter_add_cp(X, out_shared, cats, switcher, n_batches, use_shared=True)

    # Both kernels should produce identical results
    cp.testing.assert_array_equal(out_optimized, expected)
    cp.testing.assert_array_equal(out_shared, expected)
    cp.testing.assert_array_equal(out_optimized, out_shared)


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
def test_compute_lambda_kb_fixed_ridge_zero(dtype):
    """dynamic_lambda=False with ridge_lambda=0 still guards zero-denominator."""
    sentinel = dtype(_SUPPRESS_PENALTY)
    E = cp.array([[5.0, 0.0]], dtype=dtype)
    O = cp.array([[10.0, 0.0]], dtype=dtype)
    N_b = cp.array([100.0], dtype=dtype)

    result = _compute_lambda_kb(
        E,
        O=O,
        N_b=N_b,
        alpha=0.2,
        threshold=None,
        ridge_lambda=0.0,
        dynamic_lambda=False,
    )
    # (0,0): O=10 + lambda=0 = 10 → no guard, stays 0.0
    assert result[0, 0] == dtype(0.0)
    # (0,1): O=0 + lambda=0 = 0 → sentinel
    assert result[0, 1] == sentinel


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
        # Float64 agreement between correction methods is covered separately.
        (cp.float64, "fast"),
    ],
)
def test_harmony2_ircolitis_reference(
    adata_ircolitis_harmonypy2, correction_method, dtype
):
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
