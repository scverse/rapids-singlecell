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
    _correction,
    _correction_multi,
    _segments,
    _solve_spd_batched,
)
from rapids_singlecell.preprocessing._harmony._helper import (
    _factorize_joint_codes,
    _get_batch_codes,
    _get_theta_array,
    _stratified_sample_indices,
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
        sampled = cp.asnumpy(
            _stratified_sample_indices(
                gpu_offsets, gpu_indices, n_target, np.random.default_rng(17)
            )
        )
        repeated = cp.asnumpy(
            _stratified_sample_indices(
                gpu_offsets, gpu_indices, n_target, np.random.default_rng(17)
            )
        )

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

    sampled = cp.asnumpy(
        _stratified_sample_indices(offsets, cell_indices, 9, np.random.default_rng(0))
    )
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


def _dense_design_correction(X, R, cats, lambda_kb, n_batches):
    """Harmony correction with an explicit intercept + one-hot design matrix;
    pruned levels (lambda at the sentinel) are dropped from each solve."""
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
        W[retained] = np.linalg.solve(
            gram[np.ix_(retained, retained)], (weighted_design.T @ X)[retained]
        )
        W[0] = 0
        expected -= R[:, cluster, None] * (design @ W)
    return expected


@pytest.mark.parametrize("dtype", [cp.float32, cp.float64])
@pytest.mark.parametrize(
    ("method", "normalize"),
    [("batched", False), ("batched", True), ("fast", False)],
)
def test_harmony_correction_matches_dense_design(dtype, method, normalize):
    rng = np.random.default_rng(734)
    n_batches, n_pcs, n_clusters = 4, 5, 3
    np_dtype = np.dtype(dtype)
    # Cells sorted by batch; batch 2 is empty.
    cats_np = np.sort(rng.choice([0, 1, 3], size=40)).astype(np.int32)
    X_np = rng.normal(size=(40, n_pcs)).astype(np_dtype)
    R_np = rng.random(size=(40, n_clusters)).astype(np_dtype)
    R_np /= R_np.sum(axis=1, keepdims=True)
    O_np = np.zeros((n_batches, n_clusters), dtype=np_dtype)
    np.add.at(O_np, cats_np, R_np)
    lambda_np = rng.uniform(0.2, 1.0, size=O_np.shape).astype(np_dtype)
    lambda_np[2] = np_dtype.type(_SUPPRESS_PENALTY)
    lambda_np[1, 1] = np_dtype.type(_SUPPRESS_PENALTY)
    offsets = np.searchsorted(cats_np, np.arange(n_batches + 1)).astype(np.int32)
    bounds = (np.diff(offsets) * np.abs(X_np).max()).tolist()

    result = _correction(
        cp.asarray(X_np),
        R=cp.asarray(R_np),
        O=cp.asarray(O_np),
        lambda_kb=cp.asarray(lambda_np),
        correction_method=method,
        cats=cp.asarray(cats_np),
        n_batches=n_batches,
        cat_offsets=cp.asarray(offsets),
        cell_indices=cp.arange(40, dtype=cp.int32),
        normalize=normalize,
        segments=(*_segments(offsets), bounds),
    )

    expected = _dense_design_correction(X_np, R_np, cats_np, lambda_np, n_batches)
    if normalize:
        expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    atol = 2e-5 if dtype == cp.float32 else 1e-11
    cp.testing.assert_allclose(result, cp.asarray(expected), atol=atol, rtol=atol)


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

    # The correction expects cells sorted by joint category.
    joint_cats_np, joint_codes_np = np.unique(cats_np, axis=0, return_inverse=True)
    order = np.argsort(joint_codes_np, kind="stable")
    cats_np, X_np, R_np = cats_np[order], X_np[order], R_np[order]
    joint_codes_np = joint_codes_np[order]
    joint_O_np = np.zeros((len(joint_cats_np), n_clusters), dtype=np_dtype)
    np.add.at(joint_O_np, joint_codes_np, R_np)
    joint_offsets = np.searchsorted(
        joint_codes_np, np.arange(len(joint_cats_np) + 1)
    ).astype(np.int32)
    marginal = sorted(
        (level, joint) for joint, levels in enumerate(joint_cats_np) for level in levels
    )
    marginal_offsets = np.searchsorted(
        [level for level, _ in marginal], np.arange(n_batches + 1)
    ).astype(np.int32)

    def correct(normalize=False):
        return _correction_multi(
            cp.asarray(X_np),
            cp.asarray(R_np),
            O=cp.asarray(O_np),
            lambda_kb=cp.asarray(lambda_np),
            joint_O=cp.asarray(joint_O_np),
            n_batches=n_batches,
            joint_cats=cp.asarray(joint_cats_np, dtype=cp.int32),
            segments=(
                *_segments(joint_offsets),
                (np.diff(joint_offsets) * np.abs(X_np).max()).tolist(),
            ),
            marginal_joint_offsets=cp.asarray(marginal_offsets),
            marginal_joint_indices=cp.asarray(
                [joint for _, joint in marginal], dtype=cp.int32
            ),
            normalize=normalize,
        )

    result = correct()

    expected = _dense_design_correction(X_np, R_np, cats_np, lambda_np, n_batches)
    atol = 2e-5 if dtype == cp.float32 else 1e-11
    cp.testing.assert_allclose(result, cp.asarray(expected), atol=atol, rtol=atol)

    monkeypatch.setattr(
        harmony_module,
        "_multi_correction_cluster_chunk_size",
        lambda **_kwargs: 1,
    )

    chunked = correct()
    cp.testing.assert_allclose(chunked, cp.asarray(expected), atol=atol, rtol=atol)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    cp.testing.assert_allclose(
        correct(normalize=True), cp.asarray(expected), atol=atol, rtol=atol
    )


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

    # 80 cells have several optima that the reference also reaches depending
    # on its seed: one of a few seeds must reproduce the stored result.
    result = min(
        (run(seed) for seed in range(5)),
        key=lambda result: _get_measure(reference, result, "L2").max(),
    )
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


def test_harmony_unseeded_random_state():
    """``random_state=None`` means unseeded, not a crash."""
    rng = np.random.default_rng(734)
    batch = np.resize(["a", "b", "c"], 60)
    adata = ad.AnnData(
        X=None,
        obs=pd.DataFrame(
            {"batch": batch}, index=[f"cell_{index}" for index in range(60)]
        ),
        obsm={"X_pca": rng.normal(size=(60, 6)).astype(np.float32)},
    )

    rsc.pp.harmony_integrate(
        adata, "batch", n_clusters=3, max_iter_harmony=1, random_state=None
    )

    assert np.isfinite(adata.obsm["X_pca_harmony"]).all()


def _repeatability_adata(dtype):
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
            },
            index=ordinal.astype(str),
        ),
        obsm={"X_pca": rng.standard_normal((n_cells, 17)).astype(dtype)},
    )


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    "key", ["batch", ["batch", "second"], ["batch", "second", "third"]]
)
@pytest.mark.parametrize("kmeans_cells_per_cluster", [5000, 1])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_harmony_integrate_repeats_bitwise(
    monkeypatch, key, kmeans_cells_per_cluster, dtype
):
    monkeypatch.setattr(
        harmony_module, "_KMEANS_INIT_CELLS_PER_CLUSTER", kmeans_cells_per_cluster
    )
    outputs = []
    for _ in range(2):
        adata = _repeatability_adata(dtype)
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


def _integrate(dtype=np.float32, *, X_pca=None, key="batch", **kwargs):
    adata = _repeatability_adata(np.float32)
    if X_pca is not None:
        adata.obsm["X_pca"] = X_pca
    rsc.pp.harmony_integrate(
        adata, key, dtype=dtype, rng=734, n_clusters=7, max_iter_harmony=2, **kwargs
    )
    return adata.obsm["X_pca_harmony"]


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_bfloat16_assignments():
    first, second = _integrate("bfloat16"), _integrate("bfloat16")
    assert first.dtype == np.float32
    assert first.tobytes() == second.tobytes()
    assert _get_measure(first, _integrate(), "L2") < 2e-2


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
    ],
)
def test_harmony_large_theta_stays_finite(
    monkeypatch, *, flavor, theta, n_clusters, sigma, kwargs
):
    # theta=200 takes float32 penalties of evenly mixed batches to about
    # 0.5**200, far below the smallest float: log-space penalties keep the
    # assignments and the objective finite (sigma=0.02: shifted per row).
    # harmony1 empties clusters entirely; they get no correction.
    objectives = []
    convergent = harmony_module._is_convergent_harmony
    monkeypatch.setattr(
        harmony_module,
        "_is_convergent_harmony",
        lambda objs, tol: objectives.append(list(objs)) or convergent(objs, tol),
    )
    adata = _repeatability_adata(np.float32)
    rsc.pp.harmony_integrate(
        adata,
        "batch",
        flavor=flavor,
        theta=theta,
        sigma=sigma,
        rng=734,
        n_clusters=n_clusters,
        max_iter_harmony=2,
        **kwargs,
    )
    assert np.isfinite(adata.obsm["X_pca_harmony"]).all()
    assert all(0 < o < 1e6 for o in objectives[-1])


@pytest.mark.parametrize("chunk", [0, -2, 1.5])
def test_harmony_shuffle_chunk_size_rejects_invalid(chunk):
    with pytest.raises(ValueError, match="shuffle_chunk_size"):
        _integrate(shuffle_chunk_size=chunk)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_shuffle_chunk_size():
    exact = _integrate(shuffle_chunk_size=1)
    assert exact.tobytes() == _integrate(shuffle_chunk_size=1).tobytes()
    assert exact.tobytes() != _integrate(shuffle_chunk_size=8).tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_host_input_matches_device_input():
    host = _repeatability_adata(np.float32).obsm["X_pca"]
    assert (
        _integrate(X_pca=host).tobytes() == _integrate(X_pca=cp.asarray(host)).tobytes()
    )
    host[3, 2] = np.nan
    for X in (host, cp.asarray(host)):
        with pytest.raises(ValueError, match="NaN"):
            _integrate(X_pca=X)


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_small_host_input_stages_its_rows(monkeypatch):
    # Pinned and GPU staging chunks hold at most the input's rows.
    rows = []
    pinned = harmony_module._pinned
    monkeypatch.setattr(
        harmony_module,
        "_pinned",
        lambda n, d, dtype: rows.append(n) or pinned(n, d, dtype),
    )
    _integrate(X_pca=_repeatability_adata(np.float32).obsm["X_pca"])
    assert rows and max(rows) <= 480


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
def test_harmony_bfloat16_falls_back_to_float32():
    key = ["batch", "second"]  # bfloat16 assignments need one batch key
    with pytest.warns(UserWarning, match="dtype='bfloat16'"):
        fallback = _integrate("bfloat16", key=key)
    assert fallback.tobytes() == _integrate(key=key).tobytes()


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("key", ["batch", ["batch", "second"]])
def test_harmony_general_assignment_matches_fused(monkeypatch, key):
    def run():
        adata = _repeatability_adata(np.float32)
        rsc.pp.harmony_integrate(adata, key, rng=734, n_clusters=7, max_iter_harmony=2)
        return adata.obsm["X_pca_harmony"]

    fused = run()
    monkeypatch.setattr(harmony_module, "_FORCE_GENERAL_ASSIGNMENT", True)
    general = run()
    assert _get_measure(general, fused, "L2") < 1e-4


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize(
    ("key", "dtype"),
    [("batch", "float32"), ("batch", "bfloat16"), (["batch", "second"], "float32")],
)
def test_harmony_many_clusters(key, dtype):
    # More than 128 clusters take the general kernel in two sweeps.
    outputs = []
    for _ in range(2):
        adata = _repeatability_adata(np.float32)
        rsc.pp.harmony_integrate(
            adata, key, dtype=dtype, rng=734, n_clusters=300, max_iter_harmony=2
        )
        outputs.append(adata.obsm["X_pca_harmony"])
    assert np.isfinite(outputs[0]).all()
    assert outputs[0].tobytes() == outputs[1].tobytes()
