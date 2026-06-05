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
from rapids_singlecell._utils import _create_category_index_mapping
from rapids_singlecell.preprocessing._harmony import (
    _SUPPRESS_PENALTY,
    _compute_lambda_kb,
)
from rapids_singlecell.preprocessing._harmony._helper import (
    _choose_colsum_algo_benchmark,
    _choose_colsum_algo_heuristic,
    _colsum_heuristic,
    _scatter_add_cp,
    _scatter_add_cp_bias_csr,
)


def _get_measure(x, base, norm):
    assert norm in ["r", "L2"]

    if norm == "r":
        corr, _ = pearsonr(x, base)
        return corr
    else:
        return np.linalg.norm(x - base) / np.linalg.norm(base)


_HARMONY_DATA_BASE = "https://exampledata.scverse.org/rapids-singlecell/harmony_data"


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
    adata = ad.AnnData(
        X=None,
        obs=meta,
        obsm={"X_pca": X_pca.values, "harmony_org": X_pca_harmony.values},
    )
    return adata


@pytest.fixture(scope="module")
def adata_ircolitis_harmony2():
    """IRcolitis blood CD8 dataset (68k cells) with Harmony2 (R) reference output."""
    pcs_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/ircolitis_blood_cd8_pcs.tsv.gz",
        known_hash="md5:9f28afa68ed4e1fd465d53d360b58a35",
    )
    pcs = pd.read_csv(pcs_file, delimiter="\t")
    harmony2_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/ircolitis_blood_cd8_pcs_harmonized.tsv.gz",
        known_hash="md5:848f1e09016c633e044b7d93650505ba",
    )
    h2 = pd.read_csv(harmony2_file, delimiter="\t")
    obs_file = pooch.retrieve(
        f"{_HARMONY_DATA_BASE}/ircolitis_blood_cd8_obs.tsv.gz",
        known_hash="md5:46efe419e59450504c3d3b343eff8022",
    )
    obs = pd.read_csv(obs_file, delimiter="\t", low_memory=False)

    X_pca = pcs.drop(columns=["cell_barcode"]).values
    X_harmony2 = h2.drop(columns=["cell_barcode"]).values

    adata = ad.AnnData(
        X=None,
        obs=obs,
        obsm={"X_pca": X_pca, "harmony2_ref": X_harmony2},
    )
    return adata


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
@pytest.mark.parametrize("correction_method", ["fast", "original", "batched"])
def test_harmony_integrate(correction_method):
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.

    This is a pure shape/contract check: the output shape is independent of
    dtype and iteration count, so we run float32 with a single harmony
    iteration to exercise all three correction-method paths cheaply.
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
def test_harmony_integrate_algos(dtype):
    """
    Test that Harmony integrate works.

    This is a very simple test that just checks to see if the Harmony
    integrate wrapper successfully added a new field to ``adata.obsm``
    and makes sure it has the same dimensions as the original PCA table.
    """
    adata = sc.datasets.pbmc68k_reduced()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method="fast", dtype=dtype
    )
    fast = adata.obsm["X_pca_harmony"].copy()
    rsc.pp.harmony_integrate(
        adata, "bulk_labels", correction_method="original", dtype=dtype
    )
    slow = adata.obsm["X_pca_harmony"].copy()
    assert _get_measure(fast, slow, "r").min() > 0.99
    assert _get_measure(fast, slow, "L2").max() < 0.1


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
@pytest.mark.parametrize("correction_method", ["fast", "original", "batched"])
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


@pytest.mark.filterwarnings("ignore:Harmony did not converge")
@pytest.mark.parametrize("correction_method", ["original", "batched"])
@pytest.mark.parametrize("dtype", [cp.float64, cp.float32])
def test_harmony2_correction_methods_agree(
    adata_reference, *, dtype, correction_method
):
    """Harmony2 default path: correction methods produce consistent results."""
    adata = adata_reference.copy()
    rsc.pp.harmony_integrate(
        adata,
        "donor",
        correction_method=correction_method,
        dtype=dtype,
        max_iter_harmony=5,
    )
    h2 = adata.obsm["X_pca_harmony"]

    # Run the reference method (fast) for comparison
    adata_ref = adata_reference.copy()
    rsc.pp.harmony_integrate(
        adata_ref,
        "donor",
        correction_method="fast",
        dtype=dtype,
        max_iter_harmony=5,
    )
    h2_ref = adata_ref.obsm["X_pca_harmony"]

    assert _get_measure(h2, h2_ref, "r").min() > 0.99
    assert _get_measure(h2, h2_ref, "L2").max() < 0.05


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
@pytest.mark.parametrize("n_pcs", [3, 4, 5])
@pytest.mark.parametrize("offset", [0, 1])
@pytest.mark.parametrize("n_cells", [0, 1, 23, 1024])
def test_scatter_add_bias_csr_alignment(dtype, n_pcs, offset, n_cells):
    rng = np.random.default_rng(42)
    n_batches = 4
    X_base_np = rng.normal(size=n_cells * n_pcs + offset).astype(np.dtype(dtype))
    X_np = X_base_np[offset:].reshape(n_cells, n_pcs)
    bias_np = rng.random(n_cells).astype(np.dtype(dtype))
    cats_np = rng.integers(0, n_batches, size=n_cells, dtype=np.int32)

    X_base = cp.asarray(X_base_np)
    X = X_base[offset:].reshape(n_cells, n_pcs)
    bias = cp.asarray(bias_np)
    cats = cp.asarray(cats_np)
    cat_offsets, cell_indices = _create_category_index_mapping(cats, n_batches)

    out = cp.zeros((n_batches + 1, n_pcs), dtype=dtype)
    _scatter_add_cp_bias_csr(
        X,
        out,
        cat_offsets=cat_offsets,
        cell_indices=cell_indices,
        bias=bias,
        n_batches=n_batches,
    )
    cp.cuda.Device().synchronize()

    expected_np = np.zeros((n_batches + 1, n_pcs), dtype=np.dtype(dtype))
    expected_np[0] = X_np.T @ bias_np
    for batch in range(n_batches):
        mask = cats_np == batch
        expected_np[batch + 1] = X_np[mask].T @ bias_np[mask]

    atol = 1e-5 if dtype == cp.float32 else 1e-12
    cp.testing.assert_allclose(out, cp.asarray(expected_np), atol=atol, rtol=atol)


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
        (cp.float32, "original"),
        (cp.float32, "batched"),
        # float64 numeric reference for `fast` only: float64 original/batched
        # agreement with `fast` is covered by test_harmony2_correction_methods_agree
        (cp.float64, "fast"),
    ],
)
def test_harmony2_ircolitis_reference(
    adata_ircolitis_harmony2, correction_method, dtype
):
    """Harmony2 on IRcolitis (68k cells, 11 batches) matches R harmony2 reference."""
    adata = adata_ircolitis_harmony2.copy()
    rsc.pp.harmony_integrate(
        adata,
        "batch",
        correction_method=correction_method,
        dtype=dtype,
        max_iter_harmony=10,
    )

    ref = adata.obsm["harmony2_ref"]
    result = adata.obsm["X_pca_harmony"]

    assert _get_measure(ref, result, "r").min() > 0.95
    assert _get_measure(ref, result, "L2").max() < 0.1
