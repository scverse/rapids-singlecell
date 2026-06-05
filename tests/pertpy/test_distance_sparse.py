"""Parity tests for the sparse (CSR) edistance data path.

The sparse path densifies each feature window inside the kernel instead of
materializing the N x G dense matrix, so its results must match the dense
kernel bit-for-bit (direct squared differences, no norm-plus-dot rewrite).
"""

from __future__ import annotations

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np
import pandas as pd
import pytest
from anndata import AnnData
from scipy.sparse import csc_matrix, csr_matrix
from scipy.spatial.distance import cdist

from rapids_singlecell.pertpy_gpu import Distance

N_GROUPS = 4
CELLS_PER_GROUP = 30
DENSITY = 0.3


def _make_matrix(n_features: int, dtype, *, seed: int = 0) -> np.ndarray:
    """Group-separated, ~``DENSITY``-dense matrix (zeros make it genuinely sparse)."""
    rng = np.random.default_rng(seed)
    blocks = [
        (rng.normal(size=(CELLS_PER_GROUP, n_features)) + i).astype(dtype)
        for i in range(N_GROUPS)
    ]
    X = np.vstack(blocks)
    X[rng.random(X.shape) > DENSITY] = 0.0
    return X


def _obs() -> pd.DataFrame:
    labels = [f"g{i}" for i in range(N_GROUPS) for _ in range(CELLS_PER_GROUP)]
    cats = [f"g{i}" for i in range(N_GROUPS)]
    return pd.DataFrame({"group": pd.Categorical(labels, categories=cats)})


def _dense_adata(X: np.ndarray) -> AnnData:
    adata = AnnData(X.astype(np.float32).copy(), obs=_obs())
    adata.obsm["emb"] = cp.asarray(X, dtype=cp.dtype(X.dtype))
    return adata


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-4), (np.float64, 1e-9)])
@pytest.mark.parametrize("n_features", [40, 200])
def test_pairwise_sparse_matches_dense(dtype, atol, n_features) -> None:
    """Sparse X and a sparse layer both match the dense obsm path."""
    X = _make_matrix(n_features, dtype)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.astype(dtype).copy()), obs=_obs())
    sparse.layers["counts"] = csr_matrix(X.astype(dtype).copy())

    ref = Distance(metric="edistance", obsm_key="emb").pairwise(dense, groupby="group")
    via_x = Distance(metric="edistance", layer_key="X").pairwise(
        sparse, groupby="group"
    )
    via_layer = Distance(metric="edistance", layer_key="counts").pairwise(
        sparse, groupby="group"
    )

    np.testing.assert_allclose(via_x.values, ref.values, atol=atol)
    np.testing.assert_allclose(via_layer.values, ref.values, atol=atol)


def test_pairwise_sparse_matches_cdist_ground_truth() -> None:
    """Energy distance from the sparse kernel matches a CPU cdist computation."""
    X = _make_matrix(30, np.float64, seed=1)
    expected = np.zeros((N_GROUPS, N_GROUPS))
    for i in range(N_GROUPS):
        a = X[i * CELLS_PER_GROUP : (i + 1) * CELLS_PER_GROUP]
        for j in range(N_GROUPS):
            if i == j:
                continue
            b = X[j * CELLS_PER_GROUP : (j + 1) * CELLS_PER_GROUP]
            d_xy = cdist(a, b).mean()
            d_xx = cdist(a, a)[np.triu_indices(len(a), 1)].mean()
            d_yy = cdist(b, b)[np.triu_indices(len(b), 1)].mean()
            expected[i, j] = 2 * d_xy - d_xx - d_yy

    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())
    result = Distance(metric="edistance", layer_key="X").pairwise(
        sparse, groupby="group"
    )
    np.testing.assert_allclose(result.values, expected, atol=1e-9)


def test_onesided_sparse_matches_dense() -> None:
    X = _make_matrix(50, np.float32)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())

    ref = Distance(metric="edistance", obsm_key="emb").onesided_distances(
        dense, groupby="group", selected_group="g0"
    )
    got = Distance(metric="edistance", layer_key="X").onesided_distances(
        sparse, groupby="group", selected_group="g0"
    )
    np.testing.assert_allclose(got.values, ref.values, atol=1e-4)


def test_subset_of_groups_sparse_matches_dense() -> None:
    X = _make_matrix(40, np.float32)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())

    ref = Distance(metric="edistance", obsm_key="emb").pairwise(
        dense, groupby="group", groups=["g0", "g2"]
    )
    got = Distance(metric="edistance", layer_key="X").pairwise(
        sparse, groupby="group", groups=["g0", "g2"]
    )
    np.testing.assert_allclose(got.values, ref.values, atol=1e-4)


def test_contrast_distances_sparse_matches_dense() -> None:
    X = _make_matrix(40, np.float32)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())

    contrasts = Distance.create_contrasts(dense, groupby="group", selected_group="g0")
    ref = Distance(metric="edistance", obsm_key="emb").contrast_distances(
        dense, contrasts
    )
    got = Distance(metric="edistance", layer_key="X").contrast_distances(
        sparse, contrasts
    )
    np.testing.assert_allclose(
        got["edistance"].values, ref["edistance"].values, atol=1e-4
    )


def test_bootstrap_sparse_matches_dense() -> None:
    """Bootstrap resamples cell_indices, which is layout-agnostic; same seed ->
    identical resampling, so sparse and dense must agree."""
    X = _make_matrix(40, np.float32)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())

    ref_mean, ref_var = Distance(metric="edistance", obsm_key="emb").pairwise(
        dense, groupby="group", bootstrap=True, n_bootstrap=20, random_state=7
    )
    got_mean, got_var = Distance(metric="edistance", layer_key="X").pairwise(
        sparse, groupby="group", bootstrap=True, n_bootstrap=20, random_state=7
    )
    np.testing.assert_allclose(got_mean.values, ref_mean.values, atol=1e-4)
    np.testing.assert_allclose(got_var.values, ref_var.values, atol=1e-4)


def test_integer_counts_sparse_matches_dense() -> None:
    """Raw integer-count CSR (a common ``adata.X``) is cast to float32 by the
    loader before reaching the kernel."""
    rng = np.random.default_rng(5)
    counts = rng.poisson(0.5, size=(N_GROUPS * CELLS_PER_GROUP, 40)).astype(np.int32)
    dense = _dense_adata(counts.astype(np.float32))
    sparse = AnnData(csr_matrix(counts.copy()), obs=_obs())

    ref = Distance(metric="edistance", obsm_key="emb").pairwise(dense, groupby="group")
    got = Distance(metric="edistance", layer_key="X").pairwise(sparse, groupby="group")
    np.testing.assert_allclose(got.values, ref.values, atol=1e-4)


@pytest.mark.parametrize("fmt", ["csc", "gpu_csr"])
def test_non_csr_sparse_inputs(fmt) -> None:
    """CSC (host) and an already-on-GPU cupyx CSR are normalized to canonical CSR."""
    X = _make_matrix(30, np.float64, seed=1)
    dense = _dense_adata(X)
    adata = AnnData(X.astype(np.float64).copy(), obs=_obs())
    if fmt == "csc":
        adata.layers["sp"] = csc_matrix(X.copy())
    else:
        adata.layers["sp"] = cpsp.csr_matrix(cp.asarray(X))

    ref = Distance(metric="edistance", obsm_key="emb").pairwise(dense, groupby="group")
    got = Distance(metric="edistance", layer_key="sp").pairwise(adata, groupby="group")
    np.testing.assert_allclose(got.values, ref.values, atol=1e-9)
