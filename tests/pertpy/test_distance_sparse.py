"""Parity tests for the sparse (CSR) edistance data path.

The sparse path densifies each feature window inside the kernel instead of
materializing the N x G dense matrix. Both paths center the rows alike and
run the same arithmetic, so the results match bit for bit while one block
sums each group pair. Unsorted columns are sorted first and the kernels sum
duplicate entries.
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
from rapids_singlecell.pertpy_gpu._metrics._edistance import _materialize_source

N_GROUPS = 4
CELLS_PER_GROUP = 30
DENSITY = 0.3


def _make_matrix(n_features: int, dtype, *, seed: int = 0) -> np.ndarray:
    """Group-separated, ~``DENSITY``-dense matrix (zeros make it genuinely sparse)."""
    rng = np.random.default_rng(seed)
    blocks = [(rng.normal(size=(CELLS_PER_GROUP, n_features)) + i).astype(dtype) for i in range(N_GROUPS)]
    X = np.vstack(blocks)
    X[rng.random(X.shape) > DENSITY] = 0.0
    return X


def _obs() -> pd.DataFrame:
    labels = [f"g{i}" for i in range(N_GROUPS) for _ in range(CELLS_PER_GROUP)]
    cats = [f"g{i}" for i in range(N_GROUPS)]
    return pd.DataFrame({"group": pd.Categorical(labels, categories=cats)}).rename(str)


def _dense_adata(X: np.ndarray) -> AnnData:
    adata = AnnData(X.astype(np.float32).copy(), obs=_obs())
    adata.obsm["emb"] = cp.asarray(X, dtype=cp.dtype(X.dtype))
    return adata


def _unsorted(X: np.ndarray) -> csr_matrix:
    """``X`` with the columns of each row in descending order."""
    rev = csr_matrix(X[::-1])  # its arrays read backwards are X with descending columns
    return csr_matrix((rev.data[::-1], rev.indices[::-1], rev.nnz - rev.indptr[::-1]), shape=X.shape)


def _halves(m: csr_matrix) -> csr_matrix:
    """Every entry of ``m`` as two of half its value."""
    return csr_matrix((np.repeat(m.data / 2, 2), np.repeat(m.indices, 2), 2 * m.indptr), shape=m.shape)


@pytest.mark.parametrize("precision", ["float32", "float64", "float16"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sparse_equals_dense_bitwise(precision, dtype) -> None:
    """CSR input gives the dense result bit for bit, also with unsorted columns and two duplicate entries per value."""
    X = _make_matrix(72, dtype)  # more than the 64 features of one float16 window of the CSR loader
    X[:, 0], X[0, 0] = 1000, 0  # the scale comes from a zero the CSR input does not store
    kwargs = {"metric": "edistance", "precision": precision}
    csr, unsorted = csr_matrix(X), _unsorted(X)
    host = [unsorted, unsorted.copy(), _halves(csr), _halves(unsorted), unsorted.copy(), unsorted.copy()]
    device = [cpsp.csr_matrix(m) for m in host]
    assert not device[1].has_canonical_format  # cached: still False once its columns are sorted
    for name in ("data", "indices"):  # in every other element of larger buffers
        setattr(device[-2], name, cp.repeat(getattr(device[-2], name), 2)[::2])
    device[-1].data = cp.concatenate([device[-1].data[:1], device[-1].data])[1:]
    assert device[-1].data.data.ptr % 16  # values that do not start on a 16-byte boundary
    for groups in (["g0", "g2"], None):
        ref = Distance(obsm_key="emb", **kwargs).pairwise(_dense_adata(X), groupby="group", groups=groups)
        for matrix in (csr, *device):
            adata = AnnData(matrix, obs=_obs())
            got = Distance(layer_key="X", **kwargs).pairwise(adata, groupby="group", groups=groups)
            assert got.values.dtype == dtype
            np.testing.assert_array_equal(got.values, ref.values)
        # The caller's device matrix keeps its duplicates; a subset of the groups reads it as it is,
        # all groups sort its columns in place, those of a strided or unaligned matrix in a copy
        for m, expected in zip(device, host, strict=True):
            for name in ("data", "indices", "indptr"):
                np.testing.assert_array_equal(getattr(m, name).get(), getattr(expected, name))
            if m.data.flags.c_contiguous and not m.data.data.ptr % 16:
                expected.sort_indices()


@pytest.mark.parametrize("selector", [None, np.arange(0, 60, 2)])
def test_duplicates_flag(selector) -> None:
    """Only a matrix with duplicates takes the summing loop; an unsorted one is asked after the sort."""
    X = _make_matrix(72, np.float32)
    csr, unsorted = csr_matrix(X), _unsorted(X)

    def flag(m):
        return _materialize_source(m, selector, "float32").duplicates

    assert not flag(csr) and not flag(cpsp.csr_matrix(csr))
    assert flag(_halves(csr)) and flag(cpsp.csr_matrix(_halves(csr)))
    if selector is None:
        assert not flag(unsorted) and not flag(cpsp.csr_matrix(unsorted))


@pytest.mark.parametrize("which", ["indptr", "indices"])
@pytest.mark.parametrize("halves", [False, True])
def test_two_index_dtypes(which, halves) -> None:
    """int64 ``indptr`` with int32 ``indices``, or the reverse, gives the dense result; unsorted it is refused."""
    X = _make_matrix(72, np.float32)
    distance = Distance(metric="edistance", layer_key="X")
    ref = Distance(metric="edistance", obsm_key="emb").pairwise(_dense_adata(X), groupby="group")
    m = cpsp.csr_matrix(_halves(csr_matrix(X)) if halves else csr_matrix(X))
    assert m.has_sorted_indices  # cached: CuPy cannot scan two index dtypes
    setattr(m, which, getattr(m, which).astype(cp.int64))
    got = distance.pairwise(AnnData(m, obs=_obs()), groupby="group")
    np.testing.assert_array_equal(got.values, ref.values)
    m = cpsp.csr_matrix(_unsorted(X))
    assert not m.has_sorted_indices  # cached: CuPy would sort it into a wrong order
    setattr(m, which, getattr(m, which).astype(cp.int64))
    with pytest.raises(ValueError, match="one index dtype"):
        distance.pairwise(AnnData(m, obs=_obs()), groupby="group")


@pytest.mark.parametrize("precision", ["float32", "float64", "float16"])
@pytest.mark.parametrize("factor", [2.0**20, 2.0**-20, -1.0])
def test_scaled_input(precision, factor) -> None:
    """Input times a power of two, or negated, gives the same bits times that power: the scale follows the data."""
    X = _make_matrix(72, np.float32)
    X[:, 0], X[0, 0] = 1000, 0  # the largest centered value lies below the column mean, above it once negated
    kwargs = {"metric": "edistance", "precision": precision}
    ref = Distance(obsm_key="emb", **kwargs).pairwise(_dense_adata(X), groupby="group").values
    X *= np.float32(factor)
    sparse = AnnData(csr_matrix(X), obs=_obs())
    for adata, where in ((_dense_adata(X), {"obsm_key": "emb"}), (sparse, {"layer_key": "X"})):
        got = Distance(**where, **kwargs).pairwise(adata, groupby="group")
        np.testing.assert_array_equal(got.values, ref * np.float32(abs(factor)))


def test_sparse_big_endian_values() -> None:
    """Host CSR values in another byte order give the result of their native dtype."""
    X = _make_matrix(40, np.float64)
    m = csr_matrix(X)
    m.data = m.data.astype(">f8")
    got = Distance(metric="edistance", layer_key="X").pairwise(AnnData(m, obs=_obs()), groupby="group")
    ref = Distance(metric="edistance", obsm_key="emb").pairwise(_dense_adata(X), groupby="group")
    assert got.values.dtype == np.float64
    np.testing.assert_array_equal(got.values, ref.values)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("case", ["nan", "all_zero"])
def test_sparse_equals_dense_nan_and_all_zero(dtype, case) -> None:
    """As for dense input, a NaN only costs the group of its cell and an all-zero matrix gives zeros."""
    X = _make_matrix(40, dtype)
    if case == "nan":
        X[0, 3] = np.copysign(np.nan, -1)  # the sign-set NaN that arithmetic such as 0 / 0 produces
    else:
        X[:] = 0
    ref = Distance(metric="edistance", obsm_key="emb").pairwise(_dense_adata(X), groupby="group")
    got = Distance(metric="edistance", layer_key="X").pairwise(AnnData(csr_matrix(X), obs=_obs()), groupby="group")
    np.testing.assert_array_equal(got.values, ref.values)
    assert not got.values.any() if case == "all_zero" else np.isfinite(got.values[1:, 1:]).all()


@pytest.mark.parametrize("dtype,atol", [(np.float32, 1e-4), (np.float64, 1e-9)])
@pytest.mark.parametrize("n_features", [40, 200])
def test_pairwise_sparse_matches_dense(dtype, atol, n_features) -> None:
    """Sparse X and a sparse layer both match the dense obsm path."""
    X = _make_matrix(n_features, dtype)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.astype(dtype).copy()), obs=_obs())
    sparse.layers["counts"] = csr_matrix(X.astype(dtype).copy())

    ref = Distance(metric="edistance", obsm_key="emb").pairwise(dense, groupby="group")
    via_x = Distance(metric="edistance", layer_key="X").pairwise(sparse, groupby="group")
    via_layer = Distance(metric="edistance", layer_key="counts").pairwise(sparse, groupby="group")

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
    result = Distance(metric="edistance", layer_key="X").pairwise(sparse, groupby="group")
    np.testing.assert_allclose(result.values, expected, atol=1e-9)


def test_onesided_sparse_matches_dense() -> None:
    X = _make_matrix(50, np.float32)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())

    ref = Distance(metric="edistance", obsm_key="emb").onesided_distances(dense, groupby="group", selected_group="g0")
    got = Distance(metric="edistance", layer_key="X").onesided_distances(sparse, groupby="group", selected_group="g0")
    np.testing.assert_allclose(got.values, ref.values, atol=1e-4)


def test_subset_of_groups_sparse_matches_dense() -> None:
    X = _make_matrix(40, np.float32)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())

    ref = Distance(metric="edistance", obsm_key="emb").pairwise(dense, groupby="group", groups=["g0", "g2"])
    got = Distance(metric="edistance", layer_key="X").pairwise(sparse, groupby="group", groups=["g0", "g2"])
    np.testing.assert_allclose(got.values, ref.values, atol=1e-4)


def test_contrast_distances_sparse_matches_dense() -> None:
    X = _make_matrix(40, np.float32)
    dense = _dense_adata(X)
    sparse = AnnData(csr_matrix(X.copy()), obs=_obs())

    contrasts = Distance.create_contrasts(dense, groupby="group", selected_group="g0")
    ref = Distance(metric="edistance", obsm_key="emb").contrast_distances(dense, contrasts)
    got = Distance(metric="edistance", layer_key="X").contrast_distances(sparse, contrasts)
    np.testing.assert_allclose(got["edistance"].values, ref["edistance"].values, atol=1e-4)


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
