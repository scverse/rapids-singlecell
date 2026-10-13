from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as cpx_sparse
from pandas.api.types import infer_dtype
from scipy import stats
from scipy.sparse import issparse, spmatrix


def _check_precision_issues(score: cp.ndarray, dtype: np.dtype) -> None:
    """Check for numerical issues in autocorrelation scores.

    Parameters
    ----------
    score
        The computed autocorrelation scores.
    dtype
        The data type used for computation.

    Raises
    ------
    ValueError
        If nan/inf values are detected.
    """
    if cp.any(cp.isnan(score) | cp.isinf(score)):
        if dtype == np.float32:
            raise ValueError(
                "Detected nan/inf values in results with float32 precision. "
                "This can occur with genes that have low variance or are constant. "
                "Please use `dtype=np.float64` for better numerical stability."
            )
        else:
            raise ValueError(
                "Detected nan/inf values in results with float64 precision. "
                "This can occur with genes that have zero variance (constant expression). "
                "Consider filtering out such genes before running spatial autocorrelation. "
                "If this is unexpected, please file a bug report at "
                "https://github.com/scverse/rapids_singlecell/issues"
            )


### Taken from squidpy: https://github.com/scverse/squidpy/blob/main/squidpy/gr/_ppatterns.py
def _p_value_calc(
    score: np.ndarray,
    *,
    sims: np.ndarray | None,
    weights: spmatrix | np.ndarray,
    params: dict[str, Any],
):
    """
    Handle p-value calculation for spatial autocorrelation function.

    Parameters
    ----------
    score
        (n_features,).
    sims
        (n_simulations, n_features).
    params
        Object to store relevant function parameters.

    Returns
    -------
    pval_norm
        p-value under normality assumption
    pval_sim
        p-values based on permutations
    pval_z_sim
        p-values based on standard normal approximation from permutations
    """
    p_norm, var_norm = _analytic_pval(score, weights, params)
    results = {"pval_norm": p_norm, "var_norm": var_norm}

    if sims is None:
        return results

    n_perms = sims.shape[0]
    large_perm = (sims >= score).sum(axis=0)
    # subtract total perm for negative values
    large_perm[(n_perms - large_perm) < large_perm] = n_perms - large_perm[(n_perms - large_perm) < large_perm]
    # get p-value based on permutation
    p_sim: np.ndarray = (large_perm + 1) / (n_perms + 1)

    # get p-value based on standard normal approximation from permutations
    e_score_sim = sims.sum(axis=0) / n_perms
    se_score_sim = sims.std(axis=0)
    z_sim = (score - e_score_sim) / se_score_sim
    p_z_sim = np.empty(z_sim.shape)

    p_z_sim[z_sim > 0] = stats.norm.sf(z_sim[z_sim > 0])
    p_z_sim[z_sim <= 0] = stats.norm.cdf(z_sim[z_sim <= 0])

    var_sim = np.var(sims, axis=0)

    results["pval_z_sim"] = p_z_sim
    results["pval_sim"] = p_sim
    results["var_sim"] = var_sim

    return results


def _analytic_pval(score: np.ndarray, g: spmatrix | np.ndarray, params: dict[str, Any]):
    """Analytic p-value computation.

    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_ and
    `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_ implementation.
    """
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
    if params["mode"] == "gearyC":
        # Geary's C has its own normality variance (Cliff & Ord 1981; esda ``Geary``)
        Vscore_norm = ((2 * s1 + s2) * (n - 1) - 4 * s02) / (2 * (n + 1) * s02)
    else:
        n2 = n * n
        v_num = n2 * s1 - n * s2 + 3 * s02
        v_den = (n - 1) * (n + 1) * s02
        Vscore_norm = v_num / v_den - (1.0 / (n - 1)) ** 2
    seScore_norm = Vscore_norm ** (1 / 2.0)

    z_norm = (score - params["expected"]) / seScore_norm
    p_norm = np.empty(score.shape)
    p_norm[z_norm > 0] = stats.norm.sf(z_norm[z_norm > 0])
    p_norm[z_norm <= 0] = stats.norm.cdf(z_norm[z_norm <= 0])

    if params["two_tailed"]:
        p_norm *= 2.0

    return p_norm, Vscore_norm


def _g_moments(w: spmatrix | np.ndarray):
    """Compute moments of adjacency matrix for analytic p-value calculation.

    See `pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    if not cpx_sparse.isspmatrix(w):
        w = cpx_sparse.csr_matrix(w)
    w = w.tocsr()
    rows = cp.searchsorted(w.indptr[1:], cp.arange(w.nnz), side="right")
    # canonical (sorted, unique columns per row) unless a row's columns do not increase
    if bool(((w.indices[1:] <= w.indices[:-1]) & (rows[1:] == rows[:-1])).any()):
        w.sum_duplicates()
        rows = cp.searchsorted(w.indptr[1:], cp.arange(w.nnz), side="right")
    from rapids_singlecell._cuda import _autocorr_cuda as _ac

    sums = cp.zeros(3)
    indices = w.indices.astype(cp.int32, copy=False)
    _ac.moments(
        w.indptr, indices, w.data, out=sums, stream=cp.cuda.get_current_stream().ptr
    )
    s0, sum_w2, mirror = (float(v) for v in sums.get())
    # s1 = sum((w + w^T)^2) / 2 = sum(w^2) + sum_ij w_ij w_ji
    s1 = sum_w2 + mirror

    # s2
    data = w.data.astype(cp.float64)
    n = w.shape[0]
    degree = cp.bincount(rows, data, minlength=n) + cp.bincount(
        indices, data, minlength=n
    )
    s2 = float((degree * degree).sum())

    return s0, s1, s2


def _create_sparse_df(
    data,
    index=None,
    columns=None,
    fill_value=0,
):
    """
    Create a new DataFrame from a scipy sparse matrix or numpy array.

    This is the original :mod:`pandas` implementation with 2 differences:

        - allow creation also from :class:`numpy.ndarray`
        - expose ``fill_values``

    Parameters
    ----------
    data
        Must be convertible to CSC format.
    index
        Row labels to use.
    columns
        Column labels to use.

    Returns
    -------
    Each column of the DataFrame is stored as a :class:`arrays.SparseArray`.
    """
    from pandas._libs.sparse import IntIndex
    from pandas.core.arrays.sparse.accessor import (
        SparseArray,
        SparseDtype,
        SparseFrameAccessor,
    )

    if not issparse(data):
        pred = (lambda col: ~np.isnan(col)) if fill_value is np.nan else (lambda col: ~np.isclose(col, fill_value))
        dtype = SparseDtype(data.dtype, fill_value=fill_value)
        n_rows, n_cols = data.shape
        arrays = []

        for i in range(n_cols):
            mask = pred(data[:, i])
            idx = IntIndex(n_rows, np.where(mask)[0], check_integrity=False)
            arr = SparseArray._simple_new(data[mask, i], idx, dtype)
            arrays.append(arr)

        return pd.DataFrame._from_arrays(arrays, columns=columns, index=index, verify_integrity=False)

    if TYPE_CHECKING:
        assert isinstance(data, spmatrix)
    data = data.tocsc()
    sort_indices = True

    data = data.tocsc()
    index, columns = SparseFrameAccessor._prep_index(data, index, columns)
    n_rows, n_columns = data.shape
    # We need to make sure indices are sorted, as we create
    # IntIndex with no input validation (i.e. check_integrity=False ).
    # Indices may already be sorted in scipy in which case this adds
    # a small overhead.
    if sort_indices:
        data.sort_indices()

    indices = data.indices
    indptr = data.indptr
    array_data = data.data
    dtype = SparseDtype(array_data.dtype, fill_value=fill_value)
    arrays = []

    for i in range(n_columns):
        sl = slice(indptr[i], indptr[i + 1])
        idx = IntIndex(n_rows, indices[sl], check_integrity=False)
        arr = SparseArray._simple_new(array_data[sl], idx, dtype)
        arrays.append(arr)

    return pd.DataFrame._from_arrays(arrays, columns=columns, index=index, verify_integrity=False)


def _assert_categorical_obs(adata, key):
    if key not in adata.obs:
        raise KeyError(f"Cluster key `{key}` not found in `adata.obs`.")

    if not isinstance(adata.obs[key].dtype, pd.CategoricalDtype):
        raise TypeError(f"Expected `adata.obs[{key!r}]` to be `categorical`, found `{infer_dtype(adata.obs[key])}`.")


def _assert_spatial_basis(adata, key: str) -> None:
    if key not in adata.obsm:
        raise KeyError(f"Spatial basis `{key}` not found in `adata.obsm`.")


def _assert_connectivity_key(adata, key: str) -> None:
    if key not in adata.obsp:
        raise KeyError(
            f"Spatial connectivity key `{key}` not found in `adata.obsp`. "
            "Please run `rapids_singlecell.gr.spatial_neighbors_*` first."
        )
