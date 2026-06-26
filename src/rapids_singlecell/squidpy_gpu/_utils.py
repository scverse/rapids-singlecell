from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
)

import cupy as cp
import numpy as np
import pandas as pd
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
    large_perm[(n_perms - large_perm) < large_perm] = (
        n_perms - large_perm[(n_perms - large_perm) < large_perm]
    )
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
    """
    Analytic p-value computation.
    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_ and
    `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_ implementation.
    """
    s0, s1, s2 = _g_moments(g)
    n = g.shape[0]
    s02 = s0 * s0
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
    """
    Compute moments of adjacency matrix for analytic p-value calculation.
    See `pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    # s0
    s0 = w.sum()

    # s1
    t = w.transpose() + w
    t2 = t.multiply(t)  # type: ignore[union-attr]
    s1 = t2.sum() / 2.0

    # s2
    s2array: np.ndarray = np.array(w.sum(1) + w.sum(0).transpose()) ** 2
    s2 = s2array.sum()

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
        pred = (
            (lambda col: ~np.isnan(col))
            if fill_value is np.nan
            else (lambda col: ~np.isclose(col, fill_value))
        )
        dtype = SparseDtype(data.dtype, fill_value=fill_value)
        n_rows, n_cols = data.shape
        arrays = []

        for i in range(n_cols):
            mask = pred(data[:, i])
            idx = IntIndex(n_rows, np.where(mask)[0], check_integrity=False)
            arr = SparseArray._simple_new(data[mask, i], idx, dtype)
            arrays.append(arr)

        return pd.DataFrame._from_arrays(
            arrays, columns=columns, index=index, verify_integrity=False
        )

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

    return pd.DataFrame._from_arrays(
        arrays, columns=columns, index=index, verify_integrity=False
    )


def _assert_categorical_obs(adata, key):
    if key not in adata.obs:
        raise KeyError(f"Cluster key `{key}` not found in `adata.obs`.")

    if not isinstance(adata.obs[key].dtype, pd.CategoricalDtype):
        raise TypeError(
            f"Expected `adata.obs[{key!r}]` to be `categorical`, found `{infer_dtype(adata.obs[key])}`."
        )


def _assert_spatial_basis(adata, key: str) -> None:
    if key not in adata.obsm:
        raise KeyError(f"Spatial basis `{key}` not found in `adata.obsm`.")
