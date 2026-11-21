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


def _check_precision_issues(score: np.ndarray, dtype: np.dtype) -> None:
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
    if not np.isfinite(score).all():
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
    n_perms: int | None,
    sim_stats: np.ndarray | None,
    n_obs: int,
    moments: tuple[float, float, float],
    params: dict[str, Any],
):
    """
    Handle p-value calculation for spatial autocorrelation function.

    Parameters
    ----------
    score
        (n_features,).
    sim_stats
        Per-feature exceedance counts, means, and variances of the simulated score changes.
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
    p_norm, var_norm = _analytic_pval(score, n_obs, moments, params)
    results = {"pval_norm": p_norm, "var_norm": var_norm}

    if sim_stats is None:
        return results

    large_perm, mean_change, var_sim = sim_stats
    assert n_perms is not None
    large_perm = np.minimum(large_perm, n_perms - large_perm)
    # get p-value based on permutation
    p_sim: np.ndarray = (large_perm + 1) / (n_perms + 1)

    # get p-value based on standard normal approximation from permutations
    with np.errstate(divide="ignore", invalid="ignore"):
        z_sim = -mean_change / np.sqrt(var_sim)
    p_z_sim = stats.norm.sf(np.abs(z_sim))

    results["pval_z_sim"] = p_z_sim
    results["pval_sim"] = p_sim
    results["var_sim"] = var_sim

    return results


def _analytic_pval(score: np.ndarray, n: int, moments: tuple[float, float, float], params: dict[str, Any]):
    """Analytic p-value computation.

    See `Moran's I <https://pysal.org/esda/_modules/esda/moran.html#Moran>`_ and
    `Geary's C <https://pysal.org/esda/_modules/esda/geary.html#Geary>`_ implementation.
    """
    s0, s1, s2 = moments
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
    p_norm = stats.norm.sf(np.abs(z_norm))

    if params["two_tailed"]:
        p_norm *= 2.0

    return p_norm, Vscore_norm


def _g_moments(w):
    """Compute row sums, column sums, and the three moments for analytic p-values.

    See `pysal <https://pysal.org/libpysal/_modules/libpysal/weights/weights.html#W>`_ implementation.
    """
    from rapids_singlecell._cuda import _autocorr_cuda as _ac

    dtype = w.dtype if w.dtype in (np.float32, np.float64) else np.float64
    if not cpx_sparse.isspmatrix_csr(w) or w.dtype != dtype:
        if not (issparse(w) or cpx_sparse.issparse(w)):
            w = cp.asarray(w, dtype=dtype)
        w = cpx_sparse.csr_matrix(w, dtype=dtype)
    if not w.has_canonical_format:
        w = w.astype(cp.float64, copy=True)
        w.sum_duplicates()
    row = cp.empty(w.shape[0], dtype=cp.float64)
    col = cp.zeros_like(row)
    sums = cp.zeros(4, dtype=cp.float64)
    indices = w.indices.astype(cp.int32, copy=False)
    _ac.moments(w.indptr, indices, w.data, row=row, col=col, out=sums[:3], stream=cp.cuda.get_current_stream().ptr)
    degree = row + col
    cp.square(degree, out=degree)
    sums[3] = degree.sum()
    s0, sum_w2, mirror, s2 = sums.get()
    return row, col, (s0, sum_w2 + mirror, s2)


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
