# adapted from decoupler/_pre.py
from __future__ import annotations

from typing import Union

import cudf
import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy.sparse import csr_matrix as cp_csr_matrix
from cupyx.scipy.sparse import issparse as cp_issparse
from scanpy.get import _get_obs_rep
from scipy.sparse import csr_matrix, issparse
from tqdm.auto import tqdm

from rapids_singlecell._compat import DaskArray
from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.preprocessing._utils import _check_use_raw

DataType = Union[  # noqa: UP007
    AnnData, pd.DataFrame, cudf.DataFrame, tuple[np.ndarray, np.ndarray, np.ndarray]
]

DataType_matrix = Union[np.ndarray, cp.ndarray, csr_matrix, cp_csr_matrix]  # noqa: UP007

getnnz_0 = cp.ElementwiseKernel(
    "int32 idx",
    "raw int32 sum",
    """
    atomicAdd(&sum[idx], 1);
    """,
    "get_nnz_0",
)


def _validate_mat(
    mat: DataType_matrix,
    row: np.ndarray,
    col: np.ndarray,
    *,
    empty: bool = True,
    verbose: bool = False,
) -> tuple[DataType_matrix, np.ndarray, np.ndarray]:
    assert isinstance(empty, bool), "empty must be bool"

    # Skip validation for Dask arrays - validation is deferred to chunk processing
    if isinstance(mat, DaskArray):
        _log(
            "Dask array detected, skipping validation (will be validated per chunk)",
            level="info",
            verbose=verbose,
        )
        return mat, row, col

    # Accept any sparse format but transform to csr
    if issparse(mat) and not isinstance(mat, csr_matrix):
        mat = csr_matrix(mat)
    elif cp_issparse(mat) and not isinstance(mat, cp_csr_matrix):
        mat = cp_csr_matrix(mat)
    # Check for empty features
    if isinstance(mat, csr_matrix):
        msk_col = mat.getnnz(axis=0) == 0
    elif cp_issparse(mat):
        msk_col = getnnz_0(mat.indices, cp.zeros(mat.shape[1], dtype=np.int32)) == 0
    elif isinstance(mat, cp.ndarray):
        msk_col = cp.count_nonzero(mat, axis=0) == 0
    else:
        msk_col = np.count_nonzero(mat, axis=0) == 0
    n_empty_col: float = np.sum(msk_col)
    if n_empty_col > 0 and empty:
        m = f"{n_empty_col} features of mat are empty, they will be removed"
        _log(m, level="warn", verbose=verbose)
        if isinstance(msk_col, cp.ndarray):
            msk_col = msk_col.get()
        col = col[~msk_col]
        mat = mat[:, ~msk_col]
    # Check for repeated features
    assert not np.any(col[1:] == col[:-1]), (
        "mat contains repeated feature names, please make them unique"
    )

    # Check for empty samples
    if isinstance(mat, csr_matrix):
        msk_row = mat.getnnz(axis=1) == 0
    elif cp_issparse(mat):
        msk_row = cp.diff(mat.indptr) == 0
    elif isinstance(mat, cp.ndarray):
        msk_row = cp.count_nonzero(mat, axis=1) == 0
    else:
        msk_row = np.count_nonzero(mat, axis=1) == 0

    n_empty_row: float = msk_row.sum()
    if n_empty_row > 0 and empty:
        m = f"{n_empty_row} observations of mat are empty, they will be removed"
        _log(m, level="warn", verbose=verbose)
        if isinstance(msk_row, cp.ndarray):
            msk_row = msk_row.get()
        row = row[~msk_row]
        mat = mat[~msk_row]

    # Check for non finite values
    check = False
    if isinstance(mat, cp.ndarray):
        if cp.any(~cp.isfinite(mat)):
            check = True
    elif isinstance(mat, np.ndarray):
        if np.any(~np.isfinite(mat)):
            check = True
    else:
        if np.any(~np.isfinite(mat.data)):
            check = True
    if check:
        raise ValueError(
            """mat contains non finite values (nan or inf), please set them to 0 or remove them."""
        )
    return mat, row, col


def _validate_backed(
    mat: DataType_matrix,
    row: np.ndarray,
    col: np.ndarray,
    *,
    empty: bool = True,
    verbose: bool = False,
    bsize: int = 250_000,
) -> np.ndarray:
    nbatch = int(np.ceil(row.size / bsize))
    msk_col = np.zeros((nbatch, mat.shape[1]), dtype=bool)
    for i in tqdm(range(nbatch), disable=not verbose):
        srt, end = i * bsize, i * bsize + bsize
        bmat = mat[srt:end]
        if issparse(bmat):
            msk_col[i] = bmat.getnnz(axis=0) == 0
        else:
            msk_col[i] = np.count_nonzero(bmat, axis=0) == 0
        has_nonfin = np.any(~np.isfinite(bmat.data))
        assert not has_nonfin, (
            "mat contains non finite values (nan or inf), set them to 0 or remove them"
        )
    msk_col = np.logical_and.reduce(msk_col, axis=0)
    n_empty_col: float = np.sum(msk_col)
    if n_empty_col > 0 and empty:
        m = f"{n_empty_col} features of mat are empty, they will be removed"
        _log(m, level="warn", verbose=verbose)
    else:
        msk_col[:] = False
    return msk_col


def _break_ties(
    mat: DataType_matrix,
    features: np.ndarray,
) -> tuple[DataType_matrix, np.ndarray]:
    # Randomize feature order to break ties randomly
    rng = np.random.default_rng(seed=0)
    idx = np.arange(features.size)
    idx = rng.choice(idx, features.size, replace=False)
    mat, features = mat[:, idx], features[idx]
    return mat, features


def _extract(data: DataType, *, raw=None, layer=None, pre_load=False):
    """
    Processes different input types so that they can be used downstream.

    Parameters
    ----------
    data : list, pd.DataFrame or AnnData
        List of [matrix, samples, features], dataframe (samples x features) or an AnnData instance.
    use_raw : bool
        Use `raw` attribute of `adata` if present.
    dtype : type
        Type of float used.

    Returns
    -------
    m : csr_matrix
        Sparse matrix containing molecular readouts or statistics.
    r : ndarray
        Array of sample names.
    c : ndarray
        Array of feature names.
    """

    if isinstance(data, list):
        m, r, c = data
        m = np.array(m, dtype=float)
        r = np.array(r, dtype="U")
        c = np.array(c, dtype="U")
    elif isinstance(data, pd.DataFrame) or isinstance(data, cudf.DataFrame):
        m = data.values.astype(float)
        r = data.index.to_numpy(dtype="U")
        c = data.columns.to_numpy(dtype="U")
    elif isinstance(data, AnnData):
        use_raw = _check_use_raw(data, layer, use_raw=raw)
        m = _get_obs_rep(data, layer=layer, use_raw=raw)
        c = (
            data.raw.var.index.values.astype("U")
            if use_raw
            else data.var.index.values.astype("U")
        )
        r = data.obs.index.values.astype("U")

    else:
        raise ValueError("""mat must be a list of [matrix, samples, features], dataframe (samples x features) or an AnnData
        instance.""")

    if pre_load is None:
        pre_load = False
    if pre_load:
        if issparse(m):
            m = cp_csr_matrix(m, dtype=cp.float32)
        elif isinstance(m, np.ndarray):
            m = cp.array(m, dtype=cp.float32)

    if isinstance(r, cp.ndarray):
        r = cp.asnumpy(r)
    if isinstance(c, cp.ndarray):
        c = cp.asnumpy(c)

    return m, r, c


@docs.dedent
def extract(
    data: DataType,
    layer: str | None = None,
    *,
    raw: bool = False,
    empty: bool = True,
    verbose: bool = False,
    bsize: int = 250_000,
    pre_load: bool = False,
) -> (
    tuple[DataType_matrix, np.ndarray, np.ndarray]
    | tuple[tuple[DataType_matrix, np.ndarray], np.ndarray, np.ndarray]
):
    """
    Extracts matrix, rownames and colnames from data.

    Parameters
    ----------
    %(data)s
    %(layer)s
    %(raw)s
    %(empty)s
    %(verbose)s
    %(pre_load)s

    Returns
    -------
    Matrix, rownames and colnames from data.
    """
    # Extract
    mat, row, col = _extract(data, layer=layer, raw=raw, pre_load=pre_load)
    # Validate
    isbacked = hasattr(data, "isbacked") and data.isbacked
    mat_tuple: (
        tuple[np.ndarray, np.ndarray, np.ndarray]
        | tuple[tuple[np.ndarray, np.ndarray], np.ndarray, np.ndarray]
    )
    if not isbacked:
        mat, row, col = _validate_mat(mat, row, col, empty=empty, verbose=verbose)
        # Randomly sort features
        mat, col = _break_ties(mat=mat, features=col)
        mat_tuple = (mat, row, col)
    else:
        msk_col = _validate_backed(
            mat=mat, row=row, col=col, empty=empty, verbose=verbose, bsize=bsize
        )
        msk_col = ~msk_col
        col = col[msk_col]
        mat_tuple = ((mat, msk_col), row, col)
    return mat_tuple
