from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _harmony_colsum_cuda as _colsum_cuda
from rapids_singlecell._cuda import _harmony_normalize_cuda as _normalize_cuda
from rapids_singlecell._cuda import _harmony_outer_cuda as _outer_cuda

if TYPE_CHECKING:
    import pandas as pd

# Column-sum heuristic thresholds (rows x cols regions)
_COLSUM_COLS_SMALL = 200
_COLSUM_ROWS_MEDIUM = 20_000
_COLSUM_ROWS_LARGE = 100_000


def _normalize_cp_p1(X: cp.ndarray) -> cp.ndarray:
    """
    Normalize rows of a matrix using an optimized kernel with shared memory and warp shuffle.

    Parameters
    ----------
    X
        Input 2D array.

    Returns
    -------
    Row-normalized 2D array.
    """
    assert X.ndim == 2, "Input must be a 2D array."

    rows, cols = X.shape

    _normalize_cuda.normalize(
        X,
        rows=rows,
        cols=cols,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return X


def _outer_cp(
    E: cp.ndarray, Pr_b: cp.ndarray, R_sum: cp.ndarray, switcher: int
) -> None:
    n_cats, n_pcs = E.shape

    _outer_cuda.outer(
        E,
        Pr_b=Pr_b,
        R_sum=R_sum,
        n_cats=n_cats,
        n_pcs=n_pcs,
        switcher=switcher,
        stream=cp.cuda.get_current_stream().ptr,
    )


def _validate_output_buffer(
    X: cp.ndarray,
    out: cp.ndarray,
    *,
    operation: str,
) -> None:
    """Validate a caller-provided output buffer."""
    if out.shape != X.shape or out.dtype != X.dtype:
        raise ValueError(f"{operation} output must match the input shape and dtype")
    if not out.flags.c_contiguous:
        raise ValueError(f"{operation} output must be C-contiguous")


def _normalize_cp(
    X: cp.ndarray, p: int = 2, *, out: cp.ndarray | None = None
) -> cp.ndarray:
    """
    Analogous to `torch.nn.functional.normalize` for `axis = 1`, `p` in numpy is known as `ord`.
    """
    if p == 2:
        X = cp.ascontiguousarray(X)
        if out is None:
            out = cp.empty_like(X)
        else:
            _validate_output_buffer(X, out, operation="Normalization")
        rows, cols = X.shape
        _normalize_cuda.l2_row_normalize(
            X,
            dst=out,
            n_rows=rows,
            n_cols=cols,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return out

    else:
        if out is not None and out is not X:
            raise ValueError("An output buffer is only supported for L2 normalization")
        return _normalize_cp_p1(X)


def _get_batch_codes(
    batch_mat: pd.DataFrame, batch_key: str | list[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Encode each batch variable into a disjoint range of marginal codes."""
    keys = [batch_key] if isinstance(batch_key, str) else list(batch_key)
    if not keys:
        raise ValueError("batch_key must contain at least one column name")

    codes = np.empty((len(batch_mat), len(keys)), dtype=np.int32)
    n_levels = np.empty(len(keys), dtype=np.int32)
    offset = 0

    for covariate, key in enumerate(keys):
        batch_vec = batch_mat[key].astype("category")
        local_codes = batch_vec.cat.codes.to_numpy(dtype=np.int32, copy=False)
        if np.any(local_codes < 0):
            raise ValueError(f"Batch variable {key!r} contains missing values")

        n_categories = batch_vec.cat.categories.size
        n_levels[covariate] = n_categories
        codes[:, covariate] = local_codes + offset
        offset += n_categories

    return codes, n_levels


def _factorize_joint_codes(
    batch_codes: np.ndarray, n_levels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Factorize marginal category tuples in lexicographic order."""
    levels = np.asarray(n_levels, dtype=np.int64)
    if batch_codes.ndim != 2 or batch_codes.shape[1] != levels.size:
        raise ValueError("Batch codes and category levels have incompatible shapes")

    joint_cardinality = 1
    for level in levels:
        joint_cardinality *= int(level)
        if joint_cardinality > np.iinfo(np.int64).max:
            joint_cats, joint_codes = np.unique(
                batch_codes, axis=0, return_inverse=True
            )
            return joint_cats, np.asarray(joint_codes).reshape(-1)

    offsets = np.empty(levels.size, dtype=np.int64)
    offsets[0] = 0
    if levels.size > 1:
        np.cumsum(levels[:-1], out=offsets[1:])

    linear_codes = batch_codes[:, 0].astype(np.int64) - offsets[0]
    for covariate in range(1, levels.size):
        linear_codes *= levels[covariate]
        linear_codes += batch_codes[:, covariate] - offsets[covariate]

    observed_codes, joint_codes = np.unique(linear_codes, return_inverse=True)
    joint_cats = np.empty((observed_codes.size, levels.size), dtype=np.int32)
    remainder = observed_codes.copy()
    for covariate in range(levels.size - 1, -1, -1):
        joint_cats[:, covariate] = remainder % levels[covariate]
        remainder //= levels[covariate]
    joint_cats += offsets.astype(np.int32)
    return joint_cats, joint_codes


def _stratified_sample_indices(
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    n_target: int,
    rng: np.random.Generator,
) -> cp.ndarray:
    """Draw exactly ``n_target`` cells while representing every observed stratum."""
    offsets = cp.asnumpy(cat_offsets).astype(np.int64, copy=False)
    sizes = np.diff(offsets)
    nonempty = np.flatnonzero(sizes)
    n_cells = int(cell_indices.size)
    if not nonempty.size <= n_target <= n_cells:
        raise ValueError(
            "n_target must cover every nonempty stratum without exceeding n_cells"
        )

    quotas = np.zeros_like(sizes)
    quotas[nonempty] = 1
    remaining = n_target - nonempty.size
    if remaining:
        capacities = sizes - quotas
        total_capacity = int(capacities.sum())
        numerators = capacities * remaining
        additional, remainders = np.divmod(numerators, total_capacity)
        quotas += additional

        leftover = n_target - int(quotas.sum())
        if leftover:
            eligible = np.flatnonzero(remainders)
            tie_break = rng.random(eligible.size)
            order = np.lexsort((tie_break, -remainders[eligible]))
            quotas[eligible[order[:leftover]]] += 1

    picks = []
    for start, size, quota in zip(offsets[:-1], sizes, quotas, strict=True):
        start, size, quota = int(start), int(size), int(quota)
        if quota == 0:
            continue
        picks.append(start + rng.choice(size, quota, replace=False))

    selected = cell_indices[cp.asarray(np.concatenate(picks))]
    return selected[cp.asarray(rng.permutation(n_target))]


def _get_theta_array(
    theta: float | int | list[float | int] | np.ndarray | cp.ndarray,
    n_levels: int | np.ndarray,
    dtype: cp.dtype,
) -> cp.ndarray:
    """
    Normalize scalar, per-variable, or per-category theta values.
    """
    levels = np.atleast_1d(n_levels).astype(np.int64, copy=False)
    n_covariates = levels.size
    n_categories = int(levels.sum())

    try:
        theta_array = cp.asarray(theta, dtype=dtype)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "Theta must be a scalar or an array-like collection of numeric values, "
            f"got {type(theta).__name__}"
        ) from e
    if theta_array.ndim == 0:
        return cp.full(n_categories, theta_array, dtype=dtype)

    theta_array = theta_array.ravel()
    if theta_array.size == n_covariates:
        return cp.repeat(theta_array, cp.asarray(levels))
    if theta_array.size == n_categories:
        return theta_array

    raise ValueError(
        f"Theta array size ({theta_array.size}) must match the number of batch "
        f"variables ({n_covariates}) or categorical levels ({n_categories})"
    )


def _column_sum(X: cp.ndarray) -> cp.ndarray:
    """
    Sum each column of the 2D, C-contiguous float32 array A.
    Returns a 1D float32 cupy array of length A.shape[1].
    """
    rows, cols = X.shape
    if not X.flags.c_contiguous:
        return X.sum(axis=0)

    out = cp.zeros(cols, dtype=X.dtype)

    _colsum_cuda.colsum(
        X,
        out=out,
        rows=rows,
        cols=cols,
        stream=cp.cuda.get_current_stream().ptr,
    )

    return out


def _gemm_colsum(X: cp.ndarray) -> cp.ndarray:
    """
    Sum each column with cuBLAS GEMM
    """
    return X.T @ cp.ones(X.shape[0], dtype=X.dtype)


def _choose_colsum_algo_heuristic(rows: int, cols: int, algo: str | None) -> callable:
    """Choose a deterministic column reduction from the shape and device."""
    if algo in {"atomics", "benchmark"}:
        algo = None
    if algo is None:
        cc = cp.cuda.Device().compute_capability
        algo = _colsum_heuristic(rows, cols, cc)
    if algo == "columns":
        return _column_sum
    return _gemm_colsum


# TODO: Make this more robust
def _colsum_heuristic(rows: int, cols: int, compute_capability: str) -> str:
    is_data_center = compute_capability in ["100", "90"]
    if cols < _COLSUM_COLS_SMALL and rows < _COLSUM_ROWS_MEDIUM:
        return "columns"
    if cols < _COLSUM_COLS_SMALL and rows < _COLSUM_ROWS_LARGE and is_data_center:
        return "columns"
    return "gemm"
