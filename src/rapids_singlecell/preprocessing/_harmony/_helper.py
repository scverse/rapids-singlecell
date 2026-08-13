from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _harmony_colsum_cuda as _hc_cs
from rapids_singlecell._cuda import _harmony_normalize_cuda as _hc_norm
from rapids_singlecell._cuda import _harmony_outer_cuda as _hc_out
from rapids_singlecell._cuda import _harmony_scatter_cuda as _hc_sc

if TYPE_CHECKING:
    import pandas as pd

# Shared-memory scatter_add heuristics
MIN_CELLS_FOR_SHARED = 50_000
MIN_CELLS_PER_BATCH_SHARED = 10_000
MAX_SHARED_MEM_BYTES = 48 * 1024  # 48 KB shared memory budget
MIN_CELLS_PER_BLOCK = 64

# Column-sum heuristic thresholds (rows x cols regions)
_COLSUM_COLS_SMALL = 200
_COLSUM_COLS_MEDIUM = 800
_COLSUM_COLS_LARGE = 1024
_COLSUM_COLS_XLARGE = 2000
_COLSUM_ROWS_TINY = 5_000
_COLSUM_ROWS_SMALL = 10_000
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

    _hc_norm.normalize(
        X,
        rows=rows,
        cols=cols,
        stream=cp.cuda.get_current_stream().ptr,
    )
    return X


def _scatter_add_cp(
    X: cp.ndarray,
    out: cp.ndarray,
    cats: cp.ndarray,
    switcher: int,
    n_batches: int | None = None,
    *,
    use_shared: bool | None = None,
) -> None:
    """
    Scatter add operation for Harmony algorithm.

    Uses shared memory kernel when n_batches is provided and output fits
    in shared memory (< 48KB). This reduces global atomic contention.

    The shared memory kernel is only beneficial when atomic contention is high,
    which occurs when n_cells / n_batches is large (many cells per batch bucket).
    With many batches, contention is naturally low and the original kernel is faster.

    Parameters
    ----------
    X
        Input array of shape (n_cells, n_pcs)
    out
        Output array of shape (n_batches, n_pcs)
    cats
        Category indices for each cell
    switcher
        0 for subtraction, 1 for addition
    n_batches
        Number of batch categories
    use_shared
        Force shared memory kernel (True), force optimized kernel (False),
        or auto-select based on heuristics (None, default)
    """
    n_cells = X.shape[0]
    n_pcs = X.shape[1]
    n_covariates = cats.shape[1] if cats.ndim == 2 else 1

    # Determine whether to use shared memory kernel
    if use_shared is None:
        use_shared = False
        if n_batches is not None and n_cells >= MIN_CELLS_FOR_SHARED:
            cells_per_batch = n_cells * n_covariates // n_batches
            shared_mem_needed = n_batches * n_pcs * X.dtype.itemsize
            if (
                shared_mem_needed <= MAX_SHARED_MEM_BYTES
                and cells_per_batch >= MIN_CELLS_PER_BATCH_SHARED
            ):
                use_shared = True

    if use_shared:
        if n_batches is None:
            raise ValueError("n_batches must be provided when use_shared=True")
        dev = cp.cuda.Device()
        n_sm = dev.attributes["MultiProcessorCount"]
        max_blocks_by_cells = max(
            1, (n_cells + MIN_CELLS_PER_BLOCK - 1) // MIN_CELLS_PER_BLOCK
        )
        n_blocks = min(n_sm * 4, max_blocks_by_cells)

        _hc_sc.scatter_add_shared(
            X,
            cats=cats,
            n_cells=n_cells,
            n_pcs=n_pcs,
            n_batches=n_batches,
            n_covariates=n_covariates,
            switcher=switcher,
            a=out,
            n_blocks=n_blocks,
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        _hc_sc.scatter_add(
            X,
            cats=cats,
            n_cells=n_cells,
            n_pcs=n_pcs,
            n_covariates=n_covariates,
            switcher=switcher,
            a=out,
            stream=cp.cuda.get_current_stream().ptr,
        )


def _outer_cp(
    E: cp.ndarray, Pr_b: cp.ndarray, R_sum: cp.ndarray, switcher: int
) -> None:
    n_cats, n_pcs = E.shape

    _hc_out.outer(
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
        _hc_norm.l2_row_normalize(
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
    random_state: int,
) -> cp.ndarray:
    """Draw exactly ``n_target`` cells while representing every observed stratum."""

    sizes = cp.diff(cat_offsets).astype(np.int64, copy=False)
    nonempty = cp.flatnonzero(sizes)
    sizes = cp.asnumpy(sizes)
    nonempty = cp.asnumpy(nonempty)
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
            tie_break = np.random.default_rng(random_state).random(eligible.size)
            order = np.lexsort((tie_break, -remainders[eligible]))
            quotas[eligible[order[:leftover]]] += 1

    rng = cp.random.RandomState(random_state)
    parts = []
    for start, size, quota in zip(offsets[:-1], sizes, quotas, strict=True):
        start, size, quota = int(start), int(size), int(quota)
        if quota == 0:
            continue
        local = rng.choice(size, quota, replace=False)
        parts.append(cell_indices[start + local])

    selected = cp.concatenate(parts)
    order = rng.choice(n_target, n_target, replace=False)
    return selected[order]


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

    _hc_cs.colsum(
        X,
        out=out,
        rows=rows,
        cols=cols,
        stream=cp.cuda.get_current_stream().ptr,
    )

    return out


def _column_sum_atomic(X: cp.ndarray) -> cp.ndarray:
    """
    Sum each column of the 2D, C-contiguous array A.

    Uses 2D grid: blockIdx.x = column tile, blockIdx.y = row tile.
    Each thread processes multiple rows to reduce atomic contention.
    """
    assert X.ndim == 2
    rows, cols = X.shape
    if not X.flags.c_contiguous:
        return X.sum(axis=0)

    out = cp.zeros(cols, dtype=X.dtype)

    _hc_cs.colsum_atomic(
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
    """
    Returns one of:
    - _column_sum
    - _column_sum_atomic
    - _gemm_colsum
    """
    # first pick the strategy string
    if algo is None:
        cc = cp.cuda.Device().compute_capability
        algo = _colsum_heuristic(rows, cols, cc)
    if algo == "columns":
        return _column_sum
    if algo == "atomics":
        return _column_sum_atomic
    return _gemm_colsum


# TODO: Make this more robust
def _colsum_heuristic(rows: int, cols: int, compute_capability: str) -> str:
    is_data_center = compute_capability in ["100", "90"]
    if cols < _COLSUM_COLS_SMALL and rows < _COLSUM_ROWS_MEDIUM:
        return "columns"
    if cols < _COLSUM_COLS_SMALL and rows < _COLSUM_ROWS_LARGE and is_data_center:
        return "columns"
    if cols < _COLSUM_COLS_MEDIUM and rows < _COLSUM_ROWS_SMALL:
        return "atomics"
    if cols < _COLSUM_COLS_LARGE and rows < _COLSUM_ROWS_TINY:
        return "atomics"
    if cols < _COLSUM_COLS_MEDIUM and rows < _COLSUM_ROWS_MEDIUM and is_data_center:
        return "atomics"
    if cols < _COLSUM_COLS_XLARGE and rows < _COLSUM_ROWS_SMALL and is_data_center:
        return "atomics"
    return "gemm"


# TODO: Make this more robust
def _benchmark_colsum_algorithms(
    shape: tuple[int, int],
    dtype: cp.dtype = cp.float32,
    n_warmup: int = 1,
    n_trials: int = 3,
) -> callable:
    """
    Benchmark all column sum algorithms and return the fastest one.
    Parameters
    ----------
    shape
        Shape of the test matrix (rows, cols)
    dtype
        Data type for the test matrix
    n_warmup
        Number of warmup iterations
    n_trials
        Number of benchmark trials
    Returns
    -------
    Name of the fastest algorithm: 'columns', 'atomics', or 'gemm'
    """
    rows, cols = shape

    # Create test data
    X = cp.random.random(shape, dtype=dtype)

    # Ensure it's C-contiguous for fair comparison
    if not X.flags.c_contiguous:
        X = cp.ascontiguousarray(X)

    algorithms = {
        "columns": _column_sum,
        "atomics": _column_sum_atomic,
        "gemm": _gemm_colsum,
    }

    results = {}

    for name, func in algorithms.items():
        # Warmup
        for _ in range(n_warmup):
            try:
                _ = func(X)
                cp.cuda.Stream.null.synchronize()
            except Exception:  # noqa: BLE001
                # If algorithm fails, skip it
                results[name] = float("inf")
                break
        else:
            # Benchmark
            times = []
            for _ in range(n_trials):
                try:
                    start_event = cp.cuda.Event()
                    end_event = cp.cuda.Event()

                    start_event.record()
                    _ = func(X)
                    end_event.record()
                    end_event.synchronize()
                    elapsed_ms = cp.cuda.get_elapsed_time(start_event, end_event)
                    times.append(elapsed_ms)
                except Exception:  # noqa: BLE001
                    results[name] = float("inf")
                    break
            else:
                # Use median time for robustness
                results[name] = cp.median(cp.array(times))

    # Return the algorithm with minimum time
    fastest_algo = min(results.items(), key=lambda x: x[1])[0]

    return algorithms[fastest_algo], fastest_algo


def _choose_colsum_algo_benchmark(
    rows: int,
    cols: int,
    dtype: cp.dtype = cp.float32,
    *,
    verbose: bool = True,
) -> callable:
    """
    Automatically choose the best column sum algorithm by benchmarking.

    Parameters
    ----------
    rows
        Number of rows
    cols
        Number of columns
    dtype
        Data type
    verbose
        Whether to print the chosen algorithm
    Returns
    -------
    Function of the fastest algorithm
    """
    func, algo = _benchmark_colsum_algorithms((rows, cols), dtype)
    if verbose:
        print(f"Using {algo} for column sum")
    return func
