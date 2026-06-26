from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _harmony_colsum_cuda as _hc_cs
from rapids_singlecell._cuda import _harmony_normalize_cuda as _hc_norm
from rapids_singlecell._cuda import _harmony_outer_cuda as _hc_out
from rapids_singlecell._cuda import _harmony_pen_cuda as _hc_pen
from rapids_singlecell._cuda import _harmony_scatter_cuda as _hc_sc

if TYPE_CHECKING:
    import pandas as pd

# Shared-memory scatter_add heuristics
MIN_CELLS_FOR_SHARED = 50_000
MIN_CELLS_PER_BATCH_SHARED = 10_000
MAX_SHARED_MEM_BYTES = 48 * 1024  # 48 KB shared memory budget
MIN_CELLS_PER_BLOCK = 64

# Kernel selection threshold for scatter_add_bias_csr
SCATTER_BIAS_KERNEL_THRESHOLD = 300_000

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

    # Determine whether to use shared memory kernel
    if use_shared is None:
        use_shared = False
        if n_batches is not None and n_cells >= MIN_CELLS_FOR_SHARED:
            cells_per_batch = n_cells // n_batches
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
            switcher=switcher,
            a=out,
            n_blocks=n_blocks,
            stream=cp.cuda.get_current_stream().ptr,
        )
    else:
        # Use nanobind .cu kernel
        _hc_sc.scatter_add(
            X,
            cats=cats,
            n_cells=n_cells,
            n_pcs=n_pcs,
            switcher=switcher,
            a=out,
            stream=cp.cuda.get_current_stream().ptr,
        )


def _Z_correction(
    Z: cp.ndarray,
    W: cp.ndarray,
    cats: cp.ndarray,
    R: cp.ndarray,
) -> None:
    """
    Scatter add operation for Harmony algorithm.
    """
    n_cells = Z.shape[0]
    n_pcs = Z.shape[1]

    _hc_out.harmony_corr(
        Z,
        W=W,
        cats=cats,
        R=R,
        n_cells=n_cells,
        n_pcs=n_pcs,
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


def _normalize_cp(X: cp.ndarray, p: int = 2) -> cp.ndarray:
    """
    Analogous to `torch.nn.functional.normalize` for `axis = 1`, `p` in numpy is known as `ord`.
    """
    if p == 2:
        X = cp.ascontiguousarray(X)
        dst = cp.empty_like(X)
        rows, cols = X.shape
        _hc_norm.l2_row_normalize(
            X,
            dst=dst,
            n_rows=rows,
            n_cols=cols,
            stream=cp.cuda.get_current_stream().ptr,
        )
        return dst

    else:
        return _normalize_cp_p1(X)


def _get_aggregated_matrix(
    aggregated_matrix: cp.ndarray, sum: cp.ndarray, n_batches: int
) -> None:
    """
    Get the aggregated matrix for the correction step.
    """

    _hc_sc.aggregated_matrix(
        aggregated_matrix,
        sum=sum,
        top_corner=float(sum.sum()),
        n_batches=n_batches,
        stream=cp.cuda.get_current_stream().ptr,
    )


def _get_batch_codes(batch_mat: pd.DataFrame, batch_key: str | list[str]) -> pd.Series:
    if isinstance(batch_key, str):
        batch_vec = batch_mat[batch_key]

    elif len(batch_key) == 1:
        batch_key = batch_key[0]

        batch_vec = batch_mat[batch_key]

    else:
        df = batch_mat[batch_key].astype("str")
        batch_vec = df.apply(lambda row: ",".join(row), axis=1)

    return batch_vec.astype("category")


def _scatter_add_cp_bias_csr(
    X: cp.ndarray,
    out: cp.ndarray,
    *,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    bias: cp.ndarray,
    n_batches: int,
) -> None:
    n_cells = X.shape[0]
    n_pcs = X.shape[1]

    if n_cells < SCATTER_BIAS_KERNEL_THRESHOLD:
        _hc_sc.scatter_add_cat0(
            X,
            n_cells=n_cells,
            n_pcs=n_pcs,
            a=out,
            bias=bias,
            stream=cp.cuda.get_current_stream().ptr,
        )

    else:
        out[0] = X.T @ bias

    _hc_sc.scatter_add_block(
        X,
        cat_offsets=cat_offsets,
        cell_indices=cell_indices,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_batches=n_batches,
        a=out,
        bias=bias,
        stream=cp.cuda.get_current_stream().ptr,
    )


def _get_theta_array(
    theta: float | int | list[float | int] | np.ndarray | cp.ndarray,
    n_batches: int,
    dtype: cp.dtype,
) -> cp.ndarray:
    """
    Convert theta parameter to a CuPy array of appropriate shape.
    """
    # Handle scalar inputs (float, int)
    if isinstance(theta, float | int):
        return cp.ones(n_batches, dtype=dtype) * float(theta)

    # Handle array-like inputs (list, numpy array, cupy array)
    if isinstance(theta, list):
        theta_array = cp.array(theta, dtype=dtype)
    elif isinstance(theta, np.ndarray):
        theta_array = cp.array(theta, dtype=dtype)
    elif isinstance(theta, cp.ndarray):
        theta_array = theta.astype(dtype)
    else:
        raise ValueError(
            f"Theta must be float, int, list, numpy array, or cupy array, got {type(theta)}"
        )

    # Verify dimensions
    if theta_array.size != n_batches:
        raise ValueError(
            f"Theta array size ({theta_array.size}) must match number of batches ({n_batches})"
        )

    return theta_array.ravel()


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


def _fused_calc_pen_norm(
    similarities: cp.ndarray,
    penalty: cp.ndarray,
    cats: cp.ndarray,
    idx_in: cp.ndarray,
    R_out: cp.ndarray,
    *,
    term: float,
) -> None:
    """
    Fused kernel that computes _calc_R + _penalty_term + _normalize_cp in one pass.


    Parameters
    ----------
    similarities
        Full similarity matrix, shape (n_cells, n_clusters)
    penalty
        Penalty term matrix, shape (n_batches, n_clusters)
    cats
        Batch categories for current block, shape (block_size,)
    idx_in
        Cell indices for current block, shape (block_size,)
    R_out
        Output buffer, shape (block_size, n_clusters), modified in-place
    term
        Softmax temperature term (-2 / sigma)
    """
    block_size, n_clusters = R_out.shape

    # Convert idx_in to size_t (uint64) for kernel compatibility
    if idx_in.dtype != cp.uint64:
        idx_in = idx_in.astype(cp.uint64)

    _hc_pen.fused_pen_norm(
        similarities,
        penalty=penalty,
        cats=cats,
        idx_in=idx_in,
        R_out=R_out,
        term=float(term),
        n_rows=block_size,
        n_cols=n_clusters,
        stream=cp.cuda.get_current_stream().ptr,
    )


def _scatter_add_bias_batched(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    n_batches: int,
) -> cp.ndarray:
    """
    Compute Phi.T @ diag(R[:, k]) @ X for all clusters k simultaneously.

    Parameters
    ----------
    X
        Input data, shape (n_cells, n_pcs)
    R
        Cluster assignment matrix, shape (n_cells, n_clusters)
    cat_offsets
        CSR-like offsets for each batch category
    cell_indices
        CSR-like cell indices sorted by batch
    n_batches
        Number of batches

    Returns
    -------
    Phi_t_diag_R_X_all
        Shape (n_clusters, n_batches+1, n_pcs)
    """
    n_clusters = R.shape[1]
    n_pcs = X.shape[1]

    # Output array
    result = cp.zeros((n_clusters, n_batches + 1, n_pcs), dtype=X.dtype)

    # Row 0: bias contribution = R.T @ X for all clusters
    result[:, 0, :] = cp.dot(R.T, X)

    # Sort X and R by batch order once to avoid repeated fancy indexing
    X_sorted = X[cell_indices]
    R_sorted = R[cell_indices]

    # Rows 1 to n_batches: contiguous slices on sorted data
    for b in range(n_batches):
        start_idx = int(cat_offsets[b])
        end_idx = int(cat_offsets[b + 1])

        if end_idx > start_idx:
            # Contiguous slices - no copy needed!
            X_batch = X_sorted[start_idx:end_idx]
            R_batch = R_sorted[start_idx:end_idx]
            result[:, b + 1, :] = cp.dot(R_batch.T, X_batch)

    return result


def _apply_batched_correction(
    Z: cp.ndarray,
    W_all: cp.ndarray,
    cats: cp.ndarray,
    R: cp.ndarray,
) -> None:
    """
    Apply corrections from all clusters at once using a fused kernel.

    Parameters
    ----------
    Z
        Data to correct, shape (n_cells, n_pcs), modified in-place
    W_all
        All W matrices, shape (n_clusters, n_batches+1, n_pcs)
    cats
        Batch categories for each cell, shape (n_cells,)
    R
        Cluster assignment matrix, shape (n_cells, n_clusters)
    """
    n_cells, n_pcs = Z.shape
    n_clusters = R.shape[1]
    n_batches_p1 = W_all.shape[1]

    _hc_out.batched_correction(
        Z,
        W_all=W_all,
        cats=cats,
        R=R,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches_p1=n_batches_p1,
        stream=cp.cuda.get_current_stream().ptr,
    )
