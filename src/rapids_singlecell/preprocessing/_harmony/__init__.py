from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
from cuml import KMeans as CumlKMeans

from rapids_singlecell._cuda import _harmony_clustering_cuda as _hc_cl
from rapids_singlecell._cuda import _harmony_correction_batched_cuda as _hc_corr_b
from rapids_singlecell._cuda import _harmony_correction_cuda as _hc_corr
from rapids_singlecell._utils import _create_category_index_mapping

from ._fuses import (
    _calc_R,
)
from ._helper import (
    _choose_colsum_algo_benchmark,
    _choose_colsum_algo_heuristic,
    _column_sum,
    _column_sum_atomic,
    _gemm_colsum,
    _get_aggregated_matrix,
    _get_batch_codes,
    _get_theta_array,
    _normalize_cp,
    _outer_cp,
    _scatter_add_cp,
    _scatter_add_cp_bias_csr,
    _Z_correction,
)

if TYPE_CHECKING:
    import pandas as pd

COLSUM_ALGO = Literal["columns", "atomics", "gemm", "benchmark"]
_SUPPRESS_PENALTY = 1e30


def harmonize(
    Z: cp.array,
    batch_mat: pd.DataFrame,
    batch_key: str | list[str],
    *,
    n_clusters: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int = 200,
    tol_harmony: float = 1e-4,
    tol_clustering: float = 1e-5,
    ridge_lambda: float = 1.0,
    sigma: float = 0.1,
    block_proportion: float = 0.05,
    theta: float | int | list[float] | np.ndarray | cp.ndarray = 2.0,
    tau: int = 0,
    correction_method: str | None = None,
    colsum_algo: COLSUM_ALGO | None = None,
    random_state: int = 0,
    stabilized_penalty: bool = True,
    dynamic_lambda: bool = True,
    alpha: float = 0.2,
    batch_prune_threshold: float | None = 1e-5,
    verbose: bool = False,
) -> cp.array:
    """
    Integrate data using Harmony algorithm.

    Parameters
    ----------
    Z
        The input embedding with rows for cells (N) and columns for embedding coordinates (d).

    batch_mat
        The cell barcode information as data frame, with rows for cells (N) and columns for cell attributes.

    batch_key
        Cell attribute(s) from ``batch_mat`` to identify batches.

    n_clusters
        Number of clusters used in Harmony algorithm. If ``None``, choose the minimum of 100 and N / 30.

    max_iter_harmony
        Maximum iterations on running Harmony if not converged.

    max_iter_clustering
        Within each Harmony iteration, maximum iterations on the clustering step if not converged.

    tol_harmony
        Tolerance on justifying convergence of Harmony over objective function values.

    tol_clustering
        Tolerance on justifying convergence of the clustering step over objective function values within each Harmony iteration.

    ridge_lambda
        Hyperparameter of ridge regression on the correction step.

    sigma
        Weight of the entropy term in objective function.

    block_proportion
        Proportion of block size in one update operation of clustering step.

    theta
        Weight of the diversity penalty term in objective function.

    tau
        Discounting factor on ``theta``. By default, there is no discounting.

    correction_method
        Choose which method for the correction step: ``original`` for original method, ``fast`` for improved method, ``batched`` for batched processing of all clusters simultaneously (fastest but needs more memory). If ``None`` (default), automatically selects ``batched`` unless the workspace would exceed 1 GB, in which case ``fast`` is used.

    colsum_algo
        Choose which algorithm to use for column sum. If `None`, choose the algorithm based on the number of rows and columns. If `'benchmark'`, benchmark all algorithms and choose the best one.

    random_state
        Random seed for reproducing results.

    stabilized_penalty
        If ``True`` (default), use the Harmony2 stabilized diversity penalty
        that prevents overintegration when batches are absent from clusters.

    dynamic_lambda
        If ``True`` (default), use per-cluster-per-batch ridge regularization
        ``lambda_kb = alpha * E_kb`` instead of a fixed ``ridge_lambda``.

    alpha
        Scaling factor for dynamic lambda. Only used when ``dynamic_lambda=True``.

    batch_prune_threshold
        Prune batches from clusters when ``O_kb / N_b < threshold``.
        Pruned batches receive zero correction for that cluster.
        Only used when ``dynamic_lambda=True``. Set to ``None`` to disable pruning.

    verbose
        Whether to print benchmarking results for the column sum algorithm and the number of iterations until convergence.

    Returns
    -------
    The integrated embedding by Harmony, of the same shape as the input embedding.
    """

    Z_norm = _normalize_cp(Z)
    n_cells = Z.shape[0]

    # Process batch information
    batch_codes = _get_batch_codes(batch_mat, batch_key)
    n_batches = batch_codes.cat.categories.size
    N_b = cp.array(batch_codes.value_counts(sort=False).values, dtype=Z.dtype)
    Pr_b = (N_b.reshape(-1, 1) / len(batch_codes)).astype(Z.dtype)

    cats = cp.array(batch_codes.cat.codes.values, dtype=cp.int32)
    cat_offsets, cell_indices = _create_category_index_mapping(cats, n_batches)

    # Set up parameters
    if max_iter_harmony < 1:
        raise ValueError("max_iter_harmony must be >= 1")
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))
        n_clusters = max(n_clusters, 2)

    # TODO: Allow for multiple colsum algorithms in a list
    assert colsum_algo in ["columns", "atomics", "gemm", "benchmark", None]
    colsum_func_big = _choose_colsum_algo_heuristic(n_cells, n_clusters, None)
    if colsum_algo == "benchmark":
        colsum_func_small = _choose_colsum_algo_benchmark(
            int(n_cells * block_proportion), n_clusters, Z.dtype, verbose=verbose
        )
    else:
        colsum_func_small = _choose_colsum_algo_heuristic(
            int(n_cells * block_proportion), n_clusters, colsum_algo
        )
    theta_array = _get_theta_array(theta, n_batches, Z.dtype)
    if tau > 0:
        theta_array = theta_array * (1 - cp.exp(-N_b / (n_clusters * tau)) ** 2)
    theta_array = cp.ascontiguousarray(theta_array.ravel())

    # Validate parameters
    assert block_proportion > 0 and block_proportion <= 1
    if dynamic_lambda:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(
                f"alpha must be a finite positive number when dynamic_lambda=True, got {alpha}."
            )
        if batch_prune_threshold is not None and not (0 <= batch_prune_threshold <= 1):
            raise ValueError(
                f"batch_prune_threshold must be in [0, 1] or None, got {batch_prune_threshold}."
            )
    if correction_method is not None and correction_method not in {
        "fast",
        "original",
        "batched",
    }:
        raise ValueError("correction_method must be 'fast', 'original', or 'batched'.")

    # Auto-select correction method: "batched" unless inv_mats would exceed 1 GB
    if correction_method is None:
        ONE_GB = 1 << 30
        nb1 = n_batches + 1
        inv_mats_bytes = n_clusters * nb1 * nb1 * Z.dtype.itemsize
        correction_method = "batched" if inv_mats_bytes <= ONE_GB else "fast"

    # Set random seed
    cp.random.seed(random_state)

    # Initialize algorithm
    R, E, O, objectives_harmony = _initialize_centroids(
        Z_norm,
        n_clusters=n_clusters,
        sigma=sigma,
        Pr_b=Pr_b,
        theta=theta_array,
        random_state=random_state,
        cats=cats,
        n_batches=n_batches,
        colsum_func=colsum_func_big,
        stabilized_penalty=stabilized_penalty,
    )

    # Pre-allocate C++ workspace buffers (reused across harmony iterations)
    cpp_workspace = _allocate_clustering_workspace(
        n_cells,
        n_pcs=Z.shape[1],
        n_clusters=n_clusters,
        n_batches=n_batches,
        block_size=int(n_cells * block_proportion),
        dtype=Z_norm.dtype,
    )

    # Main harmony iterations
    is_converged = False

    for i in range(max_iter_harmony):
        # Clustering step
        _clustering(
            Z_norm,
            Pr_b=Pr_b,
            cats=cats,
            R=R,
            E=E,
            O=O,
            theta=theta_array,
            tol=tol_clustering,
            objectives_harmony=objectives_harmony,
            max_iter=max_iter_clustering,
            sigma=sigma,
            block_proportion=block_proportion,
            colsum_func=colsum_func_small,
            n_batches=n_batches,
            random_state=random_state + i * 1000003,
            stabilized_penalty=stabilized_penalty,
            cpp_workspace=cpp_workspace,
        )
        # Compute per-(k,b) ridge regularization
        lambda_kb = _compute_lambda_kb(
            E,
            O=O,
            N_b=N_b,
            alpha=alpha,
            threshold=batch_prune_threshold,
            ridge_lambda=ridge_lambda,
            dynamic_lambda=dynamic_lambda,
        )
        # Correction step
        Z_hat = _correction(
            Z,
            R=R,
            O=O,
            lambda_kb=lambda_kb,
            correction_method=correction_method,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
        )
        # Normalize corrected data
        Z_norm = _normalize_cp(Z_hat, p=2)
        # Check for convergence
        if _is_convergent_harmony(objectives_harmony, tol=tol_harmony):
            is_converged = True
            if verbose:
                print(f"Harmony converged in {i + 1} iterations")
            break

    if not is_converged:
        warnings.warn(
            "Harmony did not converge. Consider increasing the number of iterations"
        )

    return Z_hat


def _initialize_centroids(
    Z_norm: cp.ndarray,
    *,
    n_clusters: int,
    sigma: float,
    Pr_b: cp.ndarray,
    theta: cp.ndarray,
    random_state: int = 0,
    cats: cp.ndarray,
    n_batches: int,
    colsum_func: callable = None,
    stabilized_penalty: bool = True,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, list]:
    """
    Initialize cluster centroids and related matrices for Harmony algorithm.

    Returns:
        R: Cluster assignment matrix
        E: Expected cluster assignment by batch
        O: Observed cluster assignment by batch
        objectives_harmony: List to store objective function values
    """
    # Run k-means to get initial cluster centers
    kmeans = CumlKMeans(
        n_clusters=n_clusters,
        init="k-means||",
        n_init=1,
        max_iter=25,
        random_state=random_state,
    )
    kmeans.fit(Z_norm)
    Y = kmeans.cluster_centers_.astype(Z_norm.dtype)
    Y_norm = _normalize_cp(Y, p=2)

    # Initialize cluster assignment matrix R
    term = Z_norm.dtype.type(-2 / sigma)
    similarities = cp.dot(Z_norm, Y_norm.T)
    R = _calc_R(term, similarities)
    R = _normalize_cp(R, p=1)

    # Initialize E (expected) and O (observed) matrices
    R_sum = colsum_func(R)
    E = cp.zeros((n_batches, R.shape[1]), dtype=Z_norm.dtype)
    _outer_cp(E, Pr_b, R_sum, 1)

    O = cp.zeros((n_batches, R.shape[1]), dtype=Z_norm.dtype)
    _scatter_add_cp(R, O, cats, 1, n_batches=n_batches)

    # Initialize objectives list
    objectives_harmony = []
    _compute_objective(
        similarities,
        R=R,
        theta=theta,
        sigma=sigma,
        O=O,
        E=E,
        objective_arr=objectives_harmony,
        stabilized_penalty=stabilized_penalty,
    )

    return R, E, O, objectives_harmony


def _allocate_clustering_workspace(
    n_cells: int,
    *,
    n_pcs: int,
    n_clusters: int,
    n_batches: int,
    block_size: int,
    dtype: cp.dtype,
) -> dict:
    """Pre-allocate workspace buffers for the C++ clustering loop."""
    cub_temp_bytes = _hc_cl.get_cub_sort_temp_bytes(n_cells=n_cells)
    return {
        "Y": cp.empty((n_clusters, n_pcs), dtype=dtype),
        "Y_norm": cp.empty((n_clusters, n_pcs), dtype=dtype),
        "similarities": cp.empty((n_cells, n_clusters), dtype=dtype),
        "idx_list": cp.empty(n_cells, dtype=cp.int32),
        "idx_list_alt": cp.empty(n_cells, dtype=cp.int32),
        "sort_keys": cp.empty(n_cells, dtype=cp.uint32),
        "sort_keys_alt": cp.empty(n_cells, dtype=cp.uint32),
        "cub_temp": cp.empty(cub_temp_bytes, dtype=cp.uint8),
        "R_out_buffer": cp.empty((block_size, n_clusters), dtype=dtype),
        "cats_in": cp.empty(block_size, dtype=cp.int32),
        "R_in_sum": cp.empty(n_clusters, dtype=dtype),
        "R_out_sum": cp.empty(n_clusters, dtype=dtype),
        "penalty": cp.empty((n_batches, n_clusters), dtype=dtype),
        "obj_scalar": cp.empty(1, dtype=dtype),
        "ones_vec": cp.ones(block_size, dtype=dtype),
        "last_obj": cp.zeros(1, dtype=dtype),
    }


# Map colsum function to C++ enum: 0=columns, 1=atomics, 2=gemm
_COLSUM_MAP = {
    _column_sum: 0,
    _column_sum_atomic: 1,
    _gemm_colsum: 2,
}


def _clustering(
    Z_norm: cp.ndarray,
    *,
    Pr_b: cp.ndarray,
    cats: cp.ndarray,
    R: cp.ndarray,
    E: cp.ndarray,
    O: cp.ndarray,
    theta: cp.ndarray,
    tol: float,
    objectives_harmony: list,
    max_iter: int,
    sigma: float,
    block_proportion: float,
    colsum_func: callable = None,
    n_batches: int = 0,
    random_state: int = 0,
    stabilized_penalty: bool = True,
    cpp_workspace: dict = None,
) -> None:
    """
    Perform iterative clustering updates on normalized input data, adjusting
    cluster assignments and associated penalty terms until convergence or
    maximum iterations are reached.

    This function operates in-place to update the cluster assignment matrix (R)
    and penalty-related matrices (O and E).
    """
    n_cells = Z_norm.shape[0]
    n_clusters = R.shape[1]
    block_size = int(n_cells * block_proportion)
    colsum_algo_int = _COLSUM_MAP.get(colsum_func, 2)

    _hc_cl.clustering_loop(
        Z_norm,
        R=R,
        E=E,
        O=O,
        Pr_b=Pr_b.ravel(),
        cats=cats,
        theta=theta,
        **cpp_workspace,
        n_cells=n_cells,
        n_pcs=Z_norm.shape[1],
        n_clusters=n_clusters,
        n_batches=n_batches,
        block_size=block_size,
        colsum_algo=colsum_algo_int,
        sigma=float(sigma),
        tol=float(tol),
        max_iter=max_iter,
        seed=random_state & 0xFFFFFFFF,
        stabilized=stabilized_penalty,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )
    objectives_harmony.append(float(cpp_workspace["last_obj"][0]))


def _compute_lambda_kb(
    E: cp.ndarray,
    *,
    O: cp.ndarray,
    N_b: cp.ndarray,
    alpha: float,
    threshold: float | None,
    ridge_lambda: float,
    dynamic_lambda: bool,
) -> cp.ndarray:
    """Compute per-(k,b) ridge regularization array."""
    sentinel = E.dtype.type(_SUPPRESS_PENALTY)
    if not dynamic_lambda:
        lambda_kb = cp.full_like(E, ridge_lambda)
    else:
        lambda_kb = (alpha * E).astype(E.dtype)
        if threshold is not None:
            safe_N_b = cp.where(N_b > 0, N_b, cp.ones_like(N_b))
            prune_mask = (O / safe_N_b[:, None]) < threshold
            prune_mask |= N_b[:, None] == 0
            lambda_kb[prune_mask] = sentinel
    # Where both O and lambda_kb are zero, the kernel computes 1/(O+lambda)
    # which would divide by zero.  Both values are exactly zero here: O comes
    # from an integer scatter-add of assignments, and lambda_kb is alpha*E
    # where E is also zero for absent batch-cluster pairs.
    lambda_kb[(O + lambda_kb) == 0] = sentinel
    return lambda_kb


def _correction(
    X: cp.ndarray,
    *,
    R: cp.ndarray,
    O: cp.ndarray,
    lambda_kb: cp.ndarray,
    correction_method: str = "batched",
    cats: cp.ndarray,
    n_batches: int,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
) -> cp.ndarray:
    """
    Apply correction to the embedding based on the specified method.
    """
    if correction_method == "batched":
        return _correction_batched(
            X,
            R,
            O=O,
            lambda_kb=lambda_kb,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
        )
    elif correction_method == "fast":
        return _correction_fast(
            X,
            R,
            O=O,
            lambda_kb=lambda_kb,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
        )
    else:
        return _correction_original(
            X,
            R,
            lambda_kb=lambda_kb,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
        )


def _correction_original(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    lambda_kb: cp.ndarray,
    cats: cp.ndarray,
    n_batches: int,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
) -> cp.ndarray:
    """
    Apply the original correction method from the Harmony paper.
    """
    n_clusters = R.shape[1]

    Z = X.copy()
    for k in range(n_clusters):
        Lambda_diag = cp.zeros(n_batches + 1, dtype=X.dtype)
        Lambda_diag[1:] = lambda_kb[:, k]
        Lambda = cp.diag(Lambda_diag)
        R_col = R[:, k].copy()
        scatter_sum = cp.zeros(n_batches, dtype=R.dtype)
        cp.add.at(scatter_sum, cats, R_col)
        aggregated_matrix = cp.zeros((n_batches + 1, n_batches + 1), dtype=X.dtype)
        _get_aggregated_matrix(aggregated_matrix, scatter_sum, n_batches=n_batches)
        inv_mat = cp.linalg.inv(aggregated_matrix + Lambda)
        Phi_t_diag_R_X = cp.zeros((n_batches + 1, X.shape[1]), dtype=X.dtype)
        _scatter_add_cp_bias_csr(
            X,
            Phi_t_diag_R_X,
            cat_offsets=cat_offsets,
            cell_indices=cell_indices,
            bias=R_col,
            n_batches=n_batches,
        )
        W = cp.dot(inv_mat, Phi_t_diag_R_X)
        W[0, :] = 0
        _Z_correction(Z, W, cats, R_col)
    return Z


def _correction_fast(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    O: cp.ndarray,
    lambda_kb: cp.ndarray,
    cats: cp.ndarray,
    n_batches: int,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
) -> cp.ndarray:
    """
    Apply the fast correction method (an optimization over the original method).
    """
    n_cells = X.shape[0]
    n_pcs = X.shape[1]
    n_clusters = R.shape[1]
    nb1 = n_batches + 1
    dtype = X.dtype

    Z = cp.empty_like(X)
    inv_mat = cp.empty((nb1, nb1), dtype=dtype)
    R_col = cp.empty(n_cells, dtype=dtype)
    Phi_t_diag_R_X = cp.empty((nb1, n_pcs), dtype=dtype)
    W = cp.empty((nb1, n_pcs), dtype=dtype)
    g_factor = cp.empty(n_batches, dtype=dtype)
    g_P_row0 = cp.empty(n_batches, dtype=dtype)

    _hc_corr.correction_fast(
        X,
        R=R,
        O=O,
        cats=cats,
        cat_offsets=cat_offsets,
        cell_indices=cell_indices,
        lambda_kb=lambda_kb,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        Z=Z,
        inv_mat=inv_mat,
        R_col=R_col,
        Phi_t_diag_R_X=Phi_t_diag_R_X,
        W=W,
        g_factor=g_factor,
        g_P_row0=g_P_row0,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )
    return Z


def _correction_batched(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    O: cp.ndarray,
    lambda_kb: cp.ndarray,
    cats: cp.ndarray,
    n_batches: int,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
) -> cp.ndarray:
    """
    Batched correction method - process all clusters simultaneously.

    Single C++ call that fuses all steps: inv_mats computation, Phi_t_diag_R_X
    via cuBLAS GEMMs, W_all via strided batched GEMM, and correction kernel.
    """
    n_cells, n_pcs = X.shape
    n_clusters = R.shape[1]
    nb1 = n_batches + 1
    dtype = X.dtype

    # Allocate workspace
    Z = cp.empty_like(X)
    inv_mats = cp.empty((n_clusters, nb1, nb1), dtype=dtype)
    Phi_t_diag_R_X_all = cp.empty((n_clusters, nb1, n_pcs), dtype=dtype)
    W_all = cp.empty((n_clusters, nb1, n_pcs), dtype=dtype)
    g_factor = cp.empty((n_clusters, n_batches), dtype=dtype)
    g_P_row0 = cp.empty((n_clusters, n_batches), dtype=dtype)
    X_sorted = cp.empty((n_cells, n_pcs), dtype=dtype)
    R_sorted = cp.empty((n_cells, n_clusters), dtype=dtype)

    _hc_corr_b.correction_batched(
        X,
        R=R,
        O=O,
        cats=cats,
        cat_offsets=cat_offsets,
        cell_indices=cell_indices,
        lambda_kb=lambda_kb,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        Z=Z,
        inv_mats=inv_mats,
        Phi_t_diag_R_X_all=Phi_t_diag_R_X_all,
        W_all=W_all,
        g_factor=g_factor,
        g_P_row0=g_P_row0,
        X_sorted=X_sorted,
        R_sorted=R_sorted,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )
    return Z


def _compute_objective(
    similarities: cp.ndarray,
    *,
    R: cp.ndarray,
    theta: cp.ndarray,
    sigma: float,
    O: cp.ndarray,
    E: cp.ndarray,
    objective_arr: list,
    stabilized_penalty: bool = True,
) -> None:
    """
    Compute the objective function value for Harmony.

    Uses a fused C++ implementation that computes all three terms
    (kmeans error, entropy, diversity) in a single pass with internal
    row-normalization of R.
    """
    n_cells, n_clusters = R.shape
    n_batches = O.shape[0]
    obj_scalar = cp.zeros(1, dtype=R.dtype)
    obj = _hc_cl.compute_objective(
        R,
        similarities=similarities,
        O=O,
        E=E,
        theta=theta,
        sigma=float(sigma),
        obj_scalar=obj_scalar,
        n_cells=n_cells,
        n_clusters=n_clusters,
        n_batches=n_batches,
        stabilized=stabilized_penalty,
        stream=cp.cuda.get_current_stream().ptr,
    )
    objective_arr.append(obj)


def _is_convergent_harmony(objectives_harmony: list, tol: float) -> bool:
    """
    Check if the Harmony algorithm has converged based on the objective function values.

    Returns True if the relative improvement in objective is below tolerance.
    """
    if len(objectives_harmony) < 2:
        return False

    obj_old = objectives_harmony[-2]
    obj_new = objectives_harmony[-1]

    return (obj_old - obj_new) < tol * np.abs(obj_old)
