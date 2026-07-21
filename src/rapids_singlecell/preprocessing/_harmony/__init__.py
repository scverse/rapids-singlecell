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
    _factorize_joint_codes,
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
_CORRECTION_WORKSPACE_LIMIT_BYTES = 1 << 30
_KMEANS_MAX_CELLS = 1_000_000
_KMEANS_MAX_SAMPLE_BYTES = 512 << 20


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
        Weight of the diversity penalty term. A scalar is broadcast to every
        variable. A sequence may have one value per key or one value per
        categorical level across all keys.

    tau
        Discounting factor on ``theta``. By default, there is no discounting.

    correction_method
        Choose which method for the correction step: ``original`` for original
        method, ``fast`` for improved method, or ``batched`` for batched
        processing. With one key, ``None`` automatically selects ``batched``
        unless its workspace would exceed 1 GiB, in which case ``fast`` is
        used. Multiple keys use the exact general-design solve and process
        clusters in workspace-bounded chunks when needed.

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
    batch_codes, n_levels = _get_batch_codes(batch_mat, batch_key)
    n_covariates = int(n_levels.size)
    n_batches = int(n_levels.sum())
    batch_counts = np.bincount(batch_codes.ravel(), minlength=n_batches)
    N_b = cp.asarray(batch_counts, dtype=Z.dtype)
    Pr_b = (N_b.reshape(-1, 1) / n_cells).astype(Z.dtype)

    # Keep the established one-dimensional layout for one covariate. Multiple
    # covariates use a cell-major matrix of disjoint marginal category codes.
    cats = cp.asarray(
        batch_codes[:, 0] if n_covariates == 1 else batch_codes, dtype=cp.int32
    )
    joint_cats = None
    joint_codes = None
    joint_offsets = None
    joint_cell_indices = None
    marginal_joint_offsets = None
    marginal_joint_indices = None
    if n_covariates > 1:
        joint_cats_host, joint_codes_host = _factorize_joint_codes(
            batch_codes, n_levels
        )
        joint_cats = cp.asarray(joint_cats_host, dtype=cp.int32)
        joint_codes = cp.asarray(joint_codes_host, dtype=cp.int32)
        n_joint_categories = joint_cats.shape[0]
        joint_offsets, joint_cell_indices = _create_category_index_mapping(
            joint_codes, n_joint_categories
        )
        marginal_joint_offsets, flat_joint_indices = _create_category_index_mapping(
            joint_cats.ravel(), n_batches
        )
        marginal_joint_indices = (flat_joint_indices // n_covariates).astype(
            cp.int32, copy=False
        )
    else:
        n_joint_categories = 0
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
    theta_array = _get_theta_array(theta, n_levels, Z.dtype)
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

    # Multi-covariate correction uses its own exact cluster-chunked solve.
    # For one covariate, retain the established arrowhead auto-selection.
    if correction_method is None:
        if n_covariates > 1:
            correction_method = "batched"
        else:
            nb1 = n_batches + 1
            inv_mats_bytes = n_clusters * nb1 * nb1 * Z.dtype.itemsize
            correction_method = (
                "batched"
                if inv_mats_bytes <= _CORRECTION_WORKSPACE_LIMIT_BYTES
                else "fast"
            )

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

    block_size = int(n_cells * block_proportion)
    joint_scatter_work = 2 * block_size + n_covariates * n_joint_categories
    marginal_scatter_work = 2 * n_covariates * block_size
    joint_workspace_bytes = n_joint_categories * n_clusters * Z.dtype.itemsize
    use_joint_scatter = (
        n_covariates > 1
        and joint_scatter_work < marginal_scatter_work
        and joint_workspace_bytes <= _CORRECTION_WORKSPACE_LIMIT_BYTES
    )

    # Pre-allocate C++ workspace buffers (reused across harmony iterations).
    cpp_workspace = _allocate_clustering_workspace(
        n_cells,
        n_pcs=Z.shape[1],
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_covariates=n_covariates,
        n_joint_categories=n_joint_categories,
        use_joint_scatter=use_joint_scatter,
        block_size=block_size,
        dtype=Z_norm.dtype,
    )
    if use_joint_scatter:
        _scatter_add_cp(
            R,
            cpp_workspace["O_joint"],
            joint_codes,
            1,
            n_batches=n_joint_categories,
        )

    empty_int = cp.empty(0, dtype=cp.int32)

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
            n_covariates=n_covariates,
            joint_codes=joint_codes if joint_codes is not None else empty_int,
            marginal_joint_offsets=(
                marginal_joint_offsets
                if marginal_joint_offsets is not None
                else empty_int
            ),
            marginal_joint_indices=(
                marginal_joint_indices
                if marginal_joint_indices is not None
                else empty_int
            ),
            n_joint_categories=n_joint_categories,
            use_joint_scatter=use_joint_scatter,
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
        if n_covariates > 1:
            if joint_cats is None or joint_codes is None:
                raise RuntimeError("Multi-key correction requires joint codes")
            Z_hat = _correction_multi(
                Z,
                R,
                O=O,
                lambda_kb=lambda_kb,
                cats=cats,
                n_batches=n_batches,
                n_covariates=n_covariates,
                joint_cats=joint_cats,
                joint_codes=joint_codes,
                joint_offsets=joint_offsets,
                joint_cell_indices=joint_cell_indices,
                marginal_joint_offsets=marginal_joint_offsets,
                marginal_joint_indices=marginal_joint_indices,
                output=Z_norm,
            )
        else:
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
                output=Z_norm,
            )
        # Check for convergence
        if _is_convergent_harmony(objectives_harmony, tol=tol_harmony):
            is_converged = True
            if verbose:
                print(f"Harmony converged in {i + 1} iterations")
            break
        # The normalized embedding is only needed by another clustering pass.
        # Correction has overwritten the old normalization buffer, so normalize
        # it in place instead of retaining another full embedding.
        if i + 1 < max_iter_harmony:
            Z_norm = _normalize_cp(Z_hat, p=2, out=Z_hat)

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
    # cuML's k-means initialization is not reliable at tens of millions of
    # rows. Fit centroids on a deterministic uniform sample, then compute the
    # initial assignments against all rows below.
    max_sample_cells = min(
        _KMEANS_MAX_CELLS,
        max(1, _KMEANS_MAX_SAMPLE_BYTES // (Z_norm.shape[1] * Z_norm.dtype.itemsize)),
    )
    kmeans_input = Z_norm
    if Z_norm.shape[0] > max_sample_cells:
        sample_indices = np.random.default_rng(random_state).choice(
            Z_norm.shape[0], size=max_sample_cells, replace=False
        )
        sample_indices.sort()
        kmeans_input = cp.ascontiguousarray(
            Z_norm[cp.asarray(sample_indices, dtype=cp.int32)]
        )

    # Run k-means to get initial cluster centers
    kmeans = CumlKMeans(
        n_clusters=n_clusters,
        init="k-means||",
        n_init=1,
        max_iter=25,
        random_state=random_state,
    )
    kmeans.fit(kmeans_input)
    Y = kmeans.cluster_centers_.astype(Z_norm.dtype)
    if kmeans_input is not Z_norm:
        del kmeans_input
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
    n_covariates: int,
    n_joint_categories: int,
    use_joint_scatter: bool,
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
        "cats_in": cp.empty(block_size * n_covariates, dtype=cp.int32),
        "O_joint": cp.zeros(
            (n_joint_categories if use_joint_scatter else 1, n_clusters), dtype=dtype
        ),
        "joint_codes_in": cp.empty(
            max(1, block_size) if use_joint_scatter else 1, dtype=cp.int32
        ),
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
    n_covariates: int = 1,
    joint_codes: cp.ndarray,
    marginal_joint_offsets: cp.ndarray,
    marginal_joint_indices: cp.ndarray,
    n_joint_categories: int,
    use_joint_scatter: bool,
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
        joint_codes=joint_codes,
        marginal_joint_offsets=marginal_joint_offsets,
        marginal_joint_indices=marginal_joint_indices,
        theta=theta,
        **cpp_workspace,
        n_cells=n_cells,
        n_pcs=Z_norm.shape[1],
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_covariates=n_covariates,
        n_joint_categories=n_joint_categories,
        use_joint_scatter=use_joint_scatter,
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
    output: cp.ndarray | None = None,
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
            output=output,
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
            output=output,
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
            output=output,
        )


def _correction_multi(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    O: cp.ndarray,
    lambda_kb: cp.ndarray,
    cats: cp.ndarray,
    n_batches: int,
    n_covariates: int,
    joint_cats: cp.ndarray,
    joint_codes: cp.ndarray,
    joint_offsets: cp.ndarray | None = None,
    joint_cell_indices: cp.ndarray | None = None,
    marginal_joint_offsets: cp.ndarray | None = None,
    marginal_joint_indices: cp.ndarray | None = None,
    output: cp.ndarray | None = None,
) -> cp.ndarray:
    """Apply the exact general-design correction in bounded cluster chunks."""
    n_cells, n_pcs = X.shape
    n_clusters = R.shape[1]
    n_joint_categories = joint_cats.shape[0]
    nb1 = n_batches + 1
    if joint_offsets is None or joint_cell_indices is None:
        joint_offsets, joint_cell_indices = _create_category_index_mapping(
            joint_codes, n_joint_categories
        )
    if marginal_joint_offsets is None or marginal_joint_indices is None:
        marginal_joint_offsets, flat_joint_indices = _create_category_index_mapping(
            joint_cats.ravel(), n_batches
        )
        marginal_joint_indices = (flat_joint_indices // n_covariates).astype(
            cp.int32, copy=False
        )
    cluster_chunk_size = _multi_correction_cluster_chunk_size(
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_joint_categories=n_joint_categories,
        itemsize=X.dtype.itemsize,
    )

    Z = _correction_output(X, output)
    for cluster_start in range(0, n_clusters, cluster_chunk_size):
        cluster_stop = min(cluster_start + cluster_chunk_size, n_clusters)
        chunk_n_clusters = cluster_stop - cluster_start

        if chunk_n_clusters == n_clusters:
            R_chunk = R
            O_chunk = O
            lambda_kb_chunk = lambda_kb
        else:
            R_chunk = cp.ascontiguousarray(R[:, cluster_start:cluster_stop])
            O_chunk = cp.ascontiguousarray(O[:, cluster_start:cluster_stop])
            lambda_kb_chunk = cp.ascontiguousarray(
                lambda_kb[:, cluster_start:cluster_stop]
            )

        gram = cp.empty((chunk_n_clusters, nb1, nb1), dtype=X.dtype)
        rhs = cp.empty((chunk_n_clusters, nb1, n_pcs), dtype=X.dtype)
        joint_O = cp.empty((n_joint_categories, chunk_n_clusters), dtype=X.dtype)
        joint_rhs = cp.empty(
            (n_joint_categories, chunk_n_clusters, n_pcs), dtype=X.dtype
        )
        active_mask = cp.ascontiguousarray(
            (lambda_kb_chunk < X.dtype.type(_SUPPRESS_PENALTY)).astype(cp.uint8)
        )

        _hc_corr_b.prepare_multi(
            X,
            R=R_chunk,
            O=O_chunk,
            joint_codes=joint_codes,
            joint_cats=joint_cats,
            joint_offsets=joint_offsets,
            joint_cell_indices=joint_cell_indices,
            marginal_joint_offsets=marginal_joint_offsets,
            marginal_joint_indices=marginal_joint_indices,
            lambda_kb=lambda_kb_chunk,
            active_mask=active_mask,
            n_cells=n_cells,
            n_pcs=n_pcs,
            n_clusters=chunk_n_clusters,
            n_batches=n_batches,
            n_covariates=n_covariates,
            n_joint_categories=n_joint_categories,
            gram=gram,
            rhs=rhs,
            joint_O=joint_O,
            joint_rhs=joint_rhs,
            stream=cp.cuda.get_current_stream().ptr,
            handle=cp.cuda.device.get_cublas_handle(),
        )

        W_all = _solve_spd_batched(gram, rhs)
        W_all[:, 0, :] = 0
        _hc_corr_b.apply_multi(
            X,
            R=R_chunk,
            W_all=W_all,
            cats=cats,
            n_cells=n_cells,
            n_pcs=n_pcs,
            n_clusters=chunk_n_clusters,
            n_batches=n_batches,
            n_covariates=n_covariates,
            initialize_output=cluster_start == 0,
            Z=Z,
            stream=cp.cuda.get_current_stream().ptr,
        )
        del (
            active_mask,
            gram,
            joint_O,
            joint_rhs,
            lambda_kb_chunk,
            O_chunk,
            R_chunk,
            rhs,
            W_all,
        )
    return Z


def _solve_spd_batched(gram: cp.ndarray, rhs: cp.ndarray) -> cp.ndarray:
    """Solve the symmetric positive-definite cluster systems in one batch."""
    n_matrices, matrix_size, _ = gram.shape
    n_rhs = rhs.shape[2]
    if n_matrices == 0:
        return cp.empty_like(rhs)

    # potrf and trsm are in-place. Preserve the original inputs so an unusual
    # rank-deficient system can fall back to the general LU solve.
    gram_work = cp.array(gram, order="C", copy=True)
    rhs_work = cp.array(rhs, order="C", copy=True)
    info = cp.empty(n_matrices, dtype=cp.int32)

    matrix_offsets = cp.arange(n_matrices, dtype=cp.uint64)
    gram_ptrs = gram_work.data.ptr + matrix_offsets * cp.uint64(
        matrix_size * matrix_size * gram.dtype.itemsize
    )
    rhs_ptrs = rhs_work.data.ptr + matrix_offsets * cp.uint64(
        matrix_size * n_rhs * rhs.dtype.itemsize
    )

    if gram.dtype == cp.float32:
        potrf_batched = cp.cuda.cusolver.spotrfBatched
        trsm_batched = cp.cuda.cublas.strsmBatched
        scalar_dtype = np.float32
    elif gram.dtype == cp.float64:
        potrf_batched = cp.cuda.cusolver.dpotrfBatched
        trsm_batched = cp.cuda.cublas.dtrsmBatched
        scalar_dtype = np.float64
    else:
        raise TypeError("Batched Harmony correction requires float32 or float64")

    stream = cp.cuda.get_current_stream()
    cusolver_handle = cp.cuda.device.get_cusolver_handle()
    cublas_handle = cp.cuda.device.get_cublas_handle()
    cp.cuda.cusolver.setStream(cusolver_handle, stream.ptr)
    cp.cuda.cublas.setStream(cublas_handle, stream.ptr)
    potrf_batched(
        cusolver_handle,
        cp.cuda.cublas.CUBLAS_FILL_MODE_LOWER,
        matrix_size,
        gram_ptrs.data.ptr,
        matrix_size,
        info.data.ptr,
        n_matrices,
    )

    # A C-contiguous (n, d) RHS is a column-major (d, n) matrix. Two
    # right-side triangular solves therefore produce rhs.T @ inv(gram)
    # directly in the original C-contiguous layout, without transposes.
    one = np.ones(1, dtype=scalar_dtype)
    for operation in (
        cp.cuda.cublas.CUBLAS_OP_T,
        cp.cuda.cublas.CUBLAS_OP_N,
    ):
        trsm_batched(
            cublas_handle,
            cp.cuda.cublas.CUBLAS_SIDE_RIGHT,
            cp.cuda.cublas.CUBLAS_FILL_MODE_LOWER,
            operation,
            cp.cuda.cublas.CUBLAS_DIAG_NON_UNIT,
            n_rhs,
            matrix_size,
            one.ctypes.data,
            gram_ptrs.data.ptr,
            matrix_size,
            rhs_ptrs.data.ptr,
            n_rhs,
            n_matrices,
        )
    if np.any(cp.asnumpy(info) != 0):
        return cp.ascontiguousarray(cp.linalg.solve(gram, rhs))
    return rhs_work


def _multi_correction_cluster_chunk_size(
    *,
    n_cells: int,
    n_pcs: int,
    n_clusters: int,
    n_batches: int,
    n_joint_categories: int,
    itemsize: int,
) -> int:
    """Choose a cluster chunk that keeps multi-key scratch below 1 GiB."""
    if n_clusters < 1:
        return 0

    nb1 = n_batches + 1

    def _workspace_bytes(chunk_n_clusters: int, *, copy_cluster_slices: bool) -> int:
        # CuPy's solve preserves its inputs. Account conservatively for the
        # Gram/RHS copies, solve output, and a possible contiguous output copy.
        float_elements_per_cluster = (
            3 * nb1 * nb1 + 3 * nb1 * n_pcs + n_joint_categories * (n_pcs + 1)
        )
        if copy_cluster_slices:
            float_elements_per_cluster += n_cells + 2 * n_batches

        # Active-mask construction can temporarily hold bool and uint8 arrays;
        # LU also needs pivots and device pointer arrays per cluster.
        auxiliary_bytes_per_cluster = 2 * n_batches + 4 * nb1 + 16
        return chunk_n_clusters * (
            float_elements_per_cluster * itemsize + auxiliary_bytes_per_cluster
        )

    full_workspace = _workspace_bytes(n_clusters, copy_cluster_slices=False)
    if full_workspace <= _CORRECTION_WORKSPACE_LIMIT_BYTES:
        return n_clusters

    single_cluster_workspace = _workspace_bytes(1, copy_cluster_slices=True)
    if single_cluster_workspace > _CORRECTION_WORKSPACE_LIMIT_BYTES:
        gib = single_cluster_workspace / _CORRECTION_WORKSPACE_LIMIT_BYTES
        raise MemoryError(
            "A single multi-key Harmony correction cluster requires "
            f"approximately {gib:.2f} GiB of scratch space; reduce the "
            "number of batch levels, cells, or embedding dimensions."
        )

    return min(
        n_clusters,
        _CORRECTION_WORKSPACE_LIMIT_BYTES // single_cluster_workspace,
    )


def _correction_output(X: cp.ndarray, output: cp.ndarray | None = None) -> cp.ndarray:
    """Return a validated correction output buffer."""
    if output is None:
        return cp.empty_like(X)
    if output.shape != X.shape or output.dtype != X.dtype:
        raise ValueError("Correction output must match the input shape and dtype")
    if not output.flags.c_contiguous:
        raise ValueError("Correction output must be C-contiguous")
    return output


def _correction_original(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    lambda_kb: cp.ndarray,
    cats: cp.ndarray,
    n_batches: int,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    output: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Apply the original correction method from the Harmony paper.
    """
    n_clusters = R.shape[1]

    Z = _correction_output(X, output)
    cp.copyto(Z, X)
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
    output: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Apply the fast correction method (an optimization over the original method).
    """
    n_cells = X.shape[0]
    n_pcs = X.shape[1]
    n_clusters = R.shape[1]
    nb1 = n_batches + 1
    dtype = X.dtype

    Z = _correction_output(X, output)
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
    output: cp.ndarray | None = None,
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

    # Reuse the old normalized embedding for output. Category GEMMs gather into
    # a bounded scratch buffer instead of duplicating all N rows of X and R.
    Z = _correction_output(X, output)
    inv_mats = cp.empty((n_clusters, nb1, nb1), dtype=dtype)
    Phi_t_diag_R_X_all = cp.empty((n_clusters, nb1, n_pcs), dtype=dtype)
    W_all = cp.empty((n_clusters, nb1, n_pcs), dtype=dtype)
    g_factor = cp.empty((n_clusters, n_batches), dtype=dtype)
    g_P_row0 = cp.empty((n_clusters, n_batches), dtype=dtype)
    max_batch_cells = int(cp.max(cp.diff(cat_offsets)).item())
    # A category containing every cell can use X and R directly in C++.
    batch_chunk_size = 1 if max_batch_cells == n_cells else max_batch_cells
    X_batch = cp.empty((batch_chunk_size, n_pcs), dtype=dtype)
    R_batch = cp.empty((batch_chunk_size, n_clusters), dtype=dtype)

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
        X_batch=X_batch,
        R_batch=R_batch,
        batch_chunk_size=batch_chunk_size,
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
