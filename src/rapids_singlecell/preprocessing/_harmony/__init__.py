from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _harmony_clustering_cuda as _clustering_cuda
from rapids_singlecell._cuda import (
    _harmony_correction_batched_cuda as _correction_batched_cuda,
)
from rapids_singlecell._cuda import _harmony_correction_cuda as _correction_cuda
from rapids_singlecell._utils import _create_category_index_mapping
from rapids_singlecell._utils._random import _seed_from_rng

from ._helper import (
    _factorize_joint_codes,
    _get_batch_codes,
    _get_theta_array,
    _normalize_cp,
    _stratified_sample_indices,
    _validate_output_buffer,
)
from ._kmeans import _kmeans

if TYPE_CHECKING:
    import pandas as pd

    from rapids_singlecell._utils._random import RNGLike, SeedLike

COLSUM_ALGO = Literal["columns", "atomics", "gemm", "benchmark"]
_SUPPRESS_PENALTY = 1e30
_CORRECTION_WORKSPACE_LIMIT_BYTES = 1 << 30
_KMEANS_MAX_ITER = 25
_KMEANS_INIT_CELLS_PER_CLUSTER = 5_000
_FUSED_MAX_CLUSTERS = 128
_FUSED_MAX_PCS = 128
# Tests: route every shape through the general assignment kernel.
_FORCE_GENERAL_ASSIGNMENT = False
_UPLOAD_BYTES = 1 << 28

# Each flavor inherits the stopping rules of the implementation it reproduces:
# harmony2 follows harmonypy 2.0.0 / harmony 2.0.5 (R), harmony1 follows
# harmony-pytorch. Keys are (max_iter_clustering, tol_clustering, tol_harmony).
_STOPPING_HARMONY2 = (4, 1e-3, 1e-2)
_STOPPING_HARMONY1 = (200, 1e-5, 1e-4)


def harmonize(
    Z: cp.array,
    batch_mat: pd.DataFrame,
    batch_key: str | list[str],
    *,
    n_clusters: int | None = None,
    max_iter_harmony: int = 10,
    max_iter_clustering: int | None = None,
    tol_harmony: float | None = None,
    tol_clustering: float | None = None,
    ridge_lambda: float = 1.0,
    sigma: float = 0.1,
    block_proportion: float = 0.05,
    shuffle_chunk_size: int = 8,
    theta: float | int | list[float] | np.ndarray | cp.ndarray = 2.0,
    tau: int = 0,
    correction_method: Literal["fast", "batched"] | None = None,
    colsum_algo: COLSUM_ALGO | None = None,
    rng: SeedLike | RNGLike | None = None,
    stabilized_penalty: bool = True,
    dynamic_lambda: bool = True,
    alpha: float = 0.2,
    batch_prune_threshold: float | None = 1e-5,
    verbose: bool = False,
    bfloat16: bool = False,
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
        If ``None``, use the value of the reference implementation for the chosen
        algorithm: ``4`` for Harmony2, ``200`` for Harmony1.

    tol_harmony
        Tolerance on justifying convergence of Harmony over objective function values.
        If ``None``, ``1e-2`` for Harmony2 and ``1e-4`` for Harmony1.

    tol_clustering
        Tolerance on justifying convergence of the clustering step over objective function values within each Harmony iteration.
        If ``None``, ``1e-3`` for Harmony2 and ``1e-5`` for Harmony1.

    ridge_lambda
        Hyperparameter of ridge regression on the correction step.
        Must be finite and greater than zero when ``dynamic_lambda=False``.

    sigma
        Weight of the entropy term in objective function.

    block_proportion
        Proportion of block size in one update operation of clustering step.

    shuffle_chunk_size
        Cells per run of neighbouring same-batch cells that move between
        update blocks together (one batch key only). ``1`` shuffles every cell
        independently.

    theta
        Weight of the diversity penalty term. A scalar is broadcast to every
        variable. A sequence may have one value per key or one value per
        categorical level across all keys.

    tau
        Discounting factor on ``theta``. By default, there is no discounting.

    correction_method
        Choose ``fast`` for bounded-memory correction or ``batched`` for
        batched processing. With one key, ``None`` automatically selects ``batched``
        unless its workspace would exceed 1 GiB, in which case ``fast`` is
        used. Multiple keys use the exact general-design solve and process
        clusters in workspace-bounded chunks when needed. For multiple keys,
        ``None`` and ``batched`` select this solve; ``fast`` is ignored with a
        warning.

    colsum_algo
        No effect: column sums come from the fused clustering kernels. Kept
        for compatibility.

    rng
        Random seed or :class:`~numpy.random.Generator` for reproducing results.
        Sequential calls with identical inputs, parameters, and integer seed
        are bitwise reproducible on the same hardware and software stack in
        float32 and float64. Initialization uses k-means++ and fixed-order
        Lloyd reductions. Results may differ across GPU architectures or
        CUDA versions; concurrent CUDA streams are outside this guarantee.

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

    bfloat16
        Store the soft cluster assignments R in bfloat16; everything else
        stays in the dtype of ``Z`` (float32). Needs the CUDA 13 build with
        cuTile kernels and a supported GPU; otherwise a warning is issued and
        float32 assignments are used.

    verbose
        Whether to print benchmarking results for the column sum algorithm and the number of iterations until convergence.

    Returns
    -------
    The integrated embedding by Harmony, of the same shape as the input embedding.
    """

    stopping = (
        _STOPPING_HARMONY2
        if stabilized_penalty and dynamic_lambda
        else _STOPPING_HARMONY1
    )
    if max_iter_clustering is None:
        max_iter_clustering = stopping[0]
    if tol_clustering is None:
        tol_clustering = stopping[1]
    if tol_harmony is None:
        tol_harmony = stopping[2]

    n_cells = Z.shape[0]

    rng = np.random.default_rng(rng)
    # Process batch information
    batch_codes, n_levels = _get_batch_codes(batch_mat, batch_key)
    n_covariates = int(n_levels.size)
    n_batches = int(n_levels.sum())
    N_b = cp.bincount(batch_codes.ravel(), minlength=n_batches).astype(Z.dtype)
    Pr_b = (N_b.reshape(-1, 1) / n_cells).astype(Z.dtype)

    # Keep the established one-dimensional layout for one covariate. Multiple
    # covariates use a cell-major matrix of disjoint marginal category codes.
    cats = batch_codes[:, 0] if n_covariates == 1 else batch_codes
    order = None
    joint_cats = None
    joint_codes = None
    joint_offsets = None
    marginal_joint_offsets = None
    marginal_joint_indices = None
    if n_covariates > 1:
        joint_cats, joint_codes = _factorize_joint_codes(batch_codes, n_levels)
        joint_cats = joint_cats.astype(cp.int32, copy=False)
        joint_codes = joint_codes.astype(cp.int32, copy=False)
        n_joint_categories = joint_cats.shape[0]
        marginal_joint_offsets, flat_joint_indices = _create_category_index_mapping(
            joint_cats.ravel(), n_batches
        )
        marginal_joint_indices = (flat_joint_indices // n_covariates).astype(
            cp.int32, copy=False
        )
        groups, n_groups = joint_codes, n_joint_categories
    else:
        n_joint_categories = 0
        groups, n_groups = cats, n_batches
    # Work on cells sorted by group (batch, or joint category with several
    # keys), in a seeded random order within each group: per-group products
    # read contiguous rows, and the runs of neighbouring cells shuffled
    # together are random groups rather than input-order neighbours. The
    # result is put back in input order.
    shuffle_seed = int(rng.integers(2**63))
    keys = cp.random.default_rng(shuffle_seed).random(n_cells)
    order = cp.argsort(groups.astype(cp.float64) + keys)
    del keys
    Z = _sorted_device_copy(Z, order)
    cats = cats[order]
    # Sorted groups: offsets from the group sizes, cells in order.
    group_offsets = cp.zeros(n_groups + 1, dtype=cp.int32)
    group_offsets[1:] = cp.cumsum(cp.bincount(groups, minlength=n_groups))
    cell_indices = cp.arange(n_cells, dtype=cp.int32)
    if n_covariates > 1:
        joint_codes = joint_codes[order]
        joint_offsets = group_offsets
    else:
        cat_offsets = group_offsets
    Z_norm = _normalize_cp(Z)

    # Set up parameters
    if max_iter_harmony < 1:
        raise ValueError("max_iter_harmony must be >= 1")
    if int(shuffle_chunk_size) != shuffle_chunk_size or shuffle_chunk_size < 1:
        raise ValueError(
            f"shuffle_chunk_size must be a positive integer, got {shuffle_chunk_size}."
        )
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))
        n_clusters = max(n_clusters, 2)

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
    elif not np.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError(
            "ridge_lambda must be a finite positive number when "
            f"dynamic_lambda=False, got {ridge_lambda}."
        )
    if correction_method is not None and correction_method not in {
        "fast",
        "original",
        "batched",
    }:
        raise ValueError("correction_method must be 'fast' or 'batched'.")
    if correction_method == "original":
        if n_covariates == 1:
            replacement = "fast"
            replacement_message = (
                "Use correction_method='fast' instead; it computes the same "
                "correction with the optimized bounded-memory implementation."
            )
        else:
            replacement = "batched"
            replacement_message = (
                "With multiple batch keys, omit correction_method or use "
                "correction_method='batched' for the exact general-design solve."
            )
        warnings.warn(
            "correction_method='original' is deprecated and will be removed in "
            f"a future release. {replacement_message}",
            FutureWarning,
            stacklevel=3,
        )
        correction_method = replacement
    if n_covariates > 1 and correction_method == "fast":
        warnings.warn(
            f"correction_method={correction_method!r} is ignored when multiple "
            "batch keys are provided; using the exact general-design correction.",
            UserWarning,
            stacklevel=3,
        )

    # Multi-covariate correction uses its own exact cluster-chunked solve.
    # For one covariate, retain the established arrowhead auto-selection.
    if correction_method is None and n_covariates == 1:
        nb1 = n_batches + 1
        inv_mats_bytes = n_clusters * nb1 * nb1 * Z.dtype.itemsize
        correction_method = (
            "batched" if inv_mats_bytes <= _CORRECTION_WORKSPACE_LIMIT_BYTES else "fast"
        )

    # The clustering loop is a CUDA kernel taking a `uint32` seed, so the
    # generator collapses into an integer here; the per-iteration offset keeps
    # successive kernel launches decorrelated.
    kernel_seed = _seed_from_rng(rng, allow_none=False)

    block_size = int(n_cells * block_proportion)
    if bfloat16 and not (
        n_covariates == 1
        and Z.dtype == np.float32
        and correction_method == "batched"
        and _clustering_cuda.cutile_bf16_available(Z.shape[1], n_clusters)
        and _correction_batched_cuda.cutile_bf16_available(Z.shape[1], n_clusters)
    ):
        warnings.warn(
            "dtype='bfloat16' needs the CUDA 13 build of rapids-singlecell with "
            "cuTile kernels, a GPU of compute capability 8.0 or newer, one batch "
            "key, at most 128 clusters and 128 components; using float32 "
            "assignments instead (about 1.5x the assignment memory).",
            UserWarning,
            stacklevel=3,
        )
        bfloat16 = False
    # Buffers for all Harmony iterations.
    workspace = _allocate_clustering_workspace(
        n_cells,
        n_pcs=Z.shape[1],
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_groups=n_groups,
        n_covariates=n_covariates,
        block_size=block_size,
        dtype=Z_norm.dtype,
        bfloat16=bfloat16,
    )
    if n_covariates > 1:
        workspace.update(
            joint_codes=joint_codes,
            joint_cats=joint_cats,
            marginal_joint_offsets=marginal_joint_offsets,
            marginal_joint_indices=marginal_joint_indices,
            n_first=int(n_levels[0]),
        )
    tile_partials = workspace.get("tile_partials")
    R, E, O, objectives_harmony = _initialize_clusters(
        Z_norm,
        n_clusters=n_clusters,
        sigma=sigma,
        Pr_b=Pr_b,
        theta=theta_array,
        rng=rng,
        stabilized_penalty=stabilized_penalty,
        cat_offsets=group_offsets,
        cell_indices=cell_indices,
        workspace=workspace,
        bfloat16=bfloat16,
    )

    # Main harmony iterations
    is_converged = False

    for i in range(max_iter_harmony):
        # Clustering step
        _clustering(
            Z_norm,
            R=R,
            E=E,
            O=O,
            Pr_b=Pr_b,
            cats=cats,
            theta=theta_array,
            tol=tol_clustering,
            objectives_harmony=objectives_harmony,
            max_iter=max_iter_clustering,
            sigma=sigma,
            block_size=block_size,
            n_batches=n_batches,
            n_covariates=n_covariates,
            n_joint_categories=n_joint_categories,
            kernel_seed=kernel_seed + i * 1000003,
            partition=i == 0,
            shuffle_chunk_size=shuffle_chunk_size,
            stabilized_penalty=stabilized_penalty,
            workspace=workspace,
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
        # Convergence depends only on the clustering objective, so it is known
        # before the correction; unless this is the last round, the
        # correction writes the normalized embedding for the next clustering.
        is_converged = _is_convergent_harmony(objectives_harmony, tol=tol_harmony)
        normalize = not is_converged and i + 1 < max_iter_harmony
        # Correction step
        if n_covariates > 1:
            Z_hat = _correction_multi(
                Z,
                R,
                O=O,
                lambda_kb=lambda_kb,
                joint_O=workspace["O_joint"],
                n_batches=n_batches,
                joint_cats=joint_cats,
                joint_offsets=joint_offsets,
                marginal_joint_offsets=marginal_joint_offsets,
                marginal_joint_indices=marginal_joint_indices,
                output=Z_norm,
                normalize=normalize,
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
                normalize=normalize,
                tile_partials=tile_partials,
            )
        if is_converged:
            if verbose:
                print(f"Harmony converged in {i + 1} iterations")
            break
        # The normalized embedding is only needed by another clustering pass.
        # Correction has overwritten the old normalization buffer.
        if normalize:
            Z_norm = Z_hat
            if n_covariates == 1 and correction_method != "batched":
                Z_norm = _normalize_cp(Z_hat, out=Z_hat)

    if not is_converged:
        warnings.warn(
            "Harmony did not converge. Consider increasing the number of iterations"
        )
    if order is not None:
        # Free the working set before allocating the output.
        del R, workspace, Z
        out = cp.empty_like(Z_hat)
        out[order] = Z_hat
        return out
    return Z_hat


def _assignment_args(R: cp.ndarray, dtype) -> dict:
    """Pass R to the CUDA modules; bfloat16 R (stored as uint16) travels as
    ``R_bf16`` next to a placeholder ``R``."""
    if R.dtype == cp.uint16:
        return {"R": cp.empty((1, R.shape[1]), dtype=dtype), "R_bf16": R}
    return {"R": R}


def _sorted_device_copy(Z, order: cp.ndarray) -> cp.ndarray:
    """``Z[order]`` on the GPU. Host input is uploaded in chunks and placed
    directly, so no unsorted device copy is kept."""
    if isinstance(Z, cp.ndarray):
        return Z[order]
    out = cp.empty(Z.shape, dtype=Z.dtype)
    position = cp.empty_like(order)
    position[order] = cp.arange(order.size)
    rows = max(1, _UPLOAD_BYTES // (Z.shape[1] * Z.dtype.itemsize))
    for begin in range(0, Z.shape[0], rows):
        end = min(begin + rows, Z.shape[0])
        out[position[begin:end]] = cp.asarray(Z[begin:end])
    if cp.isnan(out).any():
        raise ValueError(
            "Input data contains NaN values. Please handle these before running harmony_integrate."
        )
    return out


def _fused_assign_smem_bytes(n_pcs: int, n_clusters: int, itemsize: int) -> int:
    """Shared memory of the fused assignment kernel (see kernels_clustering.cuh)."""
    pcs = -(-n_pcs // 4) * 4
    return (pcs * _FUSED_MAX_CLUSTERS + 8 * 8 * pcs + 8 * n_clusters) * itemsize


def _initialize_clusters(
    Z_norm: cp.ndarray,
    *,
    n_clusters: int,
    sigma: float,
    Pr_b: cp.ndarray,
    theta: cp.ndarray,
    rng: np.random.Generator,
    stabilized_penalty: bool = True,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    workspace: dict,
    bfloat16: bool = False,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, list]:
    """
    Initialize clusters: k-means centroids, then one unpenalized assignment
    pass over the cells (sorted by group) giving R, E, O and the objective.
    """
    # Fit only the initialization sample, retaining every observed stratum.
    n_init_cells = min(Z_norm.shape[0], _KMEANS_INIT_CELLS_PER_CLUSTER * n_clusters)
    if n_init_cells < cat_offsets.size - 1:
        n_strata = int(cp.count_nonzero(cp.diff(cat_offsets)))
        n_init_cells = max(n_init_cells, n_strata)
    Z_init = Z_norm
    if n_init_cells < Z_norm.shape[0]:
        sample_indices = _stratified_sample_indices(
            cat_offsets,
            cell_indices,
            n_init_cells,
            rng,
        )
        Z_init = Z_norm[sample_indices]
    Y = _kmeans(
        Z_init,
        n_clusters,
        max_iter=_KMEANS_MAX_ITER,
        rng=_seed_from_rng(rng),
    )
    Y_norm = _normalize_cp(Y)
    n_cells = Z_norm.shape[0]
    n_batches = Pr_b.shape[0]
    R = cp.empty((n_cells, n_clusters), dtype=cp.uint16 if bfloat16 else Z_norm.dtype)
    O = cp.zeros((n_batches, n_clusters), dtype=Z_norm.dtype)
    E = cp.empty_like(O)
    optional = (
        "O_joint",
        "marginal_joint_offsets",
        "marginal_joint_indices",
        "n_first",
        "y_t_general",
        "col_workspace",
        "force_general",
    )
    obj = _clustering_cuda.fused_initialize(
        Z_norm,
        Y_norm=cp.ascontiguousarray(Y_norm),
        cat_offsets=cat_offsets,
        **_assignment_args(R, Z_norm.dtype),
        O=O,
        E=E,
        Pr_b=Pr_b.ravel(),
        theta=theta,
        tiles=workspace["block_cat_offsets"][-(cat_offsets.size) :],
        assign_partial=workspace["assign_partial"],
        objective_partials=workspace["objective_partials"],
        obj_scalar=workspace["obj_scalar"],
        sigma=float(sigma),
        stabilized=stabilized_penalty,
        stream=cp.cuda.get_current_stream().ptr,
        **{key: workspace[key] for key in optional if key in workspace},
    )
    return R, E, O, [obj]


def _allocate_clustering_workspace(
    n_cells: int,
    *,
    n_pcs: int,
    n_clusters: int,
    n_batches: int,
    n_groups: int,
    n_covariates: int,
    block_size: int,
    dtype: cp.dtype,
    bfloat16: bool = False,
) -> dict:
    """Buffers of the clustering loop. Groups are batches, or joint categories
    (with counts in ``O_joint``) for several keys."""
    itemsize = np.dtype(dtype).itemsize
    n_blocks = -(-n_cells // block_size)
    n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
    # About four assignment tiles per SM, plus one partial tile per group.
    n_tiles = 4 * n_sm + n_groups + 1
    workspace = {
        "Y": cp.empty((n_clusters, n_pcs), dtype=dtype),
        "Y_norm": cp.empty((n_clusters, n_pcs), dtype=dtype),
        "idx_list": cp.empty(n_cells, dtype=cp.int32),
        "idx_list_alt": cp.empty(n_cells, dtype=cp.int32),
        "sort_keys": cp.empty(n_cells, dtype=cp.uint32),
        "sort_keys_alt": cp.empty(n_cells, dtype=cp.uint32),
        "cub_temp": cp.empty(
            _clustering_cuda.get_cub_sort_temp_bytes(n_cells=n_cells), dtype=cp.uint8
        ),
        "penalty": cp.empty((n_batches, n_clusters), dtype=dtype),
        "obj_scalar": cp.empty(1, dtype=dtype),
        "last_obj": cp.zeros(1, dtype=dtype),
        "scatter_workspace": cp.empty(
            _clustering_cuda.get_scatter_temp_bytes(
                n_rows=n_cells,
                n_cols=n_clusters,
                n_categories=n_groups,
                itemsize=itemsize,
                grouped=True,
            ),
            dtype=cp.uint8,
        ),
        "objective_partials": cp.empty(
            n_cells + _clustering_cuda.OBJECTIVE_REDUCE_BLOCKS + 1, dtype=dtype
        ),
        # Block-group offsets, then the assignment tiles of one block.
        "block_cat_offsets": cp.empty(
            n_blocks * n_groups + 1 + n_groups + 1, dtype=cp.int32
        ),
        # Each block's counts in O (the blocks are fixed per harmonize call).
        "block_counts": cp.empty((n_blocks, n_groups, n_clusters), dtype=dtype),
        "assign_partial": cp.empty(n_tiles * n_clusters, dtype=dtype),
    }
    if n_covariates > 1:
        workspace["O_joint"] = cp.zeros((n_groups, n_clusters), dtype=dtype)
        workspace["group_penalty"] = cp.empty((n_groups, n_clusters), dtype=dtype)
    if _FORCE_GENERAL_ASSIGNMENT or not (
        n_clusters <= _FUSED_MAX_CLUSTERS
        and n_pcs <= _FUSED_MAX_PCS
        and _fused_assign_smem_bytes(n_pcs, n_clusters, itemsize)
        <= cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
    ):
        # General assignment: transposed centroids and per-warp column sums.
        stride = -(-n_clusters // _FUSED_MAX_CLUSTERS) * _FUSED_MAX_CLUSTERS
        workspace["y_t_general"] = cp.empty((-(-n_pcs // 4) * 4, stride), dtype=dtype)
        workspace["col_workspace"] = cp.empty(n_tiles * 8 * n_clusters, dtype=dtype)
        workspace["force_general"] = _FORCE_GENERAL_ASSIGNMENT
    if bfloat16:
        workspace["tile_partials"] = cp.empty(
            _clustering_cuda.cutile_partials_size(n_pcs), dtype=cp.float32
        )
    return workspace


def _clustering(
    Z_norm: cp.ndarray,
    *,
    R: cp.ndarray,
    E: cp.ndarray,
    O: cp.ndarray,
    Pr_b: cp.ndarray,
    cats: cp.ndarray,
    theta: cp.ndarray,
    tol: float,
    objectives_harmony: list,
    max_iter: int,
    sigma: float,
    block_size: int,
    n_batches: int,
    n_covariates: int,
    n_joint_categories: int,
    kernel_seed: int,
    shuffle_chunk_size: int,
    stabilized_penalty: bool,
    partition: bool = True,
    workspace: dict,
) -> None:
    """
    Clustering updates of R, O and E in place: random blocks of cells are
    reassigned against the counts of all other cells until convergence or
    ``max_iter`` passes.
    """
    _clustering_cuda.clustering_loop(
        Z_norm,
        **_assignment_args(R, Z_norm.dtype),
        E=E,
        O=O,
        Pr_b=Pr_b.ravel(),
        cats=cats,
        theta=theta,
        **workspace,
        n_cells=Z_norm.shape[0],
        n_pcs=Z_norm.shape[1],
        n_clusters=R.shape[1],
        n_batches=n_batches,
        n_covariates=n_covariates,
        n_joint_categories=n_joint_categories,
        block_size=block_size,
        sigma=float(sigma),
        tol=float(tol),
        max_iter=max_iter,
        seed=kernel_seed & 0xFFFFFFFF,
        stabilized=stabilized_penalty,
        shuffle_chunk=shuffle_chunk_size,
        partition=partition,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )
    objectives_harmony.append(float(workspace["last_obj"][0]))


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
    correction_method: Literal["fast", "batched"] = "batched",
    cats: cp.ndarray,
    n_batches: int,
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    output: cp.ndarray | None = None,
    normalize: bool = False,
    tile_partials: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Apply correction to the embedding based on the specified method.

    Cells are sorted by batch. ``normalize`` asks the batched method to write
    the L2-normalized embedding; the caller normalizes after other methods.
    """
    if correction_method == "batched":
        return _correction_batched(
            X,
            R,
            O=O,
            lambda_kb=lambda_kb,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            output=output,
            normalize=normalize,
            tile_partials=tile_partials,
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
    raise ValueError("correction_method must be 'fast' or 'batched'.")


def _correction_multi(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    O: cp.ndarray,
    lambda_kb: cp.ndarray,
    joint_O: cp.ndarray,
    n_batches: int,
    joint_cats: cp.ndarray,
    joint_offsets: cp.ndarray,
    marginal_joint_offsets: cp.ndarray,
    marginal_joint_indices: cp.ndarray,
    output: cp.ndarray | None = None,
    normalize: bool = False,
) -> cp.ndarray:
    """Apply the exact general-design correction in bounded cluster chunks.

    Cells are sorted by joint category (``joint_offsets``) and ``joint_O``
    holds the per-joint cluster counts.
    """
    n_cells, n_pcs = X.shape
    n_clusters = R.shape[1]
    n_joint_categories = joint_cats.shape[0]
    nb1 = n_batches + 1
    cluster_chunk_size = _multi_correction_cluster_chunk_size(
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_joint_categories=n_joint_categories,
        itemsize=X.dtype.itemsize,
    )
    stream = cp.cuda.get_current_stream().ptr
    handle = cp.cuda.device.get_cublas_handle()
    Z = _correction_output(X, output)
    for cluster_start in range(0, n_clusters, cluster_chunk_size):
        chunk = slice(cluster_start, cluster_start + cluster_chunk_size)
        lambda_chunk = cp.ascontiguousarray(lambda_kb[:, chunk])
        R_chunk = R[:, chunk]
        k = R_chunk.shape[1]
        joint_rhs = cp.empty((n_joint_categories, k, n_pcs), dtype=X.dtype)
        gram = cp.empty((k, nb1, nb1), dtype=X.dtype)
        rhs = cp.empty((k, nb1, n_pcs), dtype=X.dtype)
        _correction_batched_cuda.prepare_multi(
            X,
            R=R_chunk,
            O=cp.ascontiguousarray(O[:, chunk]),
            joint_O=cp.ascontiguousarray(joint_O[:, chunk]),
            joint_cats=joint_cats,
            joint_offsets=joint_offsets,
            marginal_joint_offsets=marginal_joint_offsets,
            marginal_joint_indices=marginal_joint_indices,
            lambda_kb=lambda_chunk,
            active_mask=(lambda_chunk < X.dtype.type(_SUPPRESS_PENALTY)).view(cp.uint8),
            n_batches=n_batches,
            n_covariates=joint_cats.shape[1],
            gram=gram,
            rhs=rhs,
            joint_rhs=joint_rhs,
            stream=stream,
            handle=handle,
        )
        W_all = _solve_spd_batched(gram, rhs)
        del gram, rhs
        _correction_batched_cuda.apply_multi(
            X,
            R=R_chunk,
            W_all=W_all,
            joint_cats=joint_cats,
            joint_offsets=joint_offsets,
            n_batches=n_batches,
            accumulate=cluster_start > 0,
            finish=cluster_start + k == n_clusters,
            normalize=normalize,
            W_joint=joint_rhs,
            Z=Z,
            stream=stream,
            handle=handle,
        )
    return Z


def _solve_spd_batched(gram: cp.ndarray, rhs: cp.ndarray) -> cp.ndarray:
    """Solve cluster systems, using least squares when a Gram matrix is singular."""
    n_matrices, matrix_size, _ = gram.shape
    n_rhs = rhs.shape[2]
    if n_matrices == 0:
        return cp.empty_like(rhs)

    # potrf and trsm are in-place. Preserve the original inputs so an unusual
    # rank-deficient system can fall back to a minimum-norm solve.
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
    failed = np.flatnonzero(cp.asnumpy(info))
    for matrix_index in failed:
        rhs_work[matrix_index] = cp.linalg.lstsq(
            gram[matrix_index], rhs[matrix_index], rcond=None
        )[0]
    return rhs_work


def _multi_correction_cluster_chunk_size(
    *,
    n_pcs: int,
    n_clusters: int,
    n_batches: int,
    n_joint_categories: int,
    itemsize: int,
) -> int:
    """Choose a cluster chunk that keeps multi-key scratch below 1 GiB."""
    nb1 = n_batches + 1
    # Gram and RHS with the solver's copies and output, joint products, and
    # sliced O, lambda, active mask and joint counts.
    per_cluster = (
        itemsize * (3 * nb1 * nb1 + 3 * nb1 * n_pcs + n_joint_categories * (n_pcs + 1))
        + (2 * itemsize + 1) * n_batches
    )
    if per_cluster > _CORRECTION_WORKSPACE_LIMIT_BYTES:
        gib = per_cluster / _CORRECTION_WORKSPACE_LIMIT_BYTES
        raise MemoryError(
            "A single multi-key Harmony correction cluster requires "
            f"approximately {gib:.2f} GiB of scratch space; reduce the "
            "number of batch levels or embedding dimensions."
        )
    return max(1, min(n_clusters, _CORRECTION_WORKSPACE_LIMIT_BYTES // per_cluster))


def _correction_output(X: cp.ndarray, output: cp.ndarray | None = None) -> cp.ndarray:
    """Return a validated correction output buffer."""
    if output is None:
        return cp.empty_like(X)
    _validate_output_buffer(X, output, operation="Correction")
    return output


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
    """Apply the bounded-memory correction method."""
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

    _correction_cuda.correction_fast(
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
    n_batches: int,
    cat_offsets: cp.ndarray,
    output: cp.ndarray | None = None,
    normalize: bool = False,
    tile_partials: cp.ndarray | None = None,
) -> cp.ndarray:
    """
    Batched correction of cells sorted by batch: all clusters at once, one
    product per batch for the right-hand sides and for the correction.
    """
    n_cells, n_pcs = X.shape
    n_clusters = R.shape[1]
    nb1 = n_batches + 1
    dtype = X.dtype
    Z = _correction_output(X, output)
    _correction_batched_cuda.correction_batched(
        X,
        **_assignment_args(R, X.dtype),
        tile_partials=tile_partials,
        O=O,
        cat_offsets=cat_offsets,
        lambda_kb=lambda_kb,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        Z=Z,
        inv_mats=cp.empty((n_clusters, nb1, nb1), dtype=dtype),
        Phi_t_diag_R_X_all=cp.empty((n_clusters, nb1, n_pcs), dtype=dtype),
        W_all=cp.empty((n_clusters, nb1, n_pcs), dtype=dtype),
        g_factor=cp.empty((n_clusters, n_batches), dtype=dtype),
        g_P_row0=cp.empty((n_clusters, n_batches), dtype=dtype),
        normalize=normalize,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
    )
    return Z


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
