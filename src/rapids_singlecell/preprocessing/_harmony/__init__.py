from __future__ import annotations

import functools
import itertools
import math
import operator
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
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

from ._chunks_to_device import _sorted_device_copy
from ._helper import (
    _factorize_joint_codes,
    _get_batch_codes,
    _get_theta_array,
    _normalize_cp,
    _stratified_sample_indices,
    _validate_output_buffer,
)
from ._kmeans import _kmeans
from ._multi_gpu import _comm, _copy, _gather_rows, _peer_copies_work

if TYPE_CHECKING:
    import pandas as pd

    from rapids_singlecell._utils._random import RNGLike, SeedLike

COLSUM_ALGO = Literal["columns", "atomics", "gemm", "benchmark"]
_SUPPRESS_PENALTY = 1e30
_CORRECTION_WORKSPACE_LIMIT_BYTES = 1 << 30
_KMEANS_MAX_ITER = 25
_KMEANS_INIT_CELLS_PER_CLUSTER = 5_000
# Tests: assign every shape in cluster blocks (as for more than 128 clusters).
_FORCE_CLUSTER_BLOCKS = False
# Tests: whether every block's counts are kept (None: unless they outgrow R).
_KEEP_BLOCK_COUNTS = None
_UPLOAD_BYTES = 1 << 28  # largest pinned staging chunk
_UPLOAD_THREADS = 8
_COMM_VALUES = 1 << 22  # int64 values per exchange; larger sums take several
_BLOCK_UNITS = 384  # units of the block draw (see _draw_blocks)
_SEGMENT = 2048  # cells per segment of the exact sums (see _segments)
_SOLVE_ONE_AT_A_TIME = 256  # system size (see _solve_spd_batched)

# Each flavor inherits the stopping rules of the implementation it reproduces:
# harmony2 follows harmonypy 2.0.0 / harmony 2.0.5 (R), harmony1 follows
# harmony-pytorch. Keys are (max_iter_clustering, tol_clustering, tol_harmony).
_STOPPING_HARMONY2 = (4, 1e-3, 1e-2)
_STOPPING_HARMONY1 = (200, 1e-5, 1e-4)


def _prefaulted(func):
    """Zero the host output ``out`` in a thread while ``func`` runs (copying
    into touched pages is about twice as fast as into a fresh allocation) and
    join it before the call returns or raises, so it never writes ``out``
    afterwards. An output sharing memory with the host input is left alone:
    the input is read later."""

    @functools.wraps(func)
    def wrapper(Z, *args, out=None, **kwargs):
        prefault = None
        if out is not None and not (
            isinstance(Z, np.ndarray) and np.may_share_memory(out, Z)
        ):
            prefault = threading.Thread(target=out.fill, args=(0,), daemon=True)
            prefault.start()
        try:
            return func(Z, *args, out=out, _prefault=prefault, **kwargs)
        finally:
            if prefault is not None:
                prefault.join()

    return wrapper


@_prefaulted
def harmonize(
    Z: cp.ndarray,
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
    devices: list[int] | None = None,
    out: np.ndarray | None = None,
    _prefault: threading.Thread | None = None,
) -> cp.ndarray | np.ndarray:
    """
    Integrate data using Harmony algorithm.

    Parameters
    ----------
    Z
        The input embedding with rows for cells (N) and columns for embedding coordinates (d).
        Host input is uploaded in chunks, converted to ``dtype`` in the staging.

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
        Cells per run of neighbouring cells of the same batch (joint category
        with several keys) that move between update blocks together. Every
        block takes an equal share of each stretch of about n_cells / 384
        batch-sorted cells (see _block_layout). ``1`` moves cells one at a time.

    theta
        Weight of the diversity penalty term. A scalar is broadcast to every
        variable. A sequence may have one value per key or one value per
        categorical level across all keys.

    tau
        Discounting factor on ``theta``. By default, there is no discounting.

    correction_method
        ``batched`` corrects from exact right-hand sides, as many clusters per
        pass as fit in 1 GiB of workspace, ``fast`` one cluster at a time in
        bounded memory. With one key, ``None`` automatically selects
        ``batched`` unless one cluster's workspace would exceed 1 GiB, in
        which case ``fast`` is used. Multiple keys use the exact
        general-design solve and process clusters in workspace-bounded chunks
        when needed. For multiple keys, ``None`` and ``batched`` select this
        solve; ``fast`` is ignored with a warning.

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
        stays in the dtype of ``Z`` (float32). Needs one batch key and the
        batched correction; otherwise a warning is issued and float32
        assignments are used.

    verbose
        Whether to print the number of iterations until convergence.

    devices
        GPUs to split the cells across (default: the current one). Several
        GPUs give the bits of a single-GPU run; they need the batched
        correction and the host output ``out``.

    out
        Host array receiving the corrected embedding (then returned). It is
        zeroed in a thread while the GPUs work, also when the call fails.

    Returns
    -------
    The integrated embedding by Harmony, of the same shape as the input embedding.
    """
    stopping = _STOPPING_HARMONY2 if stabilized_penalty and dynamic_lambda else _STOPPING_HARMONY1
    if max_iter_clustering is None:
        max_iter_clustering = stopping[0]
    if tol_clustering is None:
        tol_clustering = stopping[1]
    if tol_harmony is None:
        tol_harmony = stopping[2]

    max_iter_harmony = operator.index(max_iter_harmony)  # NumPy integers too
    max_iter_clustering = operator.index(max_iter_clustering)
    if max_iter_harmony < 1:
        raise ValueError("max_iter_harmony must be >= 1")
    if out is not None and (
        not isinstance(out, np.ndarray) or out.shape != Z.shape or out.dtype != Z.dtype
    ):
        raise ValueError(
            f"out must be a NumPy array of shape {Z.shape} and dtype {Z.dtype}."
        )
    if devices is not None and len(devices) > 1 and out is None:
        raise ValueError("Several GPUs write the result into a host array `out`.")
    try:
        valid = int(shuffle_chunk_size) == shuffle_chunk_size >= 1
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise ValueError(f"shuffle_chunk_size must be a positive integer, got {shuffle_chunk_size!r}.")
    shuffle_chunk_size = int(shuffle_chunk_size)
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError(f"sigma must be a finite positive number, got {sigma}.")
    if not 0 < block_proportion <= 1:
        raise ValueError(f"block_proportion must be in (0, 1], got {block_proportion}.")
    if colsum_algo not in (None, "columns", "atomics", "gemm", "benchmark"):
        raise ValueError(f"Unknown colsum_algo {colsum_algo!r}.")
    n_cells = Z.shape[0]
    if n_cells > 2**31 - 2**24:  # int32 cell indices, with room to round up
        raise ValueError(f"harmonize supports at most {2**31 - 2**24} cells, got {n_cells}.")

    rng = np.random.default_rng(rng)
    # Process batch information
    batch_codes, n_levels = _get_batch_codes(batch_mat, batch_key)
    n_covariates = int(n_levels.size)
    n_batches = int(n_levels.sum())
    # Several keys solve a dense float64 system over all their levels per
    # cluster (and factor a copy); fail before the GPU work if it cannot fit.
    system_bytes = 16 * (n_batches + 1) ** 2
    if n_covariates > 1 and system_bytes > _CORRECTION_WORKSPACE_LIMIT_BYTES:
        raise MemoryError(
            f"Harmony with several batch keys solves a dense system over their "
            f"{n_batches} levels per cluster ({system_bytes / 2**30:.1f} GiB); "
            "reduce the number of levels (a key with about one level per cell "
            "is not a batch variable)."
        )
    N_b = cp.bincount(batch_codes.ravel(), minlength=n_batches).astype(Z.dtype)
    Pr_b = (N_b.reshape(-1, 1) / n_cells).astype(Z.dtype)

    # Keep the established one-dimensional layout for one covariate. Multiple
    # covariates use a cell-major matrix of disjoint marginal category codes.
    cats = batch_codes[:, 0] if n_covariates == 1 else batch_codes
    joint_cats = None
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
    # Sorted groups (the batch codes with one key) for the block draw; only one
    # batch key reads the batch codes otherwise.
    groups = groups[order]
    cats = groups if n_covariates == 1 else None
    # Sorted groups: offsets by binary search (a histogram's scratch grows
    # with the number of groups), cells in order.
    group_offsets = cp.searchsorted(
        groups, cp.arange(n_groups + 1, dtype=groups.dtype)
    ).astype(cp.int32)
    cell_indices = cp.arange(n_cells, dtype=cp.int32)
    del batch_codes
    if n_covariates > 1:
        del joint_codes  # unsorted

    # Set up parameters
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))
        n_clusters = max(n_clusters, 2)
    n_clusters = operator.index(n_clusters)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    dtype = np.dtype(Z.dtype if dtype is None else dtype)

    rng = np.random.default_rng(rng)
    # Process batch information
    n_levels, N_b, joint, offsets, order = _sort_by_batch(batch_mat, batch_key, n_cells=n_cells, dtype=dtype, rng=rng)
    n_covariates, n_batches = n_levels.size, int(n_levels.sum())
    # The sorted cells' groups, allocated before the upload (faster with a
    # non-pooled allocator)
    groups = _sorted_groups(offsets)
    X = _sorted_device_copy(Z, order, dtype)
    rhs_bounds = _rhs_bounds(X, offsets)
    Z_norm = _normalize_cp(X)

    theta_array = _get_theta_array(theta, n_levels, dtype)
    if tau > 0:
        theta_array = theta_array * (1 - cp.exp(-N_b / (n_clusters * tau)) ** 2)
    theta_array = cp.ascontiguousarray(theta_array.ravel())

    # Validate parameters
    if dynamic_lambda:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be a finite positive number when dynamic_lambda=True, got {alpha}.")
        if batch_prune_threshold is not None and not (0 <= batch_prune_threshold <= 1):
            raise ValueError(f"batch_prune_threshold must be in [0, 1] or None, got {batch_prune_threshold}.")
    elif not np.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError(
            f"ridge_lambda must be a finite positive number when dynamic_lambda=False, got {ridge_lambda}."
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
            stacklevel=5,
        )
        correction_method = replacement
    if n_covariates > 1:
        if correction_method == "fast":
            warnings.warn(
                f"correction_method={correction_method!r} is ignored when multiple "
                "batch keys are provided; using the exact general-design correction.",
                UserWarning,
                stacklevel=5,
            )
        correction_method = "batched"  # several keys: the multi-key solve

    # One key: the batched correction unless one cluster's workspace is too large.
    if correction_method is None:
        fit = _batched_cluster_chunk(Z.shape[1], n_batches, Z.dtype.itemsize)
        correction_method = "batched" if fit else "fast"

    # The clustering loop is a CUDA kernel taking a `uint32` seed, so the
    # generator collapses into an integer here; the per-iteration offset keeps
    # successive kernel launches decorrelated.
    kernel_seed = _seed_from_rng(rng, allow_none=False)

    n_blocks = -(-n_cells // max(1, math.ceil(n_cells * block_proportion)))
    # 32-bit indices over update blocks x batches (the block draw, one key)
    # and batches x clusters or PCs (counts, fast correction).
    pairs = max(
        n_blocks * (n_batches if n_covariates == 1 else 1),
        (n_batches + 1) * max(Z.shape[1], n_clusters),
    )
    if pairs > 2**31 - 2**24:
        raise ValueError(
            f"harmonize supports at most {2**31 - 2**24} update blocks x batches "
            f"or batches x clusters (or PCs), got {pairs}; raise block_proportion "
            "or use fewer batch levels."
        )
    if bfloat16 and not (n_covariates == 1 and dtype == np.float32 and correction_method == "batched"):
        warnings.warn(
            f"dtype='bfloat16' needs one batch key and the batched correction; using {dtype} assignments instead.",
            UserWarning,
            stacklevel=5,
        )
        bfloat16 = False
    # Each block's counts are kept (on every GPU) unless they would outgrow
    # R; then three blocks' are recounted from R at a time (see clustering.cu).
    keep = _KEEP_BLOCK_COUNTS
    if keep is None:
        keep = (n_blocks + 3) * n_batches * 8 <= n_cells * dtype.itemsize
    devices = devices or [cp.cuda.Device().id]
    if len(devices) > 1 and correction_method != "batched":
        warnings.warn(
            "The fast correction runs on one GPU (it is also chosen when one "
            "cluster's batched correction workspace would exceed 1 GiB).",
            UserWarning,
            stacklevel=5,
        )
        devices = devices[:1]
    # Shards: contiguous runs of the sorted cells, one per GPU, cut at
    # multiples of the block draw's units (which also cut the segments), so
    # that every GPU draws the blocks and sums the segments of a single-GPU run.
    chunk, unit = _block_layout(n_cells, n_blocks, shuffle_chunk_size)
    devices = devices[: max(1, n_cells // unit)]
    n_ranks, n_pcs = len(devices), Z.shape[1]
    bounds = [
        min(n_cells, round(n_cells * r / n_ranks / unit) * unit) for r in range(n_ranks)
    ] + [n_cells]
    peer = n_ranks > 1 and _peer_copies_work(devices)
    # Every GPU writes its rows of the output (see _write_shards).
    shard_rows = _shard_rows(order, bounds, devices)
    parts = _upload_sorted(Z, order, bounds, devices, shard_rows)
    del Z, order
    # Read after the upload, whose host staging overlaps the sort.
    if not bool(cp.isfinite(theta_array).all()):
        raise ValueError(f"theta must be finite, got {theta}.")
    host_offsets = cp.asnumpy(group_offsets)
    rhs_bounds = _rhs_bounds(parts, host_offsets, bounds)
    # With several keys the assignment tiles take any cells (in joint
    # category order, see the JOINT kernel).
    tile_groups = n_batches if n_covariates == 1 else 1
    # The normalized cells and the clustering buffers exist before the
    # k-means: its many small temporaries allocate faster (by cudaMalloc)
    # while other small buffers are held.
    normalized, workspaces = [], []
    for rank, (lo, hi) in enumerate(itertools.pairwise(bounds)):
        with cp.cuda.Device(devices[rank]):  # a loop name would hold a shard
            normalized.append(_normalize_cp(parts[rank]))
            workspaces.append(
                _allocate_clustering_workspace(
                    hi - lo,
                    n_pcs=n_pcs,
                    n_clusters=n_clusters,
                    n_batches=n_batches,
                    n_groups=tile_groups,
                    n_blocks=n_blocks,
                    n_stored=n_blocks if keep else min(n_blocks, 3),
                    dtype=dtype,
                )
            )
    Y_norm = _kmeans_centroids(
        normalized,
        bounds=bounds,
        cat_offsets=group_offsets,
        cell_indices=cell_indices,
        n_clusters=n_clusters,
        rng=rng,
    )
    # Each GPU's sorted groups: one key drops them after its block draw
    # unless it recounts block counts, several keys keep them as their cells'
    # joint categories.
    shard_groups = [groups[lo:hi] for lo, hi in itertools.pairwise(bounds)]
    del groups
    if correction_method != "fast":  # the only later reader
        cats = cell_indices = None
    comm = None
    if n_ranks > 1:
        # Exact int64 sums of block counts, centroids and right-hand sides;
        # per-group sums move only each GPU's groups.
        limbs = _limbs(dtype.itemsize)
        capacity = max(
            (n_blocks if keep else 1) * n_batches, limbs * n_groups * n_pcs, 3 * n_pcs
        )
        first = np.searchsorted(host_offsets, bounds[:-1], side="right") - 1
        end = np.searchsorted(host_offsets, np.subtract(bounds[1:], 1), side="right")
        comm, comm_buffers = _comm(
            devices,
            min(capacity * n_clusters, _COMM_VALUES),
            peer=peer,
            groups=np.column_stack([first, np.maximum(first, end)]).ravel().tolist(),
        )

    def run(rank: int):
        # One GPU's shard; several GPUs run it in parallel threads.
        lo, hi = bounds[rank], bounds[rank + 1]

        def local(a):
            return _copy(a, devices[rank])

        # The fast correction (one GPU, maybe not the current one) reads the
        # batch codes and offsets.
        codes = {
            "cats": cats,
            "cat_offsets": group_offsets,
            "cell_indices": cell_indices,
        }
        if correction_method == "fast":
            codes = {key: local(value) for key, value in codes.items()}

        X = parts[rank]
        mine, shard_groups[rank] = local(shard_groups[rank]), None
        Z_norm, normalized[rank] = normalized[rank], None
        workspace, workspaces[rank] = workspaces[rank], None
        seg_start, seg_group = _segments(host_offsets, unit, lo, hi)
        segments = (seg_start, seg_group, cp.asarray(rhs_bounds))
        # Centroid sums need no group cuts.
        workspace["seg_start"] = (
            seg_start
            if n_covariates == 1
            else _segments(host_offsets[[0, -1]], unit, lo, hi)[0]
        )
        joint = {}
        if n_covariates > 1:
            joint = {
                "joint_cats": local(joint_cats),
                "marginal_joint_offsets": local(marginal_joint_offsets),
                "marginal_joint_indices": local(marginal_joint_indices),
            }
            workspace.update(joint_cats=joint["joint_cats"], n_first=int(n_levels[0]))
        if n_covariates > 1 or not keep:  # joint runs; recounted by batch
            # a view would keep every cell's groups on this GPU
            workspace["joints"] = mine if mine.base is None else mine.copy()
        workspace["Y_norm"][...] = local(Y_norm)
        # The update blocks are drawn once, before the large arrays exist.
        _draw_blocks(
            mine if n_covariates == 1 else cp.zeros(hi - lo, dtype=cp.int32),
            n_groups=tile_groups,
            n_blocks=n_blocks,
            seed=kernel_seed,
            chunk=chunk,
            unit=unit,
            first=lo,
            workspace=workspace,
        )
        del mine
        sums = {"comm": comm.handle if comm else 0, "rank": rank}
        loop_args = {
            "Pr_b": local(Pr_b),
            "theta": local(theta_array),
            "sigma": sigma,
            "n_blocks": n_blocks,
            "n_batches": n_batches,
            "n_covariates": n_covariates,
            "n_joint_categories": n_joint_categories,
            "stabilized_penalty": stabilized_penalty,
            "workspace": workspace,
            "n_total": n_cells,
            "sums": sums,
        }
        R = cp.empty((hi - lo, n_clusters), dtype=cp.uint16 if bfloat16 else dtype)
        O = cp.empty((n_batches, n_clusters), dtype=dtype)
        E = cp.empty_like(O)
        objectives = []
        _clustering(
            Z_norm,
            R=R,
            E=E,
            O=O,
            objectives_harmony=objectives,
            max_iter=0,
            initialize=True,
            **loop_args,
        )
        N_b_local = local(N_b)
        is_converged = False
        for i in range(max_iter_harmony):
            _clustering(
                Z_norm,
                R=R,
                E=E,
                O=O,
                objectives_harmony=objectives,
                max_iter=max_iter_clustering,
                tol=tol_clustering,
                kernel_seed=kernel_seed + i * 1000003,
                **loop_args,
            )
            # Compute per-(k,b) ridge regularization
            lambda_kb = _compute_lambda_kb(
                E,
                O=O,
                N_b=N_b_local,
                alpha=alpha,
                threshold=batch_prune_threshold,
                ridge_lambda=ridge_lambda,
                dynamic_lambda=dynamic_lambda,
            )
            # Convergence depends only on the clustering objective, so it is
            # known before the correction; unless this is the last round, the
            # correction writes the normalized embedding for the next
            # clustering.
            is_converged = _is_convergent_harmony(objectives, tol=tol_harmony)
            normalize = bool(not is_converged and i + 1 < max_iter_harmony)
            if n_covariates > 1:
                Z_hat = _correction_multi(
                    X,
                    R,
                    lambda_kb=lambda_kb,
                    n_batches=n_batches,
                    n_first=int(n_levels[0]),
                    **joint,
                    segments=segments,
                    output=Z_norm,
                    normalize=normalize,
                    n_total=n_cells,
                    sums=sums,
                    n_ranks=n_ranks,
                )
            else:
                Z_hat = _correction(
                    X,
                    R=R,
                    O=O,
                    lambda_kb=lambda_kb,
                    correction_method=correction_method,
                    n_batches=n_batches,
                    **codes,
                    output=Z_norm,
                    normalize=normalize,
                    segments=segments,
                    sums=sums,
                )
            if is_converged:
                break
            # The normalized embedding is only needed by another clustering
            # pass. Correction has overwritten the old normalization buffer.
            if normalize:
                Z_norm = Z_hat
                if correction_method != "batched":
                    Z_norm = _normalize_cp(Z_hat, out=Z_hat)
        return Z_hat, is_converged, i

    errors = []  # the first is the cause; the other GPUs then fail on the abort

    def guarded(rank: int):
        with cp.cuda.Device(devices[rank]):
            try:
                result = run(rank)
                cp.cuda.get_current_stream().synchronize()  # read by other threads
                return result
            except BaseException as error:
                errors.append(error)
                comm.abort()  # release the other GPUs' threads
                raise

    if n_ranks == 1:  # in this thread, on its stream
        with cp.cuda.Device(devices[0]):
            results = [run(0)]
    else:
        # The GPUs' threads read this thread's arrays on their own streams.
        for device in {cp.cuda.Device().id, *devices}:
            with cp.cuda.Device(device):
                cp.cuda.get_current_stream().synchronize()
        pool = ThreadPoolExecutor(n_ranks)
        futures, done = [], False
        try:
            for rank in range(n_ranks):
                futures.append(pool.submit(guarded, rank))
            wait(futures)
            done = not errors
        except BaseException:  # e.g. Ctrl-C: stop the GPU threads first
            comm.abort()
            wait(futures)
            raise
        finally:
            pool.shutdown()
            if not done:  # without holding the GIL
                comm.synchronize()
            del comm, comm_buffers
        if errors:
            raise errors[0]
        results = [future.result() for future in futures]
        del futures
    is_converged, i = results[0][1:]  # the shards are released as written
    if is_converged and verbose:
        print(f"Harmony converged in {i + 1} iterations")
    if not is_converged:
        warnings.warn(
            "Harmony did not converge. Consider increasing the number of iterations",
            stacklevel=5,
        )
    parts.clear()  # free the input before the output
    shards = [r[0] for r in results]
    del results  # _write_shards releases the shards as it goes
    if out is None:  # one GPU: the result in input order, on its GPU
        with cp.cuda.Device(devices[0]):
            return shards[0][shard_rows[1][0]]
    _write_shards(shards, shard_rows, devices, out, _prefault)
    return out


def _assignment_args(R: cp.ndarray, dtype) -> dict:
    """Pass R to the CUDA modules.

    bfloat16 R (stored as uint16) travels as ``R_bf16`` next to a placeholder ``R``.
    """
    if R.dtype == cp.uint16:
        return {"R": cp.empty((1, R.shape[1]), dtype=dtype), "R_bf16": R}
    return {"R": R}


def _upload_sorted(
    Z,
    order: cp.ndarray,
    bounds: list[int],
    devices: list[int],
    shard_rows: tuple[list, list] | None = None,
) -> list[cp.ndarray]:
    """Rows ``Z[order[lo:hi]]`` of every shard on its GPU. Device input is
    gathered on its own GPU, the other GPUs' shards first and in chunks of
    _UPLOAD_BYTES, so only a shard's rows move and that GPU holds one extra
    chunk at a time. Host input is staged in pinned chunks by several host
    threads while the previous chunk is copied, then placed on the GPU (all
    GPUs at once), so no unsorted device copy is kept: one GPU reads the input
    in order, several GPUs each only their own rows (``shard_rows``, see
    _shard_rows; gathered in input order, from C-ordered input)."""
    shards = list(itertools.pairwise(bounds))
    n, d = Z.shape
    if isinstance(Z, cp.ndarray):
        source, parts = Z.device.id, [None] * len(devices)
        step = max(1, _UPLOAD_BYTES // (d * Z.itemsize))
        with cp.cuda.Device(source):
            for r in sorted(range(len(devices)), key=lambda r: devices[r] == source):
                lo, hi = shards[r]  # the temporaries die with the statement
                if devices[r] == source:
                    parts[r] = Z[_copy(order[lo:hi], source)]
                    continue
                with cp.cuda.Device(devices[r]):
                    parts[r] = cp.empty((hi - lo, d), dtype=Z.dtype)
                for a in range(lo, hi, step):
                    b = min(hi, a + step)
                    chunk = Z[_copy(order[a:b], source)]
                    _copy(chunk, devices[r], out=parts[r][a - lo : b - lo])
                    del chunk  # before the next one
            cp.cuda.get_current_stream().synchronize()  # read on other streams
        return parts
    rows = _staging_rows(n, d * Z.itemsize)
    sources, targets = shard_rows or _shard_rows(order, bounds, devices)
    if len(shards) > 1:
        Z = np.ascontiguousarray(Z)  # np.take copies other layouts whole
        # The upload threads read them on their own streams.
        cp.cuda.get_current_stream().synchronize()

    def upload(rank, fill):
        (lo, hi), source = shards[rank], sources[rank]
        with cp.cuda.Device(devices[rank]):
            target = targets[rank]
            stream = cp.cuda.get_current_stream()
            X = cp.empty((hi - lo, d), dtype=Z.dtype)
            slots = [
                [_pinned(rows, d, Z.dtype), cp.empty((rows, d), dtype=Z.dtype), None]
                for _ in range(2)
            ]
            for i, a in enumerate(range(0, hi - lo, rows)):
                host, staged, done = slots[i % 2]
                if done is not None:
                    done.synchronize()  # the slot's previous chunk is placed
                b = min(hi - lo, a + rows)
                if source is None:
                    _parallel_copy(fill, host[: b - a], Z[a:b])
                else:
                    _parallel_copy(fill, host[: b - a], Z, source[a:b])
                staged[: b - a].set(host[: b - a], stream=stream)
                X[target[a:b]] = staged[: b - a]
                slots[i % 2][2] = stream.record()
            stream.synchronize()  # the caller reads it on its own stream
            return X

    with ThreadPoolExecutor(_UPLOAD_THREADS) as fill:
        if len(devices) == 1:
            return [upload(0, fill)]
        with ThreadPoolExecutor(len(devices)) as pool:
            return list(pool.map(lambda rank: upload(rank, fill), range(len(devices))))


def _shard_rows(
    order: cp.ndarray, bounds: list[int], devices: list[int]
) -> tuple[list, list]:
    """Each shard's cells in input order: their input rows (increasing, on the
    host; ``None`` for a single shard, which has them all) and their rows in
    the shard (on its GPU)."""
    n = order.size
    position = cp.empty(n, dtype=cp.int32)
    position[order] = cp.arange(n, dtype=cp.int32)
    if len(bounds) == 2:
        return [None], [_copy(position, devices[0])]
    sources, targets = [], []
    for (lo, hi), device in zip(itertools.pairwise(bounds), devices, strict=True):
        mine = cp.flatnonzero((position >= lo) & (position < hi))
        targets.append(_copy(position[mine] - lo, device))
        sources.append(cp.asnumpy(mine.astype(cp.int32)))
    return sources, targets


def _write_shards(
    results: list[cp.ndarray],
    shard_rows: tuple[list, list],
    devices: list[int],
    out: np.ndarray,
    prefault: threading.Thread | None,
) -> None:
    """``out[order] = concatenated results`` (the shards' rows in sorted
    order): every GPU puts its rows in input order and copies them into
    theirs of ``out`` (see _shard_rows), so no row moves between the GPUs.
    ``out`` is written once ``prefault`` (see _prefaulted) has finished."""
    sources, targets = shard_rows

    def write(rank):
        with cp.cuda.Device(devices[rank]):
            values = results[rank][targets[rank]]
            results[rank] = None
            if prefault is not None:
                prefault.join()
            _download(values, out, sources[rank])

    if len(devices) == 1:
        return write(0)
    with ThreadPoolExecutor(len(devices)) as pool:
        list(pool.map(write, range(len(devices))))


def _staging_rows(n: int, row_bytes: int) -> int:
    """Rows per pinned staging chunk: about 1/32 of the data, 64-256 MiB.
    Pinning costs time on first use; every chunk costs a host sync."""
    chunk = min(_UPLOAD_BYTES, max(_UPLOAD_BYTES >> 2, n * row_bytes >> 5))
    return max(1, min(n, chunk // row_bytes))


def _pinned(rows: int, d: int, dtype) -> np.ndarray:
    """A (rows, d) host array in pinned memory (CuPy caches these blocks)."""
    dtype = np.dtype(dtype)
    memory = cp.cuda.alloc_pinned_memory(rows * d * dtype.itemsize)
    return np.frombuffer(memory, dtype, rows * d).reshape(rows, d)


def _parallel_copy(
    pool: ThreadPoolExecutor,
    dst: np.ndarray,
    src: np.ndarray,
    index: np.ndarray | None = None,
    *,
    scatter: bool = False,
) -> None:
    """``dst[...] = src``, ``src[index]`` or (``scatter``) ``dst[index] = src``
    for increasing ``index`` into C-ordered arrays, by several threads (host
    memory bound)."""
    n = len(src if scatter else dst)
    cuts = np.linspace(0, n, _UPLOAD_THREADS + 1, dtype=np.int64)

    def copy(a, b):
        if index is None:
            np.copyto(dst[a:b], src[a:b])
        elif b > a and index[b - 1] - index[a] == b - a - 1:  # consecutive rows
            run = slice(index[a], index[a] + b - a)
            np.copyto(*((dst[run], src[a:b]) if scatter else (dst[a:b], src[run])))
        elif scatter:
            dst[index[a:b]] = src[a:b]
        elif b > a:  # mode="raise" would buffer the output
            np.take(src, index[a:b], axis=0, out=dst[a:b], mode="clip")

    list(pool.map(copy, cuts[:-1], cuts[1:]))


def _download(
    src: cp.ndarray, out: np.ndarray, index: np.ndarray | None = None
) -> None:
    """``out[...] = src``, or ``out[index] = src`` for increasing ``index``,
    through pinned chunks: the next chunk is copied from the GPU while host
    threads move the previous one into ``out``."""
    n, d = src.shape
    rows = _staging_rows(n, d * src.itemsize)
    stream = cp.cuda.get_current_stream()
    hosts = [_pinned(rows, d, src.dtype) for _ in range(2)]
    with ThreadPoolExecutor(_UPLOAD_THREADS) as pool:

        def drain(chunk, a, b, done):
            done.synchronize()
            if index is None:
                _parallel_copy(pool, out[a:b], chunk)
            else:
                _parallel_copy(pool, out, chunk, index[a:b], scatter=True)

        pending = None
        for j, a in enumerate(range(0, n, rows)):
            b = min(n, a + rows)
            chunk = hosts[j % 2][: b - a]
            src[a:b].get(stream=stream, out=chunk, blocking=False)
            ready = (chunk, a, b, stream.record())
            if pending is not None:
                drain(*pending)
            pending = ready
        if pending is not None:
            drain(*pending)


def _rhs_bounds(
    parts: list[cp.ndarray], offsets: np.ndarray, bounds: list[int]
) -> np.ndarray:
    """Bounds of the exact right-hand side sums (segment_gemm.cuh): the
    exponents e (int16) of n_g max |X_gd| < 2^e for each group g of the cells
    (sorted by group, ``offsets`` on the host) and PC d, so no other group or
    PC coarsens a sum. ``parts`` hold the cells ``bounds[r] .. bounds[r + 1]``;
    maxima are exact, so every split gives the same bounds."""
    maxima = []
    for part, lo, hi in zip(parts, bounds[:-1], bounds[1:], strict=True):
        with cp.cuda.Device(part.device.id):
            seg_start, seg_group = _segments(offsets, _SEGMENT, lo, hi)
            m = cp.empty((len(offsets) - 1, part.shape[1]), dtype=part.dtype)
            _correction_batched_cuda.group_absmax(
                part,
                seg_start=seg_start,
                seg_group=seg_group,
                out=m,
                stream=cp.cuda.get_current_stream().ptr,
            )
            maxima.append(m.get())
    maxima = np.maximum.reduce(maxima).astype(np.float64)
    if not np.isfinite(maxima).all():
        raise ValueError(
            "Input data contains NaN or infinite values. Please handle these before running harmony_integrate."
        )
    n = np.diff(offsets).astype(np.float64)[:, None]
    with np.errstate(over="ignore"):  # beyond float64 near 1e308: n < 2^a, max < 2^b
        rhs = n * maxima
    exp = np.frexp(np.where(rhs > 0, rhs, 1.0))[1]
    exp = np.where(np.isinf(rhs), np.frexp(n)[1] + np.frexp(maxima)[1], exp)
    return exp.astype(np.int16)


def _segments(
    offsets: np.ndarray, unit: int = _SEGMENT, lo: int = 0, hi: int | None = None
) -> tuple[cp.ndarray, cp.ndarray]:
    """Segments of the exact sums over cells sorted by group (``offsets`` on
    the host): runs of one group cut at multiples of _SEGMENT and of ``unit``
    (the block draw's units, where several GPUs split the cells). Start rows
    (and the end) and the group of each segment of the cells ``lo .. hi``
    (multiples of ``unit``), in rows from ``lo``."""
    hi = offsets[-1] if hi is None else hi
    grid = np.arange(-(-lo // _SEGMENT) * _SEGMENT, hi, _SEGMENT)
    start = np.union1d(np.clip(offsets, lo, hi), np.arange(lo, hi, unit))
    start = np.union1d(start, grid)
    group = np.searchsorted(offsets, start[:-1], side="right") - 1
    return cp.asarray(start - lo, dtype=cp.int32), cp.asarray(group, dtype=cp.int32)


def _sorted_groups(offsets: np.ndarray) -> cp.ndarray:
    """The groups of the sorted cells (``offsets`` on the host).

    The sum of a 1 at the first cell of every later group.
    """
    n_cells, starts = int(offsets[-1]), offsets[1:]
    groups = cp.zeros(n_cells, dtype=cp.int32)
    cp.add.at(groups, cp.asarray(starts[starts < n_cells]), 1)
    cp.cumsum(groups, dtype=cp.int32, out=groups)
    return groups


def _block_layout(n_total: int, n_blocks: int, shuffle_chunk: int) -> tuple[int, int]:
    """Shuffle runs and units of the block draw (see _draw_blocks).

    About _BLOCK_UNITS units, each dealing every block the same share.
    """
    chunk = max(1, min(shuffle_chunk, n_total // n_blocks))
    step = n_blocks * chunk
    return chunk, step * -(-n_total // (step * _BLOCK_UNITS))


def _draw_blocks(
    groups: cp.ndarray,
    *,
    n_groups: int,
    n_blocks: int,
    seed: int,
    chunk: int,
    unit: int,
    first: int,
    workspace: dict,
) -> None:
    """Draw the update blocks of a harmonize call into the workspace's
    ``idx_list`` and ``block_cat_offsets`` (sort scratch only lives here).
    ``groups`` are the sorted cells ``first ..`` of a call (``chunk`` and
    ``unit`` from _block_layout); a cell's block depends only on its position
    among all cells."""
    n = groups.size
    _clustering_cuda.draw_blocks(
        groups,
        idx_list=workspace["idx_list"],
        block_cat_offsets=workspace["block_cat_offsets"],
        idx_list_alt=cp.empty(n, dtype=cp.int32),
        sort_keys=cp.empty(n, dtype=cp.uint32),
        sort_keys_alt=cp.empty(n, dtype=cp.uint32),
        cub_temp=cp.empty(_clustering_cuda.get_cub_sort_temp_bytes(n_cells=n), dtype=cp.uint8),
        n_groups=n_groups,
        n_blocks=n_blocks,
        unit=unit,
        seed=seed & 0xFFFFFFFF,
        shuffle_chunk=chunk,
        first=first,
        stream=cp.cuda.get_current_stream().ptr,
    )
    # Block b is idx_list[starts[b] : starts[b + 1]]; read once for the loop.
    workspace["block_starts"] = workspace["block_cat_offsets"][: n_blocks * n_groups + 1 : n_groups].get()


def _kmeans_centroids(
    parts: list[cp.ndarray],
    *,
    bounds: list[int],
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> cp.ndarray:
    """Normalized k-means centroids of a stratified sample of the normalized
    sorted cells (shards ``parts``), on the current GPU."""
    # Fit only the initialization sample, retaining every observed stratum.
    n_cells = bounds[-1]
    n_init_cells = min(n_cells, _KMEANS_INIT_CELLS_PER_CLUSTER * n_clusters)
    if n_init_cells < cat_offsets.size - 1:
        n_strata = int(cp.count_nonzero(cp.diff(cat_offsets)))
        n_init_cells = max(n_init_cells, n_strata)
    sample = cell_indices
    if n_init_cells < n_cells:
        sample = _stratified_sample_indices(
            cat_offsets, cell_indices, n_init_cells, rng
        )
    rows = _gather_rows(parts, bounds, sample)
    Y = _kmeans(rows, n_clusters, max_iter=_KMEANS_MAX_ITER, rng=_seed_from_rng(rng))
    return _normalize_cp(Y)


def _allocate_clustering_workspace(
    n_cells: int,
    *,
    n_pcs: int,
    n_clusters: int,
    n_batches: int,
    n_groups: int,
    n_blocks: int,
    n_stored: int,
    dtype: cp.dtype,
) -> dict:
    """Buffers of the clustering loop. The tiles' groups are the batches, or
    one group for several keys; counts are kept per batch level."""
    workspace = {
        "Y_norm": cp.empty((n_clusters, n_pcs), dtype=dtype),
        # Exact centroid sums (at most 3 limbs) and the exponents of their
        # bound, n_cells per PC (|Z_norm|, R <= 1; see segment_gemm.cuh).
        "y_acc": cp.empty((3, n_clusters, n_pcs), dtype=cp.int64),
        "y_bound": cp.full(n_pcs, np.frexp(n_cells)[1], dtype=cp.int16),
        "idx_list": cp.empty(n_cells, dtype=cp.int32),
        "penalty": cp.empty((n_batches, n_clusters), dtype=dtype),
        "objective_partials": cp.empty(n_cells + 8, dtype=dtype),
        # Block-group offsets, then each block's assignment tiles.
        "block_cat_offsets": cp.empty(n_blocks * (2 * n_groups + 1) + 1, dtype=cp.int32),
        # Fixed-point counts of each block (the blocks are fixed per harmonize
        # call) or three recounted ones, of O, and two slots of new counts.
        "block_counts": cp.empty((n_stored + 3, n_batches, n_clusters), dtype=cp.int64),
    }
    # Centroids transposed: padded PCs x cluster blocks.
    warp_clusters = _clustering_cuda.FUSED_CLUSTER_STRIDE
    stride = -(-n_clusters // warp_clusters) * warp_clusters
    workspace["y_t"] = cp.empty((_clustering_cuda.fused_padded_pcs(n_pcs), stride), dtype=dtype)
    if _FORCE_CLUSTER_BLOCKS or n_clusters > warp_clusters:
        # Cluster blocks: per-warp column sums and a group's dot products for
        # each resident CTA, and the CTAs' tile counter.
        n_slots = _clustering_cuda.assign_slots(itemsize=dtype.itemsize)
        workspace["col_workspace"] = cp.empty(n_slots * _clustering_cuda.FUSED_WARPS * n_clusters + 1, cp.int64)
        workspace["dots_workspace"] = cp.empty(n_slots * _clustering_cuda.FUSED_GROUP * stride, dtype=dtype)
    return workspace


def _clustering(
    Z_norm: cp.ndarray,
    *,
    R: cp.ndarray,
    E: cp.ndarray,
    O: cp.ndarray,
    Pr_b: cp.ndarray,
    theta: cp.ndarray,
    objectives_harmony: list,
    max_iter: int,
    sigma: float,
    n_blocks: int,
    n_batches: int,
    n_covariates: int,
    n_joint_categories: int,
    stabilized_penalty: bool,
    workspace: dict,
    tol: float = 0.0,
    kernel_seed: int = 0,
    initialize: bool = False,
    n_total: int = 0,
    sums: dict | None = None,
) -> None:
    """
    Clustering updates of R, O and E in place: the update blocks of cells are
    reassigned against the counts of all other cells until convergence or
    ``max_iter`` passes. ``initialize`` instead assigns every cell to the
    centroids in ``workspace["Y_norm"]`` without penalty. On several GPUs,
    ``sums`` (``comm`` handle and ``rank``) sums over the ``n_total`` cells.
    """
    objective = _clustering_cuda.clustering_loop(
        Z_norm,
        **_assignment_args(R, Z_norm.dtype),
        E=E,
        O=O,
        Pr_b=Pr_b.ravel(),
        theta=theta,
        **workspace,
        n_cells=Z_norm.shape[0],
        n_pcs=Z_norm.shape[1],
        n_clusters=R.shape[1],
        n_batches=n_batches,
        n_covariates=n_covariates,
        n_joint_categories=n_joint_categories,
        n_blocks=n_blocks,
        sigma=float(sigma),
        tol=float(tol),
        max_iter=max_iter,
        seed=kernel_seed & 0xFFFFFFFF,
        stabilized=stabilized_penalty,
        initialize=initialize,
        n_total=n_total,
        stream=cp.cuda.get_current_stream().ptr,
        **(sums or {}),
    )
    objectives_harmony.append(objective)


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
    output: cp.ndarray | None = None,
    normalize: bool = False,
    segments: tuple | None = None,
    sums: dict | None = None,
) -> cp.ndarray:
    """
    Apply correction to the embedding based on the specified method.

    Cells are sorted by batch. ``normalize`` writes the L2-normalized embedding.
    """
    if correction_method == "batched":
        return _correction_batched(
            X,
            R,
            O=O,
            lambda_kb=lambda_kb,
            segments=segments,
            output=output,
            normalize=normalize,
            sums=sums,
        )
    elif correction_method == "fast":
        Z = _correction_fast(
            X,
            R,
            O=O,
            lambda_kb=lambda_kb,
            cats=cats,
            n_batches=n_batches,
            cat_offsets=cat_offsets,
            output=output,
        )
        return _normalize_cp(Z, out=Z) if normalize else Z
    raise ValueError("correction_method must be 'fast' or 'batched'.")


def _correction_multi(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    lambda_kb: cp.ndarray,
    n_first: int,
    joint_cats: cp.ndarray,
    marginal_joint_offsets: cp.ndarray,
    marginal_joint_indices: cp.ndarray,
    segments: tuple,
    output: cp.ndarray | None = None,
    normalize: bool = False,
    n_total: int = 0,
    sums: dict | None = None,
    n_ranks: int = 1,
) -> cp.ndarray:
    """Apply the exact general-design correction in bounded cluster chunks.

    Cells are sorted by joint category and ``segments`` are their segments
    (see _segments, plus the right-hand side bounds). The joint counts are the
    clustering's, recomputed from R per chunk (over ``n_total`` cells on all
    ``n_ranks`` GPUs of ``sums``). Systems solved one at a time (see
    _solve_spd_batched) are split between the GPUs, which then exchange the
    coefficients.
    """
    seg_start, seg_group, bounds = segments
    n_pcs, n_clusters = X.shape[1], R.shape[1]
    n_joint_categories = joint_cats.shape[0]
    nb1 = n_batches + 1
    chunk_size = _multi_correction_cluster_chunk_size(
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        n_joint_categories=n_joint_categories,
        itemsize=X.dtype.itemsize,
    )
    starts = range(0, n_clusters, chunk_size)
    rank = (sums or {}).get("rank", 0)
    if nb1 < _SOLVE_ONE_AT_A_TIME:
        rank, n_ranks = 0, 1  # every GPU solves every system
    # Rounds of one chunk, or of one chunk per GPU when chunks are smaller:
    # GPU r solves the systems cuts[r] .. cuts[r + 1] of a round.
    per_round = 1 if chunk_size >= n_ranks else n_ranks
    stream = cp.cuda.get_current_stream().ptr
    Z = _correction_output(X, output)
    system = None
    for i, cluster_start in enumerate(starts):
        if i % per_round == 0:  # clusters first .. first + m
            first = cluster_start
            m = min(n_clusters - first, per_round * chunk_size)
            share = chunk_size if per_round > 1 else -(-m // n_ranks)
            cuts = [min(m, r * share) for r in range(n_ranks + 1)]
        chunk = slice(cluster_start, cluster_start + chunk_size)
        lambda_chunk = cp.ascontiguousarray(lambda_kb[:, chunk])
        R_chunk = R[:, chunk]
        k = R_chunk.shape[1]
        # Systems in double (see kernels_correction_multi.cuh).
        joint_rhs = cp.empty((n_joint_categories, k, n_pcs), dtype=cp.float64)
        gram = cp.empty((k, nb1, nb1), dtype=cp.float64)
        rhs = cp.empty((k, nb1, n_pcs), dtype=cp.float64)
        acc = cp.empty(
            (n_joint_categories, _limbs(X.dtype.itemsize), k, n_pcs), dtype=cp.int64
        )
        joint_O = cp.empty((n_joint_categories, k), dtype=X.dtype)
        _clustering_cuda.joint_counts(
            R_chunk,
            seg_start=seg_start,
            seg_group=seg_group,
            out=joint_O,
            acc=acc,
            n_total=n_total,
            stream=stream,
            **(sums or {}),
        )
        _correction_batched_cuda.prepare_multi(
            X,
            R=R_chunk,
            joint_O=joint_O,
            joint_cats=joint_cats,
            marginal_joint_offsets=marginal_joint_offsets,
            marginal_joint_indices=marginal_joint_indices,
            lambda_kb=lambda_chunk,
            active_mask=(lambda_chunk < X.dtype.type(_SUPPRESS_PENALTY)).view(cp.uint8),
            n_batches=n_batches,
            n_first=n_first,
            seg_start=seg_start,
            seg_group=seg_group,
            bounds=bounds,
            gram=gram,
            rhs=rhs,
            joint_rhs=joint_rhs,
            acc=acc,
            stream=stream,
            **(sums or {}),
        )
        lo, hi = (cut + first - cluster_start for cut in cuts[rank : rank + 2])
        if 0 <= lo < k:  # this GPU's systems (all in one chunk)
            system = gram[lo:hi], rhs[lo:hi]
        del gram, rhs, joint_rhs, acc, joint_O
        if (i + 1) % per_round and i + 1 < len(starts):
            continue  # the round's next chunk
        W_all = _solve_spd_batched(*system) if system else cp.empty((0, nb1, n_pcs))
        system = None
        if n_ranks > 1:
            W_all, W_mine = cp.empty((m, nb1, n_pcs)), W_all
            W_all[cuts[rank] : cuts[rank + 1]] = W_mine
            _clustering_cuda.allgather_rows(
                W_all.reshape(m, -1).view(cp.int64), starts=cuts, stream=stream, **sums
            )
        W_all = W_all.astype(X.dtype)
        # Each chunk continues the sums of the earlier clusters in Z and the
        # last one subtracts them from X, so no second buffer is needed.
        for start in range(first, first + m, chunk_size):
            R_chunk = R[:, start : start + chunk_size]
            k = R_chunk.shape[1]
            last = start + k == n_clusters
            _correction_batched_cuda.apply_multi(
                X,
                R=R_chunk,
                W_all=W_all[start - first :][:k],
                joint_cats=joint_cats,
                n_batches=n_batches,
                seg_start=seg_start,
                seg_group=seg_group,
                normalize=normalize and last,
                W_joint=cp.empty((n_joint_categories, k, n_pcs), dtype=X.dtype),
                Z=Z,
                resume=start > 0,
                finish=last,
                stream=stream,
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

    if gram.dtype != cp.float64 or rhs.dtype != cp.float64:
        raise TypeError("The multi-key systems are solved in float64")

    stream = cp.cuda.get_current_stream()
    cusolver_handle = cp.cuda.device.get_cusolver_handle()
    cublas_handle = cp.cuda.device.get_cublas_handle()
    cp.cuda.cusolver.setStream(cusolver_handle, stream.ptr)
    cp.cuda.cublas.setStream(cublas_handle, stream.ptr)
    lower = cp.cuda.cublas.CUBLAS_FILL_MODE_LOWER
    if matrix_size >= _SOLVE_ONE_AT_A_TIME:
        # Large systems (many batch levels): the blocked factorization of one
        # matrix at a time is much faster than the batched one. A symmetric
        # C-ordered matrix is its own column-major form.
        dpotrf, dpotrs = cp.cuda.cusolver.dpotrf, cp.cuda.cusolver.dpotrs
        n = matrix_size
        size = cp.cuda.cusolver.dpotrf_bufferSize(
            cusolver_handle, lower, n, gram_work.data.ptr, n
        )
        work = cp.empty(max(size, 1), dtype=cp.float64)
        solve_info = cp.empty(1, dtype=cp.int32)  # keeps the factor's info
        for i in range(n_matrices):
            factor, solution = gram_work[i], cp.asfortranarray(rhs[i])
            dpotrf(
                cusolver_handle, lower, n, factor.data.ptr, n, work.data.ptr,
                size, info[i:].data.ptr,
            )  # fmt: skip
            dpotrs(
                cusolver_handle, lower, n, n_rhs, factor.data.ptr, n,
                solution.data.ptr, n, solve_info.data.ptr,
            )  # fmt: skip
            rhs_work[i] = solution
        failed = np.flatnonzero(cp.asnumpy(info))
        for matrix_index in failed:
            rhs_work[matrix_index] = cp.linalg.lstsq(
                gram[matrix_index], rhs[matrix_index], rcond=None
            )[0]
        return rhs_work
    cp.cuda.cusolver.dpotrfBatched(
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
    one = np.ones(1, dtype=np.float64)
    for operation in (
        cp.cuda.cublas.CUBLAS_OP_T,
        cp.cuda.cublas.CUBLAS_OP_N,
    ):
        cp.cuda.cublas.dtrsmBatched(
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
    """Choose a cluster chunk that keeps multi-key scratch below 1 GiB (one cluster at a time above that)."""
    nb1 = n_batches + 1
    # Double Gram and RHS with the solver's copies, joint right-hand sides
    # (double) and coefficients, exact joint sums, and joint counts. The chunk
    # sets the results (correction_batched.cu), so the terms of buffers that
    # are gone stay: the solver's Gram copies, coefficients in X's dtype and a
    # sliced lambda and mask.
    per_cluster = (
        16 * (nb1 * nb1 + nb1 * n_pcs)
        + itemsize * nb1 * n_pcs
        + n_joint_categories * ((8 + itemsize) * n_pcs + itemsize)
        + 8 * _limbs(itemsize) * n_joint_categories * n_pcs
        + (itemsize + 1) * n_batches
    )
    return max(1, min(n_clusters, _CORRECTION_WORKSPACE_LIMIT_BYTES // per_cluster))


def _limbs(itemsize: int) -> int:
    """int64 limbs of the exact right-hand side sums (segment_gemm.cuh)."""
    return 1 if itemsize == 4 else 2


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
    output: cp.ndarray | None = None,
) -> cp.ndarray:
    """Apply the bounded-memory correction method (cells sorted by batch)."""
    n_cells = X.shape[0]
    n_pcs = X.shape[1]
    n_clusters = R.shape[1]
    nb1 = n_batches + 1
    dtype = X.dtype

    Z = _correction_output(X, output)
    _correction_cuda.correction_fast(
        X,
        R=R,
        O=O,
        cats=cats,
        cat_offsets=cat_offsets,
        lambda_kb=lambda_kb,
        n_cells=n_cells,
        n_pcs=n_pcs,
        n_clusters=n_clusters,
        n_batches=n_batches,
        Z=Z,
        R_col=cp.empty(n_cells, dtype=dtype),
        S=cp.empty((nb1, n_pcs), dtype=dtype),
        W=cp.empty((nb1, n_pcs), dtype=dtype),
        beta=cp.empty(n_pcs, dtype=cp.float64),
        stream=cp.cuda.get_current_stream().ptr,
    )
    return Z


def _correction_batched(
    X: cp.ndarray,
    R: cp.ndarray,
    *,
    O: cp.ndarray,
    lambda_kb: cp.ndarray,
    segments: tuple,
    output: cp.ndarray | None = None,
    normalize: bool = False,
    sums: dict | None = None,
) -> cp.ndarray:
    """Batched correction of cells sorted by batch, as many clusters per pass as fit in 1 GiB of workspace.

    Right-hand sides and the correction over ``segments`` (see _segments, plus the right-hand side bounds).
    """
    seg_start, seg_group, bounds = segments
    n_pcs, n_clusters, n_batches = X.shape[1], R.shape[1], len(bounds)
    dtype = X.dtype
    chunk = _batched_cluster_chunk(n_pcs, n_batches, dtype.itemsize)
    chunk = -(-n_clusters // -(-n_clusters // max(1, chunk)))  # balanced
    Z = _correction_output(X, output)
    _correction_batched_cuda.correction_batched(
        X,
        **_assignment_args(R, dtype),
        O=O,
        lambda_kb=lambda_kb,
        seg_start=seg_start,
        seg_group=seg_group,
        bounds=bounds,
        Z=Z,
        chunk=chunk,
        normalize=normalize,
        stream=cp.cuda.get_current_stream().ptr,
    )
    # Each chunk continues the sums of the earlier clusters in Z.
    for start in range(0, n_clusters, chunk):
        n = min(chunk, n_clusters - start)
        _correction_batched_cuda.correction_batched(
            X,
            **_assignment_args(R, X.dtype),
            O=O,
            lambda_kb=lambda_kb,
            seg_start=seg_start,
            seg_group=seg_group,
            bounds=bounds,
            Z=Z,
            S=S[:n],
            W=W[:n],
            rhs_acc=rhs_acc,
            normalize=normalize,
            cluster_start=start,
            resume=start > 0,
            finish=start + n == n_clusters,
            stream=cp.cuda.get_current_stream().ptr,
            **(sums or {}),
        )
    return Z


def _batched_cluster_chunk(n_pcs: int, n_batches: int, itemsize: int) -> int:
    """Clusters per pass of the batched correction that fit in 1 GiB (0: not even one).

    Counts their right-hand sides, coefficients and exact sums.
    """
    per_cluster = n_batches * n_pcs * (2 * itemsize + 8 * _limbs(itemsize))
    return _CORRECTION_WORKSPACE_LIMIT_BYTES // per_cluster


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
