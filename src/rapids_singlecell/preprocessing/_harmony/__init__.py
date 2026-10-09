from __future__ import annotations

import itertools
import math
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
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
from ._multi_gpu import _comm, _copy, _gather_rows, _peer_copies_work

if TYPE_CHECKING:
    import pandas as pd

    from rapids_singlecell._utils._random import RNGLike, SeedLike

COLSUM_ALGO = Literal["columns", "atomics", "gemm", "benchmark"]
_SUPPRESS_PENALTY = 1e30
_CORRECTION_WORKSPACE_LIMIT_BYTES = 1 << 30
_KMEANS_MAX_ITER = 25
_KMEANS_INIT_CELLS_PER_CLUSTER = 5_000
_FUSED_MAX_CLUSTERS = 128
# Tests: route every shape through the general assignment kernel.
_FORCE_GENERAL_ASSIGNMENT = False
_UPLOAD_BYTES = 1 << 28  # largest pinned staging chunk
_UPLOAD_THREADS = 8
_BLOCK_UNITS = 384  # units of the block draw (see _draw_blocks)
_SEGMENT = 2048  # cells per segment of the exact sums (see _segments)

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
    devices: list[int] | None = None,
    out: np.ndarray | None = None,
) -> cp.ndarray | np.ndarray:
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
        Cells per run of neighbouring cells of the same batch (joint category
        with several keys) that move between update blocks together. ``1``
        shuffles every cell independently.

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
        Host array receiving the corrected embedding (then returned).

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

    if max_iter_harmony < 1:
        raise ValueError("max_iter_harmony must be >= 1")
    if int(shuffle_chunk_size) != shuffle_chunk_size or shuffle_chunk_size < 1:
        raise ValueError(
            f"shuffle_chunk_size must be a positive integer, got {shuffle_chunk_size}."
        )
    shuffle_chunk_size = int(shuffle_chunk_size)
    if colsum_algo not in (None, "columns", "atomics", "gemm", "benchmark"):
        raise ValueError(f"Unknown colsum_algo {colsum_algo!r}.")
    n_cells = Z.shape[0]

    rng = np.random.default_rng(rng)
    # Fault in the host output while the GPUs work; copying into touched pages
    # is about twice as fast as into a fresh allocation. An output sharing
    # memory with the host input is left alone: the input is read later.
    prefault = None
    if out is not None and not (
        isinstance(Z, np.ndarray) and np.may_share_memory(out, Z)
    ):
        prefault = threading.Thread(target=out.fill, args=(0,), daemon=True)
        prefault.start()
    # Process batch information
    batch_codes, n_levels = _get_batch_codes(batch_mat, batch_key)
    n_covariates = int(n_levels.size)
    n_batches = int(n_levels.sum())
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
    # Sorted groups: offsets from the group sizes, cells in order.
    group_offsets = cp.zeros(n_groups + 1, dtype=cp.int32)
    group_offsets[1:] = cp.cumsum(cp.bincount(groups, minlength=n_groups))
    cell_indices = cp.arange(n_cells, dtype=cp.int32)
    del batch_codes
    if n_covariates > 1:
        del joint_codes  # unsorted
    host_offsets = cp.asnumpy(group_offsets)

    # Set up parameters
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))
        n_clusters = max(n_clusters, 2)

    dtype = np.dtype(Z.dtype)
    theta_array = _get_theta_array(theta, n_levels, dtype)
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

    n_blocks = -(-n_cells // max(1, math.ceil(n_cells * block_proportion)))
    if bfloat16 and not (
        n_covariates == 1 and dtype == np.float32 and correction_method == "batched"
    ):
        warnings.warn(
            "dtype='bfloat16' needs one batch key and the batched correction; "
            "using float32 assignments instead (twice the assignment memory).",
            UserWarning,
            stacklevel=3,
        )
        bfloat16 = False
    devices = devices or [cp.cuda.Device().id]
    if len(devices) > 1 and n_covariates == 1 and correction_method != "batched":
        warnings.warn(
            f"correction_method={correction_method!r} runs on one GPU.",
            UserWarning,
            stacklevel=3,
        )
        devices = devices[:1]
    # Shards: contiguous runs of the sorted cells, one per GPU, cut at
    # multiples of the block draw's units (which also cut the segments), so
    # that every GPU draws the blocks and sums the segments of a single-GPU run.
    unit = _block_layout(n_cells, n_blocks, shuffle_chunk_size)[1]
    devices = devices[: max(1, n_cells // unit)]
    n_ranks, n_pcs = len(devices), Z.shape[1]
    bounds = [
        min(n_cells, round(n_cells * r / n_ranks / unit) * unit) for r in range(n_ranks)
    ] + [n_cells]
    peer = n_ranks > 1 and _peer_copies_work(devices)
    parts = _upload_sorted(Z, order, bounds, devices, peer=peer)
    del Z
    # Exact right-hand side sums per group are bounded by n_g max|X|.
    extremes = []
    for part in parts:
        with cp.cuda.Device(part.device.id):
            extremes += [float(part.min()), float(part.max())]
    del part
    if not all(map(math.isfinite, extremes)):
        raise ValueError(
            "Input data contains NaN or infinite values. Please handle these "
            "before running harmony_integrate."
        )
    max_abs = max(map(abs, extremes))
    rhs_bounds = (np.diff(host_offsets) * max_abs).tolist()
    Y_norm = _kmeans_centroids(
        parts,
        bounds=bounds,
        cat_offsets=group_offsets,
        cell_indices=cell_indices,
        n_clusters=n_clusters,
        rng=rng,
        peer=peer,
    )
    if correction_method != "fast":  # the only later reader
        cats = cell_indices = None
    comm = None
    if n_ranks > 1:
        # Exact int64 sums of block counts, centroids and right-hand sides.
        capacity = max(n_blocks * n_groups, 2 * n_groups * n_pcs, 3 * n_pcs)
        comm, comm_buffers = _comm(devices, capacity * n_clusters, peer=peer)

    def run(rank: int):
        # One GPU's shard; several GPUs run it in parallel threads.
        lo, hi = bounds[rank], bounds[rank + 1]

        def local(a):
            return _copy(a, devices[rank], peer=peer)

        X = parts[rank]
        Z_norm = _normalize_cp(X)
        seg_start, seg_group = _segments(host_offsets, unit, lo, hi)
        segments = (seg_start, seg_group, rhs_bounds)
        workspace = _allocate_clustering_workspace(
            hi - lo,
            n_pcs=n_pcs,
            n_clusters=n_clusters,
            n_batches=n_batches,
            n_groups=n_groups,
            n_covariates=n_covariates,
            n_blocks=n_blocks,
            dtype=dtype,
            bfloat16=bfloat16,
        )
        joint = {}
        if n_covariates > 1:
            joint = {
                "joint_cats": local(joint_cats),
                "marginal_joint_offsets": local(marginal_joint_offsets),
                "marginal_joint_indices": local(marginal_joint_indices),
            }
            workspace.update(joint, n_first=int(n_levels[0]))
        workspace["seg_start"] = seg_start
        workspace["Y_norm"][...] = local(Y_norm)
        # The update blocks are drawn once, before the large arrays exist.
        _draw_blocks(
            local(groups[lo:hi]),
            n_groups=n_groups,
            n_blocks=n_blocks,
            seed=kernel_seed,
            shuffle_chunk=shuffle_chunk_size,
            workspace=workspace,
            n_total=n_cells,
            first=lo,
        )
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
            normalize = not is_converged and i + 1 < max_iter_harmony
            if n_covariates > 1:
                Z_hat = _correction_multi(
                    X,
                    R,
                    O=O,
                    lambda_kb=lambda_kb,
                    joint_O=workspace["O_joint"],
                    n_batches=n_batches,
                    **joint,
                    segments=segments,
                    output=Z_norm,
                    normalize=normalize,
                    sums=sums,
                )
            else:
                Z_hat = _correction(
                    X,
                    R=R,
                    O=O,
                    lambda_kb=lambda_kb,
                    correction_method=correction_method,
                    cats=cats,
                    n_batches=n_batches,
                    cat_offsets=group_offsets,
                    cell_indices=cell_indices,
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
                if n_covariates == 1 and correction_method != "batched":
                    Z_norm = _normalize_cp(Z_hat, out=Z_hat)
        return Z_hat, is_converged, i

    def guarded(rank: int):
        with cp.cuda.Device(devices[rank]):
            try:
                result = run(rank)
                cp.cuda.get_current_stream().synchronize()  # read by other threads
                return result
            except BaseException:
                if comm is not None:
                    comm.abort()  # release the other GPUs' threads
                raise

    if n_ranks == 1:
        results = [guarded(0)]
    else:
        # The GPUs' threads read this thread's arrays on their own streams.
        cp.cuda.get_current_stream().synchronize()
        with ThreadPoolExecutor(n_ranks) as pool:
            results = list(pool.map(guarded, range(n_ranks)))
        del comm, comm_buffers
    _, is_converged, i = results[0]
    if is_converged and verbose:
        print(f"Harmony converged in {i + 1} iterations")
    if not is_converged:
        warnings.warn(
            "Harmony did not converge. Consider increasing the number of iterations"
        )
    parts.clear()  # free the input before the output
    if prefault is not None:
        prefault.join()
    if n_ranks > 1:
        _write_shards([r[0] for r in results], order, bounds, devices, out, peer=peer)
        return out
    Z_hat = results[0][0]
    # Put the result back in input order, on its GPU.
    with cp.cuda.Device(Z_hat.device.id):
        result = cp.empty_like(Z_hat)
        result[_copy(order, Z_hat.device.id, peer=peer)] = Z_hat
        if out is None:
            return result
        _download(result, out)
    return out


def _assignment_args(R: cp.ndarray, dtype) -> dict:
    """Pass R to the CUDA modules; bfloat16 R (stored as uint16) travels as
    ``R_bf16`` next to a placeholder ``R``."""
    if R.dtype == cp.uint16:
        return {"R": cp.empty((1, R.shape[1]), dtype=dtype), "R_bf16": R}
    return {"R": R}


def _upload_sorted(
    Z, order: cp.ndarray, bounds: list[int], devices: list[int], *, peer: bool
) -> list[cp.ndarray]:
    """Rows ``Z[order[lo:hi]]`` of every shard on its GPU. Device input
    shards are gathered on the current GPU. Host input chunks holding a GPU's
    rows are staged in pinned memory by several host threads while the
    previous chunk is copied, then placed on the GPU (all GPUs at once), so no
    unsorted device copy is kept."""
    shards = list(itertools.pairwise(bounds))
    if isinstance(Z, cp.ndarray):
        Z = _copy(Z, order.device.id, peer=peer)
        return [
            _copy(Z[order[lo:hi]], d, peer=peer) for d, (lo, hi) in zip(devices, shards)
        ]
    n, d = Z.shape
    rows = _staging_rows(n, d * Z.itemsize)
    n_chunks = -(-n // rows)
    position = cp.empty(n, dtype=cp.int32)
    position[order] = cp.arange(n, dtype=cp.int32)
    # Sorted positions spanned by each input chunk.
    chunks = cp.full(n_chunks * rows, position[-1], dtype=cp.int32)
    chunks[:n] = position
    chunks = chunks.reshape(n_chunks, rows)
    first, last = cp.asnumpy(chunks.min(1)), cp.asnumpy(chunks.max(1))
    del chunks
    # Destination row per input row; other GPUs' rows go to a spare row.
    targets = [
        cp.where((position >= lo) & (position < hi), position - lo, hi - lo)
        for lo, hi in shards
    ]
    # The upload threads read them on their own streams.
    cp.cuda.get_current_stream().synchronize()

    def upload(rank, fill):
        lo, hi = shards[rank]
        with cp.cuda.Device(devices[rank]):
            target = _copy(targets[rank], devices[rank], peer=peer)
            stream = cp.cuda.get_current_stream()
            X = cp.empty((hi - lo + 1, d), dtype=Z.dtype)
            slots = [
                [_pinned(rows, d, Z.dtype), cp.empty((rows, d), dtype=Z.dtype), None]
                for _ in range(2)
            ]
            for i, k in enumerate(np.flatnonzero((last >= lo) & (first < hi))):
                host, staged, done = slots[i % 2]
                if done is not None:
                    done.synchronize()  # the slot's previous chunk is placed
                a, b = k * rows, min(n, (k + 1) * rows)
                _parallel_copy(fill, host[: b - a], Z[a:b])
                staged[: b - a].set(host[: b - a], stream=stream)
                X[target[a:b]] = staged[: b - a]
                slots[i % 2][2] = stream.record()
            stream.synchronize()  # the caller reads it on its own stream
            return X[: hi - lo]

    with (
        ThreadPoolExecutor(_UPLOAD_THREADS) as fill,
        ThreadPoolExecutor(len(devices)) as pool,
    ):
        return list(pool.map(lambda rank: upload(rank, fill), range(len(devices))))


def _write_shards(
    results: list[cp.ndarray],
    order: cp.ndarray,
    bounds: list[int],
    devices: list[int],
    out: np.ndarray,
    *,
    peer: bool,
) -> None:
    """``out[order] = concatenated results`` (the shards' rows in sorted
    order): each GPU's rows are split by the output range (in input order) they
    land in, moved to that range's GPU and placed there; every GPU then copies
    its range to the host."""
    shards = list(itertools.pairwise(bounds))

    def split(rank):
        lo, hi = shards[rank]
        with cp.cuda.Device(devices[rank]):
            rows = _copy(order[lo:hi], devices[rank], peer=peer)
            parts = []
            for a, b in shards:
                mine = cp.flatnonzero((rows >= a) & (rows < b))
                parts.append((results[rank][mine], rows[mine] - a))
            cp.cuda.get_current_stream().synchronize()
            return parts

    def write(rank):
        # This GPU's output rows: its own parts plus the other GPUs' (kept
        # alive until they are placed), then one host copy.
        a, b = shards[rank]
        with cp.cuda.Device(devices[rank]):
            moved = [
                [_copy(x, devices[rank], peer=peer) for x in parts[rank]]
                for parts in split_parts
            ]
            rows = cp.empty((b - a, out.shape[1]), dtype=out.dtype)
            for values, positions in moved:
                rows[positions] = values
            _download(rows, out[a:b])

    with ThreadPoolExecutor(len(devices)) as pool:
        split_parts = list(pool.map(split, range(len(devices))))
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


def _parallel_copy(pool: ThreadPoolExecutor, dst: np.ndarray, src: np.ndarray) -> None:
    """``dst[...] = src`` by several threads (host memory bound)."""
    cuts = np.linspace(0, len(src), _UPLOAD_THREADS + 1, dtype=np.int64)
    list(pool.map(lambda a, b: np.copyto(dst[a:b], src[a:b]), cuts[:-1], cuts[1:]))


def _download(src: cp.ndarray, out: np.ndarray) -> None:
    """``out[...] = src`` through pinned chunks: the next chunk is copied from
    the GPU while host threads move the previous one into ``out``."""
    n, d = src.shape
    rows = _staging_rows(n, d * src.itemsize)
    stream = cp.cuda.get_current_stream()
    hosts = [_pinned(rows, d, src.dtype) for _ in range(2)]
    with ThreadPoolExecutor(_UPLOAD_THREADS) as pool:

        def drain(chunk, a, b, done):
            done.synchronize()
            _parallel_copy(pool, out[a:b], chunk)

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


def _block_layout(n_total: int, n_blocks: int, shuffle_chunk: int) -> tuple[int, int]:
    """Shuffle runs and units of the block draw (see _draw_blocks): about
    _BLOCK_UNITS units, each dealing the same cells to every block."""
    chunk = max(1, min(shuffle_chunk, n_total // n_blocks))
    step = n_blocks * chunk
    return chunk, step * -(-n_total // (step * _BLOCK_UNITS))


def _draw_blocks(
    groups: cp.ndarray,
    *,
    n_groups: int,
    n_blocks: int,
    seed: int,
    shuffle_chunk: int,
    workspace: dict,
    n_total: int | None = None,
    first: int = 0,
) -> None:
    """Draw the update blocks of a harmonize call into the workspace's
    ``idx_list`` and ``block_cat_offsets`` (sort scratch only lives here).
    ``groups`` are the cells ``first ..`` of ``n_total`` sorted cells; a
    cell's block depends only on its position in all ``n_total``."""
    n = groups.size
    chunk, unit = _block_layout(
        n if n_total is None else n_total, n_blocks, shuffle_chunk
    )
    _clustering_cuda.draw_blocks(
        groups,
        idx_list=workspace["idx_list"],
        block_cat_offsets=workspace["block_cat_offsets"],
        idx_list_alt=cp.empty(n, dtype=cp.int32),
        sort_keys=cp.empty(n, dtype=cp.uint32),
        sort_keys_alt=cp.empty(n, dtype=cp.uint32),
        cub_temp=cp.empty(
            _clustering_cuda.get_cub_sort_temp_bytes(n_cells=n), dtype=cp.uint8
        ),
        n_groups=n_groups,
        n_blocks=n_blocks,
        unit=unit,
        seed=seed & 0xFFFFFFFF,
        shuffle_chunk=chunk,
        first=first,
        stream=cp.cuda.get_current_stream().ptr,
    )


def _kmeans_centroids(
    parts: list[cp.ndarray],
    *,
    bounds: list[int],
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
    peer: bool,
) -> cp.ndarray:
    """Normalized k-means centroids of a stratified sample of the sorted cells
    (shards ``parts``), on the current GPU."""
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
    rows = _normalize_cp(_gather_rows(parts, bounds, sample, peer=peer))
    Y = _kmeans(rows, n_clusters, max_iter=_KMEANS_MAX_ITER, rng=_seed_from_rng(rng))
    return _normalize_cp(Y)


def _allocate_clustering_workspace(
    n_cells: int,
    *,
    n_pcs: int,
    n_clusters: int,
    n_batches: int,
    n_groups: int,
    n_covariates: int,
    n_blocks: int,
    dtype: cp.dtype,
    bfloat16: bool = False,
) -> dict:
    """Buffers of the clustering loop. Groups are batches, or joint categories
    (with counts in ``O_joint``) for several keys."""
    counts_bytes = n_blocks * n_groups * n_clusters * 8
    if counts_bytes > 1 << 30:
        warnings.warn(
            f"Harmony keeps {counts_bytes / 2**30:.1f} GiB of cluster counts for "
            f"{n_blocks} update blocks x {n_groups} batch groups; a larger "
            "block_proportion needs less.",
            UserWarning,
            stacklevel=4,
        )
    n_sm = cp.cuda.Device().attributes["MultiProcessorCount"]
    # Assignment tiles: four per SM, plus one partial tile per group.
    n_tiles = 4 * n_sm + n_groups + 1
    workspace = {
        "Y_norm": cp.empty((n_clusters, n_pcs), dtype=dtype),
        # Exact centroid sums (at most 3 limbs) and their scale.
        "y_acc": cp.empty((3, n_clusters, n_pcs), dtype=cp.int64),
        "y_scale": cp.empty(1, dtype=cp.float64),
        "idx_list": cp.empty(n_cells, dtype=cp.int32),
        "penalty": cp.empty((n_batches, n_clusters), dtype=dtype),
        "objective_partials": cp.empty(n_cells + 8, dtype=dtype),
        # Block-group offsets, then the assignment tiles of one block.
        "block_cat_offsets": cp.empty(
            n_blocks * n_groups + 1 + n_groups + 1, dtype=cp.int32
        ),
        # Fixed-point counts of each block (the blocks are fixed per harmonize
        # call), of O, and two slots of new block counts.
        "block_counts": cp.empty((n_blocks + 3, n_groups, n_clusters), dtype=cp.int64),
    }
    if n_covariates > 1:
        workspace["O_joint"] = cp.zeros((n_groups, n_clusters), dtype=dtype)
        workspace["group_penalty"] = cp.empty((n_groups, n_clusters), dtype=dtype)
    # Centroids transposed: padded PCs x 128-cluster blocks.
    stride = -(-n_clusters // _FUSED_MAX_CLUSTERS) * _FUSED_MAX_CLUSTERS
    workspace["y_t"] = cp.empty((-(-n_pcs // 4) * 4, stride), dtype=dtype)
    if _FORCE_GENERAL_ASSIGNMENT or n_clusters > _FUSED_MAX_CLUSTERS:
        # General assignment: per-warp column sums.
        workspace["col_workspace"] = cp.empty(n_tiles * 8 * n_clusters, dtype=cp.int64)
        workspace["force_general"] = _FORCE_GENERAL_ASSIGNMENT
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
    cell_indices: cp.ndarray,
    output: cp.ndarray | None = None,
    normalize: bool = False,
    segments: tuple | None = None,
    sums: dict | None = None,
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
            segments=segments,
            output=output,
            normalize=normalize,
            sums=sums,
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
    marginal_joint_offsets: cp.ndarray,
    marginal_joint_indices: cp.ndarray,
    segments: tuple,
    output: cp.ndarray | None = None,
    normalize: bool = False,
    sums: dict | None = None,
) -> cp.ndarray:
    """Apply the exact general-design correction in bounded cluster chunks.

    Cells are sorted by joint category, ``segments`` are their segments (see
    _segments, plus the right-hand side bounds) and ``joint_O`` holds the
    per-joint cluster counts.
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
    stream = cp.cuda.get_current_stream().ptr
    Z = _correction_output(X, output)
    # Each chunk continues the sums of the earlier clusters in Z and the last
    # one subtracts them from X, so no second buffer is needed.
    for i, cluster_start in enumerate(starts):
        chunk = slice(cluster_start, cluster_start + chunk_size)
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
            marginal_joint_offsets=marginal_joint_offsets,
            marginal_joint_indices=marginal_joint_indices,
            lambda_kb=lambda_chunk,
            active_mask=(lambda_chunk < X.dtype.type(_SUPPRESS_PENALTY)).view(cp.uint8),
            n_batches=n_batches,
            seg_start=seg_start,
            seg_group=seg_group,
            bounds=bounds,
            gram=gram,
            rhs=rhs,
            joint_rhs=joint_rhs,
            scales=cp.empty(n_joint_categories, dtype=cp.float64),
            acc=cp.empty(
                (n_joint_categories, _limbs(X.dtype.itemsize), k, n_pcs), dtype=cp.int64
            ),
            stream=stream,
            **(sums or {}),
        )
        W_all = _solve_spd_batched(gram, rhs)
        del gram, rhs
        last = i == len(starts) - 1
        _correction_batched_cuda.apply_multi(
            X,
            R=R_chunk,
            W_all=W_all,
            joint_cats=joint_cats,
            n_batches=n_batches,
            seg_start=seg_start,
            seg_group=seg_group,
            normalize=normalize and last,
            W_joint=joint_rhs,
            Z=Z,
            resume=i > 0,
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
        + 8 * _limbs(itemsize) * n_joint_categories * n_pcs  # exact joint sums
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
    segments: tuple,
    output: cp.ndarray | None = None,
    normalize: bool = False,
    sums: dict | None = None,
) -> cp.ndarray:
    """
    Batched correction of cells sorted by batch, all clusters at once:
    right-hand sides and the correction over ``segments`` (see _segments,
    plus the right-hand side bounds).
    """
    seg_start, seg_group, bounds = segments
    n_pcs, n_clusters, nb1 = X.shape[1], R.shape[1], len(bounds) + 1
    dtype = X.dtype
    Z = _correction_output(X, output)
    _correction_batched_cuda.correction_batched(
        X,
        **_assignment_args(R, X.dtype),
        O=O,
        lambda_kb=lambda_kb,
        seg_start=seg_start,
        seg_group=seg_group,
        bounds=bounds,
        Z=Z,
        inv_mats=cp.empty((n_clusters, nb1, nb1), dtype=dtype),
        Phi_t_diag_R_X_all=cp.empty((n_clusters, nb1, n_pcs), dtype=dtype),
        W_all=cp.empty((n_clusters, nb1, n_pcs), dtype=dtype),
        g_factor=cp.empty((n_clusters, nb1 - 1), dtype=dtype),
        g_P_row0=cp.empty((n_clusters, nb1 - 1), dtype=dtype),
        scales=cp.empty(nb1 - 1, dtype=cp.float64),
        rhs_acc=cp.empty(
            (nb1 - 1, _limbs(dtype.itemsize), n_clusters, n_pcs), dtype=cp.int64
        ),
        normalize=normalize,
        stream=cp.cuda.get_current_stream().ptr,
        handle=cp.cuda.device.get_cublas_handle(),
        **(sums or {}),
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
