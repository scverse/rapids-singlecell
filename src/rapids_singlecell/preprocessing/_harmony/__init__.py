from __future__ import annotations

import functools
import math
import operator
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import contextmanager
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _harmony_clustering_cuda as _clustering_cuda
from rapids_singlecell._cuda import (
    _harmony_correction_batched_cuda as _correction_batched_cuda,
)
from rapids_singlecell._cuda import _harmony_correction_cuda as _correction_cuda
from rapids_singlecell._utils import _create_category_index_mapping
from rapids_singlecell._utils._multi_gpu import _copy_to_device
from rapids_singlecell._utils._random import _seed_from_rng

from ._chunks_to_device import _shard_rows, _upload_sorted, _write_shards
from ._helper import (
    _factorize_joint_codes,
    _get_batch_codes,
    _get_theta_array,
    _normalize_cp,
    _stratified_sample_indices,
    _validate_output_buffer,
)
from ._kmeans import _kmeans
from ._multi_gpu import _comm, _gather_rows

if TYPE_CHECKING:
    from collections.abc import Iterator

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
_COMM_VALUES = 1 << 22  # int64 values per exchange; larger sums take several
_BLOCK_UNITS = 384  # units of the block draw (see _draw_blocks)
_SEGMENT = 2048  # cells per segment of the exact sums (see _segments)

# Each flavor inherits the stopping rules of the implementation it reproduces:
# harmony2 follows harmonypy 2.0.0 / harmony 2.0.5 (R), harmony1 follows
# harmony-pytorch. Keys are (max_iter_clustering, tol_clustering, tol_harmony).
_STOPPING_HARMONY2 = (4, 1e-3, 1e-2)
_STOPPING_HARMONY1 = (200, 1e-5, 1e-4)


def _touch_pages(out: np.ndarray) -> None:
    """Map the pages of ``out``: one write per 4 KiB page (the kernel zero-fills new pages), a strided array whole."""
    if out.flags.c_contiguous or out.flags.f_contiguous:
        np.ravel(out, order="K").view(np.uint8)[::4096] = 0
    else:
        out.fill(0)


@contextmanager
def _prefaulted(out: np.ndarray | None, Z) -> Iterator[threading.Thread | None]:
    """Map the pages of the host output ``out`` in a thread while the block runs.

    Copying into mapped pages is about twice as fast as into a fresh allocation. The thread is joined when the block
    exits, so it never writes ``out`` afterwards. An output sharing memory with the host input is left alone: the
    input is read later.
    """
    if out is None or (isinstance(Z, np.ndarray) and np.may_share_memory(out, Z)):
        yield None
        return
    prefault = threading.Thread(target=_touch_pages, args=(out,), daemon=True)
    prefault.start()
    try:
        yield prefault
    finally:
        prefault.join()


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
    dtype: np.dtype | type | None = None,
    devices: list[int] | None = None,
    out: np.ndarray | None = None,
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

    dtype
        float32 or float64 for the computation; ``None`` keeps the dtype of ``Z``.

    devices
        GPUs to split the cells across (default: the current one). Several
        GPUs give the bits of a single-GPU run; they need the batched
        correction and the host output ``out``. A small input uses fewer
        of them, with a warning.

    out
        Host array receiving the corrected embedding (then returned). Its
        pages are mapped in a thread while the GPUs work.

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
    dtype = np.dtype(Z.dtype if dtype is None else dtype)
    if out is not None:
        if not isinstance(out, np.ndarray) or out.shape != Z.shape or out.dtype != dtype:
            raise ValueError(f"out must be a NumPy array of shape {Z.shape} and dtype {dtype}.")
        if not out.flags.writeable:
            raise ValueError("out must be writeable.")
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
    if n_clusters is None:
        n_clusters = int(min(100, n_cells / 30))
        n_clusters = max(n_clusters, 2)
    n_clusters = operator.index(n_clusters)
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")
    if correction_method not in {None, "fast", "original", "batched"}:
        raise ValueError("correction_method must be 'fast' or 'batched'.")
    if dynamic_lambda:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError(f"alpha must be a finite positive number when dynamic_lambda=True, got {alpha}.")
        if batch_prune_threshold is not None and not (0 <= batch_prune_threshold <= 1):
            raise ValueError(f"batch_prune_threshold must be in [0, 1] or None, got {batch_prune_threshold}.")
    elif not np.isfinite(ridge_lambda) or ridge_lambda <= 0:
        raise ValueError(
            f"ridge_lambda must be a finite positive number when dynamic_lambda=False, got {ridge_lambda}."
        )
    with _prefaulted(out, Z) as prefault:
        rng = np.random.default_rng(rng)
        # Process batch information
        n_levels, N_b, joint, offsets, order = _sort_by_batch(
            batch_mat, batch_key, n_cells=n_cells, dtype=dtype, rng=rng
        )
        n_covariates, n_batches = n_levels.size, int(n_levels.sum())
        n_groups = len(offsets) - 1  # batches, or joint categories with several keys

        theta_array = _get_theta_array(theta, n_levels, dtype)
        if tau > 0:
            theta_array = theta_array * (1 - cp.exp(-N_b / (n_clusters * tau)) ** 2)
        theta_array = cp.ascontiguousarray(theta_array.ravel())
        correction_method = _choose_method(Z, n_levels, correction_method, dtype)

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
                stacklevel=4,
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
                stacklevel=4,
            )
            devices = devices[:1]
        # Shards: contiguous runs of the sorted cells, one per GPU, cut at
        # multiples of the block draw's units (which also cut the segments), so
        # that every GPU draws the blocks and sums the segments of a single-GPU run.
        chunk, unit = _block_layout(n_cells, n_blocks, shuffle_chunk_size)
        if len(devices) > n_cells // unit:
            warnings.warn(
                f"Using {n_cells // unit} of {len(devices)} GPUs: {n_cells} cells split into at most that many shards.",
                UserWarning,
                stacklevel=4,
            )
            devices = devices[: max(1, n_cells // unit)]
        if len(devices) > 1 and out is None:
            raise ValueError("Several GPUs write the result into a host array `out`.")
        n_ranks, n_pcs = len(devices), Z.shape[1]
        bounds = [round(n_cells * r / n_ranks / unit) * unit for r in range(n_ranks)] + [n_cells]
        # Each GPU's sorted groups, allocated before the upload (faster with a
        # non-pooled allocator)
        shard_groups = [
            _sorted_groups(offsets, lo, hi, device)
            for device, lo, hi in zip(devices, bounds[:-1], bounds[1:], strict=True)
        ]
        # Exact int64 sums of block counts, right-hand sides and centroids.
        capacity = n_clusters * max(
            (n_blocks if keep else 1) * n_batches,
            _limbs(dtype.itemsize) * n_groups * n_pcs,
            3 * n_pcs,
        )

        # Every GPU writes its rows of the output (see _write_shards).
        shard_rows = _shard_rows(order, bounds, devices)
        shards = _upload_sorted(Z, order, bounds, devices, shard_rows=shard_rows, dtype=dtype)
        del Z, order
        # Read after the upload, whose host staging overlaps the sort.
        if not bool(cp.isfinite(theta_array).all()):
            raise ValueError(f"theta must be finite, got {theta}.")
        rhs_bounds = _rhs_bounds(shards, offsets, bounds)

        # Initialize algorithm
        shards = [
            _shard(
                X,
                n_clusters=n_clusters,
                n_batches=n_batches,
                n_groups=n_batches if n_covariates == 1 else 1,
                n_blocks=n_blocks,
                n_stored=n_blocks if keep else min(n_blocks, 3),
                n_total=n_cells,
            )
            for X in shards
        ]
        Y_norm = _kmeans_centroids(
            [Z_norm for _, Z_norm, _ in shards],
            bounds=bounds,
            offsets=offsets,
            n_clusters=n_clusters,
            rng=rng,
        )
        # Every GPU's copies of the shared arrays, made before the GPU threads: a
        # thread's copy from another GPU would wait for that GPU's queued work.
        shared = {**joint, "N_b": N_b, "theta": theta_array, "Y_norm": Y_norm}
        shared["rhs_bounds"] = rhs_bounds
        local = [{k: _copy_to_device(a, d) for k, a in shared.items()} for d in devices]
        del Y_norm, rhs_bounds, shared  # each GPU's copies: until its workspace has them

        def run(rank: int, comm: int) -> tuple[cp.ndarray, bool, int]:
            # Harmony on one GPU's shard; several GPUs run it at once, ``comm``
            # summing over all cells exactly, so each gives the bits of one GPU.
            X, Z_norm, workspace = shards[rank]
            lo, hi = bounds[rank], bounds[rank + 1]
            groups, shard_groups[rank] = shard_groups[rank], None
            mine, local[rank] = local[rank], None
            joint_args = {name: mine[name] for name in joint}
            N_b_local, sums = mine["N_b"], {"comm": comm, "rank": rank}
            seg_start, seg_group = _segments(offsets, unit, lo, hi)
            segments = (seg_start, seg_group, mine["rhs_bounds"])
            if n_covariates > 1:  # centroid sums need no group cuts
                seg_start = _segments(offsets[[0, -1]], unit, lo, hi)[0]
            Pr_b = (N_b_local / n_cells).astype(dtype)
            # The update blocks are drawn once per call, before the large arrays
            # exist; with several keys the assignment tiles take any cells.
            _draw_blocks(
                groups if n_covariates == 1 else cp.zeros(hi - lo, dtype=cp.int32),
                n_groups=n_batches if n_covariates == 1 else 1,
                n_blocks=n_blocks,
                seed=kernel_seed,
                chunk=chunk,
                unit=unit,
                first=lo,
                workspace=workspace,
            )
            cats = cat_offsets = None
            if correction_method == "fast":  # the batch of every cell (one GPU)
                cats, cat_offsets = groups, cp.asarray(offsets)
            # joint runs, or the groups of recounted block counts
            joints = groups if n_covariates > 1 or not keep else None
            del groups

            workspace["Y_norm"][...] = mine.pop("Y_norm")
            R = cp.empty((hi - lo, n_clusters), dtype=cp.uint16 if bfloat16 else dtype)
            O = cp.empty((n_batches, n_clusters), dtype=dtype)
            E = cp.empty_like(O)
            # Clustering updates of R, O and E in place: the update blocks of cells
            # are reassigned against the counts of all other cells until
            # convergence or max_iter passes (the workspace holds its buffers).
            clustering = functools.partial(
                _clustering_cuda.clustering_loop,
                Z_norm,
                E=E,
                O=O,
                Pr_b=Pr_b,
                theta=mine["theta"],
                seg_start=seg_start,
                joint_cats=joint_args.get("joint_cats"),
                joints=joints,
                **workspace,
                force_blocks=_FORCE_CLUSTER_BLOCKS,
                n_cells=hi - lo,
                n_total=n_cells,
                n_pcs=n_pcs,
                n_clusters=n_clusters,
                n_batches=n_batches,
                n_covariates=n_covariates,
                n_first=int(n_levels[0]),
                n_blocks=n_blocks,
                sigma=float(sigma),
                stabilized=stabilized_penalty,
                stream=cp.cuda.get_current_stream().ptr,
                comm=comm,
                rank=rank,
            )
            # Every cell assigned to the k-means centroids without penalty
            objectives = [clustering(**_assignment_args(R, dtype), max_iter=0, initialize=True)]

            # Main harmony iterations
            for i in range(max_iter_harmony):
                # Clustering step
                objectives.append(
                    clustering(
                        **_assignment_args(R, dtype),
                        tol=float(tol_clustering),
                        max_iter=max_iter_clustering,
                        seed=(kernel_seed + i * 1000003) & 0xFFFFFFFF,
                    )
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
                # Check for convergence: it depends only on the clustering
                # objective, so the correction writes the normalized embedding
                # for the next clustering (into Z_norm) unless this is the last.
                is_converged = _is_convergent_harmony(objectives, tol=tol_harmony)
                normalize = not is_converged and i + 1 < max_iter_harmony
                # Correction step
                if n_covariates > 1:
                    Z_hat = _correction_multi(
                        X,
                        R,
                        lambda_kb=lambda_kb,
                        n_first=int(n_levels[0]),
                        **joint_args,
                        segments=segments,
                        output=Z_norm,
                        normalize=normalize,
                        n_total=n_cells,
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
                        cat_offsets=cat_offsets,
                        output=Z_norm,
                        normalize=normalize,
                        segments=segments,
                        sums=sums,
                    )
                if is_converged:
                    break
            return Z_hat, is_converged, i

        shards, is_converged, i = _on_gpus(  # rebinding frees the inputs before the output
            run, devices, offsets=offsets, bounds=bounds, capacity=capacity
        )
        if is_converged and verbose:
            print(f"Harmony converged in {i + 1} iterations")
        if not is_converged:
            warnings.warn(
                "Harmony did not converge. Consider increasing the number of iterations",
                stacklevel=4,
            )
        if out is None:  # one GPU: the result in input order, on its GPU
            with cp.cuda.Device(devices[0]):
                return shards[0][shard_rows[1][0]]
        _write_shards(shards, shard_rows, devices, out, prefault)
        return out


def _sort_by_batch(
    batch_mat: pd.DataFrame,
    batch_key: str | list[str],
    *,
    n_cells: int,
    dtype: np.dtype,
    rng: np.random.Generator,
) -> tuple[np.ndarray, cp.ndarray, dict, np.ndarray, cp.ndarray]:
    """Sort the cells by group (batch, or joint category with several keys).

    Returns the levels of each key, the cells per level, the joint categories with several keys (else empty), and
    the offsets (host) and order of the sorted cells.
    """
    batch_codes, n_levels = _get_batch_codes(batch_mat, batch_key)
    n_covariates, n_batches = int(n_levels.size), int(n_levels.sum())
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
    N_b = cp.bincount(batch_codes.ravel(), minlength=n_batches)
    groups, counts, joint = batch_codes[:, 0], N_b, {}
    if n_covariates > 1:
        joint_cats, groups = _factorize_joint_codes(batch_codes, n_levels)
        if joint_cats.size > 2**31 - 2**24:  # int32 (category, key) positions
            raise ValueError(
                f"harmonize supports at most {2**31 - 2**24} joint categories x keys, got {joint_cats.size}."
            )
        counts = cp.zeros(joint_cats.shape[0], dtype=cp.int32)
        cp.add.at(counts, groups, 1)
        marginal = _create_category_index_mapping(joint_cats.ravel(), n_batches)
        joint = {
            "joint_cats": joint_cats,
            "marginal_joint_offsets": marginal[0],
            "marginal_joint_indices": (marginal[1] // n_covariates).astype(cp.int32),
        }
    # A seeded random order within each group, so the runs of neighbouring
    # cells shuffled together are random cells of a group. The counts are read
    # (waiting for the GPU) before the sort is queued.
    offsets = np.zeros(counts.size + 1, dtype=np.int32)
    np.cumsum(counts.get(), out=offsets[1:])
    keys = cp.random.default_rng(_seed_from_rng(rng, allow_none=False)).random(n_cells)
    order = cp.argsort(groups.astype(cp.float64) + keys)
    return n_levels, N_b.astype(dtype), joint, offsets, order


def _choose_method(
    Z: cp.ndarray | np.ndarray, n_levels: np.ndarray, correction_method: str | None, dtype: np.dtype
) -> str:
    """The correction method (with a warning where the choice is replaced)."""
    n_covariates, n_batches = n_levels.size, int(n_levels.sum())
    if correction_method == "original":
        correction_method = "fast" if n_covariates == 1 else "batched"
        warnings.warn(
            "correction_method='original' is deprecated and will be removed in "
            "a future release. "
            + (
                "Use correction_method='fast' instead; it computes the same "
                "correction with the optimized bounded-memory implementation."
                if n_covariates == 1
                else "With multiple batch keys, omit correction_method or use "
                "correction_method='batched' for the exact general-design solve."
            ),
            FutureWarning,
            stacklevel=5,
        )
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
        fit = _batched_cluster_chunk(Z.shape[1], n_batches, dtype.itemsize)
        correction_method = "batched" if fit else "fast"
    return correction_method


def _shard(X: cp.ndarray, **sizes) -> tuple[cp.ndarray, cp.ndarray, dict]:
    """One GPU's sorted cells, normalized, and their clustering buffers.

    The k-means' many small temporaries then allocate faster.
    """
    with cp.cuda.Device(X.device.id):
        Z_norm = _normalize_cp(X)
        workspace = _allocate_clustering_workspace(X.shape[0], n_pcs=X.shape[1], dtype=X.dtype, **sizes)
        return X, Z_norm, workspace


def _on_gpus(
    run, devices: list[int], *, offsets: np.ndarray, bounds: list[int], capacity: int
) -> tuple[list[cp.ndarray], bool, int]:
    """``run(rank, comm)`` under every GPU: one in this thread, several in a thread each.

    They exchange exact sums of up to ``capacity`` int64 values over ``comm``. The first error on any GPU is raised
    once every thread stopped.
    """
    if len(devices) == 1:
        with cp.cuda.Device(devices[0]):
            Z_hat, is_converged, i = run(0, 0)
        return [Z_hat], is_converged, i
    # Per-group sums move only each GPU's groups.
    first = np.searchsorted(offsets, bounds[:-1], side="right") - 1
    end = np.searchsorted(offsets, np.subtract(bounds[1:], 1), side="right")
    comm, comm_buffers = _comm(  # held until the GPU threads stopped
        devices,
        min(capacity, _COMM_VALUES),
        groups=np.column_stack([first, end]).ravel().tolist(),
    )
    errors = []  # the first is the cause; the other GPUs then fail on the abort

    def guarded(rank: int):
        try:
            with cp.cuda.Device(devices[rank]):
                result = run(rank, comm.handle)
                cp.cuda.get_current_stream().synchronize()  # read by other threads
                return result
        except BaseException as error:
            errors.append(error)
            comm.abort()  # release the other GPUs' threads
            raise

    # The GPU threads read this thread's arrays on their own streams.
    for device in {cp.cuda.Device().id, *devices}:
        with cp.cuda.Device(device):
            cp.cuda.get_current_stream().synchronize()
    pool, done, interrupts = ThreadPoolExecutor(len(devices)), False, 0
    try:
        futures = [pool.submit(guarded, rank) for rank in range(len(devices))]
        wait(futures)
        done = not errors
    except KeyboardInterrupt:
        interrupts += 1
        raise
    finally:
        while True:  # the GPU threads stop first; a second Ctrl-C gives up on them
            try:
                if not done:  # e.g. Ctrl-C: release the other GPUs' threads
                    comm.abort()
                pool.shutdown()  # waits for the GPU threads
                if not done:  # exchanges may be in flight: wait, without the GIL
                    comm.synchronize()
                break
            except KeyboardInterrupt:
                interrupts += 1
                if interrupts > 1:  # left behind, with the buffers they may still use
                    threading.Thread(target=lambda keep=comm_buffers: pool.shutdown(), daemon=True).start()
                    raise
        del comm_buffers  # and comm (also guarded's): no traceback holds them
        comm = None
    if interrupts and not errors:  # in a call that had not failed
        raise KeyboardInterrupt
    if errors:
        try:
            raise errors[0]
        finally:  # cycle error -> traceback -> errors/futures holds GPU arrays until gc
            errors.clear()
            futures.clear()
    results = [future.result() for future in futures]
    return [result[0] for result in results], *results[0][1:]


def _assignment_args(R: cp.ndarray, dtype) -> dict:
    """Pass R to the CUDA modules.

    bfloat16 R (stored as uint16) travels as ``R_bf16`` next to a placeholder ``R``.
    """
    if R.dtype == cp.uint16:
        return {"R": cp.empty((1, R.shape[1]), dtype=dtype), "R_bf16": R}
    return {"R": R}


def _rhs_bounds(parts: list[cp.ndarray], offsets: np.ndarray, bounds: list[int]) -> cp.ndarray:
    """Bounds of the exact right-hand side sums (segment_gemm.cuh), on the current GPU.

    The exponents e (int16) of n_g max |X_gd| < 2^e for each group g of the cells (sorted by group, ``offsets`` on
    the host) and PC d, so no other group or PC coarsens a sum. ``parts`` hold the cells
    ``bounds[r] .. bounds[r + 1]``; each GPU takes the maxima of its own groups (a group spanning GPUs from each),
    which are exact, so every split gives the same bounds.
    """
    maxima = cp.zeros((len(offsets) - 1, parts[0].shape[1]), dtype=parts[0].dtype)
    for part, lo, hi in zip(parts, bounds[:-1], bounds[1:], strict=True):
        # the groups of the cells lo .. hi - 1
        first, end = np.searchsorted(offsets, [lo, hi - 1], side="right") - [1, 0]
        with cp.cuda.Device(part.device.id):
            seg_start, seg_group = _segments(offsets, _SEGMENT, lo, hi)
            seg_group -= int(first)  # rows of m
            m = cp.empty((end - first, part.shape[1]), dtype=part.dtype)
            _correction_batched_cuda.group_absmax(
                part,
                seg_start=seg_start,
                seg_group=seg_group,
                out=m,
                stream=cp.cuda.get_current_stream().ptr,
            )
        rows = maxima[first:end]
        cp.maximum(rows, _copy_to_device(m, maxima.device.id), out=rows)
    if not bool(cp.isfinite(maxima).all()):
        raise ValueError(
            "Input data contains NaN or infinite values. Please handle these before running harmony_integrate."
        )
    n = cp.asarray(np.diff(offsets), dtype=cp.float64)[:, None]
    rhs = n * maxima  # beyond float64 near 1e308: n < 2^a, max < 2^b
    exp = cp.frexp(cp.where(rhs > 0, rhs, 1.0))[1]
    exp = cp.where(cp.isinf(rhs), cp.frexp(n)[1] + cp.frexp(maxima)[1], exp)
    return exp.astype(cp.int16)


def _segments(
    offsets: np.ndarray, unit: int = _SEGMENT, lo: int = 0, hi: int | None = None
) -> tuple[cp.ndarray, cp.ndarray]:
    """Segments of the exact sums over cells sorted by group (``offsets`` on the host).

    Runs of one group cut at multiples of _SEGMENT and of ``unit`` (the block draw's units, where several GPUs split
    the cells). Start rows (and the end) and the group of each segment of the cells ``lo .. hi`` (multiples of
    ``unit``), in rows from ``lo``.
    """
    hi = offsets[-1] if hi is None else hi
    grid = np.arange(-(-lo // _SEGMENT) * _SEGMENT, hi, _SEGMENT)
    start = np.union1d(np.clip(offsets, lo, hi), np.arange(lo, hi, unit))
    start = np.union1d(start, grid)
    group = np.searchsorted(offsets, start[:-1], side="right") - 1
    return cp.asarray(start - lo, dtype=cp.int32), cp.asarray(group, dtype=cp.int32)


def _sorted_groups(offsets: np.ndarray, lo: int, hi: int, device: int) -> cp.ndarray:
    """The groups of the sorted cells ``lo .. hi`` (``offsets`` on the host) on ``device``.

    The sum of a 1 at the first cell of every later group.
    """
    first = int(np.searchsorted(offsets, lo, side="right")) - 1
    starts = offsets[first + 1 :]
    with cp.cuda.Device(device):
        groups = cp.zeros(hi - lo, dtype=cp.int32)
        cp.add.at(groups, cp.asarray(starts[starts < hi] - lo), 1)
        cp.cumsum(groups, dtype=cp.int32, out=groups)
        groups += first
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
    """Draw the update blocks of a harmonize call into the workspace.

    Writes ``idx_list``, ``block_cat_offsets`` and the host ``block_starts`` (sort scratch only lives here).
    ``groups`` are the sorted cells ``first ..`` of a call (``chunk`` and ``unit`` from _block_layout); a cell's
    block depends only on its position among all cells.
    """
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
    offsets: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> cp.ndarray:
    """Normalized k-means centroids of a stratified sample of the normalized sorted cells.

    The sorted cells are the shards ``parts``; the centroids are on the current GPU.
    """
    # Fit only the initialization sample, retaining every observed stratum.
    n_cells = bounds[-1]
    n_init_cells = min(n_cells, _KMEANS_INIT_CELLS_PER_CLUSTER * n_clusters)
    if n_init_cells < len(offsets) - 1:
        n_init_cells = max(n_init_cells, int(np.count_nonzero(np.diff(offsets))))
    sample = None  # all cells: on one GPU the shard itself, which _kmeans only reads
    if n_init_cells < n_cells:
        sample = _stratified_sample_indices(offsets, n_init_cells, rng)
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
    n_total: int,
    dtype: cp.dtype,
) -> dict:
    """Buffers of the clustering loop of ``n_cells`` of the ``n_total`` cells (all GPUs).

    The tiles' groups are the batches, or one group for several keys; counts are kept per batch level.
    """
    workspace = {
        "Y_norm": cp.empty((n_clusters, n_pcs), dtype=dtype),
        # Exact centroid sums (at most 3 limbs) and the exponents of their
        # bound, n_total per PC (|Z_norm|, R <= 1; see segment_gemm.cuh).
        "y_acc": cp.empty((3, n_clusters, n_pcs), dtype=cp.int64),
        "y_bound": cp.full(n_pcs, np.frexp(n_total)[1], dtype=cp.int16),
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
) -> cp.ndarray:
    """Apply the exact general-design correction in bounded cluster chunks.

    The cells are sorted by joint category (``segments`` as for _correction_batched), with the joint counts of R over
    the ``n_total`` cells of all GPUs.
    """
    seg_start, seg_group, bounds = segments
    Z = _correction_output(X, output)
    _correction_batched_cuda.correction_multi(
        X,
        R=R,
        lambda_kb=lambda_kb,
        joint_cats=joint_cats,
        marginal_joint_offsets=marginal_joint_offsets,
        marginal_joint_indices=marginal_joint_indices,
        n_first=n_first,
        seg_start=seg_start,
        seg_group=seg_group,
        bounds=bounds,
        Z=Z,
        normalize=normalize,
        chunk=_multi_correction_cluster_chunk_size(
            n_pcs=X.shape[1],
            n_clusters=R.shape[1],
            n_batches=lambda_kb.shape[0],
            n_joint_categories=joint_cats.shape[0],
            itemsize=X.dtype.itemsize,
        ),
        suppress=_SUPPRESS_PENALTY,
        n_total=n_total,
        stream=cp.cuda.get_current_stream().ptr,
        blas=cp.cuda.device.get_cublas_handle(),
        solver=cp.cuda.device.get_cusolver_handle(),
        **({} if sums is None else sums),
    )
    return Z


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
        **({} if sums is None else sums),
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
