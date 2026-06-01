"""GPU-accelerated Wasserstein distance via batched Sinkhorn.

Computes entropy-regularized 2-Wasserstein between groups of cells. Pairs are
split across GPUs and solved in memory-bounded batches; each pair is oriented
with its larger group as the columns (the OT cost is symmetric, so this is
free) so the cooperative-reduction Sinkhorn kernels stay well parallelized even
for very large reference groups. See ``_sinkhorn`` and the ``_sinkhorn_cuda``
kernels.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import pandas as pd

from rapids_singlecell._cuda import _sinkhorn_cuda as _sk
from rapids_singlecell.squidpy_gpu._utils import _assert_categorical_obs

from ._base_metric import BaseMetric, parse_device_ids
from ._sinkhorn import (
    DEFAULT_CHECK_EVERY,
    DEFAULT_EPSILON_SCALE,
    DEFAULT_MAX_ITER,
    DEFAULT_RELAXATION,
    DEFAULT_TOL,
    batched_sinkhorn,
    finalize,
    make_state,
    run_async,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

# Fraction of free GPU memory to use for one Sinkhorn batch.
MEMORY_BUDGET_FRACTION = 0.2
# Bytes per cost-matrix cell: the padded cost tensor plus the transient gather
# buffers used to build it (the fused distance kernel only needs the cost tensor
# plus the small gathered-index arrays).
BYTES_PER_PAIR_CELL_OVERHEAD = 2
MIN_BATCH = 1
MAX_BATCH = 4096
# Over-relaxation bounds: omega = 1 is plain Sinkhorn, (1, 2) is the classic SOR
# convergent range. Above 2 the iteration is unstable for any problem.
MIN_RELAXATION = 1.0
MAX_RELAXATION = 2.0


def _pair_batch_size(
    max_n: int,
    max_m: int,
    itemsize: int,
) -> int:
    """Pick a batch size so the working tensors fit in the memory budget."""
    free, _ = cp.cuda.runtime.memGetInfo()
    budget = int(free * MEMORY_BUDGET_FRACTION)
    per_pair_bytes = max(max_n * max_m, 1) * itemsize * BYTES_PER_PAIR_CELL_OVERHEAD
    if per_pair_bytes == 0:
        return MAX_BATCH
    batch = max(MIN_BATCH, budget // max(per_pair_bytes, 1))
    return int(min(batch, MAX_BATCH))


def _split_positions_by_work(
    n_left_host: np.ndarray,
    n_right_host: np.ndarray,
    n_devices: int,
) -> list[int]:
    """Partition pair positions ``[0, n)`` into ``n_devices`` contiguous,
    work-balanced segments and return the ``n_devices + 1`` boundaries.

    Per-pair work is proportional to ``n_left * n_right`` (the cost-matrix
    size). Splitting by cumulative work keeps each device's solve time roughly
    equal even when groups differ a lot in size. Done entirely on the host so
    the multi-GPU launch phase never has to sync a device to plan its batches.
    """
    n = len(n_left_host)
    if n_devices <= 1 or n == 0:
        return [0, n]
    work = n_left_host.astype(np.int64) * n_right_host.astype(np.int64)
    cum = np.cumsum(work)
    total = int(cum[-1])
    bounds = [0]
    for k in range(1, n_devices):
        target = total * k / n_devices
        idx = int(np.searchsorted(cum, target, side="right"))
        bounds.append(min(max(idx, bounds[-1]), n))
    bounds.append(n)
    return bounds


def _plan_batches(
    start: int,
    stop: int,
    n_left_host: np.ndarray,
    n_right_host: np.ndarray,
    itemsize: int,
) -> list[tuple[int, int, int, int]]:
    """Split pair positions ``[start, stop)`` into memory-bounded batches.

    Returns ``(batch_start, batch_stop, max_n, max_m)`` per batch, all computed
    from the host size arrays so planning never synchronizes a device.
    """
    if stop <= start:
        return []
    batch = _pair_batch_size(
        int(n_left_host[start:stop].max()),
        int(n_right_host[start:stop].max()),
        itemsize,
    )
    plans = []
    for bstart in range(start, stop, batch):
        bstop = min(bstart + batch, stop)
        plans.append(
            (
                bstart,
                bstop,
                int(n_left_host[bstart:bstop].max()),
                int(n_right_host[bstart:bstop].max()),
            )
        )
    return plans


def _build_cost_indices(
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    pair_left: cp.ndarray,
    pair_right: cp.ndarray,
    *,
    max_n: int,
    max_m: int,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray, cp.ndarray]:
    """Per-pair gather indices and padding masks for the cost build.

    This is the cheap, allocation-y host-stream part of the build, kept separate
    from the (heavy) ``_sk.build_cost`` kernel so several devices' cost kernels
    can be launched back-to-back and overlap — interleaving the index
    ``cudaMalloc``s with the kernels serializes them (cudaMalloc takes a
    process-wide driver lock).

    ``max_n`` / ``max_m`` are supplied by the caller (host-computed) so this
    never syncs a device.

    Returns
    -------
    cidx_l, cidx_r
        ``(B, max_n)`` / ``(B, max_m)`` int32 cell-row indices into the
        embedding; padded slots use clamped (in-bounds) indices.
    mask_a, mask_b
        ``(B, max_n)`` / ``(B, max_m)`` boolean real-point masks.
    """
    group_sizes = (cat_offsets[1:] - cat_offsets[:-1]).astype(cp.int32)
    n_left = group_sizes[pair_left]
    n_right = group_sizes[pair_right]
    offs_left = cat_offsets[pair_left]
    offs_right = cat_offsets[pair_right]

    row_range_n = cp.arange(max_n, dtype=cp.int32)[None, :]
    row_range_m = cp.arange(max_m, dtype=cp.int32)[None, :]

    mask_a = cp.ascontiguousarray(row_range_n < n_left[:, None])
    mask_b = cp.ascontiguousarray(row_range_m < n_right[:, None])

    # Clamp local indices to a valid in-group position for padded slots; the
    # mask filters them out later, but indexing must stay in-bounds.
    local_n = cp.where(mask_a, row_range_n, cp.int32(0))
    local_m = cp.where(mask_b, row_range_m, cp.int32(0))

    flat_n = (offs_left[:, None] + local_n).astype(cp.intp)
    flat_m = (offs_right[:, None] + local_m).astype(cp.intp)

    cidx_l = cp.ascontiguousarray(cell_indices[flat_n].astype(cp.int32))
    cidx_r = cp.ascontiguousarray(cell_indices[flat_m].astype(cp.int32))
    return cidx_l, cidx_r, mask_a, mask_b


class WassersteinMetric(BaseMetric):
    """GPU-accelerated 2-Wasserstein distance (entropic, Sinkhorn).

    Returns OTT-JAX's ``reg_ot_cost`` value: the regularized OT objective
    at convergence with uniform marginals and squared-Euclidean cost.

    The Sinkhorn solver configuration (auto epsilon = ``0.05 * std(C)``,
    ``max_iter`` 2000, ``tol`` 1e-4) is fixed to the OTT-JAX / pertpy defaults
    and is not user-tunable, matching upstream pertpy's zero-config Wasserstein.
    The one exception is ``relaxation`` (see below), which has no pertpy
    equivalent.

    Parameters
    ----------
    layer_key
        Key in ``adata.layers`` for cell data. Mutually exclusive with
        ``obsm_key``.
    obsm_key
        Key in ``adata.obsm`` for embeddings (default: ``'X_pca'``).
    relaxation
        Over-relaxation factor for the Sinkhorn updates. ``1.0`` (default) is
        plain Sinkhorn and matches OTT-JAX. Values in ``(1, 2)`` use successive
        over-relaxation to cut the iteration count (roughly halved near
        ``1.5``), trading a small change in the converged value (~1e-5 relative)
        for speed. Too large a value (problem-dependent, often around ``1.7``)
        makes the iteration diverge — it will then hit the iteration cap and
        warn. Leave at ``1.0`` for reference-exact results.

    References
    ----------
    Cuturi, M. (2013). Sinkhorn distances: Lightspeed computation of
    optimal transport. NeurIPS.
    """

    supports_multi_gpu: bool = True

    def __init__(
        self,
        *,
        layer_key: str | None = None,
        obsm_key: str | None = "X_pca",
        relaxation: float = DEFAULT_RELAXATION,
    ):
        super().__init__(layer_key=layer_key, obsm_key=obsm_key)
        if not MIN_RELAXATION <= relaxation < MAX_RELAXATION:
            raise ValueError(
                f"relaxation must be in [{MIN_RELAXATION}, {MAX_RELAXATION}); "
                f"got {relaxation}."
            )
        # Solver config fixed to OTT-JAX / pertpy defaults (not user-tunable);
        # epsilon is always auto (epsilon_scale * std(C) per pair).
        self.epsilon_scale = DEFAULT_EPSILON_SCALE
        self.max_iter = DEFAULT_MAX_ITER
        self.tol = DEFAULT_TOL
        self.relaxation = relaxation

    def _solve_pairs(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        pair_left: list[int],
        pair_right: list[int],
        *,
        dtype,
        device_ids: list[int] | None = None,
    ) -> cp.ndarray:
        """Solve all (i, j) Sinkhorn problems, returning a flat ``(n_pairs,)`` array.

        Each pair is oriented so the **larger** group is the columns (M) — valid
        because the OT cost is symmetric, ``W(a, b) = W(b, a)`` — which keeps the
        cooperative ``update_f`` reduction on the big axis and is what makes a
        large reference group (e.g. a 75k control) fast.

        Pairs are split across ``device_ids`` (work-balanced) and each device's
        share is solved in memory-bounded batches. Within a batch-round the
        per-iteration launches are dispatched to every device's stream in turn
        (an async multi-stream work queue), so the GPUs overlap with no host
        threads; only one batch per device is resident at a time.
        """
        if device_ids is None:
            device_ids = [0]
        n_pairs = len(pair_left)
        if n_pairs == 0:
            return cp.zeros(0, dtype=dtype)

        pl_host = np.asarray(pair_left, dtype=np.int32)
        pr_host = np.asarray(pair_right, dtype=np.int32)
        group_sizes = (cat_offsets[1:] - cat_offsets[:-1]).astype(cp.int32)
        # Per-pair group sizes on the host (one sync). All batch planning below
        # is then host-only.
        n_left = group_sizes[cp.asarray(pl_host)].get()
        n_right = group_sizes[cp.asarray(pr_host)].get()
        # Orient larger group as columns (M); rows = smaller group (N).
        swap = n_left > n_right
        rows = np.where(swap, pr_host, pl_host)
        cols = np.where(swap, pl_host, pr_host)
        n_row = np.minimum(n_left, n_right)
        n_col = np.maximum(n_left, n_right)
        itemsize = cp.dtype(dtype).itemsize

        bounds = _split_positions_by_work(n_row, n_col, len(device_ids))
        plans = [
            _plan_batches(bounds[i], bounds[i + 1], n_row, n_col, itemsize)
            for i in range(len(device_ids))
        ]

        out = cp.empty(n_pairs, dtype=dtype)
        home = device_ids[0]

        # Move the shared inputs to each participating device once.
        streams: dict[int, cp.cuda.Stream] = {}
        inputs: dict[int, tuple] = {}
        for di, dev in enumerate(device_ids):
            if not plans[di]:
                continue
            with cp.cuda.Device(dev):
                streams[dev] = cp.cuda.Stream(non_blocking=True)
                with streams[dev]:
                    inputs[dev] = (
                        (embedding, cat_offsets, cell_indices)
                        if dev == home
                        else (
                            cp.asarray(embedding),
                            cp.asarray(cat_offsets),
                            cp.asarray(cell_indices),
                        )
                    )

        # Per-device reusable cost buffers (grow-only): reused across rounds so
        # the per-round cost tensor isn't a fresh synchronous cudaMalloc. Safe
        # because a round's solve finishes before the next round overwrites it.
        cost_bufs: dict[int, cp.ndarray] = {}
        converged = True
        for r in range(max((len(p) for p in plans), default=0)):
            batch = [
                (dev, *plans[di][r])
                for di, dev in enumerate(device_ids)
                if r < len(plans[di])
            ]

            # Phase 1 (index): per-device gather indices + masks. These do the
            # allocations, kept separate from the cost kernel so the kernels can
            # launch back-to-back across devices and overlap.
            preps = []
            for dev, start, stop, max_n, max_m in batch:
                _, off, idx = inputs[dev]
                with cp.cuda.Device(dev), streams[dev]:
                    cidx_l, cidx_r, mask_a, mask_b = _build_cost_indices(
                        off,
                        idx,
                        cp.asarray(rows[start:stop]),
                        cp.asarray(cols[start:stop]),
                        max_n=max_n,
                        max_m=max_m,
                    )
                preps.append(
                    (dev, start, stop, max_n, max_m, cidx_l, cidx_r, mask_a, mask_b)
                )

            # Phase 2 (cost): launch the fused cost kernels back-to-back into the
            # reusable buffers (no cudaMalloc between launches), so the devices'
            # builds overlap.
            units: list[dict] = []
            for dev, start, stop, max_n, max_m, cidx_l, cidx_r, ma, mb in preps:
                emb = inputs[dev][0]
                with cp.cuda.Device(dev), streams[dev]:
                    need = (stop - start) * max_n * max_m
                    buf = cost_bufs.get(dev)
                    if buf is None or buf.size < need:
                        buf = cp.empty(need, dtype=dtype)
                        cost_bufs[dev] = buf
                    cost = buf[:need].reshape(stop - start, max_n, max_m)
                    _sk.build_cost(
                        emb,
                        cidx_l,
                        cidx_r,
                        cost,
                        streams[dev].ptr,
                    )
                units.append(
                    {
                        "dev": dev,
                        "stream": streams[dev],
                        "start": start,
                        "stop": stop,
                        "cost": cost,
                        "mask_a": ma,
                        "mask_b": mb,
                    }
                )

            # Phase 3 (state): per-pair solver state (auto-eps + potentials).
            for u in units:
                with cp.cuda.Device(u["dev"]), u["stream"]:
                    u["state"] = make_state(
                        u["cost"],
                        u["mask_a"],
                        u["mask_b"],
                        epsilon=None,
                        epsilon_scale=self.epsilon_scale,
                    )

            # Phase 4: async multi-stream solve over this round's units, gather.
            run_async(
                units,
                max_iter=self.max_iter,
                tol=self.tol,
                check_every=DEFAULT_CHECK_EVERY,
                omega=self.relaxation,
            )
            for u in units:
                with cp.cuda.Device(u["dev"]), u["stream"]:
                    u["reg"] = finalize(u["state"])
            for u in units:
                with cp.cuda.Device(u["dev"]):
                    u["stream"].synchronize()
            with cp.cuda.Device(home):
                for u in units:
                    out[u["start"] : u["stop"]] = cp.asarray(u["reg"])
            for u in units:
                with cp.cuda.Device(u["dev"]):
                    converged = converged and bool(u["state"]["conv"].all().get())

        if not converged:
            self._warn_not_converged()
        return out

    def _bootstrap_solve(
        self,
        embedding: cp.ndarray,
        cat_offsets: cp.ndarray,
        cell_indices: cp.ndarray,
        pair_left: list[int],
        pair_right: list[int],
        *,
        n_bootstrap: int,
        random_state: int,
        dtype,
        device: int,
    ) -> tuple[cp.ndarray, cp.ndarray]:
        """Bootstrap per-pair mean/variance of W by resampling cells.

        For each pair, draw ``n_bootstrap`` resamples of both groups (cells with
        replacement, to the group's own size) and solve W for each. The resamples
        differ only in their gathered cell indices, so all ``n_pairs *
        n_bootstrap`` problems are solved together by the batched Sinkhorn (in
        memory-bounded chunks) on a single device. All resample indices are drawn
        in one up-front RNG pass so the result is reproducible regardless of how
        the solve is later chunked. Returns per-pair ``(mean, var)`` over the
        resamples (variance is population, ddof=0, matching pertpy).
        """
        if n_bootstrap < 1:
            raise ValueError(f"n_bootstrap must be >= 1, got {n_bootstrap}")
        n_pairs = len(pair_left)
        if n_pairs == 0:
            empty = cp.zeros(0, dtype=dtype)
            return empty, empty
        with cp.cuda.Device(device):
            emb = cp.ascontiguousarray(cp.asarray(embedding))
            offs = cp.asarray(cat_offsets)
            cidx = cp.asarray(cell_indices)
            group_sizes = (offs[1:] - offs[:-1]).astype(cp.int32)
            pl = cp.asarray(np.asarray(pair_left, np.int32))
            pr = cp.asarray(np.asarray(pair_right, np.int32))
            n_l = group_sizes[pl]
            n_r = group_sizes[pr]
            # Orient the larger group as columns (M); rows = smaller (N).
            swap = n_l > n_r
            row_grp = cp.where(swap, pr, pl)
            col_grp = cp.where(swap, pl, pr)
            n_row = cp.minimum(n_l, n_r)
            n_col = cp.maximum(n_l, n_r)
            # Expand each pair into n_bootstrap units (pair-major).
            nrow_u = cp.repeat(n_row, n_bootstrap)
            ncol_u = cp.repeat(n_col, n_bootstrap)
            orow_u = cp.repeat(offs[row_grp], n_bootstrap)
            ocol_u = cp.repeat(offs[col_grp], n_bootstrap)
            n_units = n_pairs * n_bootstrap
            max_n = int(n_row.max().get())
            max_m = int(n_col.max().get())

            # Resample local positions in [0, n) with replacement; one RNG pass
            # over all units keeps the draw independent of the solve chunking.
            rng = cp.random.default_rng(random_state)
            rr = cp.arange(max_n, dtype=cp.int32)[None, :]
            rc = cp.arange(max_m, dtype=cp.int32)[None, :]
            mask_a = cp.ascontiguousarray(rr < nrow_u[:, None])
            mask_b = cp.ascontiguousarray(rc < ncol_u[:, None])
            loc_n = (
                rng.random((n_units, max_n), dtype=dtype) * nrow_u[:, None]
            ).astype(cp.int32)
            loc_m = (
                rng.random((n_units, max_m), dtype=dtype) * ncol_u[:, None]
            ).astype(cp.int32)
            gl = (orow_u[:, None] + cp.where(mask_a, loc_n, cp.int32(0))).astype(
                cp.intp
            )
            gr = (ocol_u[:, None] + cp.where(mask_b, loc_m, cp.int32(0))).astype(
                cp.intp
            )
            cidx_l_all = cidx[gl].astype(cp.int32)
            cidx_r_all = cidx[gr].astype(cp.int32)

            reg = cp.empty(n_units, dtype=dtype)
            itemsize = cp.dtype(dtype).itemsize
            chunk = _pair_batch_size(max_n, max_m, itemsize)
            stream = cp.cuda.get_current_stream()
            converged = True
            for u0 in range(0, n_units, chunk):
                u1 = min(u0 + chunk, n_units)
                cidx_l = cp.ascontiguousarray(cidx_l_all[u0:u1])
                cidx_r = cp.ascontiguousarray(cidx_r_all[u0:u1])
                cost = cp.empty((u1 - u0, max_n, max_m), dtype=dtype)
                _sk.build_cost(emb, cidx_l, cidx_r, cost, stream.ptr)
                st = make_state(
                    cost,
                    cp.ascontiguousarray(mask_a[u0:u1]),
                    cp.ascontiguousarray(mask_b[u0:u1]),
                    epsilon=None,
                    epsilon_scale=self.epsilon_scale,
                )
                run_async(
                    [{"dev": device, "stream": stream, "state": st}],
                    max_iter=self.max_iter,
                    tol=self.tol,
                    check_every=DEFAULT_CHECK_EVERY,
                    omega=self.relaxation,
                )
                reg[u0:u1] = finalize(st)
                converged = converged and bool(st["conv"].all().get())
            if not converged:
                self._warn_not_converged()
            reg = reg.reshape(n_pairs, n_bootstrap)
            return reg.mean(axis=1), reg.var(axis=1)

    def bootstrap_arrays(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
    ) -> tuple[float, float]:
        """Bootstrap mean/variance of W between two arrays (``Distance.bootstrap``)."""
        Xc = cp.asarray(X)
        Yc = cp.asarray(Y)
        if len(Xc) == 0 or len(Yc) == 0:
            raise ValueError("Neither X nor Y can be empty.")
        if Xc.dtype != Yc.dtype:
            Yc = Yc.astype(Xc.dtype)
        n_a, n_b = Xc.shape[0], Yc.shape[0]
        emb = cp.ascontiguousarray(cp.concatenate([Xc, Yc], axis=0))
        cat_offsets = cp.asarray([0, n_a, n_a + n_b], dtype=cp.int32)
        cell_indices = cp.arange(n_a + n_b, dtype=cp.int32)
        mean, var = self._bootstrap_solve(
            emb,
            cat_offsets,
            cell_indices,
            [0],
            [1],
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            dtype=emb.dtype,
            device=cp.cuda.runtime.getDevice(),
        )
        return float(mean[0]), float(var[0])

    def bootstrap(
        self,
        adata: AnnData,
        groupby: str,
        group_a: str,
        group_b: str,
        *,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> tuple[float, float]:
        """Bootstrap mean/variance of W between two groups in ``adata``."""
        mean_df, var_df = self.pairwise(
            adata,
            groupby,
            groups=[group_a, group_b],
            bootstrap=True,
            n_bootstrap=n_bootstrap,
            random_state=random_state,
            multi_gpu=multi_gpu,
        )
        return float(mean_df.loc[group_a, group_b]), float(var_df.loc[group_a, group_b])

    def _warn_not_converged(self) -> None:
        warnings.warn(
            f"Sinkhorn did not converge in {self.max_iter} iterations "
            f"(tol={self.tol}).",
            RuntimeWarning,
            stacklevel=3,
        )

    def pairwise(
        self,
        adata: AnnData,
        groupby: str,
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Pairwise Wasserstein between all groups in ``groupby``.

        Returns a symmetric K×K DataFrame with zero on the diagonal. With
        ``bootstrap=True`` returns ``(mean_df, var_df)``, the per-pair mean and
        (population) variance over ``n_bootstrap`` cell resamples. Bootstrap is
        computed on a single GPU (the resamples are batched into the solver).
        """
        device_ids = parse_device_ids(multi_gpu=multi_gpu)

        _assert_categorical_obs(adata, key=groupby)
        embedding, cat_offsets, cell_indices, groups_list = self._subset_to_groups(
            adata, groupby, groups
        )
        k = len(groups_list)

        # Upper-triangle pairs (i < j); diagonal is 0
        pair_left: list[int] = []
        pair_right: list[int] = []
        for i in range(k):
            for j in range(i + 1, k):
                pair_left.append(i)
                pair_right.append(j)

        def _to_matrix(flat: cp.ndarray, name: str) -> pd.DataFrame:
            mat = cp.zeros((k, k), dtype=embedding.dtype)
            if pair_left:
                il = cp.asarray(pair_left, dtype=cp.intp)
                jr = cp.asarray(pair_right, dtype=cp.intp)
                mat[il, jr] = flat
                mat[jr, il] = flat
            df = pd.DataFrame(mat.get(), index=groups_list, columns=groups_list)
            df.index.name = groupby
            df.columns.name = groupby
            df.name = name
            return df

        if bootstrap:
            mean, var = self._bootstrap_solve(
                embedding,
                cat_offsets,
                cell_indices,
                pair_left,
                pair_right,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                dtype=embedding.dtype,
                device=device_ids[0],
            )
            return (
                _to_matrix(mean, "pairwise wasserstein"),
                _to_matrix(var, "pairwise wasserstein (var)"),
            )

        flat = self._solve_pairs(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left,
            pair_right,
            dtype=embedding.dtype,
            device_ids=device_ids,
        )
        return _to_matrix(flat, "pairwise wasserstein")

    def onesided_distances(
        self,
        adata: AnnData,
        groupby: str,
        selected_group: str | Sequence[str],
        *,
        groups: Sequence[str] | None = None,
        bootstrap: bool = False,
        n_bootstrap: int = 100,
        random_state: int = 0,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> (
        pd.Series
        | pd.DataFrame
        | tuple[pd.Series, pd.Series]
        | tuple[pd.DataFrame, pd.DataFrame]
    ):
        device_ids = parse_device_ids(multi_gpu=multi_gpu)

        selected_groups, single_control, needed = self._resolve_onesided_inputs(
            adata, groupby, selected_group, groups
        )
        embedding, cat_offsets, cell_indices, groups_list = self._subset_to_groups(
            adata, groupby, needed
        )
        k = len(groups_list)
        group_map = {v: i for i, v in enumerate(groups_list)}
        selected_indices = [group_map[sg] for sg in selected_groups]

        pair_left: list[int] = []
        pair_right: list[int] = []
        for si in selected_indices:
            for j in range(k):
                if j == si:
                    continue
                pair_left.append(si)
                pair_right.append(j)

        def _to_df(flat: cp.ndarray) -> pd.DataFrame:
            flat_cpu = flat.get()
            ed_cols: dict[str, np.ndarray] = {}
            cursor = 0
            for ii, si in enumerate(selected_indices):
                col = np.zeros(k, dtype=flat_cpu.dtype)
                for j in range(k):
                    if j == si:
                        continue
                    col[j] = flat_cpu[cursor]
                    cursor += 1
                ed_cols[selected_groups[ii]] = col
            df = pd.DataFrame(ed_cols, index=groups_list)
            df.index.name = groupby
            df.columns.name = "selected_group"
            return df

        if bootstrap:
            mean, var = self._bootstrap_solve(
                embedding,
                cat_offsets,
                cell_indices,
                pair_left,
                pair_right,
                n_bootstrap=n_bootstrap,
                random_state=random_state,
                dtype=embedding.dtype,
                device=device_ids[0],
            )
            mean_df, var_df = _to_df(mean), _to_df(var)
            if single_control:
                return mean_df[selected_groups[0]], var_df[selected_groups[0]]
            return mean_df, var_df

        flat = self._solve_pairs(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left,
            pair_right,
            dtype=embedding.dtype,
            device_ids=device_ids,
        )
        df = _to_df(flat)
        if single_control:
            return df[selected_groups[0]]
        return df

    def contrast_distances(
        self,
        adata: AnnData,
        contrasts: pd.DataFrame,
        *,
        multi_gpu: bool | list[int] | str | None = None,
    ) -> pd.DataFrame:
        device_ids = parse_device_ids(multi_gpu=multi_gpu)
        groupby, split_by = self._parse_contrasts(adata, contrasts)
        embedding_raw = self._get_embedding(adata)

        all_cols = [groupby, *split_by]
        grouped = adata.obs.groupby(all_cols, observed=True)
        group_indices = grouped.indices

        target_vals = contrasts[groupby].values
        ref_vals = contrasts["reference"].values
        split_arrays = [contrasts[col].values for col in split_by]

        cond_to_idx: dict[tuple, int] = {}
        contrast_pairs: list[tuple[int, int]] = []
        for i in range(len(contrasts)):
            if split_by:
                split_vals = tuple(arr[i] for arr in split_arrays)
                target_key = (target_vals[i], *split_vals)
                ref_key = (ref_vals[i], *split_vals)
            else:
                target_key = (target_vals[i],)
                ref_key = (ref_vals[i],)
            for key in (target_key, ref_key):
                if key not in cond_to_idx:
                    cond_to_idx[key] = len(cond_to_idx)
            contrast_pairs.append((cond_to_idx[target_key], cond_to_idx[ref_key]))

        # Build a unified (cat_offsets, cell_indices) layout — one "group" per
        # unique contrast condition. Skip empty conditions safely.
        offsets_host = [0]
        all_cells: list[np.ndarray] = []
        for key, _ in sorted(cond_to_idx.items(), key=lambda kv: kv[1]):
            lookup_key = key[0] if len(key) == 1 else key
            cell_idx = group_indices.get(lookup_key)
            if cell_idx is None:
                raise ValueError(f"No cells found for contrast condition {lookup_key}")
            all_cells.append(np.asarray(cell_idx, dtype=np.int64))
            offsets_host.append(offsets_host[-1] + len(cell_idx))

        flat_cell_idx = (
            np.concatenate(all_cells) if all_cells else np.array([], dtype=np.int64)
        )
        cat_offsets = cp.asarray(offsets_host, dtype=cp.int32)
        cell_indices = cp.asarray(flat_cell_idx, dtype=cp.int32)
        embedding = cp.asarray(embedding_raw)
        dtype = embedding.dtype

        # Deduplicate canonical pairs (i, j) with i < j
        canon_to_flat: dict[tuple[int, int], int] = {}
        pair_left: list[int] = []
        pair_right: list[int] = []
        for idx_a, idx_b in contrast_pairs:
            if idx_a == idx_b:
                continue
            canon = (min(idx_a, idx_b), max(idx_a, idx_b))
            if canon not in canon_to_flat:
                canon_to_flat[canon] = len(canon_to_flat)
                pair_left.append(canon[0])
                pair_right.append(canon[1])

        flat = self._solve_pairs(
            embedding,
            cat_offsets,
            cell_indices,
            pair_left,
            pair_right,
            dtype=dtype,
            device_ids=device_ids,
        )
        flat_cpu = flat.get()

        distances = np.empty(len(contrast_pairs), dtype=np.float64)
        for n, (idx_a, idx_b) in enumerate(contrast_pairs):
            if idx_a == idx_b:
                distances[n] = 0.0
                continue
            canon = (min(idx_a, idx_b), max(idx_a, idx_b))
            distances[n] = flat_cpu[canon_to_flat[canon]]

        result = contrasts.copy()
        result["wasserstein"] = distances
        return result

    def compute_distance(
        self,
        X: np.ndarray | cp.ndarray,
        Y: np.ndarray | cp.ndarray,
    ) -> float:
        Xc = cp.asarray(X)
        Yc = cp.asarray(Y)
        if len(Xc) == 0 or len(Yc) == 0:
            raise ValueError("Neither X nor Y can be empty.")
        if Xc.dtype != Yc.dtype:
            Yc = Yc.astype(Xc.dtype)

        n_a, n_b = Xc.shape[0], Yc.shape[0]
        sq_X = cp.sum(Xc * Xc, axis=1)
        sq_Y = cp.sum(Yc * Yc, axis=1)
        C = sq_X[:, None] + sq_Y[None, :] - 2.0 * (Xc @ Yc.T)
        cost = cp.maximum(C, 0)[None, :, :]
        mask_a = cp.ones((1, n_a), dtype=cp.bool_)
        mask_b = cp.ones((1, n_b), dtype=cp.bool_)
        out = batched_sinkhorn(
            cost,
            mask_a,
            mask_b,
            epsilon_scale=self.epsilon_scale,
            max_iter=self.max_iter,
            tol=self.tol,
            relaxation=self.relaxation,
        )
        return float(out[0])
