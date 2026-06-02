"""GPU-accelerated entropic 2-Wasserstein between groups of cells.

Pairs are split across GPUs and solved in memory-bounded batches by the ragged
batched Sinkhorn (see ``_sinkhorn`` and the ``_sinkhorn_cuda`` kernels).
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
    finalize,
    make_state,
    run_async,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

# Per-batch cost-array byte budget. Fixed, not a fraction of free memory: under
# an RMM pool allocator cudaMemGetInfo is unreliable (degenerates the batch).
COST_BUDGET_BYTES = 8 * 1024**3
MAX_BATCH = 4096
# Max per-device work excess over ideal before round-robin batch assignment is
# rejected for the work-balanced split (see _plan_device_batches).
DEVICE_WORK_IMBALANCE_TOL = 0.2
COST_TILE = 16  # must match constexpr int TILE in sinkhorn.cu
# Over-relaxation: omega 1 = plain Sinkhorn; (1, 2) is the convergent SOR range.
MIN_RELAXATION = 1.0
MAX_RELAXATION = 2.0


def _build_ragged_layout(
    cat_offsets: cp.ndarray,
    cell_indices: cp.ndarray,
    rows_h: np.ndarray,
    cols_h: np.ndarray,
    *,
    n_h: np.ndarray,
    m_h: np.ndarray,
    dtype,
    rng: cp.random.Generator | None = None,
) -> dict:
    """Build the flat ragged index layout for one batch on the current device.

    ``rows_h`` / ``cols_h`` (host group indices, smaller group as rows) and
    ``n_h`` / ``m_h`` (host sizes) drive the build. Offsets and flat lengths are
    computed host-side so the device work never syncs (lets devices' builds
    overlap). ``rng`` resamples cells with replacement for the bootstrap.
    """
    B = len(rows_h)
    n64 = n_h.astype(np.int64)
    m64 = m_h.astype(np.int64)
    cost_sizes = n64 * m64

    def _excl_cumsum(sizes: np.ndarray) -> tuple[np.ndarray, int]:
        off = np.zeros(B, dtype=np.int64)
        if B > 1:
            off[1:] = np.cumsum(sizes)[:-1]
        return off, int(sizes.sum())

    cost_off_h, total_cost = _excl_cumsum(cost_sizes)
    f_off_h, total_rows = _excl_cumsum(n64)
    g_off_h, total_cols = _excl_cumsum(m64)

    n = cp.asarray(n_h.astype(np.int32))
    m = cp.asarray(m_h.astype(np.int32))
    cost_off = cp.asarray(cost_off_h)
    f_off = cp.asarray(f_off_h)
    g_off = cp.asarray(g_off_h)

    # Flat row/column -> pair via searchsorted on the internal start offsets.
    row2pair = cp.searchsorted(
        f_off[1:], cp.arange(total_rows, dtype=cp.int64), side="right"
    ).astype(cp.int32)
    col2pair = cp.searchsorted(
        g_off[1:], cp.arange(total_cols, dtype=cp.int64), side="right"
    ).astype(cp.int32)

    # Flat tile schedule (one TILE x TILE tile per block in build_cost).
    ntn_h = (n_h.astype(np.int64) + COST_TILE - 1) // COST_TILE
    ntm_h = (m_h.astype(np.int64) + COST_TILE - 1) // COST_TILE
    tpp_h = ntn_h * ntm_h
    tile_off_h, total_tiles = _excl_cumsum(tpp_h)
    tile_off = cp.asarray(tile_off_h)
    ntm = cp.asarray(ntm_h)
    tile_pair = cp.searchsorted(
        tile_off[1:], cp.arange(total_tiles, dtype=cp.int64), side="right"
    ).astype(cp.int32)
    t_local = cp.arange(total_tiles, dtype=cp.int64) - tile_off[tile_pair]
    ntm_p = ntm[tile_pair]
    tile_i0 = ((t_local // ntm_p) * COST_TILE).astype(cp.int32)
    tile_j0 = ((t_local % ntm_p) * COST_TILE).astype(cp.int32)

    # Flat gather indices into cell_indices (per-pair group offset + local pos).
    offs_row = cat_offsets[cp.asarray(rows_h)].astype(cp.int64)
    offs_col = cat_offsets[cp.asarray(cols_h)].astype(cp.int64)
    if rng is None:
        loc_r = cp.arange(total_rows, dtype=cp.int64) - f_off[row2pair]
        loc_c = cp.arange(total_cols, dtype=cp.int64) - g_off[col2pair]
    else:
        # Resample with replacement to each group's own size (bootstrap).
        loc_r = (
            rng.random(total_rows, dtype=dtype) * n[row2pair].astype(dtype)
        ).astype(cp.int64)
        loc_c = (
            rng.random(total_cols, dtype=dtype) * m[col2pair].astype(dtype)
        ).astype(cp.int64)
    gather_f = offs_row[row2pair] + loc_r
    gather_g = offs_col[col2pair] + loc_c
    cidx_l = cp.ascontiguousarray(cell_indices[gather_f].astype(cp.int32))
    cidx_r = cp.ascontiguousarray(cell_indices[gather_g].astype(cp.int32))

    return {
        "B": B,
        "n": n,
        "m": m,
        "cost_off": cost_off,
        "f_off": f_off,
        "g_off": g_off,
        "row2pair": row2pair,
        "col2pair": col2pair,
        "tile_pair": tile_pair,
        "tile_i0": tile_i0,
        "tile_j0": tile_j0,
        "cidx_l": cidx_l,
        "cidx_r": cidx_r,
        "total_cost": total_cost,
        "total_rows": total_rows,
        "total_cols": total_cols,
        "total_tiles": total_tiles,
    }


def _launch_build_cost(
    emb: cp.ndarray, layout: dict, cost: cp.ndarray, stream_ptr: int
) -> None:
    """Fill the flat ``cost`` array for a ragged ``layout`` (one kernel launch)."""
    _sk.build_cost(
        emb,
        layout["cidx_l"],
        layout["f_off"],
        layout["cidx_r"],
        layout["g_off"],
        layout["n"],
        layout["m"],
        layout["cost_off"],
        layout["tile_pair"],
        layout["tile_i0"],
        layout["tile_j0"],
        cost,
        stream_ptr,
    )


def _split_positions_by_work(
    n_left_host: np.ndarray,
    n_right_host: np.ndarray,
    n_devices: int,
) -> list[int]:
    """Partition pair positions into ``n_devices`` contiguous segments of equal
    cumulative work (``n_left * n_right``); returns ``n_devices + 1`` boundaries.
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
    n_row_host: np.ndarray,
    n_col_host: np.ndarray,
    itemsize: int,
) -> list[tuple[int, int]]:
    """Greedily pack ``[start, stop)`` into ``(batch_start, batch_stop)`` batches,
    each capped by ``COST_BUDGET_BYTES`` of true cost cells (no padding) and
    ``MAX_BATCH`` pairs. A single oversized pair gets its own batch.
    """
    budget_cells = max(COST_BUDGET_BYTES // itemsize, 1)
    plans = []
    i = start
    while i < stop:
        cells = int(n_row_host[i]) * int(n_col_host[i])
        j = i + 1
        while j < stop and (j - i) < MAX_BATCH:
            nxt = int(n_row_host[j]) * int(n_col_host[j])
            if cells + nxt > budget_cells:
                break
            cells += nxt
            j += 1
        plans.append((i, j))
        i = j
    return plans


def _plan_device_batches(
    n_row_host: np.ndarray,
    n_col_host: np.ndarray,
    itemsize: int,
    n_devices: int,
) -> list[list[tuple[int, int]]]:
    """Assign batches to devices, choosing the split per workload.

    Round-robin dealing gives equal batch counts (equal rounds, no idle tail) and
    balances work well when batches are many and small; the work-split (equal
    cumulative ``n*m`` per device) is needed when a few big batches would
    otherwise land on one device. Prefer round-robin when its estimated work
    imbalance is within ``DEVICE_WORK_IMBALANCE_TOL``, else fall back.
    """
    n_pairs = len(n_row_host)
    if n_devices <= 1 or n_pairs == 0:
        return [_plan_batches(0, n_pairs, n_row_host, n_col_host, itemsize)]

    all_batches = _plan_batches(0, n_pairs, n_row_host, n_col_host, itemsize)
    work = n_row_host.astype(np.int64) * n_col_host.astype(np.int64)
    round_robin = [all_batches[i::n_devices] for i in range(n_devices)]
    loads = [sum(int(work[s:e].sum()) for (s, e) in plan) for plan in round_robin]
    ideal = sum(loads) / n_devices
    if ideal > 0 and max(loads) / ideal - 1 <= DEVICE_WORK_IMBALANCE_TOL:
        return round_robin

    bounds = _split_positions_by_work(n_row_host, n_col_host, n_devices)
    return [
        _plan_batches(bounds[i], bounds[i + 1], n_row_host, n_col_host, itemsize)
        for i in range(n_devices)
    ]


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
        Over-relaxation factor. ``1.0`` (default) is plain Sinkhorn and matches
        OTT-JAX. Values in ``(1, 2)`` cut the iteration count (~halved near
        ``1.5``) for a small change in the converged value; too large diverges
        (hits the iteration cap and warns). Leave at ``1.0`` for exact results.

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
        # Fixed to OTT-JAX / pertpy defaults (not user-tunable); eps is auto.
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

        Each pair is oriented larger-group-as-columns (the OT cost is symmetric)
        so the cooperative ``update_f`` reduction runs on the big axis. Pairs are
        split across ``device_ids`` and solved in batch-rounds, with per-iteration
        launches interleaved across devices' streams so the GPUs overlap.
        """
        if device_ids is None:
            device_ids = [0]
        n_pairs = len(pair_left)
        if n_pairs == 0:
            return cp.zeros(0, dtype=dtype)

        pl_host = np.asarray(pair_left, dtype=np.int32)
        pr_host = np.asarray(pair_right, dtype=np.int32)
        group_sizes = (cat_offsets[1:] - cat_offsets[:-1]).astype(cp.int32)
        # Group sizes to host (one sync) -> all batch planning is host-only.
        n_left = group_sizes[cp.asarray(pl_host)].get()
        n_right = group_sizes[cp.asarray(pr_host)].get()
        # Orient larger group as columns.
        swap = n_left > n_right
        rows = np.where(swap, pr_host, pl_host)
        cols = np.where(swap, pl_host, pr_host)
        n_row = np.minimum(n_left, n_right)
        n_col = np.maximum(n_left, n_right)
        itemsize = cp.dtype(dtype).itemsize

        plans = _plan_device_batches(n_row, n_col, itemsize, len(device_ids))

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
                        cp.ascontiguousarray(cp.asarray(embedding)),
                        cp.asarray(cat_offsets),
                        cp.asarray(cell_indices),
                    )

        # Grow-only per-device cost buffers, reused across rounds (avoids a fresh
        # cudaMalloc per round; safe since a round finishes before the next).
        cost_bufs: dict[int, cp.ndarray] = {}
        converged = True
        for r in range(max((len(p) for p in plans), default=0)):
            batch = [
                (dev, *plans[di][r])
                for di, dev in enumerate(device_ids)
                if r < len(plans[di])
            ]

            # Phases are kept separate (layout/alloc, then cost kernels, then
            # solve) so the per-device kernels launch back-to-back and overlap.
            # Phase 1: per-device ragged layout.
            preps = []
            for dev, start, stop in batch:
                _, off, idx = inputs[dev]
                with cp.cuda.Device(dev), streams[dev]:
                    layout = _build_ragged_layout(
                        off,
                        idx,
                        rows[start:stop],
                        cols[start:stop],
                        n_h=n_row[start:stop],
                        m_h=n_col[start:stop],
                        dtype=dtype,
                    )
                preps.append((dev, start, stop, layout))

            # Phase 2: build cost into the reusable buffers.
            units: list[dict] = []
            for dev, start, stop, layout in preps:
                emb = inputs[dev][0]
                with cp.cuda.Device(dev), streams[dev]:
                    need = layout["total_cost"]
                    buf = cost_bufs.get(dev)
                    if buf is None or buf.size < need:
                        buf = cp.empty(need, dtype=dtype)
                        cost_bufs[dev] = buf
                    cost = buf[:need]
                    _launch_build_cost(emb, layout, cost, streams[dev].ptr)
                units.append(
                    {
                        "dev": dev,
                        "stream": streams[dev],
                        "start": start,
                        "stop": stop,
                        "cost": cost,
                        "layout": layout,
                    }
                )

            # Phase 3: per-pair solver state.
            for u in units:
                with cp.cuda.Device(u["dev"]), u["stream"]:
                    u["state"] = make_state(
                        u["layout"],
                        u["cost"],
                        epsilon_scale=self.epsilon_scale,
                    )

            # Phase 4: solve across the round's units, then gather.
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

        Expands the pairs into ``n_pairs * n_bootstrap`` units (resampled with
        replacement) and solves them in deterministic memory-bounded chunks on one
        device. Chunk boundaries don't depend on data, so the single advancing RNG
        is reproducible for a given ``random_state``. Variance is population
        (ddof=0), matching pertpy.
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
            # Sizes/orientation on the host so the per-chunk build never syncs.
            co_h = cp.asnumpy(offs)
            sizes_h = np.diff(co_h)
            pl_h = np.asarray(pair_left, dtype=np.int64)
            pr_h = np.asarray(pair_right, dtype=np.int64)
            n_l = sizes_h[pl_h]
            n_r = sizes_h[pr_h]
            swap = n_l > n_r  # orient larger group as columns
            row_grp = np.where(swap, pr_h, pl_h)
            col_grp = np.where(swap, pl_h, pr_h)
            n_row = np.minimum(n_l, n_r)
            n_col = np.maximum(n_l, n_r)
            # Expand each pair into n_bootstrap independently-resampled units.
            rows_u = np.repeat(row_grp, n_bootstrap)
            cols_u = np.repeat(col_grp, n_bootstrap)
            nrow_u = np.repeat(n_row, n_bootstrap)
            ncol_u = np.repeat(n_col, n_bootstrap)
            n_units = n_pairs * n_bootstrap

            reg = cp.empty(n_units, dtype=dtype)
            itemsize = cp.dtype(dtype).itemsize
            plans = _plan_batches(0, n_units, nrow_u, ncol_u, itemsize)
            rng = cp.random.default_rng(random_state)
            stream = cp.cuda.get_current_stream()
            converged = True
            for u0, u1 in plans:
                layout = _build_ragged_layout(
                    offs,
                    cidx,
                    rows_u[u0:u1],
                    cols_u[u0:u1],
                    n_h=nrow_u[u0:u1],
                    m_h=ncol_u[u0:u1],
                    dtype=dtype,
                    rng=rng,
                )
                cost = cp.empty(layout["total_cost"], dtype=dtype)
                _launch_build_cost(emb, layout, cost, stream.ptr)
                st = make_state(layout, cost, epsilon_scale=self.epsilon_scale)
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

        n_a, n_b = int(Xc.shape[0]), int(Yc.shape[0])
        emb = cp.ascontiguousarray(cp.concatenate([Xc, Yc], axis=0))
        cat_offsets = cp.asarray([0, n_a, n_a + n_b], dtype=cp.int32)
        cell_indices = cp.arange(n_a + n_b, dtype=cp.int32)
        # Single pair (groups 0 and 1), smaller group oriented as rows.
        if n_a <= n_b:
            rows_h, cols_h, n_h, m_h = (0, 1, n_a, n_b)
        else:
            rows_h, cols_h, n_h, m_h = (1, 0, n_b, n_a)
        layout = _build_ragged_layout(
            cat_offsets,
            cell_indices,
            np.array([rows_h], dtype=np.int64),
            np.array([cols_h], dtype=np.int64),
            n_h=np.array([n_h], dtype=np.int64),
            m_h=np.array([m_h], dtype=np.int64),
            dtype=emb.dtype,
        )
        cost = cp.empty(layout["total_cost"], dtype=emb.dtype)
        stream = cp.cuda.get_current_stream()
        _launch_build_cost(emb, layout, cost, stream.ptr)
        st = make_state(layout, cost, epsilon_scale=self.epsilon_scale)
        run_async(
            [{"dev": cp.cuda.runtime.getDevice(), "stream": stream, "state": st}],
            max_iter=self.max_iter,
            tol=self.tol,
            check_every=DEFAULT_CHECK_EVERY,
            omega=self.relaxation,
        )
        if not bool(st["conv"].all().get()):
            self._warn_not_converged()
        return float(finalize(st)[0])
