"""Multi-GPU sharding for the host-streaming aggregation / binned paths.

Additive-over-cells → format-preserving shards: CSR/dense row-shard (partials
summed), CSC column-shard (partials concatenated). Each shard runs the
single-GPU streamer in its own thread (bindings release the GIL → parallel);
results gather onto the caller's device.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
import scipy.sparse as sp

from rapids_singlecell._cuda import _rank_stream_cuda as _rss
from rapids_singlecell._utils import parse_device_ids

from ._wilcoxon_host import _copy_gpu_array_to_device

if TYPE_CHECKING:
    from ._core import _RankGenes


# Host pageable->pinned copy is the multi-GPU bottleneck; base on core count.
_MAX_HOST_STAGING_WORKERS = os.cpu_count() or 32


def resolve_stream_devices(*, multi_gpu: bool | list[int] | str | None) -> list[int]:
    """Device ids for a host-streaming run (single current device when off)."""
    if multi_gpu is None or multi_gpu is False:
        return [cp.cuda.Device().id]
    return list(dict.fromkeys(parse_device_ids(multi_gpu=multi_gpu)))


def _limit_host_workers(n_devices: int) -> None:
    """Split the host staging pool across device shards (no CPU oversubscription)."""
    _rss._set_host_worker_limit(max(1, _MAX_HOST_STAGING_WORKERS // n_devices))


def _bands(n: int, k: int) -> list[tuple[int, int]]:
    edges = np.linspace(0, n, k + 1, dtype=np.int64)
    return [
        (int(a), int(b)) for a, b in zip(edges[:-1], edges[1:], strict=True) if b > a
    ]


def _is_col_shard(X) -> bool:
    """CSC shards by gene columns; CSR / dense shard by cell rows."""
    return sp.issparse(X) and X.format == "csc"


def _shard_view(X, b0: int, b1: int):
    """Slice a shard via direct attribute assignment (no copy, no nnz scan).

    Avoids two O(nnz) traps: scipy ``X[b0:b1]`` copies data/indices, and the
    ``sp.csr_matrix((...))`` constructor validates (~200 ms for 100M nnz).
    """
    if not sp.issparse(X):
        return X[b0:b1]
    ptr0 = int(X.indptr[b0])
    ptr1 = int(X.indptr[b1])
    indptr = X.indptr[b0 : b1 + 1] - X.indptr[b0]
    data = X.data[ptr0:ptr1]
    indices = X.indices[ptr0:ptr1]
    if X.format == "csc":
        shard = sp.csc_matrix((X.shape[0], b1 - b0), dtype=X.data.dtype)
    else:
        shard = sp.csr_matrix((b1 - b0, X.shape[1]), dtype=X.data.dtype)
    shard.data = data
    shard.indices = indices
    shard.indptr = indptr
    return shard


def _sum_to_device(parts: list[cp.ndarray], device_id: int) -> cp.ndarray:
    with cp.cuda.Device(device_id):
        total = _copy_gpu_array_to_device(parts[0], device_id).copy()
        for part in parts[1:]:
            total += _copy_gpu_array_to_device(part, device_id)
        cp.cuda.runtime.deviceSynchronize()
    return total


def _concat_to_device(parts: list[cp.ndarray], device_id: int, axis: int) -> cp.ndarray:
    with cp.cuda.Device(device_id):
        local = [_copy_gpu_array_to_device(part, device_id) for part in parts]
        out = cp.concatenate(local, axis=axis)
        cp.cuda.runtime.deviceSynchronize()
    return out


def _host_aggr_data(data: np.ndarray) -> np.ndarray:
    if data.dtype in (np.float32, np.float64):
        return data
    return data.astype(np.float32)


def aggr_host_planes(
    X,
    cats: cp.ndarray,
    n_cats: int,
    *,
    comp_pts: bool,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray | None]:
    """Stream sum / sq_sum / (count) for ``X`` on the current device.

    ``cats`` (device int32) must match ``X``'s rows; outputs are
    ``(n_cats, X.shape[1])``.
    """
    n_cells, n_genes = X.shape
    out_sum = cp.zeros((n_cats, n_genes), dtype=cp.float64)
    out_sqsum = cp.zeros((n_cats, n_genes), dtype=cp.float64)
    out_count = cp.zeros((n_cats, n_genes), dtype=cp.float64) if comp_pts else None
    kw = {"out_sum": out_sum, "out_sqsum": out_sqsum}
    if out_count is not None:
        kw["out_count"] = out_count

    if sp.issparse(X):
        if X.format not in {"csr", "csc"}:
            X = X.tocsr()
        launch = _rss.aggr_csr_host if X.format == "csr" else _rss.aggr_csc_host
        launch(
            _host_aggr_data(X.data),
            X.indices,
            X.indptr,
            cats,
            n_cells=n_cells,
            n_genes=n_genes,
            **kw,
        )
    else:
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float32)
        if not (X.flags.c_contiguous or X.flags.f_contiguous):
            X = np.ascontiguousarray(X)
        _rss.aggr_dense_host(X, cats, **kw)
    return out_sum, out_sqsum, out_count


def stream_planes_multi(
    rg: _RankGenes, device_ids: list[int]
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray | None]:
    """Sharded sum / sq_sum / (count) over groups plus the remainder."""
    X = rg.X
    n_cells, n_genes = X.shape
    n_cats = len(rg.groups_order) + 1
    codes = rg.group_codes.astype(np.int32, copy=False)
    comp_pts = rg.comp_pts
    col_shard = _is_col_shard(X)
    axis_len = n_genes if col_shard else n_cells
    bands = _bands(axis_len, len(device_ids))
    device_ids = device_ids[: len(bands)]

    def run_shard(index: int):
        device_id = device_ids[index]
        b0, b1 = bands[index]
        with cp.cuda.Device(device_id):
            _limit_host_workers(len(device_ids))
            shard = _shard_view(X, b0, b1)
            shard_cats = cp.asarray(
                codes if col_shard else codes[b0:b1], dtype=cp.int32
            )
            out = aggr_host_planes(shard, shard_cats, n_cats, comp_pts=comp_pts)
            cp.cuda.runtime.deviceSynchronize()
        return out

    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        shards = list(executor.map(run_shard, range(len(device_ids))))

    # Gather onto the caller's device so downstream stats math stays local.
    dev0 = cp.cuda.Device().id
    if col_shard:
        sums = _concat_to_device([s[0] for s in shards], dev0, axis=1)
        sqsums = _concat_to_device([s[1] for s in shards], dev0, axis=1)
        counts = (
            _concat_to_device([s[2] for s in shards], dev0, axis=1)
            if comp_pts
            else None
        )
    else:
        sums = _sum_to_device([s[0] for s in shards], dev0)
        sqsums = _sum_to_device([s[1] for s in shards], dev0)
        counts = _sum_to_device([s[2] for s in shards], dev0) if comp_pts else None
    return sums, sqsums, counts


def run_binned_hist_multi(
    X,
    group_codes_np: np.ndarray,
    device_ids: list[int],
    *,
    n_groups: int,
    n_hist_groups: int,
    n_bins: int,
    bin_low: float,
    inv_bin_width: float,
    comp_pts: bool,
    start: int,
    stop: int,
    accumulate_means: bool,
):
    """Build one sharded gene-window histogram and gather onto the caller.

    Fused group sums are optional. For CSC they cover ``[start, stop)``; for
    dense/CSR row shards they retain the full input width required by the
    aggregation kernels.
    """
    from ._wilcoxon_binned import (
        _launch_csc_host,
        _launch_csr_host,
        _launch_dense_host,
    )

    n_cells, n_genes = X.shape
    if not 0 <= start < stop <= n_genes:
        raise ValueError("invalid multi-GPU histogram gene window")
    col_shard = _is_col_shard(X)
    axis_len = stop - start if col_shard else n_cells
    bands = _bands(axis_len, len(device_ids))
    device_ids = device_ids[: len(bands)]

    def run_shard(index: int):
        device_id = device_ids[index]
        b0, b1 = bands[index]
        with cp.cuda.Device(device_id):
            _limit_host_workers(len(device_ids))
            if col_shard:
                shard = _shard_view(X, start + b0, start + b1)
                codes = cp.asarray(group_codes_np, dtype=cp.int32)
                band_genes = b1 - b0
                launch = _launch_csc_host
                launch_start, launch_stop = 0, band_genes
            else:
                shard = _shard_view(X, b0, b1)
                codes = cp.asarray(group_codes_np[b0:b1], dtype=cp.int32)
                band_genes = stop - start
                launch = _launch_csr_host if sp.issparse(X) else _launch_dense_host
                launch_start, launch_stop = start, stop
            sum_genes = band_genes if col_shard else n_genes
            gsum = (
                cp.zeros((n_groups + 1, sum_genes), dtype=cp.float64)
                if accumulate_means
                else None
            )
            gnnz = (
                cp.zeros((n_groups + 1, sum_genes), dtype=cp.float64)
                if accumulate_means and comp_pts
                else None
            )
            hist = launch(
                shard,
                codes,
                n_hist_groups,
                start=launch_start,
                stop=launch_stop,
                n_bins=n_bins,
                bin_low=bin_low,
                inv_bin_width=inv_bin_width,
                group_sums=gsum,
                group_nnz=gnnz,
            )
            cp.cuda.runtime.deviceSynchronize()
        return hist, gsum, gnnz

    with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
        shards = list(executor.map(run_shard, range(len(device_ids))))

    home = cp.cuda.Device().id
    if col_shard:
        hist = _concat_to_device([s[0] for s in shards], home, axis=0)
        gsum = (
            _concat_to_device([s[1] for s in shards], home, axis=1)
            if accumulate_means
            else None
        )
        gnnz = (
            _concat_to_device([s[2] for s in shards], home, axis=1)
            if accumulate_means and comp_pts
            else None
        )
    else:
        hist = _sum_to_device([s[0] for s in shards], home)
        gsum = (
            _sum_to_device([s[1] for s in shards], home) if accumulate_means else None
        )
        gnnz = (
            _sum_to_device([s[2] for s in shards], home)
            if accumulate_means and comp_pts
            else None
        )
    return hist, gsum, gnnz
