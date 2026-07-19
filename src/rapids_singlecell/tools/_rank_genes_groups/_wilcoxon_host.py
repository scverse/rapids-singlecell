from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc
from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as _wcs
from rapids_singlecell._utils import parse_device_ids

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ._core import _RankGenes

CUDA_HOST_REGISTER_PORTABLE = 1
MIN_SPARSE_GENE_WORK = 1
MAX_SPARSE_SPLIT_SAMPLES = 10_000_000
SPARSE_SPLIT_SAMPLE_BLOCKS = 32
MAX_HOST_STAGING_WORKERS = 64
OVO_STABLE_GROUPING_MIN_GROUPS = 256


@dataclass(frozen=True)
class _OvoHostContext:
    n_ref: int
    ref_row_ids: np.ndarray
    test_group_indices: list[int]
    all_grp_row_ids: np.ndarray
    offsets_np: np.ndarray
    n_all_grp: int
    n_test: int


_WilcoxonResult = tuple[np.ndarray, cp.ndarray, cp.ndarray, cp.ndarray | None]


def _host_sparse_kernel_source(X):
    """Normalize sparse values once while sharing index metadata."""
    data = (
        X.data if X.data.dtype == np.float64 else X.data.astype(np.float32, copy=False)
    )
    if data is X.data:
        return X
    source = copy(X)
    source.data = data
    return source


def _host_dense_matrix(matrix: np.ndarray) -> np.ndarray:
    if matrix.dtype.kind != "f" or matrix.dtype.itemsize < 4:
        return np.asarray(matrix, dtype=np.float32, order="F")
    if matrix.flags.c_contiguous or matrix.flags.f_contiguous:
        return matrix
    return np.asfortranarray(matrix)


@contextmanager
def _shared_dense_host_registration(
    matrix: np.ndarray | None,
    *,
    enabled: bool,
):
    """Best-effort portable pin shared by concurrent dense OVR workers."""
    if matrix is None or not enabled or matrix.nbytes == 0:
        yield
        return

    registered = False
    try:
        cp.cuda.runtime.hostRegister(
            matrix.ctypes.data,
            matrix.nbytes,
            CUDA_HOST_REGISTER_PORTABLE,
        )
    except cp.cuda.runtime.CUDARuntimeError:
        pass
    else:
        registered = True
    try:
        yield
    finally:
        if registered:
            cp.cuda.runtime.hostUnregister(matrix.ctypes.data)


def _build_ovo_host_context(rg: _RankGenes) -> _OvoHostContext:
    group_sizes = rg.group_sizes
    codes = rg.group_codes
    n_groups = len(rg.groups_order)
    ireference = int(rg.ireference)
    n_ref = int(group_sizes[ireference])
    test_group_indices = [i for i in range(n_groups) if i != ireference]

    if n_groups < OVO_STABLE_GROUPING_MIN_GROUPS:
        ref_row_ids = np.flatnonzero(codes == ireference).astype(np.int32, copy=False)
        row_id_parts = [
            np.flatnonzero(codes == group_index).astype(np.int32, copy=False)
            for group_index in test_group_indices
        ]
        all_grp_row_ids = (
            np.concatenate(row_id_parts)
            if row_id_parts
            else np.empty(0, dtype=np.int32)
        )
    else:
        # One stable grouping pass replaces a full-cell boolean scan per group.
        # Selected codes are [0, n_groups); the unselected-cell sentinel sorts
        # after them and is excluded by selected_total.
        grouped_rows = np.argsort(codes, kind="stable").astype(np.int32, copy=False)
        group_starts = np.empty(n_groups + 1, dtype=np.intp)
        group_starts[0] = 0
        np.cumsum(group_sizes, dtype=np.intp, out=group_starts[1:])
        ref_start = int(group_starts[ireference])
        ref_stop = int(group_starts[ireference + 1])
        ref_row_ids = grouped_rows[ref_start:ref_stop]
        selected_total = int(group_starts[-1])
        all_grp_row_ids = np.concatenate(
            (grouped_rows[:ref_start], grouped_rows[ref_stop:selected_total])
        )
    test_indices_np = np.asarray(test_group_indices, dtype=np.intp)
    offsets_np = np.empty(len(test_group_indices) + 1, dtype=np.int32)
    offsets_np[0] = 0
    offsets_np[1:] = np.cumsum(group_sizes[test_indices_np], dtype=np.int64)
    return _OvoHostContext(
        n_ref=n_ref,
        ref_row_ids=ref_row_ids,
        test_group_indices=test_group_indices,
        all_grp_row_ids=all_grp_row_ids,
        offsets_np=offsets_np,
        n_all_grp=int(all_grp_row_ids.size),
        n_test=len(test_group_indices),
    )


def _split_host_gene_ranges(
    X,
    *,
    n_devices: int,
    dense_fallback: bool,
) -> list[tuple[int, int]]:
    """Split contiguous genes, balancing sparse-native work by column nnz."""
    n_genes = int(X.shape[1])
    n_shards = min(n_devices, n_genes)
    if n_shards <= 1:
        return [(0, n_genes)]

    if isinstance(X, np.ndarray) or dense_fallback:
        weights = np.ones(n_genes, dtype=np.int64)
    elif X.format == "csc":
        weights = np.diff(X.indptr).astype(np.int64, copy=False) + MIN_SPARSE_GENE_WORK
    else:
        indices = X.indices
        if indices.size <= MAX_SPARSE_SPLIT_SAMPLES:
            weights = np.bincount(indices, minlength=n_genes).astype(
                np.int64, copy=False
            )
        else:
            # An exact CSR column histogram scans the entire index buffer. At
            # real scale that planning pass can cost more than the Wilcoxon
            # itself, so sample small contiguous blocks spread across the
            # buffer. Contiguous sampling keeps host reads bounded and the
            # evenly-spaced blocks avoid depending on one cell/group region.
            weights = np.zeros(n_genes, dtype=np.int64)
            block_size = MAX_SPARSE_SPLIT_SAMPLES // SPARSE_SPLIT_SAMPLE_BLOCKS
            last_start = indices.size - block_size
            for block_index in range(SPARSE_SPLIT_SAMPLE_BLOCKS):
                start = last_start * block_index // (SPARSE_SPLIT_SAMPLE_BLOCKS - 1)
                weights += np.bincount(
                    indices[start : start + block_size], minlength=n_genes
                )
        weights += MIN_SPARSE_GENE_WORK

    total = int(weights.sum(dtype=np.int64))
    cumulative = np.cumsum(weights, dtype=np.int64)
    bounds = [0]
    for shard_index in range(1, n_shards):
        target = (total * shard_index + n_shards - 1) // n_shards
        candidate = int(np.searchsorted(cumulative, target, side="left")) + 1
        lower = bounds[-1] + 1
        upper = n_genes - (n_shards - shard_index)
        bounds.append(min(max(candidate, lower), upper))
    bounds.append(n_genes)
    return list(zip(bounds[:-1], bounds[1:], strict=True))


def _host_csc_column_shard(X, start: int, stop: int):
    ptr_start = int(X.indptr[start])
    ptr_stop = int(X.indptr[stop])
    indptr = X.indptr[start : stop + 1] - ptr_start
    return type(X)(
        (
            X.data[ptr_start:ptr_stop],
            X.indices[ptr_start:ptr_stop],
            indptr,
        ),
        shape=(X.shape[0], stop - start),
        copy=False,
    )


def _concat_shard_stat(workers: list[_RankGenes], name: str) -> NDArray | None:
    arrays = [getattr(worker, name) for worker in workers]
    if all(array is None for array in arrays):
        return None
    if any(array is None for array in arrays):
        msg = f"Inconsistent host Wilcoxon shard statistic: {name}."
        raise RuntimeError(msg)
    return np.concatenate(arrays, axis=1)


def _copy_gpu_array_to_device(array: cp.ndarray, device_id: int) -> cp.ndarray:
    if array.device.id == device_id:
        return array
    with cp.cuda.Device(device_id):
        copied = cp.empty_like(array)
    try:
        cp.cuda.runtime.memcpyPeer(
            copied.data.ptr,
            device_id,
            array.data.ptr,
            array.device.id,
            array.nbytes,
        )
    except cp.cuda.runtime.CUDARuntimeError:
        with cp.cuda.Device(array.device.id):
            host = array.get()
        with cp.cuda.Device(device_id):
            copied.set(host)
    return copied


def _concat_gpu_shards(arrays: list[cp.ndarray], device_id: int) -> cp.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    return cp.concatenate(
        [_copy_gpu_array_to_device(array, device_id) for array in arrays],
        axis=1,
    )


def _run_host_wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    multi_gpu: bool | list[int] | str | None,
    return_u_values: bool,
    shard_runner: Callable[..., _WilcoxonResult | None],
) -> _WilcoxonResult | None:
    """Run and gather the common host-resident single/multi-GPU path."""
    X = rg.X
    n_cells, n_total_genes = X.shape
    device_ids = (
        [cp.cuda.Device().id]
        if multi_gpu is False
        else list(dict.fromkeys(parse_device_ids(multi_gpu=multi_gpu)))
    )
    ranges = _split_host_gene_ranges(
        X,
        n_devices=len(device_ids),
        dense_fallback=rg._sparse_negative_fallback,
    )
    device_ids = device_ids[: len(ranges)]
    ovo_host_context = (
        _build_ovo_host_context(rg) if rg.ireference is not None else None
    )
    dense_source = _host_dense_matrix(X) if isinstance(X, np.ndarray) else None
    csr_row_spans = [None] * len(ranges)
    if dense_source is not None:
        kernel_inputs = [dense_source] * len(ranges)
    else:
        sparse_source = _host_sparse_kernel_source(X)
        if sparse_source.format == "csr":
            kernel_inputs = [sparse_source] * len(ranges)
            if len(ranges) == 1:
                csr_row_spans = [(sparse_source.indptr[:-1], sparse_source.indptr[1:])]
            else:
                cuts = np.asarray([stop for _, stop in ranges[:-1]], dtype=np.int32)
                boundaries = np.empty(
                    (cuts.size, n_cells),
                    dtype=sparse_source.indptr.dtype,
                    order="C",
                )
                _wcs.csr_row_boundaries_host(
                    sparse_source.indices,
                    sparse_source.indptr,
                    cuts,
                    boundaries,
                    n_cols=n_total_genes,
                )
                csr_row_spans = [
                    (
                        sparse_source.indptr[:-1]
                        if shard_index == 0
                        else boundaries[shard_index - 1],
                        sparse_source.indptr[1:]
                        if shard_index == len(ranges) - 1
                        else boundaries[shard_index],
                    )
                    for shard_index in range(len(ranges))
                ]
        else:
            kernel_inputs = [
                _host_csc_column_shard(sparse_source, start, stop)
                for start, stop in ranges
            ]

    def run_shard(
        shard_index: int,
    ) -> tuple[_RankGenes, _WilcoxonResult | None]:
        worker = copy(rg)
        worker.X = kernel_inputs[shard_index]
        column_range = ranges[shard_index]
        host_workers = max(1, MAX_HOST_STAGING_WORKERS // len(device_ids))
        _wc._set_host_worker_limit(host_workers)
        _wcs._set_host_worker_limit(host_workers)
        with cp.cuda.Device(device_ids[shard_index]):
            result = shard_runner(
                worker,
                tie_correct=tie_correct,
                use_continuity=use_continuity,
                chunk_size=chunk_size,
                return_u_values=return_u_values,
                column_range=column_range,
                sparse_row_spans=csr_row_spans[shard_index],
                ovo_host_context=ovo_host_context,
            )
        return worker, result

    register_dense = dense_source is not None and rg.ireference is None
    with _shared_dense_host_registration(dense_source, enabled=register_dense):
        with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
            shards = list(executor.map(run_shard, range(len(device_ids))))

    workers = [worker for worker, _ in shards]

    rg.means = _concat_shard_stat(workers, "means")
    rg.vars = None
    rg.pts = _concat_shard_stat(workers, "pts")
    rg.means_rest = _concat_shard_stat(workers, "means_rest")
    rg.vars_rest = None
    rg.pts_rest = _concat_shard_stat(workers, "pts_rest")

    gpu_results = [result for _, result in shards]
    if all(result is None for result in gpu_results):
        return None
    if any(result is None for result in gpu_results):
        msg = "Inconsistent host Wilcoxon GPU result state."
        raise RuntimeError(msg)
    complete_gpu_results = [result for result in gpu_results if result is not None]
    group_indices = complete_gpu_results[0][0]
    if any(
        not np.array_equal(result[0], group_indices)
        for result in complete_gpu_results[1:]
    ):
        msg = "Inconsistent host Wilcoxon group ordering."
        raise RuntimeError(msg)
    result_device = device_ids[0]
    with cp.cuda.Device(result_device):
        scores = _concat_gpu_shards(
            [result[1] for result in complete_gpu_results], result_device
        )
        pvals = _concat_gpu_shards(
            [result[2] for result in complete_gpu_results], result_device
        )
        logfoldchange_parts = [result[3] for result in complete_gpu_results]
        logfoldchanges = (
            _concat_gpu_shards(
                [part for part in logfoldchange_parts if part is not None],
                result_device,
            )
            if all(part is not None for part in logfoldchange_parts)
            else None
        )
    return (
        group_indices,
        scores,
        pvals,
        logfoldchanges,
    )
