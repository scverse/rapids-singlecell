from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import cache
from typing import TYPE_CHECKING

import cupy as cp
import cupyx.scipy.sparse as cpsp
import numpy as np

from rapids_singlecell._cuda import _wilcoxon_cuda as _wc
from rapids_singlecell._cuda import _wilcoxon_sparse_cuda as _wcs
from rapids_singlecell._utils import parse_device_ids

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ._core import _RankGenes

CUDA_HOST_REGISTER_PORTABLE = 1
CUDA_ERROR_PEER_ACCESS_UNSUPPORTED = 217
CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED = 704
CUDA_ERROR_TOO_MANY_PEERS = 711
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


def _csr_column_weights(indices, n_genes: int, xp) -> np.ndarray:
    """Estimate CSR column work with exact or evenly sampled histograms."""
    if indices.size == 0:
        return np.zeros(n_genes, dtype=np.int64)
    if indices.size <= MAX_SPARSE_SPLIT_SAMPLES:
        weights = xp.bincount(indices, minlength=n_genes)
    else:
        weights = xp.zeros(n_genes, dtype=xp.int64)
        block_size = MAX_SPARSE_SPLIT_SAMPLES // SPARSE_SPLIT_SAMPLE_BLOCKS
        last_start = indices.size - block_size
        for block_index in range(SPARSE_SPLIT_SAMPLE_BLOCKS):
            start = last_start * block_index // (SPARSE_SPLIT_SAMPLE_BLOCKS - 1)
            weights += xp.bincount(
                indices[start : start + block_size], minlength=n_genes
            )
    if xp is cp:
        weights = cp.asnumpy(weights)
    return weights.astype(np.int64, copy=False)


def _split_gene_ranges(
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

    if isinstance(X, np.ndarray | cp.ndarray) or dense_fallback:
        weights = np.ones(n_genes, dtype=np.int64)
    elif X.format == "csc":
        if cpsp.issparse(X):
            with cp.cuda.Device(X.data.device.id):
                weights = cp.asnumpy(cp.diff(X.indptr)).astype(np.int64, copy=False)
        else:
            weights = np.diff(X.indptr).astype(np.int64, copy=False)
        weights += MIN_SPARSE_GENE_WORK
    else:
        indices = X.indices
        if cpsp.issparse(X):
            with cp.cuda.Device(X.data.device.id):
                weights = _csr_column_weights(indices, n_genes, cp)
        else:
            weights = _csr_column_weights(indices, n_genes, np)
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
        msg = f"Inconsistent sharded Wilcoxon statistic: {name}."
        raise RuntimeError(msg)
    return np.concatenate(arrays, axis=1)


@cache
def _enable_peer_access(device_id: int, source_device: int) -> bool:
    """Enable destination-to-source peer access."""
    with cp.cuda.Device(device_id):
        if not cp.cuda.runtime.deviceCanAccessPeer(device_id, source_device):
            return False
        try:
            cp.cuda.runtime.deviceEnablePeerAccess(source_device)
        except cp.cuda.runtime.CUDARuntimeError as error:
            if error.status == CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED:
                return True
            if error.status in {
                CUDA_ERROR_PEER_ACCESS_UNSUPPORTED,
                CUDA_ERROR_TOO_MANY_PEERS,
            }:
                return False
            raise
    return True


def _copy_gpu_array_to_device(array: cp.ndarray, device_id: int) -> cp.ndarray:
    if array.device.id == device_id:
        return array
    source_device = array.device.id
    with cp.cuda.Device(device_id):
        copied = cp.empty_like(array)
        if array.nbytes == 0:
            return copied
        if _enable_peer_access(device_id, source_device):
            # Do not hide peer-copy errors; they may report earlier async failures.
            cp.cuda.runtime.memcpyPeer(
                copied.data.ptr,
                device_id,
                array.data.ptr,
                source_device,
                array.nbytes,
            )
            return copied

    with cp.cuda.Device(array.device.id):
        host = array.get()
    with cp.cuda.Device(device_id):
        copied.set(host)
    return copied


def _device_sparse_from_arrays(X, data, indices, indptr, shape):
    result = type(X)((data, indices, indptr), shape=shape, copy=False)
    # CuPyX may narrow int64 metadata; restore the validated source dtype.
    result.data = data
    result.indices = indices
    result.indptr = indptr
    return result


def _copy_device_sparse_to_device(X, device_id: int):
    if X.data.device.id == device_id:
        return X
    data = _copy_gpu_array_to_device(X.data, device_id)
    indices = _copy_gpu_array_to_device(X.indices, device_id)
    indptr = _copy_gpu_array_to_device(X.indptr, device_id)
    with cp.cuda.Device(device_id):
        return _device_sparse_from_arrays(X, data, indices, indptr, X.shape)


def _device_csc_column_shard(X, start: int, stop: int):
    with cp.cuda.Device(X.data.device.id):
        ptr_start = int(X.indptr[start].item())
        ptr_stop = int(X.indptr[stop].item())
        indptr = X.indptr[start : stop + 1] - ptr_start
        return _device_sparse_from_arrays(
            X,
            X.data[ptr_start:ptr_stop],
            X.indices[ptr_start:ptr_stop],
            indptr,
            (X.shape[0], stop - start),
        )


def _device_csr_column_shard(X, start: int, stop: int):
    with cp.cuda.Device(X.data.device.id):
        stream = cp.cuda.get_current_stream()
        local_indptr = cp.empty_like(X.indptr)
        local_nnz = _wcs.csr_column_range_indptr_device(
            X.indices,
            X.indptr,
            local_indptr,
            col_start=start,
            col_stop=stop,
            stream=stream.ptr,
        )
        local_data = cp.empty(local_nnz, dtype=X.data.dtype)
        local_indices = cp.empty(local_nnz, dtype=X.indices.dtype)
        _wcs.csr_column_range_gather_device(
            X.data,
            X.indices,
            X.indptr,
            local_indptr,
            local_data,
            local_indices,
            col_start=start,
            col_stop=stop,
            stream=stream.ptr,
        )
        stream.synchronize()
        return _device_sparse_from_arrays(
            X,
            local_data,
            local_indices,
            local_indptr,
            (X.shape[0], stop - start),
        )


def _device_column_shard(X, start: int, stop: int, device_id: int):
    source_device = X.device.id if isinstance(X, cp.ndarray) else X.data.device.id
    if start == 0 and stop == X.shape[1] and device_id == source_device:
        return X

    with cp.cuda.Device(source_device):
        if isinstance(X, cp.ndarray):
            # Pack columns contiguously for the dense binding and peer copy.
            local = cp.asfortranarray(X[:, start:stop])
        elif X.format == "csc":
            local = _device_csc_column_shard(X, start, stop)
        else:
            local = _device_csr_column_shard(X, start, stop)

    result = (
        _copy_gpu_array_to_device(local, device_id)
        if isinstance(local, cp.ndarray)
        else _copy_device_sparse_to_device(local, device_id)
    )
    if device_id != source_device:
        # Finish remote use before releasing source-side shard buffers.
        with cp.cuda.Device(device_id):
            cp.cuda.runtime.deviceSynchronize()
    return result


def _prepare_device_column_shards(X, ranges, device_ids):
    source_device = X.device.id if isinstance(X, cp.ndarray) else X.data.device.id
    if (
        len(device_ids) == 1
        and device_ids[0] == source_device
        and ranges[0] == (0, X.shape[1])
    ):
        return [X]
    with cp.cuda.Device(source_device):
        cp.cuda.get_current_stream().synchronize()

    # Build remote shards first to limit concurrent source-side buffers.
    order = sorted(
        range(len(device_ids)), key=lambda index: device_ids[index] == source_device
    )
    shards = [None] * len(device_ids)
    for index in order:
        start, stop = ranges[index]
        shards[index] = _device_column_shard(X, start, stop, device_ids[index])
    # Finish source-stream packing before worker threads consume the shards.
    with cp.cuda.Device(source_device):
        cp.cuda.get_current_stream().synchronize()
    return shards


def _concat_gpu_shards(arrays: list[cp.ndarray], device_id: int) -> cp.ndarray:
    if len(arrays) == 1:
        return arrays[0]
    local_arrays = [_copy_gpu_array_to_device(array, device_id) for array in arrays]
    result = cp.concatenate(local_arrays, axis=1)
    # Keep peer-copy buffers alive until concatenation finishes.
    cp.cuda.runtime.deviceSynchronize()
    return result


def _run_sharded_wilcoxon(
    rg: _RankGenes,
    *,
    tie_correct: bool,
    use_continuity: bool,
    chunk_size: int | None,
    multi_gpu: bool | list[int] | str | None,
    return_u_values: bool,
    shard_runner: Callable[..., _WilcoxonResult | None],
) -> _WilcoxonResult | None:
    """Run Wilcoxon across one or more GPU shards."""
    X = rg.X
    n_cells, n_total_genes = X.shape
    is_device_input = isinstance(X, cp.ndarray) or cpsp.issparse(X)
    source_device = (
        X.device.id
        if isinstance(X, cp.ndarray)
        else X.data.device.id
        if cpsp.issparse(X)
        else None
    )
    auto_single_device = multi_gpu is None and is_device_input and rg.ireference is None
    if multi_gpu is False or auto_single_device:
        device_ids = [
            source_device if source_device is not None else cp.cuda.Device().id
        ]
    else:
        device_ids = list(dict.fromkeys(parse_device_ids(multi_gpu=multi_gpu)))
        device_ids.sort(key=lambda device_id: device_id != source_device)
    ranges = _split_gene_ranges(
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
    if is_device_input:
        kernel_inputs = _prepare_device_column_shards(X, ranges, device_ids)
    elif dense_source is not None:
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
        column_range = None if is_device_input else ranges[shard_index]
        if not is_device_input:
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
            # Synchronize ranker streams before gathering or releasing inputs.
            cp.cuda.runtime.deviceSynchronize()
        return worker, result

    register_dense = dense_source is not None and rg.ireference is None
    with _shared_dense_host_registration(dense_source, enabled=register_dense):
        if len(device_ids) == 1:
            shards = [run_shard(0)]
        else:
            with ThreadPoolExecutor(max_workers=len(device_ids)) as executor:
                shards = list(executor.map(run_shard, range(len(device_ids))))

    workers = [worker for worker, _ in shards]

    rg.means = _concat_shard_stat(workers, "means")
    rg.vars = None
    rg.pts = _concat_shard_stat(workers, "pts")
    rg.means_rest = _concat_shard_stat(workers, "means_rest")
    rg.vars_rest = None
    rg.pts_rest = _concat_shard_stat(workers, "pts_rest")
    for worker in workers:
        worker.X = None
    kernel_inputs.clear()

    gpu_results = [result for _, result in shards]
    if all(result is None for result in gpu_results):
        return None
    if any(result is None for result in gpu_results):
        msg = "Inconsistent sharded Wilcoxon GPU result state."
        raise RuntimeError(msg)
    complete_gpu_results = [result for result in gpu_results if result is not None]
    group_indices = complete_gpu_results[0][0]
    if any(
        not np.array_equal(result[0], group_indices)
        for result in complete_gpu_results[1:]
    ):
        msg = "Inconsistent sharded Wilcoxon group ordering."
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
    return group_indices, scores, pvals, logfoldchanges
