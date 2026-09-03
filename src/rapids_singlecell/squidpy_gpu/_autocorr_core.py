"""Shared GPU engine for Moran's I and Geary's C.

Both statistics reduce to the cross term ``sum_i x_i * sum_j w_ij x_j`` plus
per-gene invariants (see ``kernels_autocorr.cuh``). Numerators are accumulated
in float64 and cast to the input dtype at the end. Permutation tests pass a row
permutation of the weights to the kernel instead of materialising ``W[perm]``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import cupy as cp
from cupyx.scipy import sparse

from rapids_singlecell._cuda import _autocorr_cuda as _ac
from rapids_singlecell._utils import _copy_to_device, parse_device_ids

from ._utils import _check_precision_issues


def _copy_csr_to_device(matrix: sparse.csr_matrix, device_id: int) -> sparse.csr_matrix:
    if matrix.data.device.id == device_id:
        return matrix
    with cp.cuda.Device(device_id):
        return sparse.csr_matrix(
            (
                _copy_to_device(matrix.data, device_id),
                _copy_to_device(matrix.indices, device_id),
                _copy_to_device(matrix.indptr, device_id),
            ),
            shape=matrix.shape,
        )


def _launch_numerator(
    data,
    adj: sparse.csr_matrix,
    means: cp.ndarray,
    num: cp.ndarray,
    *,
    geary: bool,
    perm: cp.ndarray | None = None,
    den: cp.ndarray | None = None,
) -> None:
    n_samples, n_features = data.shape
    stream = cp.cuda.get_current_stream().ptr
    if sparse.isspmatrix_csr(data):
        _ac.autocorr_sparse(
            adj.indptr,
            adj.indices,
            adj.data,
            data_row_ptr=data.indptr,
            data_col_ind=data.indices,
            data_values=data.data,
            means=means,
            perm=perm,
            num=num,
            n_samples=n_samples,
            n_features=n_features,
            geary=geary,
            stream=stream,
        )
    else:
        _ac.autocorr_dense(
            data,
            means,
            adj.indptr,
            adj.indices,
            adj.data,
            perm=perm,
            num=num,
            den=den,
            geary=geary,
            stream=stream,
        )


def _autocorr_cupy(
    data: cp.ndarray | sparse.csr_matrix,
    adj_matrix_cupy: sparse.csr_matrix,
    n_permutations: int | None = 100,
    *,
    geary: bool,
    multi_gpu: bool | list[int] | str | None = None,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    is_sparse = sparse.isspmatrix_csr(data)
    if not is_sparse and not isinstance(data, cp.ndarray):
        raise ValueError("Datatype not supported")
    device_ids = parse_device_ids(multi_gpu=multi_gpu)
    n_samples, n_features = data.shape
    dtype = data.dtype
    if is_sparse and not data.has_canonical_format:
        data.sum_duplicates()  # kernels binary-search sorted column indices

    s0 = adj_matrix_cupy.data.sum(dtype=cp.float64)
    num = cp.zeros(n_features, dtype=cp.float64)
    if is_sparse:
        colsum_w = cp.asarray(adj_matrix_cupy.sum(axis=0), dtype=cp.float64).ravel()
        sum_x = cp.zeros(n_features, dtype=cp.float64)
        sum_x2 = cp.zeros(n_features, dtype=cp.float64)
        tail = cp.zeros(n_features, dtype=cp.float64)
        _ac.autocorr_sparse_stats(
            data.indptr,
            data.indices,
            data.data,
            colsum_w=colsum_w,
            sum_x=sum_x,
            sum_x2=sum_x2,
            tail=tail,
            geary=geary,
            stream=cp.cuda.get_current_stream().ptr,
        )
        means = sum_x / n_samples
        sq_dev = sum_x2 - n_samples * means**2
        invariant = tail if geary else means**2 * s0 - means * tail
        _launch_numerator(data, adj_matrix_cupy, means, num, geary=geary)
    else:
        means = data.mean(axis=0, dtype=cp.float64)
        sq_dev = cp.zeros(n_features, dtype=cp.float64)
        invariant = None
        _launch_numerator(data, adj_matrix_cupy, means, num, geary=geary, den=sq_dev)

    if geary:
        den = 2.0 * s0 * sq_dev
        scale = float(n_samples - 1)
    else:
        den = sq_dev
        scale = 1.0

    score = _finish(num, invariant, den, scale).astype(dtype)
    _check_precision_issues(score, dtype)

    perms = None
    if n_permutations:
        perms = _run_permutations(
            data,
            adj_matrix_cupy,
            means,
            invariant=invariant,
            den=den,
            scale=scale,
            geary=geary,
            n_permutations=n_permutations,
            device_ids=device_ids,
        ).astype(dtype)
    return score, perms


def _finish(num, invariant, den, scale):
    if invariant is not None:
        num = num + invariant
    return scale * num / den


def _run_permutations(
    data,
    adj: sparse.csr_matrix,
    means: cp.ndarray,
    *,
    invariant: cp.ndarray | None,
    den: cp.ndarray,
    scale: float,
    geary: bool,
    n_permutations: int,
    device_ids: list[int],
) -> cp.ndarray:
    """Row-permute the weights ``n_permutations`` times, sharded over devices.

    One host thread per device: ``cp.random.permutation`` synchronises its
    device, so a single thread would serialise the devices.
    """
    n_samples, n_features = data.shape
    is_sparse = sparse.isspmatrix_csr(data)
    source_device = data.data.device.id if is_sparse else data.device.id
    n_devices = len(device_ids)
    perms_per_device = (n_permutations + n_devices - 1) // n_devices

    def run_device(device_id: int) -> cp.ndarray:
        with cp.cuda.Device(device_id):
            dev_data = (
                _copy_csr_to_device(data, device_id)
                if is_sparse
                else _copy_to_device(data, device_id)
            )
            dev_adj = _copy_csr_to_device(adj, device_id)
            dev_means = _copy_to_device(means, device_id)
            dev_inv = (
                _copy_to_device(invariant, device_id) if invariant is not None else None
            )
            dev_den = _copy_to_device(den, device_id)
            nums = cp.zeros((perms_per_device, n_features), dtype=cp.float64)
            for p in range(perms_per_device):
                perm = cp.random.permutation(n_samples).astype(cp.int32)
                _launch_numerator(
                    dev_data, dev_adj, dev_means, nums[p], geary=geary, perm=perm
                )
            result = _finish(nums, dev_inv, dev_den, scale)
            cp.cuda.runtime.deviceSynchronize()
        return result

    if n_devices == 1:
        results = [run_device(device_ids[0])]
    else:
        with ThreadPoolExecutor(max_workers=n_devices) as executor:
            results = list(executor.map(run_device, device_ids))
    with cp.cuda.Device(source_device):
        parts = [_copy_to_device(result, source_device) for result in results]
        return cp.concatenate(parts, axis=0)[:n_permutations]
