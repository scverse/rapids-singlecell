from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
from cupyx.scipy import sparse

from rapids_singlecell._cuda import _autocorr_cuda as _ac
from rapids_singlecell._utils import parse_device_ids

from ._utils import _check_precision_issues

if TYPE_CHECKING:
    from cupyx.scipy.sparse import csr_matrix


def _morans_I_cupy_dense(
    data: cp.ndarray,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
    device_ids: list[int] | None = None,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    n_samples, n_features = data.shape
    dtype = data.dtype
    data_centered_cupy = data - data.mean(axis=0)

    # Calculate the numerator and denominator for Moran's I
    num = cp.zeros(n_features, dtype=dtype)
    stream = cp.cuda.get_current_stream().ptr
    _ac.morans_dense(
        data_centered_cupy,
        adj_row_ptr=adj_matrix_cupy.indptr,
        adj_col_ind=adj_matrix_cupy.indices,
        adj_data=adj_matrix_cupy.data,
        num=num,
        n_samples=n_samples,
        n_features=n_features,
        stream=stream,
    )

    # Calculate the denominator for Moran's I
    den = cp.sum(data_centered_cupy**2, axis=0)

    # Calculate Moran's I
    morans_I = num / den

    # Check for numerical issues before expensive permutations
    _check_precision_issues(morans_I, dtype)

    # Calculate p-values using permutation tests
    if n_permutations:
        morans_I_permutations = _run_permutations_dense(
            data_centered_cupy,
            adj_matrix_cupy,
            den,
            n_permutations=n_permutations,
            n_samples=n_samples,
            n_features=n_features,
            dtype=dtype,
            device_ids=device_ids,
        )
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _run_permutations_dense(
    data_centered_cupy: cp.ndarray,
    adj_matrix_cupy: csr_matrix,
    den: cp.ndarray,
    *,
    n_permutations: int,
    n_samples: int,
    n_features: int,
    dtype: cp.dtype,
    device_ids: list[int] | None = None,
) -> cp.ndarray:
    """Run permutation tests for Moran's I (dense data) with multi-GPU support."""
    if device_ids is None:
        device_ids = [0]

    n_devices = len(device_ids)
    streams: dict[int, cp.cuda.Stream] = {}
    device_data: list[dict] = []

    # Each device runs perms_per_device iterations
    perms_per_device = (n_permutations + n_devices - 1) // n_devices

    # Phase 1: Create streams and transfer data to all devices
    for device_id in device_ids:
        with cp.cuda.Device(device_id):
            streams[device_id] = cp.cuda.Stream(non_blocking=True)

            with streams[device_id]:
                # Copy data to this device
                if device_id == device_ids[0]:
                    dev_data = data_centered_cupy
                    dev_adj = adj_matrix_cupy
                    dev_den = den
                else:
                    dev_data = cp.asarray(data_centered_cupy)
                    dev_adj = sparse.csr_matrix(
                        (
                            cp.asarray(adj_matrix_cupy.data),
                            cp.asarray(adj_matrix_cupy.indices),
                            cp.asarray(adj_matrix_cupy.indptr),
                        ),
                        shape=adj_matrix_cupy.shape,
                    )
                    dev_den = cp.asarray(den)

                # Allocate output array for this device
                dev_perms = cp.zeros((perms_per_device, n_features), dtype=dtype)

                device_data.append(
                    {
                        "data": dev_data,
                        "adj": dev_adj,
                        "den": dev_den,
                        "perms": dev_perms,
                        "device_id": device_id,
                    }
                )

    # Phase 2: Dispatch each iteration to all GPUs in parallel
    for p in range(perms_per_device):
        for dd in device_data:
            device_id = dd["device_id"]
            with cp.cuda.Device(device_id):
                streams[device_id].synchronize()

                num_permuted = cp.zeros(n_features, dtype=dtype)

                idx_shuffle = cp.random.permutation(dd["adj"].shape[0])
                adj_matrix_permuted = dd["adj"][idx_shuffle, :]
                _ac.morans_dense(
                    dd["data"],
                    adj_row_ptr=adj_matrix_permuted.indptr,
                    adj_col_ind=adj_matrix_permuted.indices,
                    adj_data=adj_matrix_permuted.data,
                    num=num_permuted,
                    n_samples=n_samples,
                    n_features=n_features,
                    stream=cp.cuda.get_current_stream().ptr,
                )
                dd["perms"][p, :] = num_permuted / dd["den"]

        # Sync all devices after each iteration
        for dd in device_data:
            with cp.cuda.Device(dd["device_id"]):
                streams[dd["device_id"]].synchronize()

    # Phase 3: Gather results on first device and cut to exact size
    with cp.cuda.Device(device_ids[0]):
        all_perms = [cp.asarray(dd["perms"]) for dd in device_data]
        morans_I_permutations = cp.concatenate(all_perms, axis=0)[:n_permutations]

    return morans_I_permutations


def _morans_I_cupy_sparse(
    data: csr_matrix,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
    device_ids: list[int] | None = None,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    n_samples, n_features = data.shape
    dtype = data.dtype

    # Calculate the numerator for Moran's I
    num = cp.zeros(n_features, dtype=dtype)
    means = data.mean(axis=0).ravel()
    means = means.astype(dtype)
    stream = cp.cuda.get_current_stream().ptr
    _ac.morans_sparse(
        adj_matrix_cupy.indptr,
        adj_matrix_cupy.indices,
        adj_matrix_cupy.data,
        data_row_ptr=data.indptr,
        data_col_ind=data.indices,
        data_values=data.data,
        n_samples=n_samples,
        n_features=n_features,
        mean_array=means,
        num=num,
        stream=stream,
    )

    # Calculate the denominator for Moran's I
    den = cp.zeros(n_features, dtype=dtype)
    counter = cp.zeros(n_features, dtype=cp.int32)
    _ac.pre_den_sparse(
        data.indices,
        data.data,
        nnz=data.nnz,
        mean_array=means,
        den=den,
        counter=counter,
        stream=stream,
    )
    counter = n_samples - counter
    den += counter * means**2

    # Calculate Moran's I
    morans_I = num / den

    # Check for numerical issues before expensive permutations
    _check_precision_issues(morans_I, dtype)

    if n_permutations:
        morans_I_permutations = _run_permutations_sparse(
            data,
            adj_matrix_cupy,
            means,
            den,
            n_permutations=n_permutations,
            n_samples=n_samples,
            n_features=n_features,
            dtype=dtype,
            device_ids=device_ids,
        )
    else:
        morans_I_permutations = None
    return morans_I, morans_I_permutations


def _run_permutations_sparse(
    data: csr_matrix,
    adj_matrix_cupy: csr_matrix,
    means: cp.ndarray,
    den: cp.ndarray,
    *,
    n_permutations: int,
    n_samples: int,
    n_features: int,
    dtype: cp.dtype,
    device_ids: list[int] | None = None,
) -> cp.ndarray:
    """Run permutation tests for Moran's I (sparse data) with multi-GPU support."""
    if device_ids is None:
        device_ids = [0]

    n_devices = len(device_ids)
    streams: dict[int, cp.cuda.Stream] = {}
    device_data: list[dict] = []

    # Each device runs perms_per_device iterations
    perms_per_device = (n_permutations + n_devices - 1) // n_devices

    # Phase 1: Create streams and transfer data to all devices
    for device_id in device_ids:
        with cp.cuda.Device(device_id):
            streams[device_id] = cp.cuda.Stream(non_blocking=True)

            with streams[device_id]:
                # Copy data to this device
                if device_id == device_ids[0]:
                    dev_data = data
                    dev_adj = adj_matrix_cupy
                    dev_means = means
                    dev_den = den
                else:
                    dev_data = sparse.csr_matrix(
                        (
                            cp.asarray(data.data),
                            cp.asarray(data.indices),
                            cp.asarray(data.indptr),
                        ),
                        shape=data.shape,
                    )
                    dev_adj = sparse.csr_matrix(
                        (
                            cp.asarray(adj_matrix_cupy.data),
                            cp.asarray(adj_matrix_cupy.indices),
                            cp.asarray(adj_matrix_cupy.indptr),
                        ),
                        shape=adj_matrix_cupy.shape,
                    )
                    dev_means = cp.asarray(means)
                    dev_den = cp.asarray(den)

                # Allocate output array for this device
                dev_perms = cp.zeros((perms_per_device, n_features), dtype=dtype)

                device_data.append(
                    {
                        "data": dev_data,
                        "adj": dev_adj,
                        "means": dev_means,
                        "den": dev_den,
                        "perms": dev_perms,
                        "device_id": device_id,
                    }
                )

    # Phase 2: Launch all permutations on each device
    for p in range(perms_per_device):
        for dd in device_data:
            device_id = dd["device_id"]
            with cp.cuda.Device(device_id):
                streams[device_id].synchronize()

                num_permuted = cp.zeros(n_features, dtype=dtype)

                idx_shuffle = cp.random.permutation(dd["adj"].shape[0])
                adj_matrix_permuted = dd["adj"][idx_shuffle, :]
                _ac.morans_sparse(
                    adj_matrix_permuted.indptr,
                    adj_matrix_permuted.indices,
                    adj_matrix_permuted.data,
                    data_row_ptr=dd["data"].indptr,
                    data_col_ind=dd["data"].indices,
                    data_values=dd["data"].data,
                    n_samples=n_samples,
                    n_features=n_features,
                    mean_array=dd["means"],
                    num=num_permuted,
                    stream=cp.cuda.get_current_stream().ptr,
                )
                dd["perms"][p, :] = num_permuted / dd["den"]

        # Sync all devices after each iteration
        for dd in device_data:
            with cp.cuda.Device(dd["device_id"]):
                streams[dd["device_id"]].synchronize()

    # Phase 3: Gather results on first device and cut to exact size
    with cp.cuda.Device(device_ids[0]):
        all_perms = [cp.asarray(dd["perms"]) for dd in device_data]
        morans_I_permutations = cp.concatenate(all_perms, axis=0)[:n_permutations]

    return morans_I_permutations


def _morans_I_cupy(
    data: cp.ndarray | csr_matrix,
    adj_matrix_cupy: csr_matrix,
    n_permutations: int | None = 100,
    *,
    multi_gpu: bool | list[int] | str | None = None,
) -> tuple[cp.ndarray, cp.ndarray | None]:
    device_ids = parse_device_ids(multi_gpu=multi_gpu)
    if sparse.isspmatrix_csr(data):
        return _morans_I_cupy_sparse(data, adj_matrix_cupy, n_permutations, device_ids)
    elif isinstance(data, cp.ndarray):
        return _morans_I_cupy_dense(data, adj_matrix_cupy, n_permutations, device_ids)
    else:
        raise ValueError("Datatype not supported")
