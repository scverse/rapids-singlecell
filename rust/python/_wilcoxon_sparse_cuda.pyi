"""Sparse-native host Wilcoxon CUDA kernels"""

from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

def _set_host_worker_limit(limit: int) -> int: ...
@overload
def csr_row_boundaries_host(
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_col_cuts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_boundaries: Annotated[NDArray[np.int32], {"shape": (None, None), "order": "C"}],
    *,
    n_cols: int,
) -> None: ...
@overload
def csr_row_boundaries_host(
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_col_cuts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_boundaries: Annotated[NDArray[np.int64], {"shape": (None, None), "order": "C"}],
    *,
    n_cols: int,
) -> None: ...
@overload
def csr_column_range_indptr_device(
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_indptr: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> int: ...
@overload
def csr_column_range_indptr_device(
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_indptr: Annotated[NDArray[np.int64], {"order": "C", "device": "cuda"}],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> int: ...
@overload
def csr_column_range_indptr_device(
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> int: ...
@overload
def csr_column_range_indptr_device(
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> int: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    local_indices: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    local_indices: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    local_indices: Annotated[NDArray[np.int64], {"order": "C", "device": "cuda"}],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    local_data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    local_indices: Annotated[NDArray[np.int64], {"order": "C", "device": "cuda"}],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    local_indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    local_indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    local_indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def csr_column_range_gather_device(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    local_data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    local_indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    col_start: int,
    col_stop: int,
    stream: int = 0,
) -> None: ...
@overload
def ovr_sparse_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_codes: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    group_sizes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csc_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csr_device(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    ref_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_rows: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    grp_offsets: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csc_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_map: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_stats_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_ref: int,
    n_all_grp: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_total_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovr_sparse_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_indptr: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_group_codes: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_group_sizes: Annotated[NDArray[np.float64], {"shape": (None,)}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_total_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    compute_totals: bool = False,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_tie_corr: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    d_group_nnz: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float32], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
@overload
def ovo_streaming_csr_host(
    h_data: Annotated[NDArray[np.float64], {"shape": (None,), "writable": False}],
    h_indices: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_starts: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_row_stops: Annotated[NDArray[np.int64], {"shape": (None,), "writable": False}],
    h_ref_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_row_ids: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    h_grp_offsets: Annotated[NDArray[np.int32], {"shape": (None,), "writable": False}],
    d_rank_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_tie_corr: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_sums: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    d_group_nnz: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    *,
    n_cols: int,
    compute_tie_corr: bool,
    compute_nnz: bool = True,
    col_start: int = 0,
    col_stop: int = -1,
    sub_batch_cols: int = 64,
    analytic_zeros: bool = False,
) -> None: ...
