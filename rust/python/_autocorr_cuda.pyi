"""Rust/cuda-oxide GPU entry points; streams and allocations are borrowed."""

from cupy import ndarray

def morans_dense(
    data_centered: ndarray,
    *,
    adj_row_ptr: ndarray,
    adj_col_ind: ndarray,
    adj_data: ndarray,
    num: ndarray,
    n_samples: int,
    n_features: int,
    stream: int = 0,
) -> None: ...
def morans_sparse(
    adj_row_ptr: ndarray,
    adj_col_ind: ndarray,
    adj_data: ndarray,
    *,
    data_row_ptr: ndarray,
    data_col_ind: ndarray,
    data_values: ndarray,
    n_samples: int,
    n_features: int,
    mean_array: ndarray,
    num: ndarray,
    stream: int = 0,
) -> None: ...
def gearys_dense(
    data: ndarray,
    *,
    adj_row_ptr: ndarray,
    adj_col_ind: ndarray,
    adj_data: ndarray,
    num: ndarray,
    n_samples: int,
    n_features: int,
    stream: int = 0,
) -> None: ...
def gearys_sparse(
    adj_row_ptr: ndarray,
    adj_col_ind: ndarray,
    adj_data: ndarray,
    *,
    data_row_ptr: ndarray,
    data_col_ind: ndarray,
    data_values: ndarray,
    n_samples: int,
    n_features: int,
    num: ndarray,
    stream: int = 0,
) -> None: ...
def pre_den_sparse(
    data_col_ind: ndarray,
    data_values: ndarray,
    *,
    nnz: int,
    mean_array: ndarray,
    den: ndarray,
    counter: ndarray,
    stream: int = 0,
) -> None: ...
