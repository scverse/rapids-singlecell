from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_norm_res_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_norm_res(
    X: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_norm_res(
    X: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_norm_res(
    X: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_norm_res(
    X: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    sums_cells: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    sums_cells: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    sums_cells: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    sums_cells: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_sum_csc(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def csc_hvg_res(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_hvg_res(
    data: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_hvg_res(
    data: Annotated[
        NDArray[np.float64], {"order": "F", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_hvg_res(
    data: Annotated[
        NDArray[np.float32], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def dense_hvg_res(
    data: Annotated[
        NDArray[np.float64], {"order": "F", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    residuals: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    inv_sum_total: float,
    clip: float,
    inv_theta: float,
    n_genes: int,
    n_cells: int,
    stream: int = 0,
) -> None: ...
