from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def mul_dense(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    nrows: int,
    ncols: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_dense(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    nrows: int,
    ncols: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_dense(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    nrows: int,
    ncols: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_dense(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    nrows: int,
    ncols: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    nrows: int,
    target_sum: float,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda_managed"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda_managed"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda_managed"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def find_hi_genes_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_is_hi: Annotated[NDArray[np.bool_], {"order": "C", "device": "cuda_managed"}],
    max_fraction: float,
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    tsum: float,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def masked_sum_major(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    indices: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    gene_mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    sums: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    major: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    scales: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    scales: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    scales: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    scales: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    scales: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    scales: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    scales: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_csr(
    indptr: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    scales: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_dense(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    *,
    scales: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_dense(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    *,
    scales: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_dense(
    data: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda_managed"}],
    *,
    scales: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
@overload
def prescaled_mul_dense(
    data: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda_managed"}],
    *,
    scales: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    nrows: int,
    ncols: int,
    stream: int = 0,
) -> None: ...
