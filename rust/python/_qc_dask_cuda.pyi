from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def sparse_qc_csr_cells(
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
    sums_cells: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_cells(
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
    sums_cells: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_cells(
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
    sums_cells: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_cells(
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
    sums_cells: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_cells(
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
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_cells(
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
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_cells(
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
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_cells(
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
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
    index: Annotated[
        NDArray[np.int32], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
    index: Annotated[
        NDArray[np.int64], {"order": "C", "device": "cuda", "writable": False}
    ],
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
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
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
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
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
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
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_genes(
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
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    nnz: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_cells(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_cells(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_cells(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_cells(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_genes(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_genes(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_genes(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_genes(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
