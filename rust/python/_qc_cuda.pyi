from typing import Annotated, overload

import numpy as np
from numpy.typing import NDArray

@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc(
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
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr(
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
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csc_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_csr_sub(
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
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_cells: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    sums_genes: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    sums_genes: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    sums_genes: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    sums_genes: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    cell_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    gene_ex: Annotated[NDArray[np.int32], {"order": "C", "device": "cuda_managed"}],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_sub(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[NDArray[np.float32], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_sub(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda", "writable": False}
    ],
    *,
    sums_cells: Annotated[NDArray[np.float64], {"order": "C", "device": "cuda"}],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda", "writable": False}
    ],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_sub(
    data: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float32], {"order": "C", "device": "cuda_managed"}
    ],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
@overload
def sparse_qc_dense_sub(
    data: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    *,
    sums_cells: Annotated[
        NDArray[np.float64], {"order": "C", "device": "cuda_managed"}
    ],
    mask: Annotated[
        NDArray[np.bool_], {"order": "C", "device": "cuda_managed", "writable": False}
    ],
    n_cells: int,
    n_genes: int,
    stream: int = 0,
) -> None: ...
