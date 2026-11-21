from __future__ import annotations

import cupy as cp
from cupy import cublas  # type: ignore[attr-defined]
from cupy._core import _routines_linalg  # type: ignore[attr-defined]
from cupyx.scipy.sparse import isspmatrix_csr

from rapids_singlecell._cuda import _spca_cuda as _spca


def _tile_rows(X):
    return min(X.shape[0], 4096, max(1, (64 << 20) // (X.shape[1] * X.dtype.itemsize)))


def _centered_tiles(X, mean):
    """Center a dense or canonical CSR matrix in a bounded, reusable buffer."""
    n_rows, n_cols = X.shape
    tile_rows = _tile_rows(X)
    buffer = cp.empty((tile_rows, n_cols), dtype=X.dtype)
    for start in range(0, n_rows, tile_rows):
        stop = min(start + tile_rows, n_rows)
        tile = buffer[: stop - start]
        if isspmatrix_csr(X):
            _spca.pca_center_csr(
                X.indptr[start : stop + 1], X.indices, X.data, mean, tile, cp.cuda.get_current_stream().ptr
            )
        else:
            cp.subtract(X[start:stop], mean, out=tile)
        yield start, tile


def _centered_gram(X, mean):
    """Accumulate the upper triangle of the centered Gram matrix in row order."""
    n_cols = X.shape[1]
    # Delegate explicit math settings such as strict FP32 to CuPy's matmul.
    tensor_gram = (
        X.dtype == cp.float32
        and int(cp.cuda.Device().compute_capability) >= 80
        and _routines_linalg.get_compute_type(X.dtype) == _routines_linalg.COMPUTE_TYPE_DEFAULT
    )
    if tensor_gram:
        # Two half-precision planes encode the leading bits and scaled residual.
        rows = (_tile_rows(X) + 31) // 32 * 32
        cols = (n_cols + 63) // 64 * 64
        planes = cp.empty((2, rows, cols), dtype=cp.uint16)
        limits = cp.empty((2, n_cols), dtype=cp.uint32)
        gram = cp.zeros((n_cols, n_cols), dtype=cp.float64)
        for _, tile in _centered_tiles(X, mean):
            _spca.gram_tensorcore(tile, planes, limits, gram, cp.cuda.get_current_stream().ptr)
    else:
        gram = cp.zeros((n_cols, n_cols), dtype=X.dtype)
        for _, tile in _centered_tiles(X, mean):
            if X.dtype == cp.float64:
                cublas.syrk("T", tile, out=gram, beta=1)
            else:
                gram += tile.T @ tile
    return gram
