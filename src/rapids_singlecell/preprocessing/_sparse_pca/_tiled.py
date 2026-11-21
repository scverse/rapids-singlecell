from __future__ import annotations

from types import SimpleNamespace

import cupy as cp
from cupy import cublas  # type: ignore[attr-defined]
from cupy._core import _routines_linalg  # type: ignore[attr-defined]
from cupyx.scipy.sparse import isspmatrix_csr

from rapids_singlecell._cuda import _spca_cuda as _spca

from ._helper import _check_matrix_for_zero_genes

_sum_csr_duplicates = cp.ElementwiseKernel(  # type: ignore[attr-defined]
    "int64 start, int64 stop, raw int64 order, raw T data",
    "T total",
    """
    double sum = 0;
    for (long long j = start; j < stop; ++j) sum += data[order[j]];
    total = sum;
    """,
    "pca_sum_csr_duplicates",
)


def _canonical_csr_arrays(X):
    if X.has_canonical_format or not X.nnz:
        return X.indptr, X.indices, X.data
    # Stable sorting keeps repeated entries in input order. CuPy's
    # sum_duplicates uses floating-point atomics for this reduction.
    keys = X.tocoo(copy=False).row.astype(cp.int64) * X.shape[1] + X.indices
    order = cp.argsort(keys)
    bounds = cp.flatnonzero(cp.diff(keys[order], prepend=-1, append=-1))
    starts = bounds[:-1]
    data = _sum_csr_duplicates(starts, bounds[1:], order, X.data)
    indptr = cp.searchsorted(starts, X.indptr).astype(X.indptr.dtype, copy=False)
    return indptr, X.indices[order[starts]], data


def _run_tiled_pca(X, n_comps):
    """Form a centered covariance matrix and project the data in bounded row tiles."""
    n_rows, n_cols = X.shape
    sparse = isspmatrix_csr(X)
    if sparse:
        _check_matrix_for_zero_genes(X)
        indptr, indices, data = _canonical_csr_arrays(X)
        partial = cp.empty((min(1024, n_rows), n_cols), dtype=cp.float64)
        _spca.pca_column_sums(indptr, indices, data, partial, cp.cuda.get_current_stream().ptr)
        mean = partial.sum(axis=0) / n_rows
        del partial
    else:
        mean = X.mean(axis=0, dtype=cp.float64)

    tile_rows = min(4096, max(1, (64 << 20) // (n_cols * X.dtype.itemsize)))
    buffer = cp.empty((min(n_rows, tile_rows), n_cols), dtype=X.dtype)

    def tiles():
        for start in range(0, n_rows, tile_rows):
            stop = min(start + tile_rows, n_rows)
            tile = buffer[: stop - start]
            if sparse:
                _spca.pca_center_csr(
                    indptr[start : stop + 1], indices, data, mean, tile, cp.cuda.get_current_stream().ptr
                )
            else:
                cp.subtract(X[start:stop], mean, out=tile)
            yield start, tile

    tensor_gram = (
        X.dtype == cp.float32
        and int(cp.cuda.Device().compute_capability) >= 80
        and _routines_linalg.get_compute_type(X.dtype) == _routines_linalg.COMPUTE_TYPE_DEFAULT
    )
    if tensor_gram:
        # Two half-precision planes encode the leading bits and scaled residual.
        rows = (len(buffer) + 31) // 32 * 32
        cols = (n_cols + 63) // 64 * 64
        planes = cp.empty((2, rows, cols), dtype=cp.uint16)
        limits = cp.empty((2, n_cols), dtype=cp.uint32)
        gram = cp.zeros((n_cols, n_cols), dtype=cp.float64)
        for _, tile in tiles():
            _spca.gram_tensorcore(tile, planes, limits, gram, cp.cuda.get_current_stream().ptr)
        del planes, limits
    else:
        gram = cp.zeros((n_cols, n_cols), dtype=X.dtype)
        for _, tile in tiles():
            if X.dtype == cp.float64:
                cublas.syrk("T", tile, out=gram, beta=1)
            else:
                gram += tile.T @ tile
    gram /= n_rows if sparse else n_rows - 1
    values, vectors = cp.linalg.eigh(gram.astype(X.dtype, copy=False), UPLO="U")
    components = cp.ascontiguousarray(vectors[:, -n_comps:][:, ::-1].T)
    if not sparse:
        # Match cuML's orientation of each component by its largest absolute loading.
        pivot = cp.abs(components).argmax(axis=1)
        components *= cp.where(components[cp.arange(n_comps), pivot] < 0, -1, 1)[:, None]
    scores = cp.empty((n_rows, n_comps), dtype=X.dtype)
    variance = values[-n_comps:][::-1]
    for start, tile in tiles():
        cp.matmul(tile, components.T, out=scores[start : start + len(tile)])

    model = SimpleNamespace(
        components_=components.get(),
        explained_variance_=variance.get(),
        explained_variance_ratio_=(variance / values.sum()).get(),
    )
    return model, scores.get()
