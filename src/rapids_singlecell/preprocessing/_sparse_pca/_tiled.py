from __future__ import annotations

from types import SimpleNamespace

import cupy as cp
from cupy import cublas  # type: ignore[attr-defined]
from cupy._core import _routines_linalg  # type: ignore[attr-defined]
from cupyx.scipy.sparse import isspmatrix_csr

from rapids_singlecell._cuda import _spca_cuda as _spca

from ._helper import _canonical_csr, _check_matrix_for_zero_genes, _csr_mean


def _run_tiled_pca(X, n_comps):
    """Form a centered covariance matrix and project the data in bounded row tiles."""
    n_rows, n_cols = X.shape
    sparse = isspmatrix_csr(X)
    if sparse:
        _check_matrix_for_zero_genes(X)
        X = _canonical_csr(X)
        mean = _csr_mean(X)
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
                    X.indptr[start : stop + 1], X.indices, X.data, mean, tile, cp.cuda.get_current_stream().ptr
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
