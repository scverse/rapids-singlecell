from __future__ import annotations

from types import SimpleNamespace

import cupy as cp
from cupyx.scipy.sparse import isspmatrix_csr

from rapids_singlecell._cuda import _sparse2dense_cuda

from ._helper import _check_matrix_for_zero_genes


def _run_tiled_pca(X, n_comps):
    """Form a centered covariance matrix and project the data in bounded row tiles."""
    n_rows, n_cols = X.shape
    sparse = isspmatrix_csr(X)
    if sparse:
        if not X.has_canonical_format:
            X = X.copy()
            X.sum_duplicates()
        _check_matrix_for_zero_genes(X)
        max_nnz = int(cp.diff(X.indptr).max())

    tile_rows = min(4096, max(1, (64 << 20) // (n_cols * X.dtype.itemsize)))

    def tiles(center=None):
        for start in range(0, n_rows, tile_rows):
            stop = min(start + tile_rows, n_rows)
            if sparse:
                tile = cp.zeros((stop - start, n_cols), dtype=X.dtype)
                _sparse2dense_cuda.sparse2dense(
                    X.indptr[start : stop + 1],
                    X.indices,
                    X.data,
                    out=tile,
                    major=stop - start,
                    minor=n_cols,
                    c_switch=True,
                    max_nnz=max_nnz,
                    stream=cp.cuda.get_current_stream().ptr,
                )
            else:
                tile = X[start:stop]
            if center is not None:
                out = tile if sparse else cp.empty((stop - start, n_cols), dtype=X.dtype)
                tile = cp.subtract(tile, center, out=out)
            yield start, tile

    if sparse:
        mean = cp.zeros(n_cols, dtype=cp.float64)
        for _, tile in tiles():
            mean += tile.sum(axis=0, dtype=cp.float64)
        mean /= n_rows
    else:
        mean = X.mean(axis=0, dtype=cp.float64)

    gram = cp.zeros((n_cols, n_cols), dtype=X.dtype)
    for _, tile in tiles(mean):
        gram += tile.T @ tile
    gram /= n_rows if sparse else n_rows - 1
    values, vectors = cp.linalg.eigh(gram, UPLO="U")
    components = cp.ascontiguousarray(vectors[:, -n_comps:][:, ::-1].T)
    if not sparse:
        # Match cuML's orientation of each component by its largest absolute loading.
        pivot = cp.abs(components).argmax(axis=1)
        components *= cp.where(components[cp.arange(n_comps), pivot] < 0, -1, 1)[:, None]
    scores = cp.empty((n_rows, n_comps), dtype=X.dtype)
    for start, tile in tiles(mean):
        cp.matmul(tile, components.T, out=scores[start : start + len(tile)])

    variance = values[-n_comps:][::-1]
    model = SimpleNamespace(
        components_=components.get(),
        explained_variance_=variance.get(),
        explained_variance_ratio_=(variance / values.sum()).get(),
    )
    return model, scores.get()
