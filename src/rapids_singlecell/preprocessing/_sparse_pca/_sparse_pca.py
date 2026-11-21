from __future__ import annotations

import os

import cupy as cp
import dask
from cupyx.scipy.sparse import csr_matrix

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
)
from rapids_singlecell._cuda import _spca_cuda as _spca
from rapids_singlecell.preprocessing._utils import _get_mean_var

from ._helper import _canonical_csr, _check_matrix_for_zero_genes, _compute_cov, _copy_gram, _csr_mean
from ._ordered import _needs_centered_gram, _ordered_gram
from ._tiled import _centered_gram, _centered_tiles


class CovariancePCA:
    def __init__(self, n_components: int | None, *, zero_center: bool = True) -> None:
        self.n_components = n_components
        self.zero_center = zero_center
        self._project_centered_tiles = False

    def fit_transform(self, x: cp.ndarray | csr_matrix | DaskArray):
        if isinstance(x, csr_matrix):
            x = _canonical_csr(x)
        elif isinstance(x, cp.ndarray):
            if x.ndim != 2 or min(x.shape) < 2:
                raise ValueError("Dense PCA requires at least two rows and two features.")
            if self.n_components is not None and not 1 <= self.n_components <= x.shape[1]:
                raise ValueError("n_components must be between one and the number of features.")
            if x.dtype.kind not in "biuf":
                raise ValueError("Dense PCA requires real numeric input.")
            if x.dtype not in (cp.float32, cp.float64):
                x = x.astype(cp.float32)
        self._project_centered_tiles = False
        self.n_components_ = min(x.shape) if self.n_components is None else self.n_components
        if isinstance(x, DaskArray):
            covariance, self.mean_ = _cov_dask(x, self.zero_center)
        else:
            covariance = self._covariance(x)

        values, vectors = cp.linalg.eigh(covariance, UPLO="U")
        self.components_ = vectors.T[::-1][: self.n_components_].copy(order="C")
        del covariance, vectors
        self.explained_variance_ = values[::-1][: self.n_components_]
        self.explained_variance_ratio_ = self.explained_variance_ / values[::-1].sum()
        if not self.zero_center:
            self.explained_variance_ = self.explained_variance_ / (x.shape[0] - 1)
        if isinstance(x, cp.ndarray):
            pivot = cp.abs(self.components_).argmax(axis=1)
            self.components_ *= cp.where(self.components_[cp.arange(len(pivot)), pivot] < 0, -1, 1)[:, None]
        scores = self._transform_dask(x) if isinstance(x, DaskArray) else self._transform_cupy(x)
        self.components_ = self.components_.get()
        self.explained_variance_ = self.explained_variance_.get()
        self.explained_variance_ratio_ = self.explained_variance_ratio_.get()
        return scores

    def _covariance(self, x):
        n_rows, n_cols = x.shape
        sparse = isinstance(x, csr_matrix)
        if sparse:
            _check_matrix_for_zero_genes(x)
            if not self.zero_center:
                self.mean_ = None
                return _ordered_gram(x)
            self.mean_ = _csr_mean(x)
            # A sparse product visits stored pairs and avoids expanding mostly
            # empty tiles. Native double-precision products also favor CSR.
            if 6 * x.nnz < n_rows * n_cols or (
                x.dtype == cp.float64 and os.environ.get("CUBLAS_EMULATE_DOUBLE_PRECISION") != "1"
            ):
                gram = _ordered_gram(x)
                gram *= 1 / n_rows
                if not _needs_centered_gram(gram, self.mean_):
                    mean = self.mean_.astype(x.dtype, copy=False)
                    return _compute_cov(gram, gram, mean)
                del gram
        else:
            self.mean_ = x.mean(axis=0, dtype=cp.float64)
            if not cp.all(cp.isfinite(self.mean_)):
                raise ValueError("Dense PCA requires finite input.")
        self._project_centered_tiles = True
        gram = _centered_gram(x, self.mean_)
        gram /= n_rows if sparse else n_rows - 1
        return gram.astype(x.dtype, copy=False)

    def _transform_cupy(self, X):
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError("PCA input and loadings have different numbers of features")
        if self._project_centered_tiles:
            scores = cp.empty((X.shape[0], len(self.components_)), dtype=X.dtype)
            for start, tile in _centered_tiles(X, self.mean_):
                cp.matmul(tile, self.components_.T, out=scores[start : start + len(tile)])
            return scores.get()
        dtype = cp.result_type(X.dtype, self.components_.dtype)
        weights = cp.ascontiguousarray(self.components_.T, dtype=dtype)
        pcs = weights.shape[1]
        offset = self.mean_.astype(dtype, copy=False) @ weights if self.zero_center else cp.zeros(pcs, dtype=dtype)
        scores = cp.empty((X.shape[0], pcs), dtype=dtype)
        _spca.pca_project_csr(
            X.indptr,
            X.indices,
            X.data.astype(dtype, copy=False),
            weights,
            offset,
            scores,
            cp.cuda.get_current_stream().ptr,
        )
        return scores.get()

    def _transform_dask(self, X: DaskArray) -> DaskArray:
        if self.zero_center:

            def _transform(X_part, mean_, components_):
                pre_mean = mean_ @ components_.T
                mean_impact = cp.ones((X_part.shape[0], 1), dtype=X_part.dtype) @ pre_mean.reshape(1, -1)
                X_transformed = X_part.dot(components_.T) - mean_impact
                return X_transformed
        else:

            def _transform(X_part, mean_, components_):
                X_transformed = X_part.dot(components_.T)
                return X_transformed

        X_pca = X.map_blocks(
            _transform,
            mean_=self.mean_,
            components_=self.components_,
            dtype=X.dtype,
            chunks=(X.chunks[0], self.n_components_),
            meta=_meta_dense(X.dtype),
        )

        return X_pca


def _cov_dask(x, zero_center):
    n_cols = x.shape[1]
    if isinstance(x._meta, csr_matrix):
        # Gram matrix for CSR matrix
        def __gram_block(x_part):
            gram_matrix = cp.zeros((n_cols, n_cols), dtype=x.dtype)

            _spca.gram_csr_upper(
                x_part.indptr,
                x_part.indices,
                x_part.data,
                nrows=x_part.shape[0],
                ncols=n_cols,
                out=gram_matrix,
                stream=cp.cuda.get_current_stream().ptr,
            )
            return gram_matrix[None, ...]  # need new axis for summing
    else:
        # Gram matrix for DaskArray of CuPy NDArray
        def __gram_block(x_part):
            gram_matrix = x_part.T.dot(x_part)
            return gram_matrix[None, ...]

    n_blocks = x.blocks.size
    gram_matrix = x.map_blocks(
        __gram_block,
        new_axis=(1,),
        chunks=((1,) * n_blocks, (x.shape[1],), (x.shape[1],)),
        meta=cp.array([]),
        dtype=x.dtype,
    ).sum(axis=0)
    if zero_center:
        mean, _ = _get_mean_var(x, axis=0)
        gram_matrix, mean = dask.compute(gram_matrix, mean)
        mean = mean.astype(x.dtype)
    else:
        gram_matrix, mean = gram_matrix.compute(), None
    if isinstance(x._meta, csr_matrix):
        gram_matrix = _copy_gram(gram_matrix, n_cols)
    if zero_center:
        gram_matrix *= 1 / x.shape[0]
        gram_matrix = _compute_cov(gram_matrix, gram_matrix, mean)
    return gram_matrix, mean
