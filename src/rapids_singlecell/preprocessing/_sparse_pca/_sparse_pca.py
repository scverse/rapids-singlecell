# This file will be removed in Q3 2025 when in favor of the CUML implementation

from __future__ import annotations

from typing import Literal, Self, overload

import cupy as cp
import dask
from cupyx.scipy.sparse import csr_matrix, spmatrix

from rapids_singlecell._compat import (
    DaskArray,
    _meta_dense,
)
from rapids_singlecell._cuda import _spca_cuda as _spca
from rapids_singlecell.preprocessing._utils import _get_mean_var

from ._helper import _canonical_csr, _check_matrix_for_zero_genes, _compute_cov, _copy_gram, _csr_mean


class PCA_sparse:
    def __init__(self, n_components: int | None, *, zero_center: bool = True) -> None:
        self.n_components = n_components
        self.zero_center = zero_center

    def fit(self, x: spmatrix | DaskArray) -> Self:
        if self.n_components is None:
            n_rows = x.shape[0]
            n_cols = x.shape[1]
            self.n_components_ = min(n_rows, n_cols)
        else:
            self.n_components_ = self.n_components
        if not isinstance(x, DaskArray):
            _check_matrix_for_zero_genes(x)
        self.n_samples_ = x.shape[0]
        self.n_features_in_ = x.shape[1] if x.ndim == 2 else 1
        self.dtype = x.dtype

        if self.zero_center:
            covariance, self.mean_ = _cov_sparse(x)
        else:
            # For truncated SVD (uncentered), operate on the Gram matrix (X^T X)
            # We don't subtract the mean in this path
            covariance = _cov_sparse(x, return_gram=True)
            self.mean_ = None

        self.explained_variance_, self.components_ = cp.linalg.eigh(covariance, UPLO="U")

        # NOTE: We reverse the eigen vector and eigen values here
        # because cupy provides them in ascending order. Make a copy otherwise
        # it is not C_CONTIGUOUS anymore and would error when converting to
        # CumlArray
        self.explained_variance_ = self.explained_variance_[::-1]

        self.components_ = cp.flip(self.components_, axis=1)

        self.components_ = self.components_.T[: self.n_components_, :]

        self.explained_variance_ratio_ = self.explained_variance_ / cp.sum(self.explained_variance_)

        self.explained_variance_ = self.explained_variance_[: self.n_components_]
        if not self.zero_center:
            # Gram eigenvalues are S**2; report S**2 / (n - 1) like PCA_sparse_svd
            self.explained_variance_ = self.explained_variance_ / (x.shape[0] - 1)

        self.explained_variance_ratio_ = self.explained_variance_ratio_[: self.n_components_]

        return self

    def transform(self, X: spmatrix | DaskArray) -> spmatrix | DaskArray:
        if isinstance(X, DaskArray):
            X_pca = self._transform_dask(X)
        else:
            X_pca = self._transform_cupy(X)

        self.components_ = self.components_.get()
        self.explained_variance_ = self.explained_variance_.get()
        self.explained_variance_ratio_ = self.explained_variance_ratio_.get()
        return X_pca

    def _transform_cupy(self, X: spmatrix) -> spmatrix:
        if X.shape[1] != self.components_.shape[1]:
            raise ValueError("PCA input and loadings have different numbers of features")
        X = _canonical_csr(X.tocsr())
        dtype = cp.result_type(X.dtype, self.components_.dtype)
        weights = cp.ascontiguousarray(self.components_.T, dtype=dtype)
        pcs = weights.shape[1]
        offset = self.mean_ @ weights if self.zero_center else cp.zeros(pcs, dtype=dtype)
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

    def fit_transform(self, X, y=None):
        if isinstance(X, csr_matrix):
            X = _canonical_csr(X)
        return self.fit(X).transform(X)


@overload
def _cov_sparse(x: spmatrix | DaskArray, *, return_gram: Literal[False] = False) -> tuple[cp.ndarray, cp.ndarray]: ...
@overload
def _cov_sparse(x: spmatrix | DaskArray, *, return_gram: Literal[True]) -> cp.ndarray: ...


def _cov_sparse(x: spmatrix | DaskArray, *, return_gram: bool = False) -> cp.ndarray | tuple[cp.ndarray, cp.ndarray]:
    """Computes the mean and the covariance of matrix X of the form Cov(X, X) = E(XX) - E(X)E(X)

    This is a temporary fix for
    cuml issue #5475 and cupy issue #7699,
    where the operation `x.T.dot(x)` did not work for
    larger sparse matrices.

    Parameters
    ----------
    x : cupyx.scipy.sparse of size (m, n)
    return_gram : boolean (default = False)
        If True, gram matrix of the form (1 / n) * X.T.dot(X)
        will be returned.
        When True, a copy will be created
        to store the results of the covariance.
        When False, the local gram matrix result
        will be overwritten
    return_mean: boolean (default = False)
        If True, the Maximum Likelihood Estimate used to
        calculate the mean of X and X will be returned,
        of the form (1 / n) * mean(X) and (1 / n) * mean(X)

    Returns
    -------
    result : cov(X, X) when return_gram and return_mean are False
            cov(X, X), gram(X, X) when return_gram is True,
            return_mean is False
            cov(X, X), mean(X), mean(X) when return_gram is False,
            return_mean is True
            cov(X, X), gram(X, X), mean(X), mean(X)
            when return_gram is True and return_mean is True
    """
    if isinstance(x, csr_matrix):
        x = _canonical_csr(x)
    gram_matrix = _create_gram_matrix(x)

    if return_gram:
        if isinstance(x, DaskArray):
            gram_matrix = gram_matrix.compute()
        if _check_csr_meta(x) or isinstance(x, csr_matrix):
            return _copy_gram(gram_matrix, x.shape[1])
        else:
            return gram_matrix
    else:
        if isinstance(x, DaskArray):
            mean_x, _ = _get_mean_var(x, axis=0)
            gram_matrix, mean_x = dask.compute(gram_matrix, mean_x)
        else:
            mean_x = _csr_mean(x)
        if _check_csr_meta(x) or isinstance(x, csr_matrix):
            gram_matrix = _copy_gram(gram_matrix, x.shape[1])

        mean_x = mean_x.astype(x.dtype)
        gram_matrix *= 1 / x.shape[0]

        cov_result = gram_matrix
        cov_result = _compute_cov(cov_result, gram_matrix, mean_x)
        return cov_result, mean_x


def _create_gram_matrix(x):
    if isinstance(x, csr_matrix):
        from ._ordered import _ordered_gram

        gram_matrix = _ordered_gram(x)
    elif isinstance(x, DaskArray):
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
    else:
        raise ValueError(f"Unsupported matrix type: {type(x)}")
    return gram_matrix


def _check_csr_meta(x):
    # Check if the meta of the DaskArray is a CSR matrix
    if isinstance(x, DaskArray):
        return isinstance(x._meta, csr_matrix)
    else:
        return False
