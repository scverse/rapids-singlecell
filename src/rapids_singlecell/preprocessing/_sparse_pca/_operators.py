"""
Linear operators for implicit mean-centering of sparse matrices.

These operators avoid materializing the dense centered matrix by computing
the centering on-the-fly during matrix-vector products.
"""

from __future__ import annotations

import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator


def mean_centered_operator(X, mean: cp.ndarray) -> LinearOperator:
    """
    Create a linear operator for mean-centered sparse matrix.

    Computes products with (X - 1*mean.T) without forming the dense matrix.

    Parameters
    ----------
    X
        Sparse matrix in CSR format.
    mean
        Column means of shape (n_features,).

    Returns
    -------
    LinearOperator
        Operator that computes mean-centered matrix-vector products.
    """
    n_samples, n_features = X.shape
    XT = X.T  # CSC view - no copy

    def matvec(v):
        return X.dot(v) - cp.dot(mean, v)

    def rmatvec(v):
        return XT.dot(v) - mean * cp.sum(v)

    def matmat(V):
        return X.dot(V) - cp.dot(mean, V)[cp.newaxis, :]

    def rmatmat(V):
        return XT.dot(V) - cp.outer(mean, cp.sum(V, axis=0))

    return LinearOperator(
        shape=(n_samples, n_features),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=X.dtype,
    )
