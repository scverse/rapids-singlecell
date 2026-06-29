"""
Linear operators for implicit mean-centering of sparse matrices.

These operators avoid materializing the dense centered matrix by computing
the centering on-the-fly during matrix-vector products.
"""

from __future__ import annotations

import cupy as cp
from cupyx.scipy.sparse.linalg import LinearOperator


def mean_centered_operator(
    X, mean: cp.ndarray, row_offset: cp.ndarray | None = None
) -> LinearOperator:
    """
    Create a linear operator for a centered sparse matrix.

    Computes products with ``A = X - 1*mean.T - row_offset*1.T`` without forming
    the dense matrix. The optional per-row offset (constant across features, e.g.
    the CLR per-cell centering term) is a rank-1 correction along the all-ones
    feature direction, handled exactly like the column-mean term.

    Parameters
    ----------
    X
        Sparse matrix in CSR format.
    mean
        Column means of shape (n_features,).
    row_offset
        Optional per-row offset of shape (n_samples,). If `None`, the operator is
        plain mean-centering.

    Returns
    -------
    LinearOperator
        Operator that computes the centered matrix-vector products.
    """
    n_samples, n_features = X.shape
    XT = X.T  # CSC view - no copy

    if row_offset is None:

        def matvec(v):
            return X.dot(v) - cp.dot(mean, v)

        def rmatvec(v):
            return XT.dot(v) - mean * cp.sum(v)

        def matmat(V):
            return X.dot(V) - cp.dot(mean, V)[cp.newaxis, :]

        def rmatmat(V):
            return XT.dot(V) - cp.outer(mean, cp.sum(V, axis=0))
    else:
        r = row_offset

        def matvec(v):
            return X.dot(v) - r * cp.sum(v) - cp.dot(mean, v)

        def rmatvec(v):
            return XT.dot(v) - cp.dot(r, v) - mean * cp.sum(v)

        def matmat(V):
            return (
                X.dot(V)
                - cp.outer(r, cp.sum(V, axis=0))
                - cp.dot(mean, V)[cp.newaxis, :]
            )

        def rmatmat(V):
            return (
                XT.dot(V)
                - cp.dot(r, V)[cp.newaxis, :]
                - cp.outer(mean, cp.sum(V, axis=0))
            )

    return LinearOperator(
        shape=(n_samples, n_features),
        matvec=matvec,
        rmatvec=rmatvec,
        matmat=matmat,
        rmatmat=rmatmat,
        dtype=X.dtype,
    )
