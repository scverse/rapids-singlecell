"""
PCA for sparse matrices using SVD solvers.

Provides a unified interface for GPU-accelerated sparse PCA using
Lanczos bidiagonalization or randomized SVD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp

from rapids_singlecell.preprocessing._utils import _get_mean_var

from ._block_lanczos import randomized_svd
from ._operators import mean_centered_operator
from ._svd_lanczos import lanczos_svd

if TYPE_CHECKING:
    from typing import Self

    from cupyx.scipy.sparse import spmatrix

SVDSolver = Literal["lanczos", "randomized"]


class PCA_sparse_svd:
    """
    PCA for sparse matrices using SVD solvers.

    Unified interface for GPU-accelerated sparse PCA with multiple SVD backends.

    Parameters
    ----------
    n_components
        Number of principal components to compute.
    svd_solver
        SVD algorithm to use:

        - ``'lanczos'``: Lanczos bidiagonalization with implicit restarts.
          Most accurate, best when high precision is needed.
        - ``'randomized'``: Randomized SVD with GPU-optimized CholeskyQR2
          orthogonalization (Tomás et al. 2024). Fast approximate method.

    zero_center
        If True, compute standard PCA (mean-centered).
        If False, compute truncated SVD (uncentered).
    n_oversamples
        Extra random vectors for randomized method.
        Higher values improve accuracy. Default is 10.
    n_iter
        Number of power iterations for randomized SVD. Higher values improve
        accuracy for matrices with slowly decaying singular values. Default is 2.
    random_state
        Random state for reproducibility.
    """

    def __init__(
        self,
        n_components: int | None,
        *,
        svd_solver: SVDSolver = "lanczos",
        zero_center: bool = True,
        n_oversamples: int = 10,
        n_iter: int | None = None,
        random_state: int | None = 0,
        offset: cp.ndarray | None = None,
    ) -> None:
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.zero_center = zero_center
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state
        # Optional per-cell offset (CLR centering term); applied via the operator
        # and the transform as a rank-1 composite, never densified.
        self.offset_ = offset

    def fit(self, X: spmatrix) -> Self:
        """
        Fit the PCA model.

        Parameters
        ----------
        X
            Sparse matrix of shape (n_samples, n_features).

        Returns
        -------
        self
        """
        from ._helper import _check_matrix_for_zero_genes

        if self.n_components is None:
            n_rows = X.shape[0]
            n_cols = X.shape[1]
            self.n_components_ = min(n_rows, n_cols)
        else:
            self.n_components_ = self.n_components

        _check_matrix_for_zero_genes(X)
        self.n_samples_ = X.shape[0]
        self.n_features_in_ = X.shape[1] if X.ndim == 2 else 1
        self.dtype = X.dtype

        # Compute mean if zero-centering
        if self.zero_center:
            self.mean_, _ = _get_mean_var(X, axis=0)
            self.mean_ = self.mean_.astype(X.dtype)
            if self.offset_ is not None:
                # Centering M = X - r·1ᵀ shifts every gene mean by mean(r).
                self.offset_ = self.offset_.astype(X.dtype)
                self.mean_ = self.mean_ - self.offset_.mean().astype(X.dtype)
        else:
            self.mean_ = None

        # Create operator (centered or raw)
        if self.zero_center:
            X_op = mean_centered_operator(X, self.mean_, row_offset=self.offset_)
        else:
            X_op = X

        # Run SVD with the selected solver
        U, S, Vt = self._run_svd(X_op)

        # Store results
        self.components_ = Vt
        self.explained_variance_ = (S**2) / (self.n_samples_ - 1)

        # Compute total variance for variance ratio
        if self.zero_center and self.offset_ is not None:
            total_variance = self._total_variance_with_offset(X)
        elif self.zero_center:
            _, var_x = _get_mean_var(X, axis=0)
            total_variance = cp.sum(var_x)
        else:
            if hasattr(X, "data"):
                total_variance = cp.sum(X.data**2) / (self.n_samples_ - 1)
            else:
                total_variance = cp.sum(X**2) / (self.n_samples_ - 1)

        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

    def _total_variance_with_offset(self, X) -> cp.ndarray:
        """Total variance of the centered M = X - r·1ᵀ, without densifying.

        ``‖M‖_F² = ‖X‖_F² - 2·Σ rᵢ·rowsumᵢ(X) + n_features·Σ rᵢ²`` and
        ``total_var = (‖M‖_F² - n·‖mean_M‖²) / (n - 1)`` (``self.mean_`` already
        carries the ``-mean(r)`` shift).
        """
        r = self.offset_
        x_norm2 = cp.sum(X.data**2) if hasattr(X, "data") else cp.sum(X**2)
        row_sums = cp.asarray(X.sum(axis=1)).ravel()
        m_norm2 = x_norm2 - 2.0 * cp.dot(r, row_sums) + X.shape[1] * cp.dot(r, r)
        n = self.n_samples_
        return (m_norm2 - n * cp.dot(self.mean_, self.mean_)) / (n - 1)

    def _run_svd(self, X_op):
        """Run the selected SVD solver."""
        if self.svd_solver == "lanczos":
            return lanczos_svd(
                X_op,
                k=self.n_components_,
                random_state=self.random_state,
            )
        elif self.svd_solver == "randomized":
            n_iter = self.n_iter if self.n_iter is not None else 2
            return randomized_svd(
                X_op,
                k=self.n_components_,
                n_oversamples=self.n_oversamples,
                n_iter=n_iter,
                random_state=self.random_state,
            )
        else:
            raise ValueError(
                f"Unknown svd_solver '{self.svd_solver}'. "
                "Must be one of: 'lanczos', 'randomized'"
            )

    def transform(self, X: spmatrix) -> cp.ndarray:
        """
        Apply dimensionality reduction to X.

        Parameters
        ----------
        X
            Sparse matrix of shape (n_samples, n_features).

        Returns
        -------
        X_new
            Transformed data of shape (n_samples, n_components).
        """
        if self.zero_center and self.mean_ is not None:
            # X_centered @ V.T = X @ V.T - mean @ V.T
            X_transformed = X.dot(self.components_.T)
            mean_projection = cp.dot(self.mean_, self.components_.T)
            X_transformed -= mean_projection
            if self.offset_ is not None:
                # Per-cell offset impact: r ⊗ (V·1), the row-sum of each component.
                v_sum = self.components_.sum(axis=1)
                X_transformed -= self.offset_[:, None] * v_sum[None, :]
            return X_transformed
        else:
            return X.dot(self.components_.T)

    def fit_transform(self, X: spmatrix, y=None) -> cp.ndarray:
        """Fit the model and apply dimensionality reduction."""
        return self.fit(X).transform(X)
