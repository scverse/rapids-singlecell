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
    ) -> None:
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.zero_center = zero_center
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.random_state = random_state

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
        else:
            self.mean_ = None

        # Create operator (centered or raw)
        if self.zero_center:
            X_op = mean_centered_operator(X, self.mean_)
        else:
            X_op = X

        # Run SVD with the selected solver
        U, S, Vt = self._run_svd(X_op)

        # Store results
        self.components_ = Vt
        self.explained_variance_ = (S**2) / (self.n_samples_ - 1)

        # Compute total variance for variance ratio
        if self.zero_center:
            _, var_x = _get_mean_var(X, axis=0)
            total_variance = cp.sum(var_x)
        else:
            if hasattr(X, "data"):
                total_variance = cp.sum(X.data**2) / (self.n_samples_ - 1)
            else:
                total_variance = cp.sum(X**2) / (self.n_samples_ - 1)

        self.explained_variance_ratio_ = self.explained_variance_ / total_variance

        return self

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
            return X_transformed
        else:
            return X.dot(self.components_.T)

    def fit_transform(self, X: spmatrix, y=None) -> cp.ndarray:
        """Fit the model and apply dimensionality reduction."""
        return self.fit(X).transform(X)
