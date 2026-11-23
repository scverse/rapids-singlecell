"""
PCA for sparse matrices using SVD solvers.

Provides a unified interface for GPU-accelerated sparse PCA using
Lanczos bidiagonalization or randomized SVD.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
from cupyx.scipy.sparse import issparse

from ._block_lanczos import randomized_svd
from ._csr_adjoint import _CSRAdjoint
from ._operators import _canonical_csr, _centered_dot, _sum_squares, _transform_center, mean_centered_operator
from ._svd_lanczos import lanczos_svd

if TYPE_CHECKING:
    from typing import Self

    from cupyx.scipy.sparse import spmatrix

    from rapids_singlecell._utils._random import RNGLike, SeedLike

SVDSolver = Literal["lanczos", "randomized"]

_explained_variance = cp.ElementwiseKernel(
    "T singular, int64 dof, float64 total_energy",
    "T variance, R ratio",
    """
    const double component_variance = double(singular) * double(singular) / dof;
    variance = component_variance;
    ratio = component_variance / (total_energy / dof);
    """,
    "svd_explained_variance",
)


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
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.
    """

    def __init__(
        self,
        n_components: int | None,
        *,
        svd_solver: SVDSolver = "lanczos",
        zero_center: bool = True,
        n_oversamples: int = 10,
        n_iter: int | None = None,
        rng: SeedLike | RNGLike | None = None,
    ) -> None:
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.zero_center = zero_center
        self.n_oversamples = n_oversamples
        self.n_iter = n_iter
        self.rng = rng

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
        self._fit(X)
        return self

    def _fit(self, X, *, return_scores=False):
        from ._helper import _check_matrix_for_zero_genes

        if self.n_components is None:
            n_rows = X.shape[0]
            n_cols = X.shape[1]
            self.n_components_ = min(n_rows, n_cols)
        else:
            self.n_components_ = self.n_components

        X = _canonical_csr(X)
        _check_matrix_for_zero_genes(X)
        self.n_samples_ = X.shape[0]
        self.n_features_in_ = X.shape[1] if X.ndim == 2 else 1
        self.dtype = X.dtype

        adjoint = _CSRAdjoint(X)
        self._center = None
        precise_center = None
        if self.zero_center:
            if X.dtype.kind == "c":
                raise TypeError("Centered sparse PCA requires real-valued input.")
            mean, total_energy = adjoint.centered_stats()
            self.mean_ = mean.astype(X.dtype, copy=False)
            self._center = adjoint.center
            precise_center = adjoint.precise_center
        else:
            self.mean_ = None
            total_energy = _sum_squares(X.data)

        X_op = mean_centered_operator(X, self.mean_, adjoint=adjoint, precise_center=precise_center)
        left, S, Vt = self._run_svd(X_op)

        # Store results
        self.components_ = Vt
        self.explained_variance_ = cp.empty_like(S)
        self.explained_variance_ratio_ = cp.empty_like(S, dtype=cp.float64 if self.zero_center else S.dtype)
        _explained_variance(
            S, self.n_samples_ - 1, total_energy, self.explained_variance_, self.explained_variance_ratio_
        )

        if return_scores:
            # Lanczos refinement already computed the projected scores.
            return left if self.svd_solver == "lanczos" else X_op.dot(Vt.T.conj())
        return None

    def _run_svd(self, X_op):
        """Run the selected SVD solver."""
        if self.svd_solver == "lanczos":
            return lanczos_svd(
                X_op,
                k=self.n_components_,
                rng=self.rng,
                _return_scores=True,
            )
        elif self.svd_solver == "randomized":
            n_iter = self.n_iter if self.n_iter is not None else 2
            return randomized_svd(
                X_op,
                k=self.n_components_,
                n_oversamples=self.n_oversamples,
                n_iter=n_iter,
                rng=self.rng,
            )
        else:
            raise ValueError(f"Unknown svd_solver '{self.svd_solver}'. Must be one of: 'lanczos', 'randomized'")

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
        V = self.components_.T.conj()
        X = _canonical_csr(X) if issparse(X) else cp.asarray(X)
        precise_center = _transform_center(X, getattr(self, "_center", None), self.mean_)
        return _centered_dot(X, V, self.mean_, precise_center)

    def fit_transform(self, X: spmatrix, y=None) -> cp.ndarray:
        """Fit the model and apply dimensionality reduction."""
        return self._fit(X, return_scores=True)
