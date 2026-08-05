from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.sparse import linalg
from scipy.sparse import issparse

if TYPE_CHECKING:
    from anndata import AnnData


def _load_connectivities(
    adata: AnnData,
    neighbors_key: str | None = None,
) -> cp_sparse.csr_matrix:
    """Load connectivities from ``adata.obsp`` and convert to CuPy CSR.

    Parameters
    ----------
    adata
        Annotated data matrix.
    neighbors_key
        Key in ``adata.uns`` where neighbor parameters are stored.
        If ``None``, defaults to ``"neighbors"``.

    Returns
    -------
    CuPy CSR matrix of connectivities.
    """
    if neighbors_key is None:
        neighbors_key = "neighbors"

    if neighbors_key not in adata.uns:
        raise ValueError(
            f"No neighbors found in `adata.uns[{neighbors_key!r}]`. "
            "Run `pp.neighbors` first."
        )

    neighbors_dict = adata.uns[neighbors_key]

    if neighbors_key == "neighbors":
        conns_key = "connectivities"
    else:
        conns_key = neighbors_dict["connectivities_key"]

    if conns_key not in adata.obsp:
        raise ValueError(
            f"No connectivities found at `adata.obsp[{conns_key!r}]`. "
            "Run `pp.neighbors` first."
        )

    connectivities = adata.obsp[conns_key]
    if issparse(connectivities):
        return cp_sparse.csr_matrix(connectivities, dtype=cp.float32)
    return cp.asarray(connectivities, dtype=cp.float32)


def _compute_transitions(
    connectivities: cp_sparse.csr_matrix | cp.ndarray,
    *,
    density_normalize: bool = True,
) -> tuple[cp_sparse.csr_matrix | cp.ndarray, cp_sparse.dia_matrix | cp.ndarray]:
    """Compute the symmetrized transition matrix.

    Parameters
    ----------
    connectivities
        Weighted adjacency matrix (CuPy sparse or dense).
    density_normalize
        If ``True``, normalize by sampling density (Coifman & Lafon 2006).

    Returns
    -------
    A tuple ``(transitions_sym, Z)`` where *transitions_sym* is the
    symmetrized transition matrix and *Z* is the row-normalization
    diagonal (``1 / sqrt(row_sum(K))``).
    """
    W = connectivities

    if density_normalize:
        q = cp.asarray(W.sum(axis=0))
        if cp_sparse.issparse(W):
            Q = cp_sparse.spdiags(1.0 / q, 0, W.shape[0], W.shape[0])
        else:
            Q = cp.diag(1.0 / q.ravel())
        K = Q @ W @ Q
    else:
        K = W

    z = cp.sqrt(cp.asarray(K.sum(axis=0)))
    if cp_sparse.issparse(K):
        Z = cp_sparse.spdiags(1.0 / z, 0, K.shape[0], K.shape[0])
    else:
        Z = cp.diag(1.0 / z.ravel())

    transitions_sym = Z @ K @ Z
    return transitions_sym, Z


def _compute_eigen(
    transitions_sym: cp_sparse.csr_matrix | cp.ndarray,
    *,
    n_comps: int = 15,
    sort: Literal["decrease", "increase"] = "decrease",
) -> tuple[cp.ndarray, cp.ndarray]:
    """Eigendecomposition of the transition matrix.

    Parameters
    ----------
    transitions_sym
        Symmetrized transition matrix (CuPy sparse or dense).
    n_comps
        Number of eigenvalues/vectors to compute.
    sort
        Sort eigenvalues in ``"decrease"`` (default) or ``"increase"``
        order.

    Returns
    -------
    A tuple ``(eigen_values, eigen_basis)`` of CuPy arrays.
    """
    if n_comps == 0:
        evals, evecs = linalg.eigsh(transitions_sym)
    else:
        n_comps = min(transitions_sym.shape[0] - 1, n_comps)
        which = "LM" if sort == "decrease" else "SM"
        matrix = transitions_sym.astype(cp.float64)
        evals, evecs = linalg.eigsh(matrix, k=n_comps, which=which)
        evals, evecs = evals.astype(cp.float32), evecs.astype(cp.float32)

    if sort == "decrease":
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

    return evals, evecs


def diffmap(
    adata: AnnData,
    n_comps: int = 15,
    *,
    neighbors_key: str | None = None,
    sort: Literal["decrease", "increase"] = "decrease",
    density_normalize: bool = True,
) -> None:
    """
    Diffusion Maps :cite:p:`Coifman2005,Haghverdi2015`.

    Diffusion maps has been proposed for visualizing single-cell data.
    This is a reimplementation of scanpys function.

    The width ("sigma") of the connectivity kernel is implicitly determined by
    the number of neighbors used to compute the single-cell graph in
    :func:`scanpy.pp.neighbors` or :func:`~rapids_singlecell.pp.neighbors`.

    Parameters
    ----------
    adata
        Annotated data matrix.
    n_comps
        The number of dimensions of the representation.
    neighbors_key
        Key in ``adata.uns`` where neighbor parameters are stored.
        If ``None``, defaults to ``"neighbors"``.
    sort
        Leave as is for the same behavior as :func:`scanpy.tl.diffmap`.
    density_normalize
        Leave as is for the same behavior as :func:`scanpy.tl.diffmap`.

    Returns
    -------
    Updates ``adata`` with the following fields:

    ``adata.obsm['X_diffmap']`` : :class:`numpy.ndarray`
        Diffusion map representation of data, which is the right eigen basis of
        the transition matrix with eigenvectors as columns.
    ``adata.uns['diffmap_evals']`` : :class:`numpy.ndarray`
        Array of size (number of eigen vectors).
        Eigenvalues of transition matrix.
    """
    connectivities = _load_connectivities(adata, neighbors_key)
    transitions_sym, _Z = _compute_transitions(
        connectivities, density_normalize=density_normalize
    )
    evals, evecs = _compute_eigen(transitions_sym, n_comps=n_comps, sort=sort)

    adata.uns["diffmap_evals"] = evals.get()
    adata.obsm["X_diffmap"] = evecs.get()
