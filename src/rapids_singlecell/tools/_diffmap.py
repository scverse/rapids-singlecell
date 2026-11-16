from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cp_sparse
from cupyx.scipy.sparse import linalg
from scipy.sparse import issparse
from scverse_misc import deprecated_arg

from rapids_singlecell._keys import _embedding_keys
from rapids_singlecell._settings import Default
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
)
from rapids_singlecell.get._out import DEPR_KEY_ADDED, DEPR_NEIGHBORS_KEY, _out_key

from ._utils import _neighbors

if TYPE_CHECKING:
    from anndata import AnnData

    from rapids_singlecell.get._utils import GraphAcc, MultiAcc


def _load_connectivities(adata: AnnData, use=None, neighbors_key: str | None = None):
    """Load the connectivities `use` (or the deprecated `neighbors_key`) points to as CuPy CSR or dense array."""
    connectivities = _neighbors(adata, use, neighbors_key)["connectivities"]
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
    rng: np.random.Generator,
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
    rng
        Random generator seeding the Lanczos start vector.

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
        # Setting the random initial vector, like scanpy
        v0 = cp.asarray(rng.standard_normal(matrix.shape[0]), dtype=cp.float64)
        evals, evecs = linalg.eigsh(matrix, k=n_comps, which=which, v0=v0)
        evals, evecs = evals.astype(cp.float32), evecs.astype(cp.float32)

    if sort == "decrease":
        evals = evals[::-1]
        evecs = evecs[:, ::-1]

    return evals, evecs


@_accepts_legacy_random_state(0)
@deprecated_arg("neighbors_key", DEPR_NEIGHBORS_KEY)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
def diffmap(
    adata: AnnData,
    n_comps: int = 15,
    *,
    sort: Literal["decrease", "increase"] = "decrease",
    density_normalize: bool = True,
    rng: SeedLike | RNGLike | None = None,
    use: GraphAcc | str | None = None,
    out: MultiAcc | str | bool = False,
    # deprecated
    neighbors_key: str | None = None,
    key_added: str | None = None,
) -> np.ndarray | None:
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
    sort
        Leave as is for the same behavior as :func:`scanpy.tl.diffmap`.
    density_normalize
        Leave as is for the same behavior as :func:`scanpy.tl.diffmap`.
    rng
        Random seed or :class:`~numpy.random.Generator` for reproducibility.
        The superseded `random_state` argument is still accepted.
    use
        The neighborhood graph (connectivities) to use, e.g. `A.obsp['k']`.
        Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsp.k'`.
        If `None`, the one :func:`~rapids_singlecell.pp.neighbors` wrote by default.
    out
        Where to write the embedding, e.g. `A.obsm['dm']`; the eigenvalues go to `.uns['dm']['evals']`.
        If `True`, the embedding is returned instead.
        If `False`, it is written to the default places below.

    Returns
    -------
    The embedding if `out=True`, else `None`. By default sets (see `out` for other keys):

    ``adata.obsm['X_diffmap']`` : :class:`numpy.ndarray`
        Diffusion map representation of data, which is the right eigen basis of
        the transition matrix with eigenvectors as columns.
    ``adata.uns['diffmap_evals']`` : :class:`numpy.ndarray`
        Array of size (number of eigen vectors).
        Eigenvalues of transition matrix.
    """
    rng = np.random.default_rng(rng)
    key = _out_key(out, key_added, kind="obsm", default=Default())
    connectivities = _load_connectivities(adata, use, neighbors_key)
    transitions_sym, _Z = _compute_transitions(connectivities, density_normalize=density_normalize)
    evals, evecs = _compute_eigen(transitions_sym, n_comps=n_comps, sort=sort, rng=rng)

    if key is True:
        return evecs.get()
    keys = _embedding_keys("diffmap", key)
    # the scanpy 1 key holds the bare evals, the others a dict
    adata.uns[keys.uns] = evals.get() if keys.uns == "diffmap_evals" else {"evals": evals.get()}
    adata.obsm[keys.obsm] = evecs.get()
    return None
