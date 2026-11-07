from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import cuml.internals.logger as logger
import cupy as cp
import numpy as np
from cuml.manifold.umap import find_ab_params, simplicial_set_embedding
from cupyx.scipy import sparse
from scanpy.tools._utils import get_init_pos_from_paga
from scipy import sparse as sc_sparse
from scverse_misc import deprecated_arg

from rapids_singlecell._compat import _rng_kwargs
from rapids_singlecell._keys import _embedding_keys
from rapids_singlecell._settings import Default
from rapids_singlecell._utils import _get_logger_level
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _legacy_random_state,
    _LegacyRng,
)
from rapids_singlecell.get._out import DEPR_KEY_ADDED, DEPR_NEIGHBORS_KEY, _out_key, deprecated_copy

from ._utils import _neighbors, _validate_init_pos

if TYPE_CHECKING:
    from anndata import AnnData

    from rapids_singlecell.get._utils import GraphAcc, MultiAcc

_InitPos = Literal["auto", "spectral", "random", "paga"]

_H2D_CHUNK = 1 << 28  # indices per host-to-device copy


def _device_coo(graph) -> sparse.coo_matrix:
    """Canonical device COO, expanded on the GPU for a host CSR."""
    if not (sc_sparse.issparse(graph) and graph.format == "csr" and graph.has_canonical_format):
        return sparse.coo_matrix(graph)
    row = cp.zeros(graph.nnz, dtype=cp.int32)
    starts = cp.asarray(graph.indptr[1:-1])
    cp.add.at(row, starts[starts < graph.nnz], 1)
    cp.cumsum(row, out=row)
    col = cp.empty(graph.nnz, dtype=cp.int32)
    for start in range(0, graph.nnz, _H2D_CHUNK):
        col[start : start + _H2D_CHUNK] = cp.asarray(graph.indices[start : start + _H2D_CHUNK])
    data = cp.asarray(graph.data, dtype=cp.float32)
    coo = sparse.coo_matrix((data, (row, col)), shape=graph.shape)
    coo.has_canonical_format = True
    return coo


@_accepts_legacy_random_state(0)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_arg("neighbors_key", DEPR_NEIGHBORS_KEY)
@deprecated_copy
def umap(
    adata: AnnData,
    *,
    min_dist: float = 0.5,
    spread: float = 1.0,
    n_components: int = 2,
    maxiter: int | None = None,
    alpha: float = 1.0,
    negative_sample_rate: int = 5,
    init_pos: _InitPos | np.ndarray | cp.ndarray | str | None = "auto",
    rng: SeedLike | RNGLike | None = None,
    a: float | None = None,
    b: float | None = None,
    use: GraphAcc | str | None = None,
    rep: MultiAcc | str | None = None,
    out: MultiAcc | str | bool = False,
    # deprecated
    key_added: str | None = None,
    neighbors_key: str | None = None,
    copy: bool = False,
) -> AnnData | np.ndarray | None:
    r"""Embed the neighborhood graph using UMAP :cite:p:`McInnes2018` :cite:p:`Nolet2021`.

    UMAP (Uniform Manifold Approximation and Projection) is a manifold learning
    technique suitable for visualizing high-dimensional data. Besides tending to
    be faster than tSNE, it optimizes the embedding such that it best reflects
    the topology of the data, which we represent throughout rapids-singlecell using a
    neighborhood graph. tSNE, by contrast, optimizes the distribution of
    nearest-neighbor distances in the embedding such that these best match the
    distribution of distances in the high-dimensional space.

    Parameters
    ----------
    adata
        Annotated data matrix.
    min_dist
        The effective minimum distance between embedded points. Smaller values
        will result in a more clustered/clumped embedding where nearby points on
        the manifold are drawn closer together, while larger values will result
        on a more even dispersal of points. The value should be set relative to
        the ``spread`` value, which determines the scale at which embedded
        points will be spread out.
    spread
        The effective scale of embedded points. In combination with `min_dist`
        this determines how clustered/clumped the embedded points are.
    n_components
        The number of dimensions of the embedding.
    maxiter
        The number of iterations (epochs) of the optimization. Called `n_epochs`
        in the original UMAP. Defaults to 500 for up to 10,000 cells, else 200.
    alpha
        The initial learning rate for the embedding optimization.
    negative_sample_rate
        The number of negative edge/1-simplex samples to use per positive
        edge/1-simplex sample in optimizing the low dimensional embedding.
    init_pos
        How to initialize the low dimensional embedding. Called `init` in the
        original UMAP. Options are:

            * 'auto': chooses 'spectral' for `'n_samples' < 1000000`, 'random' otherwise.
            * 'spectral': use a spectral embedding of the graph.
            * 'random': assign initial embedding positions at random.
            * 'paga': use the :func:`~scanpy.tl.paga` layout as initial embedding positions.
            * Array of shape (n_obs, 2)
            * Any key for :attr:`~anndata.AnnData.obsm`

        .. note::
            If your embedding looks odd it's recommended setting `init_pos` to 'random'.

    rng
        Random seed or :class:`~numpy.random.Generator` used by the random
        number generator.
        The superseded `random_state` argument is still accepted.
        `rng=None` runs unseeded, which is faster but not reproducible.
    a
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    b
        More specific parameters controlling the embedding. If `None` these
        values are set automatically as determined by `min_dist` and
        `spread`.
    use
        The neighborhood graph (connectivities) to embed, e.g. `A.obsp['k']` after
        `rsc.pp.neighbors(adata, out=A.obsp['k'])`.
        Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsp.k'`.
        If `None`, the one :func:`~rapids_singlecell.pp.neighbors` wrote by default.
    rep
        Accepted for compatibility with :func:`scanpy.tl.umap`, which lays out the graph's
        connected components with it. cuML lays them out from the graph alone, so it is not read.
    out
        Where to write the embedding, e.g. `A.obsm['umap2']`; the parameters go to `.uns` under the same key.
        If `True`, the embedding is returned instead.
        If `False`, it is written to `.obsm['X_umap']` and `.uns['umap']`
        (`'umap'` for both if ``rapids_singlecell.settings.preset`` is `scanpy-v2-preview`).

    Returns
    -------
    The embedding if `out=True`, else `None`. By default sets (see `out` for other keys):

        `adata.obsm['X_umap']` : :class:`~numpy.ndarray` (dtype `float`)
            UMAP coordinates of data.
        `adata.uns['umap']['params']` : :class:`dict`
            UMAP parameters `a`, `b`, and `random_state` (if specified).
    """
    # a seed forces cuML's slower deterministic optimizer
    unseeded = rng is None or (isinstance(rng, _LegacyRng) and rng.arg is None)
    rng = np.random.default_rng(rng)

    key = _out_key(out, key_added, kind="obsm", default=Default())
    neighbors = _neighbors(adata, use, neighbors_key)
    if a is None or b is None:
        a, b = find_ab_params(spread, min_dist)

    # store params for adata.uns
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}
    stored_params = {"a": a, "b": b, **meta_random_state}

    n_obs = adata.shape[0]
    # like scanpy and umap-learn: fewer epochs suffice for larger datasets
    n_epochs = (500 if n_obs <= 10_000 else 200) if maxiter is None else maxiter

    match init_pos:
        case str() if init_pos in adata.obsm:
            init_coords = adata.obsm[init_pos]
        case str() if init_pos == "paga":
            init_coords = get_init_pos_from_paga(
                adata,
                **_rng_kwargs(get_init_pos_from_paga, rng),
                adjacency=neighbors["connectivities"],
            )
        case str() if init_pos == "auto":
            init_coords = "spectral" if n_obs < 1000000 else "random"
        case _:
            init_coords = init_pos

    if hasattr(init_coords, "dtype"):
        init_coords = _validate_init_pos(init_coords)
        if init_coords.shape[1] != n_components:
            raise ValueError(f"Expected {n_components} columns but got {init_coords.shape[1]} columns.")

    logger_level = _get_logger_level(logger)
    X_umap = simplicial_set_embedding(
        # `data` is only used for its number of rows: the layout is optimized
        # from `graph` alone, so we pass a placeholder instead of materializing
        # the representation on the GPU.
        data=cp.zeros((n_obs, 1), dtype=cp.float32),
        graph=_device_coo(neighbors["connectivities"]),
        n_components=n_components,
        initial_alpha=alpha,
        a=a,
        b=b,
        negative_sample_rate=negative_sample_rate,
        n_epochs=n_epochs,
        init=init_coords,
        random_state=None if unseeded else _legacy_random_state(rng, always_state=True),
    )
    logger.set_level(logger_level)
    X_umap = cp.asarray(X_umap).get()
    if key is True:
        return X_umap
    keys = _embedding_keys("umap", key)
    adata.obsm[keys.obsm] = X_umap
    adata.uns[keys.uns] = {"params": stored_params}
    return None
