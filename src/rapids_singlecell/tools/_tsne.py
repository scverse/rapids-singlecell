from __future__ import annotations

from typing import TYPE_CHECKING

import cuml.internals.logger as logger
import numpy as np
from cuml.manifold import TSNE
from scverse_misc import deprecated_arg

from rapids_singlecell._keys import _embedding_keys
from rapids_singlecell._settings import Default
from rapids_singlecell._utils import _get_logger_level
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _LegacyRng,
    _seed_from_rng,
)
from rapids_singlecell.get import _get_arr
from rapids_singlecell.get._out import (
    DEPR_KEY_ADDED,
    DEPR_USE,
    _out_key,
    _resolve_obs,
    _use_to_uns,
    deprecated_copy,
)
from rapids_singlecell.get._utils import LayerAcc

from ._utils import _choose_representation

if TYPE_CHECKING:
    import cupy as cp
    from anndata import AnnData

    from rapids_singlecell.get._utils import MultiAcc


@_accepts_legacy_random_state(0)
@deprecated_arg("use_rep", DEPR_USE)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_copy
def tsne(
    adata: AnnData,
    n_pcs: int = None,
    *,
    use: LayerAcc | MultiAcc | str | None = None,
    perplexity: int = 30,
    early_exaggeration: int = 12,
    learning_rate: int = 200,
    method: str = "barnes_hut",
    metric: str = "euclidean",
    rng: SeedLike | RNGLike | None = None,
    out: MultiAcc | str | bool = False,
    # deprecated
    key_added: str | None = None,
    copy: bool = False,
    use_rep: str | None = None,
) -> AnnData | cp.ndarray | None:
    r"""
    t-SNE :cite:p:`vanDerMaaten2008` :cite:p:`Chan2018`.

    Performs t-distributed stochastic neighborhood embedding (tSNE) using cuml library.

    Parameters
    ----------
        adata
            Annotated data matrix.
        n_pcs
            Use this many PCs. If `n_pcs==0` use `.X` if `use_rep is None`.
        use
            The representation to embed, e.g. `A.obsm['pca']`, `A.X` or `A.layers['k']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsm.pca'`.
            If None, the representation is chosen automatically: For .n_vars < 50, .X
            is used, otherwise the PCA embedding, under whichever key the active
            ``rapids_singlecell.settings.preset`` uses. If no PCA embedding is
            present, it's computed with default parameters or `n_pcs` if present.
        perplexity
            The perplexity is related to the number of nearest neighbors that is used
            in other manifold learning algorithms. Larger datasets usually require a larger
            perplexity. Consider selecting a value between 5 and 50. The choice is not extremely
            critical since t-SNE is quite insensitive to this parameter.
        early_exaggeration
            Controls how tight natural clusters in the original space are in the embedded space
            and how much space will be between them. For larger values, the space between natural
            clusters will be larger in the embedded space. Again, the choice of this parameter is
            not very critical. If the cost function increases during initial optimization, the early
            exaggeration factor or the learning rate might be too high.
        learning_rate
            Note that the R-package “Rtsne” and cuML uses a default of 200. The learning rate can be
            a critical parameter. It should be between 100 and 1000. If the cost function increases
            during initial optimization, the early exaggeration factor or the learning rate might
            be too high. If the cost function gets stuck in a bad local minimum increasing the
            learning rate helps sometimes.
        method
            'barnes_hut' and 'fft' are fast approximations. 'exact' is more accurate but slower.
        metric
            Distance metric to use. Supported distances are ['l1, 'cityblock', 'manhattan', 'euclidean',
            'l2', 'sqeuclidean', 'minkowski', 'chebyshev', 'cosine', 'correlation']
        rng
            Random seed or :class:`~numpy.random.Generator` seeding cuML's t-SNE.
            Note that cuML's t-SNE is not fully deterministic, so a fixed seed
            narrows but does not eliminate run-to-run variation.
            The superseded `random_state` argument is still accepted.
        out
            Where to write the embedding, e.g. `A.obsm['k']` (or `'obsm.k'`);
            the parameters then go to `.uns['k']`.
            If `True`, the embedding is returned instead of written.
            If `False`, it is written to `.obsm['X_tsne']` and the parameters to `.uns['tsne']`
            (`'tsne'` for both under the scanpy v2 preset).

    Returns
    -------
        The embedding if `out=True`, else `None`. By default sets the following fields:

            `adata.obsm['X_tsne']` : :class:`~cupy.ndarray` (dtype `float`)
                tSNE coordinates of data.
            `adata.uns['tsne']['params']` : :class:`dict`
                tSNE parameters `perplexity`, `early_exaggeration`,
                `learning_rate`, `method`, `metric`, and `use_rep`.
    """
    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}
    key = _out_key(out, key_added, kind="obsm", default=Default())
    if (use := _resolve_obs(use)) is not None and use_rep is not None:
        raise TypeError("Pass either `use` or `use_rep`, not both.")
    if isinstance(use, LayerAcc):
        X = _get_arr(adata, use, dim="obs")
    else:
        X = _choose_representation(adata, use_rep=use_rep if use is None else use.k, n_pcs=n_pcs)
    logger_level = _get_logger_level(logger)
    X_tsne = TSNE(
        perplexity=perplexity,
        early_exaggeration=early_exaggeration,
        learning_rate=learning_rate,
        method=method,
        metric=metric,
        # cuML's TSNE is seeded, so draw the seed right here
        random_state=_seed_from_rng(rng),
    ).fit_transform(X)
    logger.set_level(logger_level)
    if key is True:
        return X_tsne
    keys = _embedding_keys("tsne", key)
    adata.obsm[keys.obsm] = X_tsne
    adata.uns[keys.uns] = {
        "params": {
            k: v
            for k, v in {
                "perplexity": perplexity,
                "early_exaggeration": early_exaggeration,
                "learning_rate": learning_rate,
                "method": method,
                "metric": metric,
                "use_rep": use_rep,
                **_use_to_uns(use, rep=True),
            }.items()
            if v is not None
        }
        | meta_random_state
    }
