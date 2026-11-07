from __future__ import annotations

from typing import TYPE_CHECKING

import cudf
import cupy as cp
import numpy as np
from scanpy.tools._utils import get_init_pos_from_paga
from scverse_misc import deprecated_arg

from rapids_singlecell._compat import _rng_kwargs
from rapids_singlecell._keys import _embedding_keys
from rapids_singlecell._settings import Default
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _LegacyRng,
    _seed_from_rng,
)
from rapids_singlecell.get._out import DEPR_KEY_ADDED, _out_key

from ._clustering import _create_graph
from ._utils import _use_graph, _validate_init_pos

if TYPE_CHECKING:
    from anndata import AnnData

    from rapids_singlecell.get._utils import GraphAcc, MultiAcc


@_accepts_legacy_random_state(0)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
def draw_graph(
    adata: AnnData,
    *,
    init_pos: str | bool | None = None,
    max_iter: int = 500,
    rng: SeedLike | RNGLike | None = None,
    use: GraphAcc | str | None = None,
    out: MultiAcc | str | bool = False,
    # deprecated
    key_added: str | None = None,
) -> np.ndarray | None:
    """
    Force-directed graph drawing :cite:p:`Fruchterman1991,Jacomy2014`.

    Uses cugraph's implementation of Force Atlas 2.
    This is a reimplementation of scanpys function for GPU compute.

    Parameters
    ----------
        adata
            annData object with 'neighbors' field.

        init_pos
            `'paga'`/`True`, `None`/`False`, or any valid 2d-`.obsm` key.
            Use precomputed coordinates for initialization.
            If `False`/`None` (the default), initialize randomly.
        max_iter
            This controls the maximum number of levels/iterations of the
            Force Atlas algorithm. When specified the algorithm will terminate
            after no more than the specified number of iterations.
            No error occurs when the algorithm terminates in this manner.
            Good short-term quality can be achieved with 50-100 iterations.
            Above 1000 iterations is discouraged.
        rng
            Random seed or :class:`~numpy.random.Generator` used when
            initializing layout and generating samples. Defaults to 0. If
            `None` is passed, a hash of process id, time, and hostname is
            used by `cugraph`.
            The superseded `random_state` argument is still accepted.
        use
            The neighborhood graph (connectivities) to lay out, e.g. `A.obsp['k']`.
            Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsp.k'`.
            If `None`, the one :func:`~rapids_singlecell.pp.neighbors` wrote by default.
        out
            Where to write the coordinates, e.g. `A.obsm['fa']`; the parameters go to `.uns` under the same key.
            If `True`, the coordinates are returned instead.
            If `False`, they are written to `.obsm['X_draw_graph_fa']` and `.uns['draw_graph']`.

    Returns
    -------
        The coordinates if `out=True`, else updates `adata` with the following fields (see `out` for other keys).

            X_draw_graph_layout_fa : `adata.obsm`
                Coordinates of graph layout.
    """
    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}

    from cugraph.layout import force_atlas2

    key = _out_key(out, key_added, kind="obsm", default=Default())
    if out is not False and not isinstance(key, bool):  # keys from `out` are literal, not templates
        key = key.replace("{", "{{").replace("}", "}}")
    adjacency = _use_graph(adata, use)
    g = _create_graph(adjacency, use_weights=False, dtype=np.float32)
    # Get Initial Positions
    match init_pos:
        case str() if init_pos in adata.obsm:
            init_coords = adata.obsm[init_pos]
        case str("paga") | True:
            init_coords = get_init_pos_from_paga(
                adata,
                **_rng_kwargs(get_init_pos_from_paga, rng),
                adjacency=adjacency,
            )
        case False:
            init_coords = None
        case _:
            init_coords = init_pos
    if hasattr(init_coords, "dtype"):
        init_coords = _validate_init_pos(init_coords)
        if init_coords.shape[1] != 2:
            raise ValueError(f"Expected 2 columns but got {init_coords.shape[1]} columns.")

    if init_coords is not None:
        x, y = np.hsplit(init_coords, init_coords.shape[1])
        inital_df = cudf.DataFrame({"x": x.ravel(), "y": y.ravel()})
        inital_df["vertex"] = inital_df.index
    else:
        inital_df = None
    # Run cugraphs Force Atlas 2
    positions = force_atlas2(
        input_graph=g,
        pos_list=inital_df,
        max_iter=max_iter,
        outbound_attraction_distribution=False,
        lin_log_mode=False,
        edge_weight_influence=1.0,
        # Performance
        jitter_tolerance=1.0,  # Tolerance
        barnes_hut_optimize=True,
        barnes_hut_theta=1.2,
        # Tuning
        scaling_ratio=2.0,
        strong_gravity_mode=False,
        gravity=1.0,
        # cuGraph's force atlas is seeded, so draw the seed right here
        random_state=_seed_from_rng(rng),
    )
    positions = positions.sort_values("vertex").reset_index(drop=True)
    positions = cp.vstack((positions["x"].to_cupy(), positions["y"].to_cupy())).T
    layout = "fa"
    if key is True:
        return positions.get()
    keys = _embedding_keys("draw_graph", key, layout=layout)
    adata.uns[keys.uns] = {"params": {"layout": layout, **meta_random_state}}
    adata.obsm[keys.obsm] = positions.get()  # Format output
    return None
