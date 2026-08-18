from __future__ import annotations

from typing import TYPE_CHECKING

import cudf
import cupy as cp
import numpy as np
from scanpy.tools._utils import get_init_pos_from_paga

from rapids_singlecell._compat import _rng_kwargs
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
    _LegacyRng,
    _seed_from_rng,
)

from ._clustering import _create_graph
from ._utils import _validate_init_pos

if TYPE_CHECKING:
    from anndata import AnnData


@_accepts_legacy_random_state(0)
def draw_graph(
    adata: AnnData,
    *,
    init_pos: str | bool | None = None,
    max_iter: int = 500,
    rng: SeedLike | RNGLike | None = None,
) -> None:
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

    Returns
    -------
        updates `adata` with the following fields.

            X_draw_graph_layout_fa : `adata.obsm`
                Coordinates of graph layout.
    """
    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}

    from cugraph.layout import force_atlas2

    # Adjacency graph
    adjacency = adata.obsp["connectivities"]
    g = _create_graph(adjacency, use_weights=False, dtype=np.float32)
    # Get Initial Positions
    match init_pos:
        case str() if init_pos in adata.obsm:
            init_coords = adata.obsm[init_pos]
        case str() if init_pos == "paga":
            init_coords = get_init_pos_from_paga(
                adata,
                **_rng_kwargs(get_init_pos_from_paga, rng),
            )
        case _:
            init_coords = init_pos
    if hasattr(init_coords, "dtype"):
        init_coords = _validate_init_pos(init_coords)
        if init_coords.shape[1] != 2:
            raise ValueError(
                f"Expected 2 columns but got {init_coords.shape[1]} columns."
            )

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
    adata.uns["draw_graph"] = {}
    adata.uns["draw_graph"]["params"] = {"layout": layout, **meta_random_state}
    key_added = f"X_draw_graph_{layout}"
    adata.obsm[key_added] = positions.get()  # Format output
