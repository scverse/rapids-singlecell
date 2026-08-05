from __future__ import annotations

from typing import TYPE_CHECKING

import cudf
import cupy as cp
import numpy as np
from scanpy.tools._utils import get_init_pos_from_paga

from rapids_singlecell._compat import _random_state_kwargs

from ._clustering import _create_graph
from ._utils import _validate_init_pos

if TYPE_CHECKING:
    from anndata import AnnData


def draw_graph(
    adata: AnnData,
    *,
    init_pos: str | bool | None = None,
    max_iter: int = 500,
    random_state: int | None = 0,
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
        random_state
            Random state to use when initializing layout and generating
            samples. Defaults to 0. If `None` is passed, a hash of process id,
            time, and hostname is used by `cugraph`.

    Returns
    -------
        updates `adata` with the following fields.

            X_draw_graph_layout_fa : `adata.obsm`
                Coordinates of graph layout.
    """
    from cugraph.layout import force_atlas2

    # Adjacency graph
    adjacency = adata.obsp["connectivities"]
    g = _create_graph(adjacency, use_weights=False, dtype=np.float32)
    # Get Initial Positions
    match init_pos:
        case str() if init_pos in adata.obsm:
            init_coords = adata.obsm[init_pos]
        case str() if init_pos == "paga":
            if random_state is None:
                random_state = np.random.default_rng()
            init_coords = get_init_pos_from_paga(
                adata,
                **_random_state_kwargs(get_init_pos_from_paga, random_state),
                neighbors_key="connectivities",
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
        random_state=random_state,
    )
    positions = positions.sort_values("vertex").reset_index(drop=True)
    positions = cp.vstack((positions["x"].to_cupy(), positions["y"].to_cupy())).T
    layout = "fa"
    adata.uns["draw_graph"] = {}
    adata.uns["draw_graph"]["params"] = {"layout": layout, "random_state": random_state}
    key_added = f"X_draw_graph_{layout}"
    adata.obsm[key_added] = positions.get()  # Format output
