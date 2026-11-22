from __future__ import annotations

import math
import traceback
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, suppress
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
import scverse_misc
from cupyx.scipy import sparse as cpx_sparse
from natsort import natsorted
from scanpy import logging as logg
from scanpy.tools._utils_clustering import restrict_adjacency
from scipy import sparse
from scverse_misc import deprecated_arg

import rapids_singlecell
from rapids_singlecell._cuda import _leiden_cuda as _ld
from rapids_singlecell._utils._multi_gpu import _copy_to_device as _copy_to_gpu
from rapids_singlecell._utils._random import (
    _accepts_legacy_random_state,
    _LegacyRng,
    _seed_from_rng,
)
from rapids_singlecell.get._out import DEPR_KEY_ADDED, DEPR_NEIGHBORS_KEY, DEPR_OBSP, _out_key, deprecated_copy

from ._clustering import (
    _DEPR_ADJACENCY,
    _check_dtype,
    _cluster_labels,
    _create_graph,
    _create_graph_dask,
    _import_cugraph,
    _rename_groups,
)
from ._utils import _use_graph

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

    from rapids_singlecell._utils._random import RNGLike, SeedLike
    from rapids_singlecell.get._utils import AdRef, GraphAcc

# Host input is read in place on GPUs with ATS (e.g. GB10); elsewhere it is
# copied to the device in chunks through pinned buffers.
_HOST_ZERO_COPY = True
_H2D_CHUNK_BYTES = 1 << 23
_WORKSPACE_CACHE_BYTES = 1 << 31  # see `_WORKSPACE_CACHE`; 0 disables it
_N_ITER_WARN = 20  # above this: warn and run the -1 rule
_MAX_RESOLUTION = 2.0**20  # bound of the integer no-overflow proof
# cuGraph returns the refined partition when it hits `max_iter` levels, so
# `n_iterations` is not forwarded and the cap stays well above what it needs.
_CUGRAPH_MAX_ITER = 100
# The native implementation is the default; this also starts the deprecation
# of flavor="cugraph" and use_dask (FutureWarning).
_DEFAULT_FLAVOR: Literal["rapids", "cugraph"] = "rapids"

# Point warnings at the caller outside the public API's deprecation wrappers.
_SKIP = tuple(str(Path(m.__file__).parent) for m in (rapids_singlecell, scverse_misc))

_CLUSTERING_ARGS = {
    "rapids": frozenset({"beta", "initial_membership", "objective_function"}),
    "cugraph": frozenset({"max_iter", "objective_function"}),
}


@_accepts_legacy_random_state(0)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_arg("adjacency", _DEPR_ADJACENCY)
@deprecated_arg("neighbors_key", DEPR_NEIGHBORS_KEY)
@deprecated_arg("obsp", DEPR_OBSP)
@deprecated_copy
def leiden(
    adata: AnnData | MuData,
    resolution: float | Sequence[float] | Mapping[str, float] = 1.0,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    directed: bool | None = None,
    use_weights: bool = True,
    n_iterations: int = 2,
    flavor: Literal["rapids", "cugraph"] = _DEFAULT_FLAVOR,
    theta: float | None = None,
    dtype: str | np.dtype = np.float32,
    use_dask: bool = False,
    use: GraphAcc | str | Sequence[GraphAcc | str] | Mapping[str, GraphAcc | str | None] | None = None,
    layer_weights: Sequence[float] | Mapping[str, float] | None = None,
    out: AdRef | str | bool = False,
    # deprecated
    key_added: str | None = None,
    adjacency: sparse.spmatrix | sparse.sparray | cpx_sparse.spmatrix | None = None,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    copy: bool = False,
    **clustering_args,
) -> AnnData | MuData | pd.Categorical | dict[float, pd.Categorical] | None:
    """Cluster cells into subgroups using the Leiden algorithm :cite:p:`Traag2019`.

    Leiden is an improved version of the Louvain algorithm :cite:p:`Blondel2008`
    that yields connected communities. This requires having run
    :func:`~rapids_singlecell.pp.neighbors` first.

    Parameters
    ----------
    adata
        Annotated data matrix, or a MuData object whose selected modalities share the same cells in the same order.
    resolution
        A parameter value controlling the coarseness of the clustering (called
        gamma in the modularity formula). Higher values lead to more clusters.
        On one graph a list runs a separate clustering for each value and writes
        one column per value. With multiple graphs, a scalar applies to every
        layer and a sequence supplies one value per layer for a single partition.
        For MuData, a mapping may give each modality's resolution by name.
        Each value must be finite and in ``[0, 2**20]``.
    restrict_to
        Restrict the clustering to the categories within the key for sample
        annotation, tuple needs to contain ``(obs_key, list_of_categories)``.
    rng
        Random seed or :class:`~numpy.random.Generator` changing the
        initialization of the optimization. Defaults to 0.
        The superseded ``random_state`` argument is still accepted.
    use
        Neighborhood graph to cluster, e.g. ``A.obsp['k']`` or ``'obsp.k'``.
        Defaults to the connectivities written by :func:`~rapids_singlecell.pp.neighbors`.
        ``flavor='rapids'`` sums duplicate entries, clusters
        an asymmetric matrix as A + A.T (with a warning) and ignores the
        diagonal (self-loops). For multiplex Leiden, pass a sequence of at
        least two graph accessors or paths, e.g. ``["obsp.expression", "obsp.spatial"]``.
        With MuData, a mapping selects each modality's graph by accessor or
        ``.obsp`` key, e.g.
        ``use={"rna": None, "atac": None}`` uses their default connectivities.
        Omitting ``use`` clusters the joint graph of the MuData object.
    layer_weights
        Multipliers for the graphs in ``use``, in the same order or keyed by
        modality. Defaults to one per graph. Multiplex Leiden finds one partition
        by maximizing the weighted sum of the layers' unnormalized modularity
        objectives, using each graph's own strengths and resolution. Weights
        must be finite and non-negative, with at least one positive weight.
    directed
        Must not be ``True``: the graph is treated as undirected.
    use_weights
        If ``True``, use the graph's edge weights. Native Leiden requires every
        stored weight to be finite and non-negative in this mode. If ``False``,
        native Leiden counts each nonzero off-diagonal entry as a unit-weight edge.
    n_iterations
        How many iterations of the Leiden algorithm to perform, as in
        ``scanpy.tl.leiden``; each iteration restarts from the partition of the
        previous one :cite:p:`Traag2019`. ``-1`` iterates until an iteration
        moves no cell or improves the modularity by less than 1e-6, then runs
        a last one (at most 20 in all). Values above 20 behave like ``-1``,
        with a warning. On multiple layers, ``-1`` stops when the joint quality
        stops improving or after 20 iterations.
        ``flavor='cugraph'`` ignores it and runs up to 100 levels.
    flavor
        Which implementation to use. ``'rapids'`` (default) is the native CUDA
        implementation of rapids-singlecell, bitwise reproducible for a given
        seed; ``'cugraph'`` (deprecated) uses :func:`cugraph.leiden` and,
        if it runs out of memory on one GPU, reruns in CUDA managed memory
        (slower, with a warning). On a single graph, ``'rapids'`` needs about 140 bytes of
        device memory per cell plus what its coarser levels hold, allocated
        as they need it: on k-NN graphs 14-18 bytes per stored entry at the
        peak (about 22 at 100M cells), in all 2.5-3.5x the float32 CSR.
        GPUs that cannot read host memory directly also need a device copy
        of host input. If that does not fit,
        it raises :class:`MemoryError` (RMM managed memory can exceed the
        GPU). Under an allocator that does not pool (the default of
        rapids-singlecell), it keeps the per-cell workspace for the next call
        if it is at most 2 GiB and 1/16 of the GPU memory.
    out
        Where to write the cluster labels, e.g. ``A.obs['clusters']``. The
        parameters and modularity go to ``.uns`` under the same key.
        ``False`` writes to the default key ``'leiden'`` (``'leiden_R'`` with
        ``restrict_to``). ``True`` returns the labels, as a dict keyed by
        numeric resolution when several are requested.
    theta
        Deprecated. cuGraph's refinement randomness, only used by
        ``flavor='cugraph'`` (``None`` means 1.0). ``flavor='rapids'`` ignores
        it with a warning; see ``beta``.
    dtype
        Weight dtype of the cuGraph graph (``'float32'`` or ``'float64'``).
        Only used by the deprecated ``flavor='cugraph'``; the native
        implementation reads float32 or float64 weights. On a single graph,
        it rounds them to one 64-bit fixed-point grid.
    use_dask
        Deprecated. If ``True``, run cuGraph's Dask implementation on all GPUs
        (this implies ``flavor='cugraph'``).
    **clustering_args
        Further arguments for the clustering. ``flavor='rapids'`` accepts
        ``beta`` (randomness of the refinement, theta in :cite:t:`Traag2019`;
        only the default 0, which merges along the best edge, is supported),
        ``initial_membership`` and ``objective_function='modularity'``.
        ``flavor='cugraph'`` accepts ``max_iter`` (maximum number of levels,
        default 100) and ``objective_function='modularity'``.

    Returns
    -------
    With ``out=True``, returns the categorical labels or a dict of them by
    resolution. Otherwise writes the following fields and returns ``None``
    (``copy=True`` returns a copy of ``adata``):

    ``adata.obs['leiden' | key_added]`` : :class:`pandas.Series` (dtype ``category``)
        Array of dim (number of samples) that stores the subgroup id
        (``'0'``, ``'1'``, ...) for each cell. Cluster ``'0'`` is the largest;
        clusters of equal size are ordered by their first cell. With several
        resolutions, the keys are ``f'{key_added}_{resolution}'``. With
        ``restrict_to`` and the default key, the key is ``'leiden_R'``.
    ``adata.uns['leiden' | key_added]['params']`` : :class:`dict`
        A dict with the values for the parameters ``resolution``,
        ``n_iterations``, and ``random_state`` (if applicable).
    ``adata.uns['leiden' | key_added]['modularity']`` : :class:`float`
        The modularity of the returned clustering with the resolution as
        gamma (a list for several resolutions). On one graph it is computed exactly without
        self-loops and with the weights rounded to one 64-bit fixed-point
        grid, which differs from float64 sums only for weights spanning many
        orders of magnitude or a resolution near ``2**20``
        (``flavor='cugraph'``: cuGraph's value). Use
        :func:`scanpy.metrics.modularity` for the score with gamma = 1 (it
        counts self-loops). For multiplex Leiden this is a list of per-layer
        modularities in input order. The unnormalized weighted sum optimized by
        the algorithm is recorded in ``adata.uns[key_added]['quality']``.
    """
    flavor = _validate_flavor(
        flavor,
        directed=directed,
        use_dask=use_dask,
        theta=theta,
        clustering_args=clustering_args,
    )
    dtype = _check_dtype(dtype)
    use_weights = bool(use_weights)  # the bindings take a Python bool
    _validate_n_iterations(n_iterations, flavor=flavor)
    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}

    start = logg.info("running Leiden clustering")
    key = _out_key(out, key_added, kind="obs", default="leiden")
    if restrict_to is not None and out is False and key == "leiden":
        key = "leiden_R"
    if isinstance(use, Mapping | Sequence) and not isinstance(use, str):
        if flavor != "rapids" or directed:
            raise ValueError("Multiplex Leiden requires undirected graphs and flavor='rapids' on one GPU.")
        if adjacency is not None or obsp is not None or neighbors_key is not None:
            raise TypeError("Pass multiplex graphs through `use` alone.")
        return _leiden_layers(
            adata,
            use,
            resolution,
            layer_weights,
            use_weights=use_weights,
            restrict_to=restrict_to,
            key=key,
            out=out,
            rng=rng,
            n_iterations=n_iterations,
            meta_random_state=meta_random_state,
            **clustering_args,
        )
    if layer_weights is not None:
        raise ValueError("`layer_weights` requires multiple graphs in `use`.")
    if isinstance(resolution, Mapping):
        raise ValueError("A resolution mapping requires a mapping of modalities in `use`.")
    resolutions = _validate_resolutions(resolution)
    if out is True and len(set(resolutions)) != len(resolutions):
        msg = "out=True requires distinct numeric resolutions for the returned dict."
        raise ValueError(msg)
    restrict_indices = None
    adjacency = _use_graph(adata, use, adjacency=adjacency, obsp=obsp, neighbors_key=neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    try:
        if flavor == "rapids":
            parts = _leiden_rapids(
                adjacency,
                resolutions,
                rng=rng,
                n_iterations=n_iterations,
                use_weights=use_weights,
                **clustering_args,
            )
        else:
            parts = _leiden_cugraph(
                adjacency,
                resolutions,
                rng=rng,
                theta=theta,
                dtype=dtype,
                use_dask=use_dask,
                use_weights=use_weights,
                **clustering_args,
            )
    except BaseException as err:  # a kept traceback (e.g. IPython's) must not hold the graph copies
        traceback.clear_frames(err.__traceback__)
        del adjacency
        raise
    labels = {}
    for res, part in zip(resolutions, parts, strict=True):
        cat = _leiden_labels(adata, part, restrict_to, restrict_indices)
        # Column names preserve the spelling of a resolution (e.g. 1 and 1.0).
        labels[res if out is True else f"{res}"] = cat
    params = {"n_iterations": n_iterations, **meta_random_state}
    if flavor == "cugraph":
        params["max_iter"] = clustering_args.get("max_iter", _CUGRAPH_MAX_ITER)
    logg.info(
        "    finished",
        time=start,
        deep=f"found {', '.join(str(p.n_clusters) for p in parts)} clusters",
    )
    return _cluster_labels(
        (key, key), resolutions, labels, out=out, adata=adata, params=params, modularities=[p.modularity for p in parts]
    )


def _leiden_labels(adata, part, restrict_to, indices):
    if restrict_to is None:
        return pd.Categorical.from_codes(part.membership, categories=[str(i) for i in range(part.n_clusters)])
    groups = _rename_groups(
        adata,
        restrict_key=restrict_to[0],
        restrict_categories=restrict_to[1],
        restrict_indices=indices,
        groups=part.membership,
    )
    return pd.Categorical(values=groups.astype("U"), categories=natsorted(map(str, np.unique(groups))))


def _layer_values(value, use, name):
    if isinstance(value, Mapping):
        if not isinstance(use, Mapping) or value.keys() != use.keys():
            raise ValueError(f"`{name}` must have the same modality keys as `use`.")
        return [value[k] for k in use]
    values = [value] * len(use) if isinstance(value, Real) else list(value)
    if len(values) != len(use):
        raise ValueError(f"`{name}` needs one value per layer ({len(use)} layers).")
    return values


def _multiplex_graph_key(graph):
    from rapids_singlecell.get._out import _resolve_obs
    from rapids_singlecell.get._utils import A, GraphAcc

    if A is None and isinstance(graph, str) and graph.startswith("obsp."):
        return graph.removeprefix("obsp.")
    if graph is None:
        raise TypeError("Each graph in a multiplex sequence must be an `obsp` accessor or path.")
    return _resolve_obs(graph, kinds=(GraphAcc,)).k


def _leiden_layers(
    adata,
    use,
    resolution,
    layer_weights,
    *,
    use_weights,
    restrict_to,
    key,
    out,
    rng,
    n_iterations,
    meta_random_state,
    beta=0.0,
    initial_membership=None,
    objective_function="modularity",
):
    from ._leiden_multiplex import leiden_multiplex

    if len(use) < 2:
        raise ValueError("Multiplex Leiden needs at least two graphs in `use`.")
    _validate_rapids_options(beta, objective_function)
    resolutions = _validate_resolutions(_layer_values(resolution, use, "resolution"), multiplex=True)
    weights = _layer_values(1.0 if layer_weights is None else layer_weights, use, "layer_weights")
    if not all(
        isinstance(w, Real) and not isinstance(w, bool) and math.isfinite(w) and w >= 0 for w in weights
    ) or not any(weights):
        raise ValueError("`layer_weights` must be finite and non-negative, with at least one positive value.")
    names, graphs = [], []
    if isinstance(use, Mapping):
        if not hasattr(adata, "mod"):
            raise TypeError("A mapping in `use` requires a MuData object.")
        for name, graph in use.items():
            mod = adata.mod[name]
            if not mod.obs_names.equals(adata.obs_names):
                raise ValueError(
                    f"Modality {name!r} must contain the same cells in the same order as the MuData object."
                )
            names.append(name)
            if graph is None and "neighbors" not in mod.uns and "connectivities" in mod.obsp:
                graph = "connectivities"
            if graph is None:
                graphs.append(_use_graph(mod))
            else:
                key_obsp = graph if isinstance(graph, str) and graph in mod.obsp else _multiplex_graph_key(graph)
                graphs.append(mod.obsp[key_obsp])
    else:
        names = [_multiplex_graph_key(graph) for graph in use]
        graphs = [adata.obsp[name] for name in names]

    indices = None
    if restrict_to is not None:
        graphs[0], indices = restrict_adjacency(
            adata, restrict_to[0], restrict_categories=restrict_to[1], adjacency=graphs[0]
        )
        graphs[1:] = [g[indices, :][:, indices] for g in graphs[1:]]
    try:
        part = leiden_multiplex(
            graphs,
            resolutions,
            weights,
            use_weights=use_weights,
            n_iterations=-1 if n_iterations > _N_ITER_WARN else n_iterations,
            rng=rng,
            initial_membership=initial_membership,
        )
    except BaseException as err:
        traceback.clear_frames(err.__traceback__)
        del graphs
        raise
    labels = _leiden_labels(adata, part, restrict_to, indices)
    logg.info(f"    finished multiplex Leiden with {part.n_clusters} clusters")
    if out is True:
        return labels
    adata.obs[key] = labels
    adata.uns[key] = {
        "params": {
            "layers": names,
            "resolution": resolutions,
            "layer_weights": weights,
            "n_iterations": n_iterations,
            **meta_random_state,
        },
        "modularity": part.modularity,
        "quality": part.quality,
    }
    return None


def _validate_flavor(
    flavor: str,
    *,
    directed: bool | None,
    use_dask: bool,
    theta: float | None,
    clustering_args: dict,
) -> Literal["rapids", "cugraph"]:
    deprecate_cugraph = _DEFAULT_FLAVOR == "rapids"
    match flavor:
        case "rapids" | "cugraph" if use_dask:
            if deprecate_cugraph or flavor == "rapids":
                msg = "use_dask is deprecated and runs the legacy cuGraph implementation, which will be removed in 0.20; use flavor='rapids' instead."
                warnings.warn(msg, FutureWarning, skip_file_prefixes=_SKIP)
            flavor = "cugraph"
        case "cugraph":
            if deprecate_cugraph:
                msg = "flavor='cugraph' is deprecated and will be removed in 0.20."
                warnings.warn(msg, FutureWarning, skip_file_prefixes=_SKIP)
        case "rapids":
            if theta is not None:
                msg = "theta is a deprecated cuGraph parameter (it had no effect there) and is ignored by flavor='rapids'; see beta."
                warnings.warn(msg, FutureWarning, skip_file_prefixes=_SKIP)
        case _:
            msg = f"flavor must be either 'rapids' or 'cugraph', but {flavor!r} was passed."
            raise ValueError(msg)
    if directed:
        msg = f"Cannot use flavor={flavor!r} with a directed graph."
        raise ValueError(msg)
    if "partition_type" in clustering_args:
        msg = f"Do not pass in partition_type argument when using flavor={flavor!r}."
        raise ValueError(msg)
    if unknown := set(clustering_args) - _CLUSTERING_ARGS[flavor]:
        msg = f"leiden() got unexpected keyword arguments {sorted(unknown)} for flavor={flavor!r}."
        raise TypeError(msg)
    if flavor == "rapids" and (_ld is None or not hasattr(_ld, "leiden")):
        msg = "flavor='rapids' is not available: this build of rapids-singlecell does not include the native Leiden extension (`_leiden_cuda`). To use flavor='cugraph', install the optional cuGraph dependency."
        raise NotImplementedError(msg)
    return flavor


def _validate_n_iterations(n_iterations: int, *, flavor: Literal["rapids", "cugraph"]) -> None:
    if (
        isinstance(n_iterations, bool)
        or not isinstance(n_iterations, Integral)
        or n_iterations == 0
        or n_iterations < -1
    ):
        msg = f"n_iterations must be a positive integer or -1, but {n_iterations!r} was passed."
        raise ValueError(msg)
    # flavor="cugraph" ignores n_iterations (see _CUGRAPH_MAX_ITER)
    if flavor == "rapids" and n_iterations > _N_ITER_WARN:
        msg = f"n_iterations counts Leiden iterations, not cuGraph levels; values above {_N_ITER_WARN} run until stable (as -1); 2 is recommended."
        warnings.warn(msg, UserWarning, skip_file_prefixes=_SKIP)


def _validate_resolutions(resolution: float | Sequence[float], *, multiplex: bool = False) -> list[float]:
    resolutions = [resolution] if isinstance(resolution, Real) else list(resolution)
    if not resolutions:
        msg = "resolution must be a number or a non-empty sequence of numbers."
        raise ValueError(msg)
    if not all(
        isinstance(res, Real) and not isinstance(res, bool) and math.isfinite(res) and 0 <= res <= _MAX_RESOLUTION
        for res in resolutions
    ):
        msg = f"resolution must be finite and in [0, 2**20], but {resolution!r} was passed."
        raise ValueError(msg)
    if not multiplex and len({f"{res}" for res in resolutions}) < len(
        resolutions
    ):  # each writes the obs column f"{key}_{res}"
        msg = f"resolution must not repeat a value, but {resolution!r} was passed."
        raise ValueError(msg)
    return resolutions


def _validate_rapids_options(beta, objective_function):
    if objective_function.lower() != "modularity":
        msg = "flavor='rapids' supports objective_function='modularity' (CPM: planned)."
        raise NotImplementedError(msg)
    if not (math.isfinite(beta) and beta >= 0):
        msg = f"beta must be finite and >= 0, but {beta!r} was passed."
        raise ValueError(msg)
    if beta > 0:
        msg = "flavor='rapids' merges along the best edge (beta=0); randomized refinement (beta > 0) is planned."
        raise NotImplementedError(msg)


def _leiden_rapids(
    adjacency,
    resolutions: list[float],
    *,
    rng: np.random.Generator,
    n_iterations: int,
    use_weights: bool,
    beta: float = 0.0,
    initial_membership=None,
    objective_function: str = "modularity",
) -> list[SimpleNamespace]:
    """Run every resolution in one native call (`_leiden_cuda.leiden`)."""
    _validate_rapids_options(beta, objective_function)
    if n_iterations > _N_ITER_WARN:  # warned about in `leiden`
        n_iterations = -1
    home = cp.cuda.Device().id
    source = _input_device(adjacency)
    with _cuda_memory_errors():
        with cp.cuda.Device(home if source is None else source):
            csr = _as_csr(adjacency)
        if source not in (None, home):
            from rapids_singlecell.tools._rank_genes_groups._wilcoxon_host import _copy_device_sparse_to_device

            csr = _copy_device_sparse_to_device(csr, home)
        n = csr.shape[0]
        graph, info = None, {"nnz_counted": 0}
        if csr.nnz:
            host_input = _HOST_ZERO_COPY and source is None and _reads_host_memory(home)
            graph, info, use_weights = _ingest(csr, use_weights=use_weights, host_input=host_input)
        del csr
        membership = _compact_initial_membership(initial_membership, n)
    if info["nnz_counted"] == 0:
        codes = np.arange(n, dtype=np.int32)
        return [SimpleNamespace(membership=codes, modularity=0.0, n_clusters=n) for _ in resolutions]
    args = {
        "indices": graph.indices,
        "data": graph.data,
        "use_weights": use_weights,
        "check": info,
        "resolutions": [float(res) for res in resolutions],
        "seeds": [_seed_from_rng(rng, allow_none=False) & 0xFFFFFFFF for _ in resolutions],
        "n_iterations": int(n_iterations),
        "initial_membership": membership,
        "stream": cp.cuda.get_current_stream().ptr,
    }
    return _leiden_single(graph, args, n)


@contextmanager
def _cuda_memory_errors():
    try:
        yield
    except cp.cuda.runtime.CUDARuntimeError as err:
        if err.status != 2:  # cudaErrorMemoryAllocation
            raise
        raise MemoryError(str(err)) from err


class _ArenaBlocks:
    """Device blocks of the native engine's level arena, allocated as levels need them and freed as they end.

    Leaving the context frees what a failed call left and raises the engine's own CUDA
    out-of-memory errors (RuntimeError) as MemoryError, as the allocators do.
    """

    def __init__(self) -> None:
        self.live: dict[int, cp.ndarray] = {}

    def __enter__(self) -> _ArenaBlocks:
        return self

    def __exit__(self, kind, err, tb) -> None:
        self.live.clear()  # the call is done with them: it synchronises before it raises
        if isinstance(err, RuntimeError) and "out of memory" in str(err):
            raise MemoryError(str(err)) from err

    def alloc(self, nbytes: int, device: int) -> int:
        with cp.cuda.Device(device):
            block = cp.empty(nbytes, dtype=cp.uint8)
        self.live[block.data.ptr] = block
        return block.data.ptr

    def free(self, ptr: int) -> None:
        del self.live[ptr]


@_cuda_memory_errors()
def _leiden_single(graph: SimpleNamespace, args: dict, n: int) -> list[SimpleNamespace]:
    """One GPU: size the workspace and run; the level arena grows by blocks of exactly what a level needs."""
    pinned = cp.cuda.alloc_pinned_memory(_ld.PINNED_BYTES)  # readback buffer
    # the memory kind of the call's other device arrays, which the workspace must share
    arrays = (graph.indptr, args["initial_membership"])
    kind = next((_memory_kind(a) for a in arrays if isinstance(a, cp.ndarray)), None)
    nbytes = _ld.leiden(graph.indptr, **args)["required_bytes"]  # a call without a workspace sizes it
    workspace = _allocate_workspace(nbytes, kind)
    codes = _host_labels((len(args["resolutions"]), n))
    with _ArenaBlocks() as blocks:
        out = _ld.leiden(
            graph.indptr,
            **args,
            workspace=workspace,
            pinned=pinned.ptr,
            labels_out=codes,
            arena_alloc=blocks.alloc,
            arena_free=blocks.free,
        )
    _cache_workspace(workspace)
    return [
        SimpleNamespace(membership=c, modularity=float(q), n_clusters=int(k))
        for c, q, k in zip(codes, out["modularity"], out["n_clusters"], strict=True)
    ]


def _input_device(adjacency) -> int | None:
    """The GPU that device input (CuPy / cupyx) lives on; None for host input."""
    if isinstance(adjacency, cp.ndarray):
        return adjacency.device.id
    if cpx_sparse.issparse(adjacency):
        return adjacency.data.device.id
    return None


def _check_csr_storage(csr):
    if csr.data.ndim != 1 or csr.indices.ndim != 1 or csr.data.size != csr.indices.size:
        msg = "data and indices must have equal length"
        raise ValueError(msg)
    if csr.indptr.shape != (csr.shape[0] + 1,) or int(csr.indptr[-1]) != csr.indices.size:
        msg = "leiden: indptr must end at nnz"
        raise ValueError(msg)


def _as_csr(adjacency):
    """CSR with float weights, kept on the host or device it lives on."""
    device = cpx_sparse.issparse(adjacency)
    if (device or sparse.issparse(adjacency)) and adjacency.format == "coo":
        # tocsr sums duplicates in the data dtype (cupyx: with float atomics, in
        # any order): keep them for `_canonicalize`
        xp, sp = (cp, cpx_sparse) if device else (np, sparse)
        order = xp.argsort(adjacency.row, kind="stable")
        indptr = xp.zeros(adjacency.shape[0] + 1, dtype=xp.int64)
        xp.cumsum(xp.bincount(adjacency.row, minlength=adjacency.shape[0]), out=indptr[1:])
        csr = sp.csr_matrix((adjacency.data[order], adjacency.col[order], indptr), shape=adjacency.shape)
    elif device or sparse.issparse(adjacency):
        csr = adjacency.tocsr()
    elif isinstance(adjacency, cp.ndarray):
        csr = cpx_sparse.csr_matrix(adjacency if adjacency.dtype.kind in "fc" else adjacency.astype(cp.float32))
    else:
        csr = sparse.csr_matrix(np.asarray(adjacency))
    if csr.shape[0] != csr.shape[1]:
        msg = f"adjacency must be square, but has shape {csr.shape}."
        raise ValueError(msg)
    # SciPy takes nnz from the last row pointer, even if the backing arrays disagree.
    if csr.indptr.shape != (csr.shape[0] + 1,) or csr.nnz != csr.indices.size:
        msg = "leiden: indptr must end at nnz and have n + 1 entries"
        raise ValueError(msg)
    if not csr.indices.size and (csr.data.size or csr.indptr.any()):
        msg = "leiden: an empty CSR must have no stored entries and an all-zero indptr"
        raise ValueError(msg)
    if csr.dtype not in (np.float32, np.float64):
        # Sparse astype may coalesce duplicates before we validate or count them.
        _check_csr_storage(csr)
        csr = type(csr)((csr.data.astype(np.float32), csr.indices, csr.indptr), shape=csr.shape)
    # the bindings take int32 / int64 offsets and indices
    if not all(a.dtype in (np.int32, np.int64) for a in (csr.indptr, csr.indices)):
        _check_csr_storage(csr)
        csr = csr.copy()
        csr.indptr = csr.indptr.astype(np.int64)
        csr.indices = csr.indices.astype(np.int64)
    return csr


def _ingest(csr, *, use_weights: bool, host_input: bool):
    """Check the input natively and fix it up (before sizing the workspace).

    Returns the arrays the driver reads, their ingest check and the weight mode.
    """
    n = csr.shape[0]
    if cpx_sparse.issparse(csr):
        arrays = (csr.indptr, csr.indices, csr.data)
        # Each binding takes contiguous arrays of a single memory kind.
        kind = _memory_kind(cp.empty(1, dtype=cp.uint8))
        arrays = tuple(a if a.flags.c_contiguous and _memory_kind(a) == kind else a.copy(order="C") for a in arrays)
        graph = SimpleNamespace(indptr=arrays[0], indices=arrays[1], data=arrays[2], host=False)
    elif host_input:
        graph = SimpleNamespace(
            indptr=np.ascontiguousarray(csr.indptr),
            indices=np.ascontiguousarray(csr.indices),
            data=np.ascontiguousarray(csr.data),
            host=True,
        )
    else:
        graph = _to_device(csr)
    fixed = set()
    while True:
        graph.nnz = int(graph.indices.shape[0])
        info = _ld.ingest_check(
            graph.indptr,
            indices=graph.indices,
            data=graph.data,
            use_weights=use_weights,
            stream=cp.cuda.get_current_stream().ptr,
        )
        if info["neg"] or info["nonfinite"]:
            msg = "adjacency weights must be finite and non-negative."
            raise ValueError(msg)
        problem = next((p for p in ("canonical", "symmetric") if not info[p]), None)
        if problem is None:
            counted = info["nnz_counted"]
            if 0 < counted < graph.nnz and _ld.n_replicas(graph.nnz, n) < _ld.n_replicas(counted, n):
                graph = _drop_ignored_entries(graph)
                continue
            return graph, info, use_weights
        if problem in fixed:  # each fix-up runs at most once
            msg = f"The adjacency is still not {problem} after fixing it."
            raise RuntimeError(msg)
        fixed.add(problem)
        if not use_weights:
            # Duplicates and A + A.T must add up unit weights (multiplicities),
            # so the fix-ups and the driver read the unit values as weights.
            graph.data = (graph.data != 0).astype(np.float32)
            use_weights = True
        if problem == "canonical":
            graph = _canonicalize(graph, n)
        else:
            msg = "adjacency is not symmetric; clustering A + A.T instead."
            warnings.warn(msg, UserWarning, skip_file_prefixes=_SKIP)
            # The ingest maximum excludes diagonal entries; a large loop can overflow too.
            max_weight = info["wmax"] if info["nnz_counted"] == graph.nnz else None
            graph = _symmetrize(graph, n, max_weight)


def _drop_ignored_entries(graph: SimpleNamespace) -> SimpleNamespace:
    """Keep replica selection independent of padding without copying the padding into each replica."""
    xp = np if graph.host else cp
    # The graph is canonical, so at most nnz_counted + n entries are nonzero.
    # Bound the temporary scan of the original (possibly much larger) data.
    step = 1 << 20
    keep = xp.concatenate([xp.flatnonzero(graph.data[i : i + step]) + i for i in range(0, graph.nnz, step)])
    keep = keep[graph.indices[keep] != xp.searchsorted(graph.indptr, keep, side="right") - 1]
    return SimpleNamespace(
        indptr=xp.searchsorted(keep, graph.indptr).astype(graph.indptr.dtype, copy=False),
        indices=graph.indices[keep],
        data=graph.data[keep],
        host=graph.host,
    )


def _to_device(csr) -> SimpleNamespace:
    """Copy host input to the device (GPUs that cannot read it in place).

    int64 column indices are narrowed to int32 on the device chunk by chunk;
    values outside ``[0, n)`` become -1, which `_ld.ingest_check` rejects.
    """
    narrow = csr.indices.dtype == np.int64
    indices = cp.empty(csr.indices.shape[0], dtype=cp.int32 if narrow else csr.indices.dtype)
    data = cp.empty(csr.data.shape[0], dtype=csr.data.dtype)
    _copy_to_device(csr.indices, indices, narrow_n=csr.shape[0] if narrow else None)
    _copy_to_device(csr.data, data)
    return SimpleNamespace(indptr=cp.asarray(csr.indptr), indices=indices, data=data, host=False)


def _copy_to_device(src: np.ndarray, dst: cp.ndarray, *, narrow_n=None) -> None:
    """Copy ``src`` into ``dst`` through two pinned buffers.

    The copies overlap. With ``narrow_n``, the int64 ``src`` is narrowed into the int32 ``dst``.
    """
    stream = cp.cuda.get_current_stream()
    src = np.ascontiguousarray(src)
    count = src.shape[0]
    step = max(1, min(count, _H2D_CHUNK_BYTES // src.itemsize))
    n_buffers = min(2, -(-count // step))
    pinned = [cp.cuda.alloc_pinned_memory(step * src.itemsize) for _ in range(n_buffers)]
    staging = [cp.empty(step, dtype=src.dtype) for _ in range(n_buffers)] if narrow_n is not None else None
    copied = [None] * n_buffers  # event: the copy out of pinned[b] is done
    for i, lo in enumerate(range(0, count, step)):
        hi, b = min(lo + step, count), i % n_buffers
        if copied[b] is not None:
            copied[b].synchronize()
        host = np.frombuffer(pinned[b], dtype=src.dtype, count=hi - lo)
        host[...] = src[lo:hi]
        target = dst[lo:hi] if staging is None else staging[b][: hi - lo]
        cp.cuda.runtime.memcpyAsync(
            target.data.ptr,
            host.ctypes.data,
            host.nbytes,
            cp.cuda.runtime.memcpyHostToDevice,
            stream.ptr,
        )
        if staging is not None:
            _ld.narrow_indices(target, dst=dst[lo:hi], n=narrow_n, stream=stream.ptr)
        copied[b] = stream.record()
    stream.synchronize()


def _canonicalize(graph: SimpleNamespace, n: int) -> SimpleNamespace:
    """Sort rows and sum duplicates (fp64, stored order) on the device."""
    indptr = cp.empty(n + 1, dtype=cp.int64)
    indices = cp.empty(graph.nnz, dtype=cp.int32)
    data = cp.empty(graph.nnz, dtype=cp.float64)
    # rows longer than the sort chunk are sorted alone and need the longest row
    max_row_nnz = int((graph.indptr[1:] - graph.indptr[:-1]).max()) if n else 0
    scratch = cp.empty(
        _ld.canonicalize_scratch_bytes(n=n, nnz=graph.nnz, max_row_nnz=max_row_nnz),
        dtype=cp.uint8,
    )
    nnz = _ld.canonicalize(
        cp.asarray(graph.indptr),
        indices=cp.asarray(graph.indices),
        data=cp.asarray(graph.data),
        out_indptr=indptr,
        out_indices=indices,
        out_data=data,
        scratch=scratch,
        stream=cp.cuda.get_current_stream().ptr,
    )
    data = data[:nnz]
    if graph.data.dtype == np.float32 and nnz == graph.nnz:  # nothing summed: the same fp32 values
        data = data.astype(cp.float32)
    return SimpleNamespace(indptr=indptr, indices=indices[:nnz], data=data, host=False)


def _symmetrize(graph: SimpleNamespace, n: int, max_weight: float | None) -> SimpleNamespace:
    """A + A.T (two-operand sums: deterministic) on the device, or on the host for host input and graphs too large for cupyx."""
    host = graph.host or 2 * graph.nnz >= 2**31  # cupyx indexes up to 2 * nnz entries with int32
    if host:
        xp, sp = np, sparse
        data, indices, indptr = (cp.asnumpy(a) for a in (graph.data, graph.indices, graph.indptr))
    else:
        xp, sp = cp, cpx_sparse
        data = graph.data
        indices, indptr = (x.astype(cp.int32, copy=False) for x in (graph.indices, graph.indptr))
    if data.dtype == np.float32:
        if max_weight is None:
            max_weight = float(data.max())
        if max_weight > np.finfo(np.float32).max / 2:
            data = data.astype(np.float64)
    a = sp.csr_matrix((data, indices, indptr), shape=(n, n))
    a = (a + a.T).tocsr()
    if host and not graph.host:
        return _to_device(a)
    return SimpleNamespace(
        indptr=xp.ascontiguousarray(a.indptr),
        indices=xp.ascontiguousarray(a.indices),
        data=xp.ascontiguousarray(a.data),
        host=graph.host,
    )


# One workspace buffer per device, kept between calls. Under an allocator that
# does not pool (rsc's default), every workspace costs a cudaMalloc and a
# cudaFree (on the GB10 about 50 ms per GB). A call takes the cached buffer if
# it is large enough and leaves its own if it is at most
# `_WORKSPACE_CACHE_BYTES` and 1/16 of the device memory. The driver never
# reads workspace bytes the call has not written, so reuse cannot change a
# result.
_WORKSPACE_CACHE: dict[int, cp.ndarray] = {}
with suppress(ImportError):
    from rmm import register_reinitialize_hook

    register_reinitialize_hook(_WORKSPACE_CACHE.clear)


def _workspace_cache_applies() -> bool:
    """True if the current CuPy allocator makes a cudaMalloc per allocation."""
    if _WORKSPACE_CACHE_BYTES <= 0:
        return False
    try:
        import rmm
        from rmm.allocators.cupy import rmm_cupy_allocator
    except ImportError:
        return False
    return cp.cuda.get_allocator() is rmm_cupy_allocator and type(rmm.mr.get_current_device_resource()) in (
        rmm.mr.CudaMemoryResource,
        rmm.mr.ManagedMemoryResource,
    )


def _allocate_workspace(nbytes: int, kind: int | None = None) -> cp.ndarray:
    """A workspace of at least ``nbytes``: the cached one if large enough and in memory of ``kind`` (`_memory_kind`; None: any)."""
    buffer = _WORKSPACE_CACHE.pop(cp.cuda.Device().id, None)
    reuse = buffer is not None and buffer.nbytes >= nbytes and kind in (None, _memory_kind(buffer))
    if not (reuse and _workspace_cache_applies()):
        buffer = None  # free the cached buffer first
        buffer = cp.empty(nbytes, dtype=cp.uint8)
    return buffer


def _cache_workspace(buffer: cp.ndarray) -> None:
    device = cp.cuda.Device()
    limit = min(_WORKSPACE_CACHE_BYTES, device.mem_info[1] // 16)
    if buffer.nbytes <= limit and _workspace_cache_applies():
        _WORKSPACE_CACHE[device.id] = buffer


def _memory_kind(array: cp.ndarray) -> int:
    """DLPack device type of a device array: CUDA or CUDA managed memory."""
    return array.__dlpack_device__()[0]


def _reads_host_memory(device: int) -> bool:
    """Whether the GPU reads pageable host memory directly.

    True through the host page tables (ATS, e.g. GB10); HMM-only systems and discrete GPUs get a device copy.
    """
    rt = cp.cuda.runtime
    return all(
        rt.deviceGetAttribute(attr, device)
        for attr in (
            rt.cudaDevAttrPageableMemoryAccess,
            rt.cudaDevAttrPageableMemoryAccessUsesHostPageTables,
        )
    )


def _host_labels(shape: tuple[int, int]) -> np.ndarray:
    """int32 labels the native call copies each resolution into, in pages touched first (on the GB10 up to 50x faster)."""
    codes = np.empty(shape, dtype=np.int32)
    codes.reshape(-1).view(np.uint8)[::4096] = 0
    return codes


def _compact_initial_membership(initial_membership, n: int) -> cp.ndarray | None:
    """Validate and rank the ids (order-preserving, as the driver's COMPACT)."""
    if initial_membership is None:
        return None
    if isinstance(initial_membership, cp.ndarray):
        membership = _copy_to_gpu(initial_membership, cp.cuda.Device().id)
    else:
        membership = cp.asarray(np.asarray(initial_membership))
    if membership.shape != (n,) or membership.dtype.kind not in "iu":
        msg = f"initial_membership must hold one integer cluster id for each of the {n} cells."
        raise ValueError(msg)
    if n and int(membership.min()) < 0:
        msg = "initial_membership must not contain negative cluster ids."
        raise ValueError(msg)
    _, ranks = cp.unique(membership, return_inverse=True)
    return cp.ascontiguousarray(ranks.ravel(), dtype=cp.int32)


def _leiden_cugraph(
    adjacency,
    resolutions: list[float],
    *,
    rng: np.random.Generator,
    theta: float | None,
    dtype: str | np.dtype,
    use_dask: bool,
    use_weights: bool,
    max_iter: int = _CUGRAPH_MAX_ITER,
    objective_function: str = "modularity",
) -> list[SimpleNamespace]:
    """Run :func:`cugraph.leiden` once per resolution."""
    if objective_function.lower() != "modularity":
        msg = "flavor='cugraph' only supports objective_function='modularity'."
        raise NotImplementedError(msg)
    # cuGraph's leiden is seeded: one seed per resolution, kept for a retry
    seeds = [_seed_from_rng(rng) for _ in resolutions]
    kw = {"seeds": seeds, "theta": theta, "dtype": dtype, "use_weights": use_weights, "max_iter": max_iter}
    if use_dask:  # an out-of-memory worker leaves the others hanging: no retry
        groups = _cugraph_groups(adjacency, resolutions, **kw, use_dask=True)
    else:
        try:
            groups = _cugraph_groups(adjacency, resolutions, **kw, use_dask=False)
        except (MemoryError, RuntimeError) as err:
            if not isinstance(err, MemoryError) and "out_of_memory" not in str(err):
                raise
            groups = None  # the retry starts after the failed call has been freed
        if groups is None:
            msg = "cuGraph ran out of GPU memory; retrying in CUDA managed memory (slower)."
            warnings.warn(msg, UserWarning, skip_file_prefixes=_SKIP)
            with _managed_memory():
                groups = _cugraph_groups(adjacency, resolutions, **kw, use_dask=False)
    parts = []
    for group, modularity in groups:
        membership, n_clusters = _size_ordered_codes(group)
        # cuGraph's modularity (not always that of the returned partition)
        parts.append(SimpleNamespace(membership=membership, modularity=float(modularity), n_clusters=n_clusters))
    return parts


def _cugraph_groups(
    adjacency, resolutions, *, seeds, theta, dtype, use_weights, max_iter, use_dask
) -> list[tuple[np.ndarray, float]]:
    """Communities and cuGraph's modularity of every resolution; cuGraph's graph is freed on return."""
    culeiden = _import_cugraph(use_dask=use_dask).leiden
    create_graph = _create_graph_dask if use_dask else _create_graph
    g = create_graph(adjacency, dtype, use_weights=use_weights)
    groups = []
    try:
        for resolution, seed in zip(resolutions, seeds, strict=True):
            leiden_parts, modularity = culeiden(
                g,
                resolution=resolution,
                random_state=seed,
                theta=1.0 if theta is None else theta,
                max_iter=max_iter,
            )
            if use_dask:
                leiden_parts = leiden_parts.to_backend("pandas").compute()
            else:
                leiden_parts = leiden_parts.to_pandas()
            groups.append((_vertex_groups(leiden_parts, adjacency.shape[0]), modularity))
    finally:
        # tear down the comms even on failure: later Dask calls would fail
        if use_dask:
            import cugraph.dask.comms.comms as Comms

            Comms.destroy()
    return groups


_MANAGED_MEMORY_USERS: dict[int, tuple] = {}
_MANAGED_MEMORY_LOCK = Lock()


@contextmanager
def _managed_memory():
    """Route cuGraph's and CuPy's allocations through RMM to CUDA managed memory, which can exceed the GPU."""
    import rmm
    from rmm.allocators.cupy import rmm_cupy_allocator

    # Concurrent retries share one managed resource per device until the last one leaves.
    device = cp.cuda.Device().id
    with _MANAGED_MEMORY_LOCK:
        previous, users = _MANAGED_MEMORY_USERS.get(device, (None, 0))
        if not users:
            previous = rmm.mr.get_per_device_resource(device)
            rmm.mr.set_per_device_resource(device, rmm.mr.ManagedMemoryResource())
        _MANAGED_MEMORY_USERS[device] = previous, users + 1
    try:
        with cp.cuda.using_allocator(rmm_cupy_allocator):
            yield
    finally:
        with _MANAGED_MEMORY_LOCK:
            _, users = _MANAGED_MEMORY_USERS[device]
            if users == 1:
                rmm.mr.set_per_device_resource(device, previous)
                del _MANAGED_MEMORY_USERS[device]
            else:
                _MANAGED_MEMORY_USERS[device] = previous, users - 1


def _vertex_groups(leiden_parts: pd.DataFrame, n: int) -> np.ndarray:
    """Community of each vertex from cuGraph's (vertex, partition) frame."""
    partition = leiden_parts["partition"].to_numpy().astype(np.int64)
    # vertices missing from the frame (isolated ones in a Dask graph) stay alone
    offset = int(partition.max()) + 1 if partition.size else 0
    groups = np.arange(n, dtype=np.int64) + offset
    groups[leiden_parts["vertex"].to_numpy()] = partition
    return groups


def _size_ordered_codes(groups: np.ndarray) -> tuple[np.ndarray, int]:
    """Relabel to 0..C-1 by decreasing size; equal sizes by their first vertex."""
    _, first, inverse, counts = np.unique(groups, return_index=True, return_inverse=True, return_counts=True)
    order = np.lexsort((first, -counts))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank[inverse].astype(np.int32), int(order.size)
