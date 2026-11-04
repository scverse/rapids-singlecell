from __future__ import annotations

import math
import warnings
from contextlib import contextmanager
from numbers import Integral, Real
from types import SimpleNamespace
from typing import TYPE_CHECKING, Literal

import cupy as cp
import numpy as np
import pandas as pd
from cupyx.scipy import sparse as cpx_sparse
from natsort import natsorted
from scanpy import logging as logg
from scanpy.tools._utils import _choose_graph
from scanpy.tools._utils_clustering import rename_groups, restrict_adjacency
from scipy import sparse

from rapids_singlecell._cuda import _leiden_cuda as _ld
from rapids_singlecell._utils._multi_gpu import _copy_to_device as _copy_to_gpu
from rapids_singlecell._utils._multi_gpu import (
    _peer_copies_work,
    resolve_stream_devices,
)
from rapids_singlecell._utils._random import (
    _accepts_legacy_random_state,
    _LegacyRng,
    _seed_from_rng,
)

from ._clustering import _check_dtype, _create_graph, _create_graph_dask

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

    from rapids_singlecell._utils._random import RNGLike, SeedLike

# Host input is read in place on GPUs with ATS (e.g. GB10); elsewhere it is
# copied to the device in chunks through pinned buffers.
_HOST_ZERO_COPY = True
_H2D_CHUNK_BYTES = 1 << 23
_WORKSPACE_CACHE_BYTES = 1 << 31  # see `_WORKSPACE_CACHE`; 0 disables it
_N_ITER_WARN = 20  # above this: warn and run the -1 rule
# Test knobs: `_N_PARTITIONS` forces the number of partitions (round-robin
# over the devices, several per device); `_MAX_REPLICAS` caps the replicas of
# small graphs (1: none, which partitioned runs require).
_N_PARTITIONS: int | None = None
_MAX_REPLICAS: int | None = None
_MAX_RESOLUTION = 2.0**20  # bound of the integer no-overflow proof
# cuGraph returns the refined partition when it hits `max_iter` levels, so
# `n_iterations` is not forwarded and the cap stays well above what it needs.
_CUGRAPH_MAX_ITER = 100
# The native implementation is the default; this also starts the deprecation
# of flavor="cugraph" and use_dask (FutureWarning).
_DEFAULT_FLAVOR: Literal["rapids", "cugraph"] = "rapids"

_CLUSTERING_ARGS = {
    "rapids": frozenset({"beta", "initial_membership", "objective_function"}),
    "cugraph": frozenset({"max_iter", "objective_function"}),
}


@_accepts_legacy_random_state(0)
def leiden(
    adata: AnnData,
    resolution: float | Sequence[float] = 1.0,
    *,
    restrict_to: tuple[str, Sequence[str]] | None = None,
    rng: SeedLike | RNGLike | None = None,
    key_added: str = "leiden",
    adjacency: sparse.spmatrix | sparse.sparray | cpx_sparse.spmatrix | None = None,
    directed: bool | None = None,
    use_weights: bool = True,
    n_iterations: int = 2,
    neighbors_key: str | None = None,
    obsp: str | None = None,
    flavor: Literal["rapids", "cugraph"] = _DEFAULT_FLAVOR,
    copy: bool = False,
    theta: float | None = None,
    dtype: str | np.dtype = np.float32,
    use_dask: bool = False,
    multi_gpu: bool | list[int] | str | None = None,
    **clustering_args,
) -> AnnData | None:
    """Cluster cells into subgroups using the Leiden algorithm :cite:p:`Traag2019`.

    Leiden is an improved version of the Louvain algorithm :cite:p:`Blondel2008`
    that yields connected communities. This requires having run
    :func:`~rapids_singlecell.pp.neighbors` first.

    Parameters
    ----------
    adata
        Annotated data matrix.
    resolution
        A parameter value controlling the coarseness of the clustering (called
        gamma in the modularity formula). Higher values lead to more clusters.
        If a list of values is given, the clustering runs for each of them and
        writes one column per value. Each value must be finite and in
        ``[0, 2**20]``.
    restrict_to
        Restrict the clustering to the categories within the key for sample
        annotation, tuple needs to contain ``(obs_key, list_of_categories)``.
    rng
        Random seed or :class:`~numpy.random.Generator` changing the
        initialization of the optimization. Defaults to 0.
        The superseded ``random_state`` argument is still accepted.
    key_added
        ``adata.obs`` key under which to add the cluster labels.
    adjacency
        Sparse adjacency matrix of the graph, defaults to neighbors
        connectivities. ``flavor='rapids'`` sums duplicate entries, clusters
        an asymmetric matrix as A + A.T (with a warning) and ignores the
        diagonal (self-loops); every stored weight must be finite and
        non-negative.
    directed
        Must not be ``True``: the graph is treated as undirected.
    use_weights
        If ``True``, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    n_iterations
        How many iterations of the Leiden algorithm to perform, as in
        ``scanpy.tl.leiden``; each iteration restarts from the partition of the
        previous one :cite:p:`Traag2019`. ``-1`` iterates until an iteration
        moves no cell or improves the modularity by less than 1e-6, then runs
        a last one (at most 20 in all). Values above 20 behave like ``-1``,
        with a warning.
        ``flavor='cugraph'`` ignores it and runs up to 100 levels.
    neighbors_key
        If not specified, ``leiden`` looks at ``.obsp['connectivities']`` for
        neighbors connectivities. If specified, ``leiden`` looks at
        ``.obsp[.uns[neighbors_key]['connectivities_key']]`` for neighbors
        connectivities.
    obsp
        Use ``.obsp[obsp]`` as adjacency. You can't specify both ``obsp`` and
        ``neighbors_key`` at the same time.
    flavor
        Which implementation to use. ``'rapids'`` (default) is the native CUDA
        implementation of rapids-singlecell, bitwise reproducible for a given
        seed; ``'cugraph'`` (deprecated) uses :func:`cugraph.leiden`.
        ``'rapids'`` needs about 140 bytes of device memory per cell plus
        what its coarser levels hold, allocated as they need it: on k-NN
        graphs 14-18 bytes per stored entry at the peak (about 22 at 100M
        cells), in all 2.5-3.5x the float32 CSR, which is also copied to the
        GPU if it is on the host. If that does not fit, it raises
        :class:`MemoryError` (RMM managed memory can exceed the GPU). Under an
        allocator that does not pool (the default of rapids-singlecell), it
        keeps the per-cell workspace for the next call if it is at most 2 GiB
        and 1/16 of the GPU memory.
    copy
        Whether to copy ``adata`` or modify it in place.
    theta
        Deprecated. cuGraph's refinement randomness, only used by
        ``flavor='cugraph'`` (``None`` means 1.0). ``flavor='rapids'`` ignores
        it with a warning; see ``beta``.
    dtype
        Weight dtype of the cuGraph graph (``'float32'`` or ``'float64'``).
        Only used by the deprecated ``flavor='cugraph'``; the native
        implementation reads float32 or float64 weights and rounds them to one
        64-bit fixed-point grid.
    use_dask
        Deprecated. If ``True``, run cuGraph's Dask implementation on all GPUs
        (this implies ``flavor='cugraph'``).
    multi_gpu
        GPUs of ``flavor='rapids'`` (not supported by ``flavor='cugraph'``):

        - ``None`` or ``False`` (default): the current GPU.
        - ``True``: all GPUs, but only for a graph that runs out of memory
          on the current GPU (it reruns split, from the start); others run
          there.
        - ``list[int]`` or comma-separated ``str`` (e.g. ``"0,2"``): these
          GPUs, always split (one GPU: run on it).

        A split run gives each GPU one contiguous range of the cells,
        balanced by stored entries: it stores the rows of its cells, their
        contraction as it needs it and about 140 bytes of state per cell of
        the graph. One GPU (the first listed one, or the next if the first
        holds a CuPy adjacency; with ``True``, the one with the most free
        memory) also holds the coarser levels, so 2 GPUs fit graphs only
        about 1.1x as large and 4 GPUs about 1.2x. A split run synchronises
        the GPUs many times per level, so it is slower than one GPU; labels
        and modularity are bitwise identical. Graphs of at most ``2**18``
        counted entries, which one GPU clusters as several replicas, warn and
        run on one GPU.
    **clustering_args
        Further arguments for the clustering. ``flavor='rapids'`` accepts
        ``beta`` (randomness of the refinement, theta in :cite:t:`Traag2019`;
        only the default 0, which merges along the best edge, is supported),
        ``initial_membership`` and ``objective_function='modularity'``.
        ``flavor='cugraph'`` accepts ``max_iter`` (maximum number of levels,
        default 100) and ``objective_function='modularity'``.

    Returns
    -------
    Returns ``None`` if ``copy=False``, else returns an ``AnnData`` object. Sets
    the following fields:

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
        gamma (a list for several resolutions), computed exactly without
        self-loops and with the weights rounded to one 64-bit fixed-point
        grid, which differs from float64 sums only for weights spanning many
        orders of magnitude or a resolution near ``2**20``
        (``flavor='cugraph'``: cuGraph's value). Use
        :func:`scanpy.metrics.modularity` for the score with gamma = 1 (it
        counts self-loops).
    """
    flavor = _validate_flavor(
        flavor,
        directed=directed,
        use_dask=use_dask,
        theta=theta,
        clustering_args=clustering_args,
    )
    if flavor == "cugraph" and multi_gpu is not None and multi_gpu is not False:
        msg = "multi_gpu is supported by flavor='rapids'; flavor='cugraph' runs on all GPUs with use_dask=True."
        raise ValueError(msg)
    dtype = _check_dtype(dtype)
    use_weights = bool(use_weights)  # the bindings take a Python bool
    _validate_n_iterations(n_iterations, flavor=flavor)
    resolutions = _validate_resolutions(resolution)
    rng = np.random.default_rng(rng)
    meta_random_state = {"random_state": rng.arg} if isinstance(rng, _LegacyRng) else {}

    start = logg.info("running Leiden clustering")
    adata = adata.copy() if copy else adata
    # are we clustering a user-provided graph or the default AnnData one?
    if adjacency is None:
        adjacency = _choose_graph(adata, obsp, neighbors_key)
    if restrict_to is not None:
        restrict_key, restrict_categories = restrict_to
        adjacency, restrict_indices = restrict_adjacency(
            adata,
            restrict_key,
            restrict_categories=restrict_categories,
            adjacency=adjacency,
        )
    if flavor == "rapids":
        parts = _leiden_rapids(
            adjacency,
            resolutions,
            rng=rng,
            n_iterations=n_iterations,
            use_weights=use_weights,
            multi_gpu=multi_gpu,
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
    # store output into adata.obs
    if restrict_to is not None and key_added == "leiden":
        key_added += "_R"
    for res, part in zip(resolutions, parts, strict=True):
        key = key_added if len(resolutions) == 1 else f"{key_added}_{res}"
        if restrict_to is not None:
            groups = rename_groups(
                adata,
                restrict_key,
                key_added=key,
                restrict_categories=restrict_categories,
                restrict_indices=restrict_indices,
                groups=part.membership,
            )
            adata.obs[key] = pd.Categorical(
                values=groups.astype("U"),
                categories=natsorted(map(str, np.unique(groups))),
            )
        else:
            # the codes are size-ordered, so this equals the natsorted strings
            adata.obs[key] = pd.Categorical.from_codes(
                part.membership,
                categories=[str(i) for i in range(part.n_clusters)],
            )
    # store information on the clustering parameters
    adata.uns[key_added] = {}
    adata.uns[key_added]["params"] = {
        "resolution": resolutions[0] if len(resolutions) == 1 else resolutions,
        "n_iterations": n_iterations,
        **meta_random_state,
    }
    adata.uns[key_added]["modularity"] = parts[0].modularity if len(parts) == 1 else [p.modularity for p in parts]
    logg.info(
        "    finished",
        time=start,
        deep=(
            f"found {', '.join(str(p.n_clusters) for p in parts)} clusters and added\n"
            f"    {key_added!r}, the cluster labels (adata.obs, categorical)"
        ),
    )
    return adata if copy else None


def _validate_flavor(
    flavor: str,
    *,
    directed: bool | None,
    use_dask: bool,
    theta: float | None,
    clustering_args: dict,
) -> Literal["rapids", "cugraph"]:
    # stacklevel 4: this helper, `leiden`, the legacy random_state wrapper
    deprecate_cugraph = _DEFAULT_FLAVOR == "rapids"
    match flavor:
        case "rapids" | "cugraph" if use_dask:
            if deprecate_cugraph or flavor == "rapids":
                msg = "use_dask is deprecated and runs the legacy cuGraph implementation, which will be removed in 0.20; use flavor='rapids' instead."
                warnings.warn(msg, FutureWarning, stacklevel=4)
            flavor = "cugraph"
        case "cugraph":
            if deprecate_cugraph:
                msg = "flavor='cugraph' is deprecated and will be removed in 0.20."
                warnings.warn(msg, FutureWarning, stacklevel=4)
        case "rapids":
            if theta is not None:
                msg = "theta is a deprecated cuGraph parameter (it had no effect there) and is ignored by flavor='rapids'; see beta."
                warnings.warn(msg, FutureWarning, stacklevel=4)
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
        msg = "flavor='rapids' is not available: this build of rapids-singlecell does not include the native Leiden extension (`_leiden_cuda`). Use flavor='cugraph'."
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
        warnings.warn(msg, UserWarning, stacklevel=4)


def _validate_resolutions(resolution: float | Sequence[float]) -> list[float]:
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
    if len(set(resolutions)) < len(resolutions):  # each writes one obs column
        msg = f"resolution must not repeat a value, but {resolution!r} was passed."
        raise ValueError(msg)
    return resolutions


def _leiden_rapids(
    adjacency,
    resolutions: list[float],
    *,
    rng: np.random.Generator,
    n_iterations: int,
    use_weights: bool,
    multi_gpu: bool | list[int] | str | None = None,
    beta: float = 0.0,
    initial_membership=None,
    objective_function: str = "modularity",
) -> list[SimpleNamespace]:
    """Run every resolution in one native call (`_leiden_cuda.leiden`, or `leiden_mg` split over GPUs)."""
    if objective_function.lower() != "modularity":
        msg = "flavor='rapids' supports objective_function='modularity' (CPM: planned)."
        raise NotImplementedError(msg)
    if not (math.isfinite(beta) and beta >= 0):
        msg = f"beta must be finite and >= 0, but {beta!r} was passed."
        raise ValueError(msg)
    if beta > 0:
        msg = "flavor='rapids' merges along the best edge (beta=0); randomized refinement (beta > 0) is planned."
        raise NotImplementedError(msg)
    if n_iterations > _N_ITER_WARN:  # warned about in `leiden`
        n_iterations = -1
    devices = resolve_stream_devices(multi_gpu=multi_gpu)
    # a device list is always split (one device: run there); True splits a
    # graph that runs out of memory on the current GPU
    explicit = isinstance(multi_gpu, list | str)
    home = devices[0] if explicit else cp.cuda.Device().id
    source = _input_device(adjacency)
    with cp.cuda.Device(home if source is None else source):
        csr = _as_csr(adjacency)
    with cp.cuda.Device(home):
        if source not in (None, home):  # device input on another GPU
            from rapids_singlecell.tools._rank_genes_groups._wilcoxon_host import _copy_device_sparse_to_device

            csr = _copy_device_sparse_to_device(csr, home)
        n = csr.shape[0]
        membership = _compact_initial_membership(initial_membership, n)
        host_input = _HOST_ZERO_COPY and not cpx_sparse.issparse(csr) and _reads_host_memory(cp.cuda.Device().id)
        graph, info = None, {"nnz_counted": 0}
        if csr.nnz:  # an empty graph needs no check (its size-0 arrays can report another memory kind)
            graph, info, use_weights = _ingest(csr, use_weights=use_weights, host_input=host_input)
        if info["nnz_counted"] == 0:  # no edge: every cell is its own cluster
            codes = np.arange(n, dtype=np.int32)
            return [SimpleNamespace(membership=codes, modularity=0.0, n_clusters=n) for _ in resolutions]
        # the native kernels are seeded: one seed per resolution, whatever the GPUs
        seeds = [_seed_from_rng(rng, allow_none=False) & 0xFFFFFFFF for _ in resolutions]
        max_replicas = 4 if _MAX_REPLICAS is None else _MAX_REPLICAS
        args = {
            "indices": graph.indices,
            "data": graph.data,
            "use_weights": use_weights,
            "check": info,
            "resolutions": [float(res) for res in resolutions],
            "seeds": seeds,
            "n_iterations": int(n_iterations),
            "initial_membership": membership,
            "max_replicas": max_replicas,
            "stream": cp.cuda.get_current_stream().ptr,
        }
        n_parts = _N_PARTITIONS if _N_PARTITIONS is not None else len(devices) if explicit else 1
        if n_parts > 1 and _ld.n_replicas(info["nnz_counted"], n, max_replicas) > 1:
            msg = "multi_gpu: graphs this small run as replicas, which split runs do not have; running on one GPU."
            warnings.warn(msg, UserWarning, stacklevel=4)
            n_parts = 1
        if n_parts == 1:
            try:
                return _leiden_single(graph, args, n)
            except MemoryError:
                if multi_gpu is not True or len(devices) == 1:
                    raise
            n_parts = len(devices)
        for device in devices:  # a split run does not use the cached single-GPU workspaces
            _WORKSPACE_CACHE.pop(device, None)
        if multi_gpu is True:  # partition 0 (levels >= 1: most of one GPU's need) where most is free
            devices = sorted(devices, key=lambda d: cp.cuda.Device(d).mem_info[0], reverse=True)
        owned = cpx_sparse.issparse(adjacency) and graph.indices is getattr(adjacency, "indices", None)
        parts = _split_rows(graph, n_parts, devices, owned=owned)
        del graph, csr, args  # the partitions hold copies of a fixed-up graph
        return _leiden_partitioned(
            parts,
            info,
            n=n,
            use_weights=use_weights,
            resolutions=resolutions,
            seeds=seeds,
            n_iterations=n_iterations,
            membership=membership,
        )


class _ArenaBlocks:
    """Device blocks of the native engine's level arena, allocated as levels need them and freed as they end.

    With ``ready``, a block is usable from any stream when returned (`leiden_mg`
    runs on its own streams, unordered with the allocator's).
    """

    def __init__(self, *, ready: bool = False) -> None:
        self.live: dict[int, cp.ndarray] = {}
        self.ready = ready

    def alloc(self, nbytes: int, device: int) -> int:
        with cp.cuda.Device(device):
            block = cp.empty(nbytes, dtype=cp.uint8)
            if self.ready:
                cp.cuda.get_current_stream().synchronize()
        self.live[block.data.ptr] = block
        return block.data.ptr

    def free(self, ptr: int) -> None:
        del self.live[ptr]


def _leiden_single(graph: SimpleNamespace, args: dict, n: int) -> list[SimpleNamespace]:
    """One GPU: size the workspace and run; the level arena grows by blocks of exactly what a level needs."""
    pinned = cp.cuda.alloc_pinned_memory(_ld.PINNED_BYTES)  # readback buffer
    # the memory kind of the call's other device arrays, which the workspace must share
    arrays = (graph.indptr, args["initial_membership"])
    kind = next((_memory_kind(a) for a in arrays if isinstance(a, cp.ndarray)), None)
    shape = (len(args["resolutions"]), n)
    nbytes = _ld.leiden(graph.indptr, **args)["required_bytes"]  # a call without a workspace sizes it
    buffer, workspace, labels = _allocate_workspace(nbytes, shape, kind)
    blocks = _ArenaBlocks()
    try:
        out = _ld.leiden(
            graph.indptr,
            **args,
            workspace=workspace,
            pinned=pinned.ptr,
            labels_out=labels,
            arena_alloc=blocks.alloc,
            arena_free=blocks.free,
        )
    except BaseException:  # a kept traceback (e.g. IPython's) must not hold the memory
        del buffer, workspace, labels
        raise
    finally:
        blocks.live.clear()
    codes = _labels_to_host(labels)
    _cache_workspace(buffer)
    return [
        SimpleNamespace(membership=c, modularity=float(q), n_clusters=int(k))
        for c, q, k in zip(codes, out["modularity"], out["n_clusters"], strict=True)
    ]


def _row_bounds(indptr: np.ndarray, n_parts: int) -> list[int]:
    """Row bounds of the partitions, balanced by stored entries.

    Every bound but n is a multiple of 32, so each word of an activity bitmap
    has one owner; a 32-row block of hubs can leave partitions empty.
    """
    n, nnz = indptr.shape[0] - 1, int(indptr[-1])
    targets = np.arange(1, n_parts, dtype=np.float64) * nnz / n_parts
    inner = -(-np.searchsorted(indptr, targets, side="left") // 32) * 32
    return [0, *np.maximum.accumulate(np.minimum(inner, n)).tolist(), n]


def _split_rows(graph: SimpleNamespace, n_parts: int, devices: list[int], *, owned: bool) -> SimpleNamespace:
    """The rows of each partition (`_row_bounds`) on its device (`leiden_mg`'s level 0).

    Partition p holds a full-length int64 indptr (other rows empty), int32
    indices and the data of its rows; it views user-owned device input on its
    device, else holds copies, so the caller can free the fixed-up graph.
    Partitions go round-robin over ``devices``; the first (which also runs
    levels >= 1) avoids the device of user-owned input.
    """
    host = isinstance(graph.indptr, np.ndarray)
    indptr = np.asarray(graph.indptr if host else cp.asnumpy(graph.indptr), dtype=np.int64)
    bounds = _row_bounds(indptr, n_parts)
    source = None if host else graph.indices.device.id
    if owned and len(devices) > 1 and devices[0] == source:
        devices = devices[1:] + devices[:1]
    devs = [devices[p % len(devices)] for p in range(n_parts)]
    indptrs, indices, data = [], [], []
    for p, device in enumerate(devs):
        e0, e1 = int(indptr[bounds[p]]), int(indptr[bounds[p + 1]])
        with cp.cuda.Device(device):
            indptrs.append(cp.asarray(np.clip(indptr, e0, e1) - e0))
        indices.append(_rows_on(graph.indices[e0:e1], device, view=owned, dtype=np.int32))
        data.append(_rows_on(graph.data[e0:e1], device, view=owned))
    return SimpleNamespace(bounds=bounds, devices=devs, indptrs=indptrs, indices=indices, data=data)


def _rows_on(array, device: int, *, view: bool, dtype=None) -> cp.ndarray:
    """A partition's entries on its device: a view of user-owned device input there, else a copy."""
    if array.shape[0] == 0:  # a slice: new empty arrays report CUDA memory also under a managed allocator
        with cp.cuda.Device(device):
            return cp.empty(1, dtype=dtype or array.dtype)[:0]
    if isinstance(array, np.ndarray):
        with cp.cuda.Device(device):
            return cp.asarray(np.ascontiguousarray(array, dtype=dtype))
    with cp.cuda.Device(array.device.id):
        if dtype is not None and array.dtype != dtype:
            array, view = array.astype(dtype), True  # a fresh copy
        if array.device.id == device:
            return array if view else array.copy()
    return _copy_to_gpu(array, device)


def _leiden_partitioned(
    parts: SimpleNamespace,
    info: dict,
    *,
    n: int,
    use_weights: bool,
    resolutions: list[float],
    seeds: list[int],
    n_iterations: int,
    membership: cp.ndarray | None,
) -> list[SimpleNamespace]:
    """Every resolution in one `_leiden_cuda.leiden_mg` call over the partitions of `_split_rows`."""
    if len({_memory_kind(a) for a in (*parts.indptrs, *parts.indices, *parts.data)}) > 1:
        msg = "multi_gpu needs one kind of memory on all its GPUs, e.g. RMM managed memory on each: rmm.reinitialize(managed_memory=True, devices=[0, 1])."
        raise ValueError(msg)
    root = parts.devices[0]
    if membership is not None and membership.device.id != root:
        membership = _copy_to_gpu(membership, root)
    peer = _peer_copies_work(sorted(set(parts.devices)))
    args = {
        "indices": parts.indices,
        "data": parts.data,
        "devices": parts.devices,
        "bounds": parts.bounds,
        "use_weights": use_weights,
        "check": info,
        "peer": peer,
        "resolutions": [float(res) for res in resolutions],
        "seeds": seeds,
        "n_iterations": int(n_iterations),
        "initial_membership": membership,
    }
    sizes = _ld.leiden_mg(parts.indptrs, **args, workspaces=[])["required_bytes"]
    workspaces = []
    for device, nbytes in zip(parts.devices, sizes, strict=True):
        with cp.cuda.Device(device):
            workspaces.append(cp.empty(nbytes, dtype=cp.uint8))
    with cp.cuda.Device(root):
        labels = cp.empty((len(resolutions), n), dtype=cp.int32)
    # leiden_mg runs on its own streams: the inputs and buffers must be ready
    for device in {*parts.devices, cp.cuda.Device().id}:
        with cp.cuda.Device(device):
            cp.cuda.get_current_stream().synchronize()
    blocks = _ArenaBlocks(ready=True)
    try:
        out = _ld.leiden_mg(
            parts.indptrs,
            **args,
            workspaces=workspaces,
            labels_out=labels,
            arena_alloc=blocks.alloc,
            arena_free=blocks.free,
        )
    except BaseException:  # a kept traceback (e.g. IPython's) must not hold the memory
        del workspaces, labels
        raise
    finally:
        blocks.live.clear()
    with cp.cuda.Device(root):
        codes = _labels_to_host(labels)
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


def _as_csr(adjacency):
    """CSR with float weights, kept on the host or device it lives on."""
    if isinstance(adjacency, cpx_sparse.coo_matrix):
        # cupyx's tocsr sums duplicates with float atomics (in any order): keep them for `_canonicalize`
        order = cp.argsort(adjacency.row, kind="stable")
        indptr = cp.zeros(adjacency.shape[0] + 1, dtype=cp.int64)
        cp.cumsum(cp.bincount(adjacency.row, minlength=adjacency.shape[0]), out=indptr[1:])
        csr = cpx_sparse.csr_matrix((adjacency.data[order], adjacency.col[order], indptr), shape=adjacency.shape)
    elif cpx_sparse.issparse(adjacency) or sparse.issparse(adjacency):
        csr = adjacency.tocsr()
    elif isinstance(adjacency, cp.ndarray):
        csr = cpx_sparse.csr_matrix(adjacency if adjacency.dtype.kind in "fc" else adjacency.astype(cp.float32))
    else:
        csr = sparse.csr_matrix(np.asarray(adjacency))
    if csr.shape[0] != csr.shape[1]:
        msg = f"adjacency must be square, but has shape {csr.shape}."
        raise ValueError(msg)
    if csr.dtype not in (np.float32, np.float64):
        csr = csr.astype(np.float32)
    # the bindings take int32 / int64 offsets and indices
    if not all(a.dtype in (np.int32, np.int64) for a in (csr.indptr, csr.indices)):
        csr = csr.copy()
        csr.indptr = csr.indptr.astype(np.int64)
        csr.indices = csr.indices.astype(np.int64)
    return csr


def _ingest(csr, *, use_weights: bool, host_input: bool):
    """Check the input natively and fix it up (before sizing the workspace).

    Returns the arrays the driver reads, their `_ld.ingest_check` result and
    the ``use_weights`` they are read with. Device input and, with
    ``host_input``, host input are read in place; other host input is copied.
    """
    n = csr.shape[0]
    if cpx_sparse.issparse(csr):
        arrays = (csr.indptr, csr.indices, csr.data)
        # the bindings take one memory kind per call: copy input of another allocator
        if _memory_kind(csr.indptr) != _memory_kind(cp.empty(1, dtype=cp.uint8)):
            arrays = tuple(a.copy() for a in arrays)
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
            warnings.warn(msg, UserWarning, stacklevel=5)
            graph = _symmetrize(graph, n)


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
    step = max(1, _H2D_CHUNK_BYTES // src.itemsize)
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


def _symmetrize(graph: SimpleNamespace, n: int) -> SimpleNamespace:
    """A + A.T (two-operand sums: deterministic) on the device, or on the host for host input and graphs too large for cupyx."""
    if graph.host or 2 * graph.nnz >= 2**31:  # cupyx indexes A + A.T (up to 2 nnz entries) with int32
        arrays = (cp.asnumpy(graph.data), cp.asnumpy(graph.indices), cp.asnumpy(graph.indptr))
        a = sparse.csr_matrix(arrays, shape=(n, n))
        a = (a + a.T).tocsr()
        if not graph.host:
            return _to_device(a)
        xp = np
    else:
        indices, indptr = (x.astype(cp.int32, copy=False) for x in (graph.indices, graph.indptr))
        a = cpx_sparse.csr_matrix((graph.data, indices, indptr), shape=(n, n))
        a = (a + a.T).tocsr()
        xp = cp
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


def _allocate_workspace(
    nbytes: int, labels_shape: tuple[int, int], kind: int | None = None
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """One buffer holding the workspace (``nbytes``) and the int32 labels.

    A cached buffer is reused only in memory of ``kind`` (`_memory_kind`; None: any).
    """
    offset = -(-nbytes // 256) * 256
    total = offset + 4 * labels_shape[0] * labels_shape[1]
    buffer = _WORKSPACE_CACHE.pop(cp.cuda.Device().id, None)
    reuse = buffer is not None and buffer.nbytes >= total and kind in (None, _memory_kind(buffer))
    if not (reuse and _workspace_cache_applies()):
        buffer = None  # free the cached buffer first
        buffer = cp.empty(total, dtype=cp.uint8)
    labels = buffer[offset:total].view(cp.int32).reshape(labels_shape)
    return buffer, buffer[:nbytes], labels


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


def _labels_to_host(labels: cp.ndarray) -> np.ndarray:
    """D2H copy into pages touched first (on the GB10 up to 50x faster)."""
    codes = np.empty(labels.shape, dtype=labels.dtype)
    codes.fill(0)
    labels.get(out=codes)
    return codes


def _compact_initial_membership(initial_membership, n: int) -> cp.ndarray | None:
    """Validate and rank the ids (order-preserving, as the driver's COMPACT)."""
    if initial_membership is None:
        return None
    membership = cp.asarray(
        initial_membership if isinstance(initial_membership, cp.ndarray) else np.asarray(initial_membership)
    )
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
            warnings.warn(msg, ResourceWarning, stacklevel=4)
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
    if use_dask:
        from cugraph.dask import leiden as culeiden

        g = _create_graph_dask(adjacency, dtype, use_weights=use_weights)
    else:
        from cugraph import leiden as culeiden

        g = _create_graph(adjacency, dtype, use_weights=use_weights)
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


@contextmanager
def _managed_memory():
    """Route RMM allocations (cuGraph's, CuPy's) to CUDA managed memory, which can exceed the GPU."""
    import rmm

    previous = rmm.mr.get_current_device_resource()
    rmm.mr.set_current_device_resource(rmm.mr.ManagedMemoryResource())
    try:
        yield
    finally:
        rmm.mr.set_current_device_resource(previous)


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
