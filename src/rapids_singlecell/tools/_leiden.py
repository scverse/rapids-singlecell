from __future__ import annotations

import math
import warnings
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

# Workspace arena in units of 8 * nnz bytes. A level that does not fit ends
# the call; the (rare, bitwise identical) rerun grows the factor by
# `_WORKSPACE_GROWTH`, or to the driver's request plus slack.
_ARENA_FACTOR = 1.9
_WORKSPACE_GROWTH = 2.0
_REQUIRED_SLACK = 1.25
_MAX_RERUNS = 6
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
        connectivities.
    directed
        Must not be ``True``: the graph is treated as undirected.
    use_weights
        If ``True``, edge weights from the graph are used in the computation
        (placing more emphasis on stronger edges).
    n_iterations
        How many iterations of the Leiden algorithm to perform (igraph
        semantics: each iteration restarts from the previous partition).
        ``-1`` iterates until the modularity gain of an iteration drops below
        1e-6 (at most 20 iterations); unlike igraph's ``-1`` this does not wait
        until no cell changes its cluster. Values above 20 behave like ``-1``,
        with a warning. ``flavor='cugraph'`` ignores it and runs up to 100
        levels.
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
        ``'rapids'`` needs a device workspace of about 4x the graph (float32
        CSR) and raises if it cannot be allocated. Under an allocator that does
        not pool (the default of rapids-singlecell), it keeps the workspace for
        the next call if it is at most 2 GiB and 1/16 of the GPU memory.
    copy
        Whether to copy ``adata`` or modify it in place.
    theta
        cuGraph's refinement randomness, only used by ``flavor='cugraph'``
        (``None`` means 1.0). ``flavor='rapids'`` ignores it with a warning;
        see ``beta``.
    dtype
        Weight dtype of the cuGraph graph (``'float32'`` or ``'float64'``).
        Only used by ``flavor='cugraph'``; the native implementation reads the
        weights in their own precision.
    use_dask
        Deprecated. If ``True``, run cuGraph's Dask implementation on all GPUs
        (this implies ``flavor='cugraph'``).
    **clustering_args
        Further arguments for the clustering. ``flavor='rapids'`` accepts
        ``beta`` (randomness of the refinement in igraph's units; the default 0
        merges along the best edge), ``initial_membership`` and
        ``objective_function='modularity'``. ``flavor='cugraph'`` accepts
        ``max_iter`` (maximum number of levels, default 100) and
        ``objective_function='modularity'``.

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
        gamma, computed in float64 from the graph (a list for several
        resolutions). Use :func:`scanpy.metrics.modularity` for the score
        with gamma = 1.
    """
    flavor = _validate_flavor(
        flavor,
        directed=directed,
        use_dask=use_dask,
        theta=theta,
        clustering_args=clustering_args,
    )
    dtype = _check_dtype(dtype)
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
    adata.uns[key_added]["modularity"] = (
        parts[0].modularity if len(parts) == 1 else [p.modularity for p in parts]
    )
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
                msg = "use_dask is deprecated and runs the legacy cuGraph implementation; use flavor='rapids' instead."
                warnings.warn(msg, FutureWarning, stacklevel=4)
            flavor = "cugraph"
        case "cugraph":
            if deprecate_cugraph:
                msg = "flavor='cugraph' is deprecated and will be removed in 0.20."
                warnings.warn(msg, FutureWarning, stacklevel=4)
        case "rapids":
            if theta is not None:
                msg = "theta is a cuGraph parameter (it had no effect there) and is ignored by flavor='rapids'; see beta."
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


def _validate_n_iterations(
    n_iterations: int, *, flavor: Literal["rapids", "cugraph"]
) -> None:
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
        msg = f"n_iterations counts Leiden iterations (igraph semantics), not cuGraph levels; values above {_N_ITER_WARN} run until stable (as -1); 2 is recommended."
        warnings.warn(msg, UserWarning, stacklevel=4)


def _validate_resolutions(resolution: float | Sequence[float]) -> list[float]:
    resolutions = [resolution] if isinstance(resolution, Real) else list(resolution)
    if not resolutions:
        msg = "resolution must be a number or a non-empty sequence of numbers."
        raise ValueError(msg)
    if not all(
        isinstance(res, Real) and math.isfinite(res) and 0 <= res <= _MAX_RESOLUTION
        for res in resolutions
    ):
        msg = f"resolution must be finite and in [0, 2**20], but {resolution!r} was passed."
        raise ValueError(msg)
    return resolutions


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
    csr = _as_csr(adjacency)
    n = csr.shape[0]
    membership = _compact_initial_membership(initial_membership, n)
    host_input = (
        _HOST_ZERO_COPY
        and not cpx_sparse.issparse(csr)
        and _reads_host_memory(cp.cuda.Device().id)
    )
    graph, info, use_weights = _ingest(
        csr, use_weights=use_weights, host_input=host_input
    )
    if info["nnz_counted"] == 0:  # no edge: every cell is its own cluster
        codes = np.arange(n, dtype=np.int32)
        return [
            SimpleNamespace(membership=codes, modularity=0.0, n_clusters=n)
            for _ in resolutions
        ]
    args = {
        "indices": graph.indices,
        "data": graph.data,
        "use_weights": use_weights,
        "check": info,
        "resolutions": [float(res) for res in resolutions],
        # the native kernels are seeded: one seed per resolution
        "seeds": [
            _seed_from_rng(rng, allow_none=False) & 0xFFFFFFFF for _ in resolutions
        ],
        "n_iterations": n_iterations,
        "initial_membership": membership,
        "stream": cp.cuda.get_current_stream().ptr,
    }
    pinned = cp.cuda.alloc_pinned_memory(_ld.PINNED_BYTES)  # readback buffer
    factor = _ARENA_FACTOR
    for _ in range(_MAX_RERUNS + 1):
        # a call without a workspace sizes it
        nbytes = _ld.leiden(graph.indptr, **args, arena_factor=factor)
        buffer, workspace, labels = _allocate_workspace(
            nbytes["required_bytes"], (len(resolutions), n)
        )
        out = _ld.leiden(
            graph.indptr,
            **args,
            workspace=workspace,
            pinned=pinned.ptr,
            labels_out=labels,
            arena_factor=factor,
        )
        if out["status"] == "ok":
            codes = _labels_to_host(labels)
            _cache_workspace(buffer)
            break
        # rare: a level did not fit; free the buffer before allocating anew
        del buffer, workspace, labels
        factor = max(
            _WORKSPACE_GROWTH * factor, _REQUIRED_SLACK * out["required_arena_factor"]
        )
    else:
        msg = f"Native Leiden reported workspace_too_small {_MAX_RERUNS + 1} times in a row."
        raise RuntimeError(msg)
    return [
        SimpleNamespace(membership=c, modularity=float(q), n_clusters=int(k))
        for c, q, k in zip(codes, out["modularity"], out["n_clusters"], strict=True)
    ]


def _as_csr(adjacency):
    """CSR with float weights, kept on the host or device it lives on."""
    if cpx_sparse.issparse(adjacency) or sparse.issparse(adjacency):
        csr = adjacency.tocsr()
    elif isinstance(adjacency, cp.ndarray):
        csr = cpx_sparse.csr_matrix(adjacency)
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
        graph = SimpleNamespace(
            indptr=csr.indptr, indices=csr.indices, data=csr.data, host=False
        )
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
    indices = cp.empty(
        csr.indices.shape[0], dtype=cp.int32 if narrow else csr.indices.dtype
    )
    data = cp.empty(csr.data.shape[0], dtype=csr.data.dtype)
    _copy_to_device(csr.indices, indices, narrow_n=csr.shape[0] if narrow else None)
    _copy_to_device(csr.data, data)
    return SimpleNamespace(
        indptr=cp.asarray(csr.indptr), indices=indices, data=data, host=False
    )


def _copy_to_device(src: np.ndarray, dst: cp.ndarray, *, narrow_n=None) -> None:
    """``dst[:] = src`` through two pinned buffers (copies overlap); with
    ``narrow_n``, the int64 ``src`` is narrowed into the int32 ``dst``."""
    stream = cp.cuda.get_current_stream()
    src = np.ascontiguousarray(src)
    count = src.shape[0]
    step = max(1, _H2D_CHUNK_BYTES // src.itemsize)
    n_buffers = min(2, -(-count // step))
    pinned = [
        cp.cuda.alloc_pinned_memory(step * src.itemsize) for _ in range(n_buffers)
    ]
    staging = (
        [cp.empty(step, dtype=src.dtype) for _ in range(n_buffers)]
        if narrow_n is not None
        else None
    )
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
    return SimpleNamespace(
        indptr=indptr, indices=indices[:nnz], data=data[:nnz], host=False
    )


def _symmetrize(graph: SimpleNamespace, n: int) -> SimpleNamespace:
    """A + A.T on the side the arrays live on (two-operand sums: deterministic)."""
    if graph.host:
        a = sparse.csr_matrix((graph.data, graph.indices, graph.indptr), shape=(n, n))
        xp = np
    else:
        if graph.nnz >= 2**31:
            msg = "Symmetrizing an asymmetric adjacency with >= 2**31 entries is not supported."
            raise ValueError(msg)
        a = cpx_sparse.csr_matrix(
            (graph.data, graph.indices.astype(cp.int32), graph.indptr.astype(cp.int32)),
            shape=(n, n),
        )
        xp = cp
    a = (a + a.T).tocsr()
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
    return cp.cuda.get_allocator() is rmm_cupy_allocator and type(
        rmm.mr.get_current_device_resource()
    ) in (rmm.mr.CudaMemoryResource, rmm.mr.ManagedMemoryResource)


def _allocate_workspace(
    nbytes: int, labels_shape: tuple[int, int]
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    """One buffer holding the workspace (``nbytes``) and the int32 labels."""
    offset = -(-nbytes // 256) * 256
    total = offset + 4 * labels_shape[0] * labels_shape[1]
    buffer = _WORKSPACE_CACHE.pop(cp.cuda.Device().id, None)
    if buffer is None or buffer.nbytes < total or not _workspace_cache_applies():
        buffer = None  # free the cached buffer first
        buffer = cp.empty(total, dtype=cp.uint8)
    labels = buffer[offset:total].view(cp.int32).reshape(labels_shape)
    return buffer, buffer[:nbytes], labels


def _cache_workspace(buffer: cp.ndarray) -> None:
    device = cp.cuda.Device()
    limit = min(_WORKSPACE_CACHE_BYTES, device.mem_info[1] // 16)
    if buffer.nbytes <= limit and _workspace_cache_applies():
        _WORKSPACE_CACHE[device.id] = buffer


def _reads_host_memory(device: int) -> bool:
    """The GPU reads pageable memory through the host page tables (ATS, e.g.
    GB10); HMM-only systems and discrete GPUs get a device copy."""
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
        initial_membership
        if isinstance(initial_membership, cp.ndarray)
        else np.asarray(initial_membership)
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
    if use_dask:
        from cugraph.dask import leiden as culeiden

        g = _create_graph_dask(adjacency, dtype, use_weights=use_weights)
    else:
        from cugraph import leiden as culeiden

        g = _create_graph(adjacency, dtype, use_weights=use_weights)
    if not hasattr(_ld, "ingest_check"):  # no native extension: float64 sums
        graph, info, weighted = _as_csr(adjacency), None, use_weights
    else:
        # cuGraph clusters an asymmetric adjacency as an undirected graph:
        # score A + A.T without the warning of flavor='rapids'
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            graph, info, weighted = _ingest(
                _as_csr(adjacency), use_weights=use_weights, host_input=False
            )
    modularity = _exact_modularity(graph, info, use_weights=weighted)
    parts = []
    try:
        for resolution in resolutions:
            leiden_parts, _ = culeiden(
                g,
                resolution=resolution,
                # cuGraph's leiden is seeded, so draw the seed right here
                random_state=_seed_from_rng(rng),
                theta=1.0 if theta is None else theta,
                max_iter=max_iter,
            )
            if use_dask:
                leiden_parts = leiden_parts.to_backend("pandas").compute()
            else:
                leiden_parts = leiden_parts.to_pandas()
            membership, n_clusters = _size_ordered_codes(
                _vertex_groups(leiden_parts, adjacency.shape[0])
            )
            parts.append(
                SimpleNamespace(
                    membership=membership,
                    # cuGraph's modularity is not that of the returned partition
                    modularity=modularity(membership, n_clusters, resolution),
                    n_clusters=n_clusters,
                )
            )
    finally:
        # tear down the comms even on failure: later Dask calls would fail
        if use_dask:
            import cugraph.dask.comms.comms as Comms

            Comms.destroy()
    return parts


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
    _, first, inverse, counts = np.unique(
        groups, return_index=True, return_inverse=True, return_counts=True
    )
    order = np.lexsort((first, -counts))
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    return rank[inverse].astype(np.int32), int(order.size)


def _exact_modularity(graph, info: dict | None, *, use_weights: bool):
    """Modularity of compact labels with the native driver's arithmetic.

    Takes `_ingest`'s results: weights quantised at the resolution's scale,
    exact integer sums and a TREE1024 sum of the volume terms, as in
    ``flavor='rapids'`` (up to the label order of that sum). With
    ``info=None`` (no native extension) the weights are summed in float64.
    """
    indptr = cp.asarray(graph.indptr, dtype=cp.int64)
    indices = cp.asarray(graph.indices, dtype=cp.int64)
    data = cp.asarray(graph.data, dtype=cp.float64)
    n = indptr.shape[0] - 1
    rows = cp.searchsorted(indptr[1:], cp.arange(indices.shape[0]), side="right")
    counted = (rows != indices) & ((data > 0) if use_weights else (data != 0))
    value = data if use_weights else cp.ones_like(data)

    def modularity(membership: np.ndarray, n_clusters: int, resolution: float):
        if info is None:
            q = cp.where(counted, value, 0.0)
        else:
            s = _scale_exponent(info["nnz_counted"], info["wmax"], resolution)
            q = cp.where(counted, cp.maximum(cp.rint(cp.ldexp(value, s)), 1), 0)
            q = q.astype(cp.int64)
        khat = cp.zeros(n, q.dtype)
        cp.add.at(khat, rows, q)
        labels = cp.asarray(membership, dtype=cp.int64)
        volumes = cp.zeros(n_clusters, q.dtype)
        cp.add.at(volumes, labels, khat)
        two_m = float(khat.sum().item())
        if two_m == 0:  # no counted entry
            return 0.0
        inner = float(q[labels[rows] == labels[indices]].sum().item())
        x = volumes.get().astype(np.float64) / two_m
        return float(inner / two_m - float(resolution) * _tree1024(x * x))

    return modularity


def _scale_exponent(nnz_counted: int, wmax: float, resolution: float) -> int:
    """Fixed-point scale exponent, as the native `scale_for_gamma`."""
    if not (nnz_counted > 0 and wmax > 0):
        return 0
    f_m, e_m = math.frexp(float(wmax))
    f_w, e_w = math.frexp(float(nnz_counted) * f_m)
    e_q = math.frexp(1.0 / f_w)[1]
    s = 58 - (e_w + e_m) + (e_q - 1)
    if resolution > 16.0:
        f, e = math.frexp(resolution / 16.0)
        s -= e - 1 if f == 0.5 else e
    return s


def _tree1024(t: np.ndarray) -> float:
    """Fixed-order float64 sum: 1024 strided partials, then a tree."""
    acc = np.zeros(1024)
    for start in range(0, t.shape[0], 1024):
        chunk = t[start : start + 1024]
        acc[: chunk.shape[0]] += chunk
    width = 512
    while width:
        acc[:width] += acc[width : 2 * width]
        width //= 2
    return float(acc[0])
