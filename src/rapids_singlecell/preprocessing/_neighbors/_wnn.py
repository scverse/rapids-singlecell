from __future__ import annotations

import itertools
import operator
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal, get_args
from warnings import warn

import cupy as cp
import numpy as np
from cupyx.scipy import sparse as cp_sparse
from scipy import sparse as sc_sparse

from rapids_singlecell._utils._random import RNGLike, SeedLike, _accepts_legacy_random_state, _LegacyRng
from rapids_singlecell.preprocessing._neighbors._helper import _check_neighbors_X
from rapids_singlecell.preprocessing._neighbors._neighbors import (
    KNN_ALGORITHMS,
    _Algorithms,
    _build_sparse_distances,
    _calc_connectivities,
    _large_coo_to_host_csr,
)
from rapids_singlecell.tools._utils import _choose_representation

if TYPE_CHECKING:
    from anndata import AnnData
    from mudata import MuData

_DEFAULT_SEED = 0
# Partner keys per chunk for the rows whose shared-neighbor (SNN) partners do not fit in
# shared memory; they are sorted globally.
_SNN_CHUNK_PAIRS = 1 << 25
_SPREAD3 = ((16, 0x030000FF), (8, 0x0300F00F), (4, 0x030C30C3), (2, 0x09249249))
# all_neighbors pads unfilled neighbor slots with sqrt(FLT_MAX) distances
_PAD_DIST = 1e18


def _stream() -> int:
    return cp.cuda.get_current_stream().ptr


def _to_host(x: cp.ndarray) -> np.ndarray:
    """Copy into host pages the CPU touched first (on GB10, GPU-faulted fresh pages are ~100x slower)."""
    out = np.empty(x.shape, dtype=x.dtype)
    out.reshape(-1).view(np.uint8)[::4096] = 0
    return x.get(out=out)


def _ensure_self_first(knn: cp.ndarray) -> cp.ndarray:
    """Put every cell first in its own neighbor list, as Seurat assumes.

    Exact duplicates can push it back or out of the list; the others keep their order.
    """
    n, k = knn.shape
    rows = cp.arange(n, dtype=knn.dtype)
    bad = cp.flatnonzero(knn[:, 0] != rows)
    if bad.size == 0:
        return knn
    sub = knn[bad]
    is_self = sub == rows[bad, None]
    has_self = is_self.any(axis=1)
    order = cp.argsort(cp.where(is_self, -1, cp.arange(k)), axis=1)
    moved = cp.take_along_axis(sub, order, axis=1)
    shifted = cp.concatenate((rows[bad, None], sub[:, :-1]), axis=1)
    knn[bad] = cp.where(has_self[:, None], moved, shifted)
    return knn


def _snn_max_keys(s: int, key_bytes: int) -> int:
    """Most SNN partners per row, a power of two, that fit in shared memory."""
    smem = cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]
    # the overlap histogram, and 4 KiB for the kernels' static shared memory
    fit = (smem - 4 * (s + 1) - 4096) // key_bytes
    return 1 << (fit.bit_length() - 1) if fit >= 128 else 0


def _snn_launches(knn: cp.ndarray, s: int, *, key_bytes: int, emb: cp.ndarray | None = None):
    """Return a function yielding the SNN kernel arguments for all rows.

    Row ``i``'s shared-neighbor (SNN) partners are the reverse neighbor lists
    of its first ``s`` neighbors: a cell sharing ``c`` neighbors with ``i``
    appears ``c`` times, the overlap Seurat's ``ComputeSNN`` counts. Rows
    whose partners (``key_bytes`` each) fit in shared memory are batched by
    the power of two ``P`` they need. The partners of longer rows are sorted
    globally, in chunks; row ``r`` of a chunk has ``sorted[seg[r]:seg[r + 1]]``.
    With ``emb``, rows follow the Z-order of a fixed random 3-D projection:
    cells close in this order share partners, so the bandwidth kernel finds
    most partner embeddings in the L2 cache. Only the speed depends on it.
    """
    from rapids_singlecell._cuda._wnn_cuda import snn_emit, snn_postings

    n = knn.shape[0]
    counts = cp.zeros(n + 1, dtype=cp.int64)
    # count pass (no postings), then the append pass
    snn_postings(knn, s=s, cursor=counts[1:], postings=None, stream=_stream())
    post_off = cp.cumsum(counts)
    postings = cp.empty(n * s, dtype=cp.int32)
    cursor = post_off[:-1].copy()
    snn_postings(knn, s=s, cursor=cursor, postings=postings, stream=_stream())
    n_keys = counts[1:][knn[:, :s]].sum(axis=1)
    caps = 128 << np.arange(max(_snn_max_keys(s, key_bytes).bit_length() - 7, 0))
    group = cp.searchsorted(cp.asarray(caps), n_keys)
    key = group.astype(cp.uint64) << 30
    if emb is not None:
        proj = emb @ cp.asarray(np.random.default_rng(0).standard_normal((emb.shape[1], 3)), emb.dtype)
        lo, hi = proj.min(axis=0), proj.max(axis=0)
        q = ((proj - lo) / cp.maximum(hi - lo, 1e-30) * 1023).astype(cp.uint64)
        for shift, bits in _SPREAD3:  # bit b of the 10-bit q to bit 3b
            q = (q | q << shift) & bits
        key |= q[:, 0] | q[:, 1] << 1 | q[:, 2] << 2
    order = cp.argsort(key).astype(cp.int32)
    sizes = cp.bincount(group, minlength=caps.size + 1).get()
    bounds = np.concatenate(([0], np.cumsum(sizes)))
    base = {"post_off": post_off, "postings": postings}
    batches = [
        base | {"rows": order[a:b], "P": int(P), "sorted": None, "seg": None}
        for P, a, b in zip(caps, bounds[:-2], bounds[1:-1], strict=True)
        if b > a
    ]
    rest = order[bounds[-2] :]
    ends = cp.cumsum(n_keys[rest]).get()
    if rest.size and np.diff(ends, prepend=0).max() >= 1 << 31:
        raise ValueError("Too many shared neighbors; remove duplicated cells.")

    def launches():
        yield from batches
        a = 0
        while a < rest.size:
            start = ends[a - 1] if a else 0
            end = np.searchsorted(ends, start + _SNN_CHUNK_PAIRS, side="right")
            rows = rest[a : max(end, a + 1)]
            seg = cp.concatenate((cp.zeros(1, cp.int64), cp.cumsum(n_keys[rows])))
            keys = cp.empty(int(ends[a + rows.size - 1] - start), dtype=cp.uint64)
            snn_emit(knn, s=s, rows=rows, **base, seg=seg, keys=keys, stream=_stream())
            keys.sort()  # by row r, then partner: keys are r << 32 | partner
            sorted_keys = keys.astype(cp.uint32)
            yield base | {"rows": rows, "P": 0, "sorted": sorted_keys, "seg": seg}
            a += rows.size

    return launches


def _snn_bandwidth(knn: cp.ndarray, emb: cp.ndarray, nearest: cp.ndarray, *, s_nn: int) -> cp.ndarray:
    """Seurat's ``ComputeSNNwidth``: mean distance to the least shared neighbors."""
    from rapids_singlecell._cuda._wnn_cuda import snn_bandwidth

    sigma = cp.empty(knn.shape[0], dtype=cp.float32)
    out = {"emb": emb, "nearest": nearest, "sigma": sigma, "stream": _stream()}
    for args in _snn_launches(knn, s_nn, key_bytes=8, emb=emb)():
        presorted = args["sorted"] is not None
        dist = cp.empty(args["sorted"].size, cp.float32) if presorted else None
        snn_bandwidth(knn, s=s_nn, **args, **out, dist=dist)
    return sigma


def _snn_graph(knn: cp.ndarray, *, prune: float) -> sc_sparse.csr_matrix:
    """Seurat's ``ComputeSNN``: Jaccard index of the neighbor sets, pruned."""
    from rapids_singlecell._cuda._wnn_cuda import snn_graph

    n, k = knn.shape
    launches = _snn_launches(knn, k, key_bytes=4)
    # count the kept partners of every row into indptr[1:], then write them
    indptr = cp.zeros(n + 1, dtype=cp.int64)
    cols = vals = None
    for fill in (False, True):
        if fill:
            indptr = cp.cumsum(indptr)
            cols = cp.empty(int(indptr[-1]), dtype=cp.int32)
            vals = cp.empty(cols.size, dtype=cp.float32)
            if not cols.size:  # every edge was pruned
                break
        out = {"indptr": indptr, "cols": cols, "vals": vals, "fill": fill}
        for args in launches():
            snn_graph(knn, s=k, **args, **out, prune=prune, stream=_stream())
    if cols.size <= np.iinfo(np.int32).max:
        indptr = indptr.astype(cp.int32)
    snn = sc_sparse.csr_matrix((_to_host(vals), _to_host(cols), _to_host(indptr)), shape=(n, n))
    snn.has_canonical_format = True
    return snn


def _get_reps(adata, use_rep: Sequence[str] | Mapping[str, str], n_pcs) -> tuple[dict[str, dict], list[cp.ndarray]]:
    """Per-modality `use_rep` and resolved `n_pcs`, and the representations."""
    is_mudata = hasattr(adata, "mod")
    if isinstance(use_rep, str):
        raise TypeError("`use_rep` needs one representation per modality.")
    if is_mudata and not isinstance(use_rep, Mapping):
        raise TypeError("For MuData, pass `use_rep` as a mapping of modality to `.obsm` key, e.g. `{'rna': 'X_pca'}`.")
    names = list(use_rep)
    keys = list(use_rep.values()) if isinstance(use_rep, Mapping) else names
    if len(keys) < 2:
        raise ValueError("WNN needs at least two modalities.")
    if len(set(names)) != len(names) or not all(isinstance(n, str) for n in names):
        raise ValueError(f"Modality names {names} must be unique strings; pass `use_rep` as a mapping.")
    if np.ndim(n_pcs) == 0:  # None or one number for all modalities
        n_pcs = [None if n_pcs is None else operator.index(n_pcs)] * len(keys)
    elif len(n_pcs) != len(keys):
        raise ValueError("`n_pcs` needs one entry per modality.")

    modalities, reps = {}, []
    for name, key, npc in zip(names, keys, n_pcs, strict=True):
        mod = adata.mod[name] if is_mudata else adata
        if is_mudata and not mod.obs_names.equals(adata.obs_names):
            raise ValueError(
                f"Modality {name!r} does not hold the cells of the MuData object in the same order. Run "
                "`muon.pp.intersect_obs` and reorder it, "
                f"e.g. `mdata.mod[{name!r}] = mdata[{name!r}][mdata.obs_names].copy()`."
            )
        X = _choose_representation(mod, use_rep=key, n_pcs=npc)
        if cp_sparse.issparse(X) or sc_sparse.issparse(X):
            raise TypeError(f"Representation {key!r} must be dense.")
        reps.append(cp.ascontiguousarray(cp.asarray(X, dtype=cp.float32)))
        modalities[name] = {"use_rep": key, "n_pcs": X.shape[1]}
    return modalities, reps


@_accepts_legacy_random_state(_DEFAULT_SEED)
def wnn(
    adata: AnnData | MuData,
    use_rep: Sequence[str] | Mapping[str, str],
    n_pcs: int | Sequence[int | None] | None = None,
    *,
    n_neighbors: int = 20,
    knn_range: int = 200,
    l2_norm: bool = True,
    sd_scale: float = 1.0,
    cross_constant: float | Sequence[float] = 1e-4,
    smooth: bool = False,
    prune_snn: float = 1 / 15,
    algorithm: _Algorithms = "brute",
    algorithm_kwds: Mapping[str, Any] = MappingProxyType({}),
    method: Literal["umap", "gauss", "jaccard"] = "umap",
    rng: SeedLike | RNGLike | None = None,
    key_added: str | None = None,
    weight_key: str | None = "{}_weight",
    copy: bool = False,
) -> AnnData | MuData | None:
    r"""Weighted nearest neighbor (WNN) graph across modalities :cite:p:`Hao2021`.

    GPU implementation of Seurat's ``FindMultiModalNeighbors``. Each cell gets a weight per modality from how well
    its neighbors there predict its embedding, compared to the other modalities' neighbors. The multimodal neighbors
    maximize the weighted sum of per-modality affinity kernels, with widths from shared-nearest-neighbor structure.

    Parameters
    ----------
    adata
        Annotated data matrix, or a ``MuData`` object whose modalities share the same cells.
    use_rep
        One dimensional reduction per modality, e.g. ``['X_pca', 'X_apca']`` (keys of `.obsm`, or `'X'`). A mapping
        names the modalities; for MuData it maps each modality to an `.obsm` key of it, e.g.
        ``{'rna': 'X_pca', 'prot': 'X_pca'}``.
    n_pcs
        Number of dimensions to use, per modality or for all (Seurat's ``dims.list``). `None` uses all dimensions.
    n_neighbors
        Number of multimodal neighbors (Seurat's ``k.nn``), also used for the modality weights.
    knn_range
        Neighbors searched per modality, whose union are the multimodal candidates (Seurat's ``knn.range``).
    l2_norm
        L2-normalize the embeddings of every cell.
    sd_scale
        Scaling factor of the kernel widths.
    cross_constant
        Constant to avoid dividing by zero when comparing within- and cross-modality predictions, per modality or
        for all (Seurat's cross-modality constant list).
    smooth
        Average the modality scores over each cell's neighbors in that modality.
    prune_snn
        Jaccard cutoff for edges of the shared nearest neighbor graph (Seurat's ``prune.SNN``).
    algorithm
        The kNN search within each modality, see :func:`~rapids_singlecell.pp.neighbors`. ``'brute'`` (default) is
        exact and bit-identical across runs up to ``knn_range=256`` (beyond, cuVS orders exactly tied neighbors
        arbitrarily). ``'ivfflat'`` is much faster, yet closer to exact than Seurat's Annoy search; ``'cagra'``
        needs ``knn_range <= 64``.
    algorithm_kwds
        Options for the kNN search, see :func:`~rapids_singlecell.pp.neighbors`. Incomplete results raise a
        :class:`ValueError` (for IVF, increase ``n_probes``). For ``'nn_descent'``, ``intermediate_graph_degree``
        defaults to ``max(128, 1.5 * knn_range)``.
    method
        Method for the `connectivities`, see :func:`~rapids_singlecell.pp.neighbors`.
    rng
        Random seed or :class:`~numpy.random.Generator` for the UMAP connectivities.
        The superseded `random_state` argument is still accepted.
    key_added
        If specified, the neighbors data is stored in `.uns[key_added]` and the graphs in
        `.obsp[f'{key_added}_distances']`, `.obsp[f'{key_added}_connectivities']` and `.obsp[f'{key_added}_snn']`,
        else in `.uns['neighbors']`, `.obsp['distances']`, `.obsp['connectivities']` and `.obsp['snn']`.
    weight_key
        Format string for the `.obs` columns of the modality weights, filled with the modality name (the `.obsm` key
        or the mapping key of `use_rep`). ``'{}:mod_weight'`` gives muon's names; `None` skips the weights.
    copy
        Return a copy instead of writing to `adata`.

    Returns
    -------
    Depending on `copy`, updates or returns `adata` with the following:

    **distances** : sparse matrix of dtype `float32`.
        The `n_neighbors` multimodal neighbors of every cell, excluding itself, with Seurat's WNN distance
        :math:`\sqrt{(1 - s)/2}` of the weighted kernel similarity :math:`s` (0 for the nearest neighbor in all
        modalities).
    **connectivities** : sparse matrix of dtype `float32`.
        Weighted adjacency matrix of the multimodal neighbors and the cell itself (``n_neighbors + 1`` as in
        Scanpy), computed with `method`, for :func:`~rapids_singlecell.tl.umap` and
        :func:`~rapids_singlecell.tl.leiden`.
    **snn** : sparse matrix of dtype `float32`.
        Seurat's ``wsnn`` graph, the Jaccard index of the multimodal neighbor sets, with unit diagonal. Seurat
        clusters on it, e.g. ``rsc.tl.leiden(adata, obsp='snn')``; Leiden (``flavor='rapids'``) and Seurat's
        default Louvain ignore the diagonal.
    **weights** : `.obs[weight_key.format(name)]`
        The weight of each modality per cell.

    ``.uns[neighbors_key]['params']['modalities']`` holds each modality's representation key and number of
    dimensions. Its standard ``use_rep`` and ``n_pcs`` entries refer to the first modality with AnnData (for
    Scanpy's layout initialization) and map the modalities to them with MuData, as in muon.
    :func:`~rapids_singlecell.tl.ingest` rejects WNN references; :func:`scanpy.tl.ingest` uses the first modality.
    """
    from rapids_singlecell._cuda._wnn_cuda import impute_dist, multimodal_knn

    rng = np.random.default_rng(rng)
    meta_random_state = {}
    if isinstance(rng, _LegacyRng) and isinstance(rng.arg, int | np.integer | None):  # only plain seeds
        meta_random_state["random_state"] = None if rng.arg is None else int(rng.arg)
    if method != "umap" and meta_random_state.get("random_state") != _DEFAULT_SEED:
        # only the UMAP connectivities are randomized
        warn(f"Parameter `rng`/`random_state` ignored if `method={method!r}`.", UserWarning, stacklevel=2)
        meta_random_state.pop("random_state", None)

    if algorithm not in get_args(_Algorithms):
        raise ValueError(f"Invalid algorithm '{algorithm}' for KNN. Valid options are: {get_args(_Algorithms)}.")
    adata = adata.copy() if copy else adata

    modalities, reps = _get_reps(adata, use_rep, n_pcs)
    names = list(modalities)
    # Scanpy's UMAP initializes disconnected components from one representation, muon's per modality
    layout_rep = modalities[names[0]]
    if hasattr(adata, "mod"):
        layout_rep = {f: {n: m[f] for n, m in modalities.items()} for f in ("use_rep", "n_pcs")}
    # validate before any work, so that errors leave `adata` unchanged
    weight_cols = [] if weight_key is None else [weight_key.format(n) for n in names]
    if len(set(weight_cols)) < len(weight_cols) or "" in weight_cols:
        raise ValueError(f"`weight_key` {weight_key!r} must give a distinct column per modality, e.g. '{{}}_weight'.")
    n_mod, n_obs = len(reps), adata.n_obs
    n_neighbors, knn_range = operator.index(n_neighbors), operator.index(knn_range)
    if n_neighbors < 2:
        raise ValueError("`n_neighbors` must be at least 2.")
    if knn_range <= n_neighbors:
        raise ValueError("`knn_range` must be larger than `n_neighbors`.")
    if knn_range > n_obs:
        raise ValueError(f"`knn_range` must be at most the number of cells ({n_obs}).")
    if n_mod * (knn_range - 1) > 4096:
        raise ValueError("`knn_range * n_modalities` must be at most 4096.")
    if algorithm == "cagra" and knn_range > 64:
        raise ValueError("`algorithm='cagra'` supports `knn_range` up to 64; use e.g. 'ivfflat'.")
    if np.ndim(cross_constant) == 0:
        cross_constant = [float(cross_constant)] * n_mod
    elif len(cross_constant) != n_mod:
        raise ValueError("`cross_constant` needs one entry per modality.")

    if l2_norm:
        norms = [cp.linalg.norm(X, axis=1, keepdims=True) for X in reps]
        reps = [X / cp.where(norm > 0, norm, 1) for X, norm in zip(reps, norms, strict=True)]

    # kNN within each modality: the first `n_neighbors` give the weights, all `knn_range` the candidates
    search_kwds = dict(algorithm_kwds)
    if algorithm == "nn_descent":  # cuVS fails if this is below the graph degree
        search_kwds.setdefault("intermediate_graph_degree", max(128, int(1.5 * knn_range)))
    kw = {"metric": "euclidean", "metric_kwds": {}, "algorithm_kwds": search_kwds}
    knns = cp.empty((n_mod, n_obs, knn_range), dtype=cp.int32)
    for m, X in enumerate(reps):
        Xc = _check_neighbors_X(X, algorithm)
        knn, knn_dist = KNN_ALGORITHMS[algorithm](Xc, Xc, k=knn_range, **kw)
        knn, knn_dist = cp.asarray(knn), cp.asarray(knn_dist)
        # IVF can pad incomplete lists with sentinel IDs (or repeated IDs with infinite
        # distances), all_neighbors with huge distances; check before narrowing IDs or indexing kernels.
        valid = (knn >= 0) & (knn < n_obs) & (knn_dist < _PAD_DIST)
        if not bool(valid.all()):
            raise ValueError(
                f"The {algorithm!r} search returned incomplete or invalid neighbors for modality {names[m]!r}. "
                "Increase `n_probes` in `algorithm_kwds` for IVF searches (lower `n_lists` for 'all_neighbors'), "
                "reduce `knn_range`, or use `algorithm='brute'`."
            )
        knns[m] = _ensure_self_first(knn.astype(cp.int32, copy=False))
        del knn, knn_dist, valid, Xc

    # dist[m, r]: distance of modality m to its prediction from modality r's neighbors
    nearest = cp.empty((n_mod, n_obs), dtype=cp.float32)
    sigma = cp.empty((n_mod, n_obs), dtype=cp.float32)
    dist = cp.empty((n_mod, n_mod, n_obs), dtype=cp.float32)
    zero = cp.zeros(n_obs, dtype=cp.float32)
    for m, X in enumerate(reps):
        # the multimodal kernel's arithmetic, so a neighbor nearest in every modality gets WNN distance 0
        impute_dist(X, knns[m], k=2, nearest=zero, out=nearest[m], stream=_stream())
        for r in range(n_mod):
            impute_dist(X, knns[r], k=n_neighbors, nearest=nearest[m], out=dist[m, r], stream=_stream())
        sigma[m] = _snn_bandwidth(knns[m], X, nearest[m], s_nn=n_neighbors)
    sigma *= sd_scale
    # a zero width (duplicated cells) makes R produce NaN weights
    sigma = cp.maximum(sigma, cp.finfo(cp.float32).tiny)

    kernel = cp.exp(-dist.astype(cp.float64) / sigma[:, None, :])
    exp_scores = cp.zeros((n_mod, n_obs), dtype=cp.float64)
    for m, r in itertools.permutations(range(n_mod), 2):
        score = cp.clip(kernel[m, m] / (kernel[m, r] + cross_constant[m]), 0, 200)
        if smooth:
            score = score[knns[m, :, 1:n_neighbors]].mean(axis=1)
        exp_scores[m] += cp.exp(score)
    weights = (exp_scores / exp_scores.sum(axis=0)).astype(cp.float32)
    del kernel, exp_scores, dist

    emb = cp.concatenate(reps, axis=1)
    dim_off = cp.asarray(np.cumsum([0] + [X.shape[1] for X in reps]), dtype=cp.int32)
    del reps
    knn_indices = cp.empty((n_obs, n_neighbors), dtype=cp.int32)
    knn_dissim = cp.empty((n_obs, n_neighbors), dtype=cp.float32)
    out = {"k_out": n_neighbors, "out_idx": knn_indices, "out_dissim": knn_dissim, "stream": _stream()}
    multimodal_knn(
        emb, dim_off=dim_off, knn=knns, knn_range=knn_range, weight=weights, sigma=sigma, nearest=nearest, **out
    )
    del emb, knns
    # Seurat's sqrt((1 - s) / 2) of the weighted kernel similarity s
    knn_dist = cp.sqrt(cp.clip(knn_dissim / 2, 0, 1))

    distances = _build_sparse_distances(knn_indices, knn_dist, n_obs=n_obs)
    snn = _snn_graph(knn_indices, prune=prune_snn)
    # the connectivity methods expect the cell itself in column 0
    knn_indices = cp.column_stack((cp.arange(n_obs, dtype=cp.int32), knn_indices))
    knn_dist = cp.column_stack((cp.zeros(n_obs, dtype=cp.float32), knn_dist))
    connectivities = _calc_connectivities(
        knn_indices, knn_dist, n_obs=n_obs, n_neighbors=n_neighbors + 1, rng=rng, method=method
    )
    if connectivities.nnz < np.iinfo(np.int32).max:
        connectivities = connectivities.tocsr().get()
    elif connectivities.format == "coo":
        connectivities = _large_coo_to_host_csr(connectivities)
    else:  # gauss and jaccard return CSR
        connectivities = connectivities.get().tocsr()

    if adata.is_view:  # only now, so that errors leave a view untouched
        adata._init_as_actual(adata.copy())
    neighbors_key = "neighbors" if key_added is None else key_added
    prefix = "" if neighbors_key == "neighbors" else f"{neighbors_key}_"
    graphs = {"connectivities": connectivities, "distances": distances, "snn": snn}
    params = {"n_neighbors": n_neighbors, "method": method, **meta_random_state, "metric": "euclidean", **layout_rep}
    params |= {"modalities": modalities, "knn_range": knn_range, "l2_norm": l2_norm, "sd_scale": sd_scale}
    params |= {"smooth": smooth, "prune_snn": prune_snn}
    if algorithm_kwds:
        params["algorithm_kwds"] = dict(algorithm_kwds)
    adata.uns[neighbors_key] = {f"{name}_key": f"{prefix}{name}" for name in graphs} | {"params": params}
    for name, graph in graphs.items():
        adata.obsp[f"{prefix}{name}"] = graph
    if weight_cols:  # weight_key=None skips them
        for col, w in zip(weight_cols, weights.get(), strict=True):
            adata.obs[col] = w

    return adata if copy else None
