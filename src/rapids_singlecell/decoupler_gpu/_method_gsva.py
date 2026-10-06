from __future__ import annotations

import math

import cupy as cp
import numba as nb
import numpy as np
from cupyx.scipy import sparse as csps

from rapids_singlecell._cuda import _gsva_cuda as _gs
from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta
from rapids_singlecell.preprocessing._utils import _sparse_to_dense


def _precomputed_cdf() -> np.ndarray:
    x = np.arange(10001, dtype=np.float64) / 1000
    z = x / np.sqrt(2)
    t = 1 / (1 + 0.3275911 * z)
    erf = 1 - (
        (
            (((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t
            + 0.254829592
        )
        * t
        * np.exp(-(z**2))
    )
    erf[0] = 0
    return 0.5 * (1 + erf)


_PRECOMPUTED_CDF = _precomputed_cdf()


@cp.memoize(for_each_device=True)
def _precomputed_cdf_gpu() -> cp.ndarray:
    return cp.asarray(_PRECOMPUTED_CDF)


@nb.njit(cache=True)
def _poisson_cdfs(values: np.ndarray, start: int = 0, end: int = -1) -> np.ndarray:
    """Scalar CDF constants with upstream's libm arithmetic and summation order."""
    if end < 0:
        end = values.size
    cdfs = np.zeros((end - start, values.size))
    for column in range(values.size):
        lam = values[column] + 0.5
        if lam < 0:
            continue
        cdf = 0.0
        count = 0
        for row in range(start, end):
            while count <= int(values[row]):
                if count == 0:
                    cdf += np.exp(-lam)
                else:
                    cdf += np.exp(-lam + count * np.log(lam) - math.lgamma(count + 1))
                count += 1
            cdfs[row - start, column] = min(cdf, 1.0)
    return cdfs


@cp.memoize(for_each_device=True)
def _poisson_cdfs_gpu() -> cp.ndarray:
    return cp.asarray(_poisson_cdfs(np.arange(65, dtype=np.float64)))


def _poisson_density_columns(
    mat: cp.ndarray, *, max_entries: int = 1 << 20
) -> cp.ndarray:
    """Keep CDF tables bounded for inputs with many distinct counts."""
    nobs, nvar = mat.shape
    density = cp.empty(mat.shape, dtype=cp.float64)
    stream = cp.cuda.get_current_stream().ptr
    for column in range(nvar):
        values, counts = cp.unique(mat[:, column], return_inverse=True)
        values = values.get().astype(np.float64)
        # Tile queries only: each density still sums observations in input order.
        bsize = max(1, max_entries // values.size)
        for start in range(0, values.size, bsize):
            end = min(start + bsize, values.size)
            cdfs = cp.asarray(_poisson_cdfs(values, start, end))
            lookup = cp.empty(end - start, dtype=cp.float64)
            _gs.poisson_lookup(counts, cdfs, lookup, nobs, 1, stream)
            _gs.poisson_gather_tile(
                counts, lookup, start, end, density[:, column], stream
            )
    return density


def _density(mat: cp.ndarray, kcdf: str | None) -> cp.ndarray:
    assert (
        isinstance(kcdf, str) and kcdf in {"gaussian", "poisson"}
    ) or kcdf is None, "kcdf must be gaussian, poisson or None"
    nobs, nvar = mat.shape
    stream = cp.cuda.get_current_stream().ptr
    if kcdf == "gaussian":
        bw = cp.empty(nvar, dtype=cp.float64)
        _gs.bandwidth(mat, bw, stream)
        if bool(cp.any(bw == 0)):
            raise ZeroDivisionError("division by zero")
        scaled = mat / bw
        density = cp.empty(mat.shape, dtype=cp.float64)
        _gs.gaussian_density(scaled, _precomputed_cdf_gpu(), density, stream)
        return density
    if kcdf == "poisson":
        assert float(cp.sum(mat, dtype=cp.float64).get()).is_integer(), (
            "when kcdf=poisson input data must be integers (e.g. 3, 4, etc.), "
            "not decimal values (e.g. 3.5, 4.9, etc.)"
        )
        bounds = cp.stack((cp.min(mat), cp.max(mat))).get()
        max_count = int(bounds[1])
        if bounds[0] >= 0 and max_count <= 64 and bool(cp.all(mat == cp.floor(mat))):
            cdfs = _poisson_cdfs_gpu()
            count_ids = mat
            ncounts = max_count + 1
        else:
            values = cp.unique(mat)
            if values.size > 256:
                return _poisson_density_columns(mat)
            cdfs = cp.asarray(_poisson_cdfs(values.get().astype(np.float64)))
            count_ids = cp.searchsorted(values, mat).astype(cp.int32)
            ncounts = values.size
        lookup = cp.empty((ncounts, nvar), dtype=cp.float64)
        density = cp.empty(mat.shape, dtype=cp.float64)
        _gs.poisson_lookup(count_ids, cdfs, lookup, nobs, nvar, stream)
        _gs.poisson_gather(count_ids, lookup, density, nvar, stream)
        return density
    sorted_mat = cp.sort(mat, axis=0)
    density = cp.empty(mat.shape, dtype=cp.float32)
    _gs.ecdf(mat, sorted_mat, density, stream)
    return density


def _sparse_ecdf_values(
    mat: csps.csr_matrix,
) -> tuple[cp.ndarray, cp.ndarray]:
    """Return exact ECDF integer ranks for stored values and implicit zeros."""
    nobs, nvar = mat.shape
    if mat.nnz == 0:
        return cp.empty(0, dtype=cp.int32), cp.full(nvar, nobs, dtype=cp.int32)
    stream = cp.cuda.get_current_stream().ptr
    if not mat.has_sorted_indices:
        mat.sort_indices()
    coo = mat.tocoo(copy=False)
    counts = cp.bincount(coo.col, minlength=nvar).astype(cp.int32)
    starts = cp.empty(nvar + 1, dtype=cp.int64)
    starts[0] = 0
    cp.cumsum(counts, dtype=cp.int64, out=starts[1:])
    keys = cp.empty(coo.nnz, dtype=cp.uint64)
    _gs.sparse_ecdf_keys(coo.data, coo.col, keys, stream)
    order = cp.argsort(keys)
    keys = keys[order]
    zero_keys = (cp.arange(nvar, dtype=cp.uint64) << np.uint64(32)) | np.uint64(
        0x80000000
    )
    zero_ranks = cp.searchsorted(keys, zero_keys, side="right") - starts[:-1]
    zero_ranks = (zero_ranks + nobs - counts).astype(cp.int32)
    values = cp.empty(coo.nnz, dtype=cp.int32)
    _gs.sparse_ecdf_ranks(keys, order, starts, counts, nobs, values, stream)
    return values, zero_ranks


@cp.memoize(for_each_device=True)
def _rank_shared_limit() -> int:
    return cp.cuda.Device().attributes["MaxSharedMemoryPerBlockOptin"]


def _rank_sparse_ecdf(
    mat: csps.csr_matrix,
    cnct: cp.ndarray,
) -> tuple[cp.ndarray, cp.ndarray, cp.ndarray]:
    nobs, nvar = mat.shape
    stream = cp.cuda.get_current_stream().ptr
    if not mat.has_canonical_format:
        mat = mat.copy()
        mat.sum_duplicates()
    row_sizes = cp.diff(mat.indptr).get()
    # Packed keys use 16-bit feature indices; larger rows/features use the
    # general dense path instead of truncating keys or exceeding shared memory.
    capacity_limit = min(8192, 1 << ((_rank_shared_limit() // 8).bit_length() - 1))
    if (
        nvar > 65536
        or row_sizes.max(initial=0) > capacity_limit
        or mat.indptr.dtype != cp.int32
        or mat.indices.dtype != cp.int32
    ):
        dos, srs = _rankmat(_density(_sparse_to_dense(mat), None))
        return dos, srs, cnct
    values, zero_ranks = _sparse_ecdf_values(mat)
    target_genes = cp.unique(cnct)
    target_lookup = cp.full(nvar, -1, dtype=cp.int32)
    target_lookup[target_genes] = cp.arange(target_genes.size, dtype=cp.int32)
    cnct = target_lookup[cnct]
    base_keys = cp.empty(nvar, dtype=cp.uint64)
    _gs.sparse_base_keys(zero_ranks, nobs, base_keys, stream)
    base_keys.sort()
    dos = cp.empty((nobs, target_genes.size), dtype=cp.int32)
    srs = cp.empty_like(dos)
    lower = -1
    for capacity in (1024, 2048, 4096, 8192):
        row_ids = np.flatnonzero((row_sizes > lower) & (row_sizes <= capacity)).astype(
            np.int32
        )
        if row_ids.size:
            row_ids = cp.asarray(row_ids)
            _gs.sparse_rank(
                mat.indptr,
                mat.indices,
                values,
                zero_ranks,
                base_keys,
                target_genes,
                row_ids,
                capacity,
                dos,
                srs,
                stream,
            )
        lower = capacity
    return dos, srs, cnct


def _rankmat(mat: cp.ndarray) -> tuple[cp.ndarray, cp.ndarray]:
    nobs, nvar = mat.shape
    if (
        mat.dtype == cp.float32
        and nvar <= 65536
        and _gs is not None
        # Reserve space for the shared GSEA sort queue as well as the indices.
        and nvar * 2 + 2048 <= _rank_shared_limit()
    ):
        dos = cp.empty(mat.shape, dtype=cp.int32)
        _gs.rank(mat, dos, cp.cuda.get_current_stream().ptr)
    else:
        # A stable sort over reversed columns reproduces Decoupler's decreasing
        # feature-index order for equal values without a lexicographic key.
        order64 = cp.argsort(mat[:, ::-1], axis=1, kind="stable")
        order = order64.astype(cp.int32)
        del order64
        cp.subtract(np.int32(nvar - 1), order, out=order)
        ranks = cp.empty(mat.shape, dtype=cp.int32)
        ranks[cp.arange(nobs)[:, None], order] = cp.arange(1, nvar + 1, dtype=cp.int32)
        del order
        cp.subtract(np.int32(nvar + 1), ranks, out=ranks)
        dos = ranks
    # Upstream stores this half-rank distance in an integer output matrix.
    srs = dos.copy()
    cp.multiply(srs, 2, out=srs)
    cp.subtract(srs, nvar + 2, out=srs)
    cp.abs(srs, out=srs)
    cp.floor_divide(srs, 2, out=srs)
    return dos, srs


def _score_sets(
    dos: cp.ndarray,
    srs: cp.ndarray,
    cnct: cp.ndarray,
    starts: cp.ndarray,
    offsets: cp.ndarray,
    *,
    maxdiff: bool,
    absrnk: bool,
    tau: int | float,
    nvar: int | None = None,
) -> cp.ndarray:
    nobs, nfeat = dos.shape
    if nvar is None:
        nvar = nfeat
    nsrc = starts.size
    scores = cp.empty((nobs, nsrc), dtype=cp.float64)
    sizes = cp.asnumpy(offsets)

    def launch(source_ids, capacity):
        if not source_ids.size:
            return
        source_ids = cp.asarray(source_ids, dtype=cp.int32)
        _gs.score(
            dos,
            srs,
            cnct,
            starts,
            offsets,
            source_ids,
            scores,
            nvar,
            capacity,
            maxdiff,
            absrnk,
            tau,
            cp.cuda.get_current_stream().ptr,
        )

    launch(np.flatnonzero(sizes <= 32), 32)
    lower = 32
    for capacity in (64, 128, 256, 512, 1024, 2048):
        launch(
            np.flatnonzero((sizes > lower) & (sizes <= capacity)),
            capacity,
        )
        lower = capacity
    launch(np.flatnonzero(sizes > 2048), 0)
    return scores


@docs.dedent
def _func_gsva(
    mat,
    *,
    cnct,
    starts,
    offsets,
    kcdf: str | None = "gaussian",
    maxdiff: bool = True,
    absrnk: bool = False,
    tau: int | float = 1,
    verbose: bool = False,
):
    r"""
    Gene Set Variation Analysis (GSVA).

    Transform features with a Gaussian, Poisson, or empirical cumulative
    distribution, then calculate feature-set enrichment with a random walk.

    %(notest)s

    %(params)s
    kcdf
        Density transformation. One of ``"gaussian"``, ``"poisson"``, or
        ``None`` for the empirical cumulative distribution.
    maxdiff
        Whether to combine the maximum positive and negative deviations.
    absrnk
        If ``maxdiff=True``, whether enrichment at either extreme is positive.
    tau
        Exponent applied to the rank statistic.

    Notes
    -----
    In-memory and Dask inputs use all observations together for density
    estimation. Backed inputs use ``bsize`` observations per batch, matching
    Decoupler; changing that batch size can change the scores.

    %(returns)s
    """
    is_sparse = csps.issparse(mat)
    if is_sparse and not mat.has_canonical_format:
        mat = mat.copy()
        mat.sum_duplicates()
    if is_sparse and kcdf is not None:
        mat = _sparse_to_dense(mat)
        is_sparse = False
    if not is_sparse:
        mat = cp.ascontiguousarray(mat, dtype=cp.float32)
    nvar = mat.shape[1]
    if mat.shape[0] > 1:
        _log(
            f"gsva - computing density with kcdf={kcdf}", level="info", verbose=verbose
        )
        if is_sparse:
            dos, srs, cnct = _rank_sparse_ecdf(mat, cnct)
        else:
            mat = _density(mat, kcdf)
            dos, srs = _rankmat(mat)
    elif is_sparse:
        mat = _sparse_to_dense(mat)
        dos, srs = _rankmat(mat)
    else:
        dos, srs = _rankmat(mat)
    _log(
        f"gsva - calculating {starts.size} scores with maxdiff={maxdiff}, absrnk={absrnk}",
        level="info",
        verbose=verbose,
    )
    es = _score_sets(
        dos,
        srs,
        cnct,
        starts,
        offsets,
        maxdiff=maxdiff,
        absrnk=absrnk,
        tau=tau,
        nvar=nvar,
    )
    return es.get(), None


_gsva = MethodMeta(
    name="gsva",
    desc="Gene Set Variation Analysis (GSVA)",
    func=_func_gsva,
    stype="numerical",
    adj=False,
    weight=False,
    test=False,
    limits=(-1, +1),
    reference="https://doi.org/10.1186/1471-2105-14-7",
)
gsva = Method(_method=_gsva, default_bsize=250_000, batch_on="backed", dense=False)
