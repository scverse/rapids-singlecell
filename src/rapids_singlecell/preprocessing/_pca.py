from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING, Any, NamedTuple, TypedDict

import cupy as cp
import numpy as np
from anndata import AnnData
from cupyx.scipy.sparse import issparse as cpissparse
from cupyx.scipy.sparse import issparse as issparse_cupy
from cupyx.scipy.sparse import isspmatrix_csr
from scipy.sparse import issparse
from scverse_misc import Deprecation, deprecated_arg

from rapids_singlecell._compat import DaskArray
from rapids_singlecell._keys import _embedding_keys
from rapids_singlecell._settings import Default, settings
from rapids_singlecell._utils._random import (
    RNGLike,
    SeedLike,
    _accepts_legacy_random_state,
)
from rapids_singlecell.get import X_to_GPU, _check_mask, _get_arr
from rapids_singlecell.get._out import (
    DEPR_KEY_ADDED,
    DEPR_USE,
    _mask_arg,
    _out_dests,
    _ref_to_json,
    _use_args,
    _use_to_uns,
    _write_dests,
    deprecated_copy,
)

from ._utils import _check_gpu_X

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from rapids_singlecell._utils import ArrayTypesDask
    from rapids_singlecell.get._utils import AdRef, LayerAcc, MultiAcc

_empty = Default(repr="adata.var.get('highly_variable')")


class PcaResult(NamedTuple):
    """Outputs of :func:`~rapids_singlecell.pp.pca`."""

    embedding: Any
    """PCA representation of the observations, shape `(n_obs, n_comps)`."""
    loadings: Any
    """Principal components, shape `(n_vars, n_comps)` (rows outside `mask` are zero)."""


class PcaOut(TypedDict, total=False):
    """Where a mapping `out` of :func:`~rapids_singlecell.pp.pca` writes each output."""

    embedding: MultiAcc | str
    loadings: MultiAcc | str


def _pca_dests(key: str | Default | None, *, multi: bool) -> PcaResult:
    keys = _embedding_keys("pca", key)
    return PcaResult({"obsm": keys.obsm}, None if multi else {"varm": keys.varm})


@_accepts_legacy_random_state(0)
@deprecated_arg("mask_var", Deprecation("0.19.0", "Use `mask` instead."))
@deprecated_arg("layer", DEPR_USE)
@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_copy
def pca(
    data: AnnData | ArrayTypesDask,
    n_comps: int | None = None,
    *,
    use: LayerAcc | MultiAcc | str | None = None,
    zero_center: bool = True,
    svd_solver: str | None = None,
    rng: SeedLike | RNGLike | None = None,
    mask: NDArray[np.bool] | AdRef | str | Default | None = _empty,
    dtype: str = "float32",
    chunked: bool = False,
    chunk_size: int = None,
    return_info: bool = False,
    out: MultiAcc | str | PcaOut | bool = False,
    # deprecated
    key_added: str | None = None,
    layer: str | None = None,
    copy: bool = False,
    mask_var: NDArray[np.bool] | str | None = None,
    **kwargs,
) -> None | AnnData | PcaResult | ArrayTypesDask:
    r"""Principal component analysis using GPU acceleration :cite:p:`Halko2009,Tomas2024`.

    Uses the following implementations based on data type (defaults for `svd_solver` in parentheses):

    .. list-table::
       :header-rows: 1

       * -
         - Dense
         - Sparse
         - Dask
       * - `zero_center=True`
         - cuML PCA (`'full'`)
         - Custom (`'lanczos'` if n_vars > 8k, else `'covariance_eigh'`)
         - Custom (`'covariance_eigh'`)
       * - `zero_center=False`
         - cuML TruncatedSVD (`'full'`)
         - Custom (`'lanczos'` if n_vars > 8k, else `'covariance_eigh'`)
         - Custom (`'covariance_eigh'`)
       * - `chunked=True`
         - cuML IncrementalPCA
         - cuML IncrementalPCA
         - Not supported

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`. Rows correspond
        to cells and columns to genes. If a matrix is passed instead of an
        :class:`~anndata.AnnData` object, the PCA representation is returned.

    n_comps
        Number of principal components to compute. Defaults to 50, or 1 - minimum
        dimension size of selected representation.

    use
        Which matrix to run PCA on, e.g. `A.X`, `A.layers['counts']` or `A.obsm['x']`.
        Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'layers.counts'`.

    zero_center
        If `True`, compute standard PCA from covariance matrix.
        If `False`, omit zero-centering variables (truncated SVD).

    svd_solver
        SVD solver to use. See table above for which implementation is used based on
        data type, as well as the default solver when `svd_solver=None`.

        `None`
            Choose automatically based on data type (see table above).
        `'covariance_eigh'`
            Eigendecomposition of the covariance matrix. Fast for sparse matrices
            with fewer than ~8,000 features. Works with Dask arrays.
        `'lanczos'`
            Lanczos bidiagonalization with implicit restarts. Memory efficient for
            large sparse matrices (>8,000 features). Best singular value accuracy.
            Does not support Dask arrays.
        `'randomized'`
            Randomized SVD (Halko et al. 2009) with CholeskyQR2 orthogonalization
            (Tomás et al. 2024). Faster than Lanczos but approximate.
            Does not support Dask arrays.
        `'full'`
            cuML: Full eigendecomposition of covariance matrix. For dense arrays only.
        `'jacobi'`
            cuML: Jacobi iterative solver. Faster but less accurate. For dense arrays only.

    rng
        Random seed or :class:`~numpy.random.Generator` for initialization.
        The superseded `random_state` argument is still accepted.

    mask
        To run only on a certain set of genes, given by a boolean array or a reference like
        `A.var['selected']`. Strings are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'var.selected'`.
        By default, uses `.var['highly_variable']` if available, else everything.

    dtype
        Numpy data type string to which to convert the result.

    chunked
        If `True`, perform an incremental PCA on segments of `chunk_size`.
        The incremental PCA automatically zero centers and ignores settings of
        `rng` and `svd_solver`. If `False`, perform a full PCA.

    chunk_size
        Number of observations to include in each chunk.
        Required if `chunked=True` was passed.

    return_info
        Only relevant when passing a matrix instead of an
        :class:`~anndata.AnnData`: see "Returns".

    out
        Where to write the outputs. A single accessor writes `embedding` to it and names the
        others after its key `k`, e.g. `out=A.obsm['k']` writes `.varm['k']` and `.uns['k']`.
        A mapping (:class:`~rapids_singlecell.pp.PcaOut`) writes only the outputs it names,
        e.g. `{'embedding': A.obsm['pca']}`; it has to include `embedding`, after which the `.uns`
        entry is named. If `True`, they are returned as a :class:`~rapids_singlecell.pp.PcaResult`
        (without the `.uns` entry). If `False`, they are written to `.obsm['X_pca']` and
        `.varm['PCs']` (`'pca'` under the v2 preset). If `use` points to `.obsm`, the loadings
        are stored in the `.uns` entry instead.

    **kwargs
        Additional arguments for specific SVD solvers.
        For `svd_solver='randomized'`:

        - `n_oversamples`: Extra random vectors for better approximation.
          Higher values improve accuracy. Default is 10.
        - `n_iter`: Number of power iterations. Higher values improve
          accuracy for matrices with slowly decaying singular values. Default is 2.

    Returns
    -------
        If a matrix is passed and `return_info=False`, the PCA representation is returned. If a matrix is passed and `return_info=True`, a tuple of ``(X_pca, components, variance_ratio, variance)`` is returned.

        If an AnnData object is passed, returns a :class:`~rapids_singlecell.pp.PcaResult` if `out=True`,
        else `None`. By default sets (see `out` for others):

            `.obsm['X_pca']`
                PCA representation of data.
            `.varm['PCs']`
                The principal components containing the loadings.
            `.uns['pca']['variance_ratio']`
                Ratio of explained variance.
            `.uns['pca']['variance']`
                Explained variance, equivalent to the eigenvalues of the covariance matrix.
    """
    if "use_highly_variable" in kwargs:
        raise TypeError("pca() got an unexpected keyword argument 'use_highly_variable'")

    rng = np.random.default_rng(rng)

    if not isinstance(data, AnnData):
        if layer is not None or use is not None or out is not False:
            raise ValueError("`use` and `out` can only be used with an AnnData object.")
        X = data
        mask = _mask_arg(mask, mask_var, dim="var")
        mask = None if isinstance(mask, Default) else _check_mask(X, mask, "var")
        X = X[:, mask] if mask is not None else X
        pca_func, X_pca, _ = _pca_compute(
            X,
            n_comps,
            zero_center=zero_center,
            svd_solver=svd_solver,
            rng=rng,
            chunked=chunked,
            chunk_size=chunk_size,
            dtype=dtype,
            kwargs=kwargs,
        )
        if return_info:
            return (
                X_pca,
                pca_func.components_,
                pca_func.explained_variance_ratio_,
                pca_func.explained_variance_,
            )
        return X_pca

    adata = data
    src = _use_args(use, layer=layer)
    multi = "obsm" in src  # `.obsm` columns aren't variables, so the loadings can't go to `.varm`
    dests = _out_dests(
        out,
        key_added,
        dests=partial(_pca_dests, multi=multi),
        kinds=PcaResult({"obsm"}, {"varm"}),
        uns=True,
        default=Default(),
    )
    mask = _mask_arg(mask, mask_var, dim="var")
    if multi and not isinstance(mask, Default) and mask is not None:
        raise ValueError("Argument `mask` is incompatible with `.obsm` arrays.")

    X = _get_arr(adata, dim="obs", **src)

    if isinstance(mask, Default):
        hvg = not multi and "highly_variable" in adata.var.columns
        mask = _mask_arg(None, "highly_variable", dim="var") if hvg else None
    mask_param, mask_var = _ref_to_json(mask), _check_mask(adata, mask, "var")
    X = X[:, mask_var] if mask_var is not None else X

    pca_func, X_pca, n_comps = _pca_compute(
        X,
        n_comps,
        zero_center=zero_center,
        svd_solver=svd_solver,
        rng=rng,
        chunked=chunked,
        chunk_size=chunk_size,
        dtype=dtype,
        kwargs=kwargs,
    )

    loadings = _as_numpy(pca_func.components_.T)
    if mask_var is not None:
        loadings = np.zeros(shape=(adata.n_vars, n_comps))
        loadings[mask_var] = _as_numpy(pca_func.components_.T)
    vals = PcaResult(X_pca, loadings)
    if dests is True:
        return vals
    key, dests = dests
    adata.uns[_embedding_keys("pca", key).uns] = {
        "params": {
            "zero_center": zero_center,
            "mask_var": mask_param,
            **({"layer": layer} if layer is not None else {}),
            **_use_to_uns(use, rep=False),
        },
        "variance": _as_numpy(pca_func.explained_variance_),
        "variance_ratio": _as_numpy(pca_func.explained_variance_ratio_),
        **({"components": loadings} if multi else {}),
    }
    _write_dests(adata, vals, dests)
    return None


def _pca_compute(
    X,
    n_comps,
    *,
    zero_center: bool,
    svd_solver: str | None,
    rng: np.random.Generator,
    chunked: bool,
    chunk_size: int | None,
    dtype: str,
    kwargs: dict,
):
    if n_comps is None:
        min_dim = min(X.shape[0], X.shape[1])
        if settings.N_PCS >= min_dim:
            n_comps = min_dim - 1
        else:
            n_comps = settings.N_PCS

    # Auto-select sparse solver based on matrix dimensions
    # Lanczos is faster for large feature counts (>8000)
    # Covariance is faster for smaller matrices due to optimized kernels
    if svd_solver is None and (cpissparse(X) or issparse(X)):
        n_vars = X.shape[1]
        if n_vars > 8000:
            svd_solver = "lanczos"
        else:
            svd_solver = "covariance_eigh"

    if isinstance(X, DaskArray):
        if chunked:
            raise ValueError("Dask arrays are not supported for chunked PCA computation.")
        _check_gpu_X(X, allow_dask=True)
        if svd_solver in ("lanczos", "randomized"):
            raise NotImplementedError(
                f"'{svd_solver}' SVD solver does not support Dask arrays. Use svd_solver='covariance_eigh' instead."
            )
        if svd_solver is None:
            svd_solver = "covariance_eigh"
        if isinstance(X._meta, cp.ndarray) and svd_solver != "covariance_eigh" and zero_center:
            pca_func, X_pca = _run_cuml_pca_dask(X, n_comps, svd_solver=svd_solver)
        else:
            pca_func, X_pca = _run_covariance_pca(X, n_comps, zero_center)

    elif zero_center:
        if chunked:
            pca_func, X_pca = _run_chunked_pca(X, n_comps, chunk_size)
        elif cpissparse(X) or issparse(X):
            if svd_solver in ("lanczos", "randomized"):
                pca_func, X_pca = _run_sparse_svd_pca(
                    X,
                    n_comps,
                    svd_solver,
                    zero_center=zero_center,
                    rng=rng,
                    n_oversamples=kwargs.get("n_oversamples"),
                    n_iter=kwargs.get("n_iter"),
                )
            else:
                pca_func, X_pca = _run_covariance_pca(X, n_comps, zero_center)
        else:
            pca_func, X_pca = _run_cuml_pca(X, n_comps, svd_solver=svd_solver)

    else:  # not zero_center
        if cpissparse(X) or issparse(X):
            if svd_solver in ("lanczos", "randomized"):
                pca_func, X_pca = _run_sparse_svd_pca(
                    X,
                    n_comps,
                    svd_solver,
                    zero_center=zero_center,
                    rng=rng,
                    n_oversamples=kwargs.get("n_oversamples"),
                    n_iter=kwargs.get("n_iter"),
                )
            else:
                pca_func, X_pca = _run_covariance_pca(X, n_comps, zero_center)
        else:
            pca_func, X_pca = _run_cuml_tsvd(X, n_comps, svd_solver=svd_solver)

    if X_pca.dtype.descr != np.dtype(dtype).descr:
        X_pca = X_pca.astype(dtype)

    return pca_func, X_pca, n_comps


def _as_numpy(X):
    if isinstance(X, cp.ndarray):
        return X.get()
    else:
        return X


def _run_covariance_pca(X, n_comps, zero_center):
    """Run PCA using covariance matrix eigendecomposition."""
    if issparse(X):
        X = X_to_GPU(X)
    from ._sparse_pca._sparse_pca import PCA_sparse

    if not isspmatrix_csr(X) and not isinstance(X, DaskArray):
        X = X.tocsr()
    if issparse_cupy(X):
        X.sort_indices()

    pca_func = PCA_sparse(n_components=n_comps, zero_center=zero_center)
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_sparse_svd_pca(
    X,
    n_comps,
    svd_solver,
    *,
    zero_center: bool = True,
    rng: np.random.Generator,
    n_oversamples: int | None = None,
    n_iter: int | None = None,
):
    """Run PCA using SVD solvers (lanczos, randomized)."""
    if issparse(X):
        X = X_to_GPU(X)
    from ._sparse_pca._sparse_svd_pca import PCA_sparse_svd

    if not isspmatrix_csr(X):
        X = X.tocsr()
    if issparse_cupy(X):
        X.sort_indices()

    # Build kwargs for PCA_sparse_svd
    kwargs = {
        "n_components": n_comps,
        "svd_solver": svd_solver,
        "zero_center": zero_center,
        "rng": rng,
    }
    if n_oversamples is not None:
        kwargs["n_oversamples"] = n_oversamples
    if n_iter is not None:
        kwargs["n_iter"] = n_iter

    pca_func = PCA_sparse_svd(**kwargs)
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_chunked_pca(X, n_comps, chunk_size):
    from cuml.decomposition import IncrementalPCA

    X_pca = np.zeros((X.shape[0], n_comps), X.dtype)

    pca_func = IncrementalPCA(n_components=n_comps, output_type="numpy", batch_size=chunk_size)
    pca_func.fit(X)

    n_batches = math.ceil(X.shape[0] / chunk_size)
    for batch in range(n_batches):
        start_idx = batch * chunk_size
        stop_idx = min(batch * chunk_size + chunk_size, X.shape[0])
        chunk = X[start_idx:stop_idx, :]
        X_pca[start_idx:stop_idx] = pca_func.transform(chunk)

    return pca_func, X_pca


def _run_cuml_pca(X, n_comps, *, svd_solver: str | None):
    from cuml.decomposition import PCA

    if svd_solver is None or svd_solver == "covariance_eigh":
        svd_solver = "auto"
    pca_func = PCA(
        n_components=n_comps,
        svd_solver=svd_solver,
        output_type="numpy",
    )
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_cuml_tsvd(X, n_comps, *, svd_solver: str | None = None):
    from cuml.decomposition import TruncatedSVD

    if svd_solver is None:
        svd_solver = "auto"
    pca_func = TruncatedSVD(
        n_components=n_comps,
        algorithm=svd_solver,
        output_type="numpy",
    )
    X_pca = pca_func.fit_transform(X)
    return pca_func, X_pca


def _run_cuml_pca_dask(X, n_comps, *, svd_solver: str):
    from cuml.dask.decomposition import PCA

    pca_func = PCA(n_components=n_comps, svd_solver=svd_solver, whiten=False)
    X_pca = pca_func.fit_transform(X)
    # cuml-issue #5883
    X_pca = X_pca.compute_chunk_sizes()
    return pca_func, X_pca
