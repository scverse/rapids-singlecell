from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable

from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta


def _ridx(
    times: int,
    nvar: int,
    seed: int | None,
):
    idx = cp.tile(cp.arange(nvar), (times, 1))
    if seed:
        rng = np.random.default_rng(seed=seed)
        idx = idx.get()
        for i in idx:
            rng.shuffle(i)
        idx = cp.array(idx)
    return idx


def _wsum(x: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
    out = cp.zeros((x.shape[0], w.shape[1]), dtype=x.dtype, order="F")
    cp.cublas.gemm("N", "N", a=x, b=w, out=out)
    return out


def _wmean(x: cp.ndarray, w: cp.ndarray) -> cp.ndarray:
    agg = _wsum(x, w)
    div = cp.sum(cp.abs(w), axis=0)
    return agg / div


def _fun(
    f: Callable,
    *,  # keyword-only arguments after this
    verbose: bool = False,
):
    def _f(mat, adj):
        es = f(mat, adj)
        return es

    _f.__name__ = f.__name__
    if _f.__name__ not in _cfuncs:
        _cfuncs[f.__name__] = _f
        m = f"waggr - using {_f.__name__}"
        _log(m, level="info", verbose=verbose)


_fun_dict = {
    "wsum": _wsum,
    "wmean": _wmean,
}

_cfuncs: dict = {}


def _validate_args(
    fun: Callable,
    *,  # keyword-only arguments after this
    verbose: bool,
) -> Callable:
    args = inspect.signature(fun).parameters
    required_args = ["x", "w"]
    for arg in required_args:
        if arg not in args:
            assert AssertionError(), (
                f"fun={fun.__name__} must contain arguments x and w"
            )
    # Check if any additional arguments have default values
    for param in args.values():
        if param.name not in required_args and param.default == inspect.Parameter.empty:
            assert AssertionError(), (
                f"fun={fun.__name__} has an argument {param.name} without a default value"
            )
    return fun


def _validate_func(
    fun: Callable,
    *,  # keyword-only arguments after this
    verbose: bool,
) -> None:
    fun = _validate_args(fun=fun, verbose=verbose)
    x = cp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    w = cp.array([[-1.0, 3.0], [0.0, 4.0], [2.0, 5.0]])
    try:
        res = fun(x=x, w=w)
        assert isinstance(res, cp.ndarray), "output of fun must be a cp.ndarray"
        assert res.shape == (x.shape[0], w.shape[1]), (
            "output of fun must be a cp.ndarray with shape (x.shape[0], w.shape[1])"
        )
    except Exception as err:
        raise ValueError(
            f"fun failed to run with test data: fun(x={x}), w={w}"
        ) from err
    m = f"waggr - using function {fun.__name__}"
    _log(m, level="info", verbose=verbose)
    _fun(f=fun, verbose=verbose)


def _perm(
    fun: Callable,
    es: cp.ndarray,
    mat: cp.ndarray,
    *,
    adj: cp.ndarray,
    idx: cp.ndarray,
):
    # Init
    nobs, nvar = mat.shape
    nvar, nsrc = adj.shape
    times, nvar = idx.shape
    es_abs = cp.abs(es)
    # Initialize accumulators for statistics
    sum_null = cp.zeros((nobs, nsrc), dtype=mat.dtype, order="C")
    sum_null_sq = cp.zeros((nobs, nsrc), dtype=mat.dtype, order="C")
    extreme_count = cp.zeros((nobs, nsrc), dtype=cp.int32, order="C")

    mat = mat.T.copy()
    adj = cp.asfortranarray(adj)
    # Permute
    for i in range(times):
        mat_perm = mat[idx[i], :].T
        # mat_perm = mat[:, idx[i]]
        # Apply the function
        perm_result = fun(mat_perm, adj)
        # Update running statistics
        sum_null += perm_result
        sum_null_sq += perm_result**2
        extreme_count += cp.abs(perm_result) > es_abs

    # Compute final statistics
    null_mean = sum_null / times
    # Var(X) = E[X²] - (E[X])²
    null_var = (sum_null_sq / times) - (null_mean * null_mean)
    null_std = cp.sqrt(cp.maximum(null_var, 1e-10))

    # Compute NES
    nes = cp.where(
        null_std > 1e-10,
        (es - null_mean) / null_std,
        cp.where(cp.abs(es) > 1e-10, cp.sign(es) * 1e6, 0.0),
    )

    # Compute empirical p-value
    pvals = extreme_count.astype(cp.float32)
    pvals = cp.where(pvals == 0.0, 1.0, pvals)
    pvals = cp.where(pvals == times, times - 1, pvals)
    pvals = pvals / times
    pvals = cp.where(pvals >= 0.5, 1 - pvals, pvals)
    pvals = pvals * 2  # Two-tailed test

    return nes.astype(cp.float32), pvals


@docs.dedent
def _func_waggr(
    mat: cp.ndarray,
    adj: cp.ndarray,
    fun: str | Callable = "wmean",
    *,  # keyword-only arguments after this
    times: int | float = 1000,
    seed: int | float = 42,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Weighted Aggregate (WAGGR).

    This approach aggregates the molecular features :math:`x_i` from one observation :math:`i` with
    the feature weights :math:`w` of a given feature set :math:`j` into an enrichment score :math:`ES`.

    This method can use any aggregation function, which by default is the weighted mean.

    .. math::

        ES = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}

    Another simpler option is the weighted sum.

    .. math::

        ES = \sum_{i=1}^{n} w_i x_i

    Alternatively, this method can also take any defined function :math:`f` as long at it aggregates :math:`x_i` and
    :math:`w` into a single :math:`ES`.

    .. math::

        ES = f(w_i, x_i)

    This functionality makes it relatively easy to implement and try new enrichment methods.

    When multiple random permutations are done (``times > 1``), statistical significance is assessed via empirical testing.

    .. math::

        p_{value}=\frac{ES_{rand} \geq ES}{P}

    Where:

    - :math:`ES_{rand}` are the enrichment scores of the random permutations
    - :math:`P` is the total number of permutations

    Additionally, :math:`ES` is updated to a normalized enrichment score :math:`NES`.

    .. math::

        NES = \frac{ES - \mu(ES_{rand})}{\sigma(ES_{rand})}

    Where:

    - :math:`\mu` is the mean
    - :math:`\sigma` is the standard deviation

    %(yestest)s

    %(params)s

    fun
        Function to compute enrichment statistic from omics readouts (``x``) and feature weights (``w``).
        Provided function must contain ``x`` and ``w`` arguments and output a single float.
        By default, 'wmean' and 'wsum' are implemented.
    %(times)s
    %(seed)s

    %(returns)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata, net = dc.ds.toy()
        dc.mt.waggr(adata, net, tmin=3)
    """
    assert isinstance(fun, str) or callable(fun), "fun must be str or callable"
    if isinstance(fun, str):
        assert fun in _fun_dict, "when fun is str, it must be wmean or wsum"
        f_fun = _fun_dict[fun]
    else:
        f_fun = fun
    _validate_func(f_fun, verbose=verbose)
    vfun = _cfuncs[f_fun.__name__]
    assert isinstance(times, int | float) and times >= 0, (
        "times must be numeric and >= 0"
    )
    assert isinstance(seed, int | float) and seed >= 0, "seed must be numeric and >= 0"
    times, seed = int(times), int(seed)
    nobs, nvar = mat.shape
    nvar, nsrc = adj.shape
    m = f"waggr - calculating scores for {nsrc} sources across {nobs} observations"
    _log(m, level="info", verbose=verbose)
    es = vfun(mat, adj)
    if times > 1:
        m = f"waggr - comparing estimates against {times} random permutations"
        _log(m, level="info", verbose=verbose)
        idx = _ridx(times=times, nvar=nvar, seed=seed)
        es, pv = _perm(fun=vfun, es=es, mat=mat, adj=adj, idx=idx)
    else:
        pv = cp.ones(es.shape)
    return es.get(), pv.get()


_waggr = MethodMeta(
    name="waggr",
    desc="Weighted Aggregate (WAGGR)",
    func=_func_waggr,
    stype="numerical",
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference="https://doi.org/10.1093/bioadv/vbac016",
)
waggr = Method(_method=_waggr)
