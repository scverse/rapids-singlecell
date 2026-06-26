from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy.special import betainc

from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta


def _cov(A: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    return cp.dot(b.T - b.mean(), A - A.mean(axis=0)) / (b.shape[0] - 1)


def _cor(A: cp.ndarray, b: cp.ndarray) -> cp.ndarray:
    cov = _cov(A, b)
    ssd = cp.std(A, axis=0, ddof=1) * cp.std(b, axis=0, ddof=1).reshape(-1, 1)
    return cov / ssd


def _tval(r: cp.ndarray, df: float) -> cp.ndarray:
    return r * cp.sqrt(df / ((1.0 - r + 1.0e-16) * (1.0 + r + 1.0e-16)))


@docs.dedent
def _func_ulm(
    mat: cp.ndarray,
    adj: cp.ndarray,
    *,
    tval: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Univariate Linear Model (ULM).

    This approach uses the molecular features from one observation as the population of samples
    and it fits a linear model with a single covariate, which is the feature weights of a set :math:`F`.

    .. math::

        y_i = \beta_0 + \beta_1 x_i + \varepsilon, \quad i = 1, 2, \ldots, n

    Where:

    - :math:`y_i` is the observed feature statistic (e.g. gene expression, :math:`log_{2}FC`, etc.) for feature :math:`i`
    - :math:`x_i` is the weight of feature :math:`i` in feature set :math:`F`. For unweighted sets, membership in the set is indicated by 1, and non-membership by 0.
    - :math:`\beta_0` is the intercept
    - :math:`\beta_1` is the slope coefficient
    - :math:`\varepsilon` is the error term for feature :math:`i`

    The enrichment score :math:`ES` is then calculated as the t-value of the slope coefficient.

    .. math::

        ES = t_{\beta_1} = \frac{\hat{\beta}_1}{\mathrm{SE}(\hat{\beta}_1)}

    Where:

    - :math:`t_{\beta_1}` is the t-value of the slope
    - :math:`\mathrm{SE}(\hat{\beta}_1)` is the standard error of the slope

    Next, :math:`p_{value}` are obtained by evaluating the two-sided survival function
    (:math:`sf`) of the Student’s t-distribution.

    .. math::

        p_{value} = 2 \times \mathrm{sf}(|ES|, \text{df})

    %(yestest)s

    %(params)s
    %(tval)s

    %(returns)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata, net = dc.ds.toy()
        rsc.dcg.ulm(adata, net, tmin=3)
    """
    # Get degrees of freedom
    n_var, n_src = adj.shape
    df = n_var - 2
    m = f"ulm - fitting {n_src} univariate models of {n_var} observations (targets) with {df} degrees of freedom"
    _log(m, level="info", verbose=verbose)
    # Compute R value for all
    r = _cor(adj, mat.T)
    # Compute t-value
    t = _tval(r, df)
    # Compute two-sided p-value via regularized incomplete beta function
    # betainc(df/2, 0.5, df/(t²+df)) = 2·sf(|t|, df) avoids 1-CDF cancellation
    pv = betainc(df / 2, 0.5, df / (t**2 + df))
    if tval:
        es = t
    else:
        # Compute coef
        es = r * (
            cp.std(mat.T, ddof=1, axis=0).reshape(-1, 1) / cp.std(adj, ddof=1, axis=0)
        )
    return es.get(), pv.get()


_ulm = MethodMeta(
    name="ulm",
    desc="Univariate Linear Model (ULM)",
    func=_func_ulm,
    stype="numerical",
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference="https://doi.org/10.1093/bioadv/vbac016",
)
ulm = Method(_method=_ulm)
