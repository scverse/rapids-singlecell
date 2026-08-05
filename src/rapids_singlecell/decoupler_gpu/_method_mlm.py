from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy.special import betainc

from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta


def _fit(X: cp.ndarray, y: cp.ndarray, inv: cp.ndarray, df: float) -> cp.ndarray:
    X = cp.ascontiguousarray(X)
    coef, sse, _, _ = cp.linalg.lstsq(X, y, rcond=-1)
    if len(sse) == 0:
        raise ValueError(
            """Couldn't fit a multivariate linear model. This can happen because there are more sources
        (covariates) than unique targets (samples), or because the network\'s matrix rank is smaller than the number of
        sources."""
        )
    sse = sse / df
    inv = cp.diag(inv)
    sse = cp.reshape(sse, (sse.shape[0], 1))
    inv = cp.reshape(inv, (1, inv.shape[0]))
    se = cp.sqrt(sse * inv)
    coef = coef.T
    tval = coef / se
    return coef[:, 1:], tval[:, 1:]


@docs.dedent
def _func_mlm(
    mat: cp.ndarray,
    adj: cp.ndarray,
    *,
    tval: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    r"""
    Multivariate Linear Model (MLM).

    This approach uses the molecular features from one observation as the population of samples
    and it fits a linear model with with multiple covariates, which are the weights of all feature sets :math:`F`.

    .. math::

        y^i = \beta_0 + \beta_1 x_{1}^{i} + \beta_2 x_{2}^{i} + \cdots + \beta_p x_{p}^{i} + \varepsilon

    Where:

    - :math:`y^i` is the observed feature statistic (e.g. gene expression, :math:`log_{2}FC`, etc.) for feature :math:`i`
    - :math:`x_{p}^{i}` is the weight of feature :math:`i` in feature set :math:`F_p`. For unweighted sets, membership in the set is indicated by 1, and non-membership by 0.
    - :math:`\beta_0` is the intercept
    - :math:`\beta_p` is the slope coefficient for feature set :math:`F_p`
    - :math:`\varepsilon` is the error term for feature :math:`i`

    The enrichment score :math:`ES` for each :math:`F` is then calculated as the t-value of the slope coefficients.

    .. math::

        ES = t_{\beta_1} = \frac{\hat{\beta}_1}{\mathrm{SE}(\hat{\beta}_1)}

    Where:

    - :math:`t_{\beta_1}` is the t-value of the slope
    - :math:`\mathrm{SE}(\hat{\beta}_1)` is the standard error of the slope

    Next, :math:`p_{value}` are obtained by evaluating the two-sided survival function
    (:math:`sf`) of the Student’s t-distribution.

    .. math::

        p_{value} = 2 \times \mathrm{sf}(|ES|, \text{df})


    %(params)s
    %(tval)s

    %(returns)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata, net = dc.ds.toy()
        rsc.dcg.mlm(adata, net, tmin=3)
    """
    # Get dims
    n_features, n_fsets = adj.shape
    # Add intercept
    adj = cp.column_stack((cp.ones((n_features,), dtype=cp.float32), adj))
    # Compute inv and df for lm
    inv = cp.linalg.inv(cp.dot(adj.T, adj))
    df = n_features - n_fsets - 1

    m = f"mlm - fitting {n_fsets} multivariate models of {n_features} observations with {df} degrees of freedom"
    _log(m, level="info", verbose=verbose)

    # Compute tval
    coef, t = _fit(adj, mat.T, inv, df)
    # Compute two-sided p-value via regularized incomplete beta function
    # betainc(df/2, 0.5, df/(t²+df)) = 2·sf(|t|, df) avoids 1-CDF cancellation
    pv = betainc(df / 2, 0.5, df / (t**2 + df))
    # Return coef or tval
    if tval:
        es = t
    else:
        es = coef
    return es.get(), pv.get()


_mlm = MethodMeta(
    name="mlm",
    desc="Multivariate Linear Model (MLM)",
    func=_func_mlm,
    stype="numerical",
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference="https://doi.org/10.1093/bioadv/vbac016",
)

mlm = Method(_method=_mlm)
