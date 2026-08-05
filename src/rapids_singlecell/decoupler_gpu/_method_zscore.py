from __future__ import annotations

import cupy as cp
import numpy as np
from cupyx.scipy.special import erfc

from rapids_singlecell.decoupler_gpu._helper._docs import docs
from rapids_singlecell.decoupler_gpu._helper._log import _log
from rapids_singlecell.decoupler_gpu._helper._Method import Method, MethodMeta


@docs.dedent
def _func_zscore(
    mat: cp.ndarray,
    adj: cp.ndarray,
    *,
    flavor: str = "RoKAI",
    verbose: bool = False,
) -> tuple[cp.ndarray, cp.ndarray]:
    r"""
    Z-score (ZSCORE).

    This approach computes the mean value of the molecular features for known targets,
    optionally subtracts the overall mean of all measured features,
    and normalizes the result by the standard deviation of all features and the square
    root of the number of targets.

    This formulation was originally introduced in KSEA, which explicitly includes the
    subtraction of the global mean to compute the enrichment score :math:`ES`.

    .. math::

        ES = \frac{(\mu_s-\mu_p) \times \sqrt m }{\sigma}

    Where:

    - :math:`\mu_s` is the mean of targets
    - :math:`\mu_p` is the mean of all features
    - :math:`m` is the number of targets
    - :math:`\sigma` is the standard deviation of all features

    However, in the RoKAI implementation, this global mean subtraction was omitted.

    .. math::

        ES = \frac{\mu_s \times \sqrt m }{\sigma}

    A two-sided :math:`p_{value}` is then calculated from the consensus score using
    the survival function :math:`sf` of the standard normal distribution.

    .. math::

        p = 2 \times \mathrm{sf}\bigl(\lvert \mathrm{ES} \rvert \bigr)

    %(yestest)s

    %(params)s

    flavor
        Which flavor to use when calculating the z-score, either KSEA or RoKAI.

    %(returns)s

    Example
    -------
    .. code-block:: python

        import decoupler as dc

        adata, net = dc.ds.toy()
        rsc.dcg.zscore(adata, net, tmin=3)
    """

    assert isinstance(flavor, str) and flavor in ["KSEA", "RoKAI"], (
        "flavor must be str and KSEA or RoKAI"
    )
    nobs, nvar = mat.shape
    nvar, nsrc = adj.shape
    m = f"zscore - calculating {nsrc} scores with flavor={flavor}"
    _log(m, level="info", verbose=verbose)
    stds = cp.std(mat, axis=1, ddof=1, keepdims=True)
    if flavor == "RoKAI":
        mean_all = cp.mean(mat, axis=1, keepdims=True)
    elif flavor == "KSEA":
        mean_all = cp.zeros(stds.shape, dtype=mat.dtype)
    n = cp.sqrt(cp.count_nonzero(adj, axis=0))
    mean = mat.dot(adj) / cp.sum(cp.abs(adj), axis=0)
    es = ((mean - mean_all) * n) / stds
    pv = erfc(cp.abs(es) / cp.sqrt(2.0))
    return es.get(), pv.get()


_zscore = MethodMeta(
    name="zscore",
    desc="Z-score (ZSCORE)",
    func=_func_zscore,
    stype="numerical",
    adj=True,
    weight=True,
    test=True,
    limits=(-np.inf, +np.inf),
    reference="https://doi.org/10.1038/s41467-021-21211-6",
)
zscore = Method(_method=_zscore)
