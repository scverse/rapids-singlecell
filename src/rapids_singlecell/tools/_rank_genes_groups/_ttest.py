from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from ._core import _RankGenes


def t_test(
    rg: _RankGenes, method: Literal["t-test", "t-test_overestim_var"]
) -> list[tuple[int, NDArray, NDArray]]:
    """Compute t-test statistics using Welch's t-test."""
    from scipy import stats

    rg._basic_stats()

    results: list[tuple[int, NDArray, NDArray]] = []

    for group_index in range(len(rg.groups_order)):
        if rg.ireference is not None and group_index == rg.ireference:
            continue

        mean_group = rg.means[group_index]
        var_group = rg.vars[group_index]
        ns_group = int(rg.group_sizes[group_index])

        if rg.ireference is not None:
            mean_rest = rg.means[rg.ireference]
            var_rest = rg.vars[rg.ireference]
            ns_other = int(rg.group_sizes[rg.ireference])
        else:
            mean_rest = rg.means_rest[group_index]
            var_rest = rg.vars_rest[group_index]
            ns_other = rg.X.shape[0] - ns_group

        if method == "t-test":
            ns_rest = ns_other
        elif method == "t-test_overestim_var":
            # Hack for overestimating the variance for small groups
            ns_rest = ns_group
        else:
            msg = "Method does not exist."
            raise ValueError(msg)

        # Welch's t-test using pre-computed stats
        with np.errstate(invalid="ignore"):
            scores, pvals = stats.ttest_ind_from_stats(
                mean1=mean_group,
                std1=np.sqrt(var_group),
                nobs1=ns_group,
                mean2=mean_rest,
                std2=np.sqrt(var_rest),
                nobs2=ns_rest,
                equal_var=False,  # Welch's
            )

        # Handle NaN values (when means are the same and vars are 0)
        scores[np.isnan(scores)] = 0
        pvals[np.isnan(pvals)] = 1

        results.append((group_index, scores, pvals))

    return results
