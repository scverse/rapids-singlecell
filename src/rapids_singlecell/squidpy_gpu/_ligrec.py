from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from itertools import product
from typing import (
    Literal,
)

import cupy as cp
import numpy as np
import pandas as pd
from anndata import AnnData
from cupyx.scipy import sparse
from cupyx.scipy.sparse import issparse as cpissparse
from scipy.sparse import csc_matrix, issparse

from ._utils import _assert_categorical_obs, _create_sparse_df

SOURCE = "source"
TARGET = "target"


def _get_interactions(
    interactions_params={},
    transmitter_params={"categories": "ligand"},
    receiver_params={"categories": "receptor"},
):
    try:
        from omnipath.interactions import import_intercell_network
    except ImportError:
        raise ImportError("Please install omnipath package via `pip install omnipath")

    interactions = import_intercell_network(
        interactions_params=interactions_params,
        transmitter_params=transmitter_params,
        receiver_params=receiver_params,
    )

    # we don't really care about these
    if SOURCE in interactions.columns:
        interactions.pop(SOURCE)
    if TARGET in interactions.columns:
        interactions.pop(TARGET)
    interactions.rename(
        columns={
            "genesymbol_intercell_source": SOURCE,
            "genesymbol_intercell_target": TARGET,
        },
        inplace=True,
    )

    interactions[SOURCE] = interactions[SOURCE].str.replace("^COMPLEX:", "", regex=True)
    interactions[TARGET] = interactions[TARGET].str.replace("^COMPLEX:", "", regex=True)
    return interactions


def _fdr_correct(
    pvals: pd.DataFrame,
    corr_method: str,
    corr_axis: Literal["interactions", "clusters"],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Correct p-values for FDR along specific axis in ``pvals``."""
    from pandas.core.arrays.sparse import SparseArray
    from statsmodels.stats.multitest import multipletests

    def fdr(pvals: pd.Series):
        _, qvals, _, _ = multipletests(
            np.nan_to_num(pvals.values, copy=True, nan=1.0),
            method=corr_method,
            alpha=alpha,
            is_sorted=False,
            returnsorted=False,
        )
        qvals[np.isnan(pvals.values)] = np.nan

        return SparseArray(qvals, dtype=qvals.dtype, fill_value=np.nan)

    if corr_axis == "clusters":
        # clusters are in columns
        pvals = pvals.apply(fdr)
    elif corr_axis == "interactions":
        pvals = pvals.T.apply(fdr).T
    else:
        raise NotImplementedError(
            f"FDR correction for `{corr_axis}` is not implemented."
        )

    return pvals


def _check_tuple_needles(needles, haystack, *, msg: str, reraise: bool = True):
    filtered = []

    for needle in needles:
        if not isinstance(needle, Sequence):
            raise TypeError(f"Expected a `Sequence`, found `{type(needle).__name__}`.")
        if len(needle) != 2:
            raise ValueError(
                f"Expected a `tuple` of length `2`, found `{len(needle)}`."
            )
        a, b = needle

        if a not in haystack:
            if reraise:
                raise ValueError(msg.format(a))
            else:
                continue
        if b not in haystack:
            if reraise:
                raise ValueError(msg.format(b))
            else:
                continue

        filtered.append((a, b))

    return filtered


def ligrec(
    adata: AnnData,
    cluster_key: str,
    *,
    clusters: list | None = None,
    interactions: pd.DataFrame | Mapping | Sequence | None = None,
    complex_policy: Literal["min", "all"] = "min",
    threshold: float = 0.01,
    corr_method: str | None = None,
    corr_axis: Literal["interactions", "clusters"] = "clusters",
    alpha: float = 0.05,
    use_raw: bool = True,
    copy: bool = False,
    key_added: str | None = None,
    gene_symbols: str | None = None,
    n_perms: int = 1000,
    interactions_params: dict = {},
    transmitter_params: dict = {"categories": "ligand"},
    receiver_params: dict = {"categories": "receptor"},
) -> pd.DataFrame | None:
    """\
    Perform the permutation test as described in [Efremova et al., 2020].

    Parameters
    ----------
        adata
            Annotated data object.

        cluster_key
            Key in :attr:`~anndata.AnnData.obs` where clustering is stored.

        clusters
            Clusters from :attr:`~anndata.AnnData.obs` `['{cluster_key}']`. \
            Can be specified either as a sequence of :class:`tuple` or just a \
            sequence of cluster names, in which case all combinations \
            considered.

        interactions
            Interaction to test. The type can be one of:
                * :class:`pandas.DataFrame` - must contain at least 2 columns named `source` and `target`.
                * :class:`dict` - dictionary with at least 2 keys named `source` and `target`.
                * :class:`typing.Sequence` - Either a sequence of str, in which case \
                    all combinations are produced, or a sequence of `tuple` of 2 `str` \
                    or a `tuple` of 2 sequences.

            If `None`, the interactions are extracted from omnipath. Protein complexes can \
                be specified by delimiting the components with `_`, such as `alpha_beta_gamma`.

        complex_policy
            Policy on how to handle complexes. Valid options are:
                * `min` - select gene with the minimum average expression. \
                This is the same as in [Efremova et al., 2020].
                * `all` - select all possible combinations between `source` and `target` complexes.

        threshold
            Do not perform permutation test if any of the interacting components is being \
            expressed in less than `threshold` percent of cells within a given cluster.

        corr_method
            Correction method for multiple testing. See \
            :func:`statsmodels.stats.multitest.multipletests()` for valid options.

        corr_axis
            Axis over which to perform the FDR correction. Only used when `corr_method != None`. Valid options are:
                * `interactions` - correct interactions by performing FDR correction across the clusters.
                * `clusters` - correct clusters by performing FDR correction across the interactions.

        alpha
            Significance level for FDR correction. Only used when `corr_method != None`.

        use_raw
            Whether to access :attr:`~anndata.AnnData.raw`

        copy
            If `True`, return the result, otherwise save it to the `adata` object.

        key_added
            Key in :attr:`~anndata.AnnData.uns` where the result is stored if copy = False. \
            If None, '{cluster_key}_ligrec' will be used.

        gene_symbols
            Key in :attr:`~anndata.AnnData.var` to use instead of :attr:`~anndata.AnnData.var_names`.

        n_perms
            Number of permutations for the permutation test.

        interactions_params
            Keyword arguments for :func:`omnipath.interactions.import_intercell_network()` defining the \
            interactions. These datasets from [Türei et al., 2016] are used by default: \
            omnipath, pathwayextra, kinaseextra and ligrecextra.

        transmitter_params
            Keyword arguments for :func:`omnipath.interactions.import_intercell_network()` \
            defining the transmitter side of intercellular connections.

        receiver_params
            Keyword arguments for :func:`omnipath.interactions.import_intercell_network()` \
            defining the receiver side of intercellular connections.

    Returns
    -------
    If `copy = True`, returns a dict with following keys:
        * `means` -
            :class:`pandas.DataFrame` containing the mean expression.
        * `pvalues` -
            :class:`pandas.DataFrame` containing the possibly corrected p-values.
        * `metadata` -
            :class:`pandas.DataFrame` containing interaction metadata.
    Otherwise, modifies the adata object with the following key:
        * :attr:`~anndata.AnnData.uns` `['{key_added}']` -
            the above mentioned dict.

    NaN p-values mark combinations for which the mean expression of one of the \
    interacting components was 0 or it didn't pass the threshold percentage of \
    cells being expressed within a given cluster.
    """
    # Get and Check interactions
    if interactions is None:
        interactions = _get_interactions(
            interactions_params, transmitter_params, receiver_params
        )

    if isinstance(interactions, Mapping):
        interactions = pd.DataFrame(interactions)

    if isinstance(interactions, pd.DataFrame):
        if SOURCE not in interactions.columns:
            raise KeyError(f"Column `{SOURCE!r}` is not in `interactions`.")
        if TARGET not in interactions.columns:
            raise KeyError(f"Column `{TARGET!r}` is not in `interactions`.")
    elif isinstance(interactions, Iterable):
        interactions = tuple(interactions)
        if not len(interactions):
            raise ValueError("No interactions were specified.")

        if isinstance(interactions[0], str):
            interactions = list(product(interactions, repeat=2))
        elif len(interactions) == 2:
            interactions = tuple(zip(*interactions))

        if not all(len(i) == 2 for i in interactions):
            raise ValueError("Not all interactions are of length `2`.")

        interactions = pd.DataFrame(interactions, columns=[SOURCE, TARGET])
    else:
        raise TypeError(
            f"Expected either a `pandas.DataFrame`, `dict` or `iterable`, found `{type(interactions).__name__}`"
        )

    assert isinstance(interactions, pd.DataFrame)

    if corr_axis:
        if corr_axis not in {"clusters", "interactions"}:
            raise ValueError(f"Invalid option `{corr_axis}` for `CorrAxis`.")

    if n_perms <= 0:
        raise ValueError(f"Expected `n_perms` to be positive, found `{n_perms}`.")

    if interactions.empty:
        raise ValueError("The interactions are empty")
    # Prepare adata
    if not isinstance(adata, AnnData):
        raise TypeError(
            f"Expected `adata` to be of type `anndata.AnnData`, found `{type(adata).__name__}`."
        )
    if not adata.n_obs:
        raise ValueError("No cells are in `adata.obs_names`.")
    if not adata.n_vars:
        raise ValueError("No genes are in `adata.var_names`.")

    if use_raw:
        if adata.raw is None:
            raise AttributeError(
                "No `.raw` attribute found. Try specifying `use_raw=False`."
            )
        var_names = adata.raw.var_names
        X = adata.raw.X
    else:
        var_names = adata.var_names
        X = adata.X

    data = pd.DataFrame.sparse.from_spmatrix(
        csc_matrix(X), index=adata.obs_names, columns=var_names
    )

    if gene_symbols:
        if use_raw:
            new_genes = adata.raw.var[gene_symbols]
        else:
            new_genes = adata.var[gene_symbols]
        data = data.rename(columns=new_genes)

    interactions[SOURCE] = interactions[SOURCE].str.upper()
    interactions[TARGET] = interactions[TARGET].str.upper()
    interactions.dropna(subset=(SOURCE, TARGET), inplace=True, how="any")
    interactions.drop_duplicates(subset=(SOURCE, TARGET), inplace=True, keep="first")
    data.columns = data.columns.str.upper()
    n_genes_prior = data.shape[1]
    data = data.loc[:, ~data.columns.duplicated()]
    if data.shape[1] != n_genes_prior:
        print(f"WARNING: Removed `{n_genes_prior - data.shape[1]}` duplicate gene(s)")

    def find_min_gene_in_complex(_complex: str | None) -> str | None:
        if _complex is None:
            return None
        if "_" not in _complex:
            return _complex
        complexes = [c for c in _complex.split("_") if c in data.columns]
        if not len(complexes):
            return None
        if len(complexes) == 1:
            return complexes[0]

        df = data[complexes].mean()
        try:
            return str(df.index[df.argmin()])
        except ValueError as e:
            if "attempt to get argmin of an empty sequence" in str(e):
                return str(df.index[0])
            else:
                raise ValueError(e)

    if complex_policy == "min":
        interactions[SOURCE] = interactions[SOURCE].apply(find_min_gene_in_complex)
        interactions[TARGET] = interactions[TARGET].apply(find_min_gene_in_complex)
    elif complex_policy == "all":
        src = interactions.pop(SOURCE).apply(lambda s: str(s).split("_")).explode()
        src.name = SOURCE
        tgt = interactions.pop(TARGET).apply(lambda s: str(s).split("_")).explode()
        tgt.name = TARGET

        interactions = pd.merge(
            interactions, src, how="left", left_index=True, right_index=True
        )
        interactions = pd.merge(
            interactions, tgt, how="left", left_index=True, right_index=True
        )
    else:
        raise ValueError(f"Invalid option `{complex_policy!r}` for `ComplexPolicy`.")

    interactions = interactions[
        interactions[SOURCE].isin(data.columns)
        & interactions[TARGET].isin(data.columns)
    ]

    _assert_categorical_obs(adata, key=cluster_key)
    if interactions.empty:
        raise ValueError("After filtering by genes, no interactions remain.")
    filtered_data = data.loc[
        :, list(set(interactions[SOURCE]) | set(interactions[TARGET]))
    ]
    if len(adata.obs[cluster_key].cat.categories) <= 1:
        raise ValueError(
            f"Expected at least `2` clusters, found `{len(adata.obs[cluster_key].cat.categories)}`."
        )
    filtered_data["clusters"] = (
        adata.obs.copy()[cluster_key].astype("string").astype("category").values
    )
    interactions.drop_duplicates(subset=(SOURCE, TARGET), inplace=True, keep="first")
    if clusters is None:
        clusters = list(map(str, adata.obs[cluster_key].cat.categories))
    if all(isinstance(c, str) for c in clusters):
        clusters = list(product(clusters, repeat=2))
    clusters = sorted(
        _check_tuple_needles(
            clusters,
            filtered_data["clusters"].cat.categories,
            msg="Invalid cluster `{0!r}`.",
            reraise=True,
        )
    )

    clusters_named = clusters.copy()
    clusters_flat = list({c for cs in clusters for c in cs})
    filter_obs = np.isin(filtered_data["clusters"], clusters_flat)

    if gene_symbols:
        if use_raw:
            filter_var = np.isin(
                adata.raw.var[gene_symbols].str.upper(), filtered_data.columns
            )
            genes = adata.raw.var[gene_symbols][filter_var].str.upper()
        else:
            filter_var = np.isin(
                adata.var[gene_symbols].str.upper(), filtered_data.columns
            )
            genes = adata.var[gene_symbols][filter_var].str.upper()
    else:
        if use_raw:
            filter_var = np.isin(adata.raw.var_names.str.upper(), filtered_data.columns)
            genes = adata.raw.var_names[filter_var].str.upper()
        else:
            filter_var = np.isin(adata.var_names.str.upper(), filtered_data.columns)
            genes = adata.var_names[filter_var].str.upper()

    if use_raw:
        mat = adata.raw[filter_obs, filter_var].X
    else:
        mat = adata[filter_obs, filter_var].X
    cluster_obs = adata.obs.loc[filter_obs, cluster_key]

    cluster_obs = cluster_obs.cat.remove_unused_categories()
    cat = cluster_obs.cat

    interactions_all = interactions.copy()
    interactions = pd.DataFrame(interactions, columns=[SOURCE, TARGET])

    cluster_mapper = dict(zip(cat.categories, range(len(cat.categories))))
    gene_mapper = dict(zip(genes, range(len(genes))))  # -1 for 'clusters'

    genes = [gene_mapper[c] if c != "clusters" else c for c in genes]
    clusters_ = np.array(
        [[cluster_mapper[c1], cluster_mapper[c2]] for c1, c2 in clusters],
        dtype=np.uint32,
    )

    data["clusters"] = cat.rename_categories(cluster_mapper)
    # much faster than applymap (tested on 1M interactions)
    interactions_ = np.vectorize(lambda g: gene_mapper[g])(interactions.values)

    if issparse(mat):
        data_cp = sparse.csr_matrix(mat.tocsr())
    else:
        data_cp = cp.array(mat)

    # Convert the 'clusters' column to a CuPy array
    clusters = cp.array(data["clusters"].values, dtype=cp.int32)

    # Find the unique clusters and the number of clusters
    unique_clusters = cp.unique(clusters)
    n_clusters = unique_clusters.size

    # Calculate the total counts per cluster
    total_counts = cp.bincount(clusters)

    from rapids_singlecell._cuda import _ligrec_cuda as _lc

    if not cpissparse(data_cp):
        sum_gt0 = cp.zeros((data_cp.shape[1], n_clusters), dtype=cp.float32)
        count_gt0 = cp.zeros((data_cp.shape[1], n_clusters), dtype=cp.int32)

        _lc.sum_count_dense(
            data_cp,
            clusters=clusters,
            sum=sum_gt0,
            count=count_gt0,
            rows=data_cp.shape[0],
            cols=data_cp.shape[1],
            ncls=n_clusters,
            stream=cp.cuda.get_current_stream().ptr,
        )

        mean_cp = sum_gt0 / total_counts
        mask_cp = count_gt0 / total_counts >= threshold
        del sum_gt0, count_gt0
    else:
        sum_gt0 = cp.zeros((data_cp.shape[1], n_clusters), dtype=cp.float32, order="C")
        count_gt0 = cp.zeros((data_cp.shape[1], n_clusters), dtype=cp.int32, order="C")
        _lc.sum_count_sparse(
            data_cp.indptr,
            data_cp.indices,
            data_cp.data,
            clusters=clusters,
            sum=sum_gt0,
            count=count_gt0,
            rows=data_cp.shape[0],
            ncls=n_clusters,
            stream=cp.cuda.get_current_stream().ptr,
        )
        mean_cp = sum_gt0 / total_counts
        mask_cp = count_gt0 / total_counts >= threshold
        del sum_gt0, count_gt0

    interactions_ = cp.array(interactions_)
    interaction_clusters = cp.array(clusters_)
    clustering_use = clusters.copy()
    n_cls = mean_cp.shape[1]

    interactions_ = interactions_.astype(cp.int32, order="C")
    mean_cp = mean_cp.astype(cp.float32, order="C")
    mask_cp = mask_cp.astype(cp.bool_, order="C")
    total_counts = total_counts.astype(cp.float32)
    res = cp.zeros(
        (len(interactions_), len(interaction_clusters)), dtype=np.float32, order="C"
    )
    if cpissparse(data_cp):
        for _i in range(n_perms):
            cp.random.shuffle(clustering_use)
            g = cp.zeros((data_cp.shape[1], n_cls), dtype=cp.float32, order="C")
            _lc.mean_sparse(
                data_cp.indptr,
                data_cp.indices,
                data_cp.data,
                clusters=clustering_use,
                g=g,
                rows=data_cp.shape[0],
                ncls=n_clusters,
                stream=cp.cuda.get_current_stream().ptr,
            )

            _lc.elementwise_diff(
                g,
                total_counts=total_counts,
                n_genes=data_cp.shape[1],
                n_clusters=n_cls,
                stream=cp.cuda.get_current_stream().ptr,
            )
            _lc.interaction(
                interactions_,
                interaction_clusters=interaction_clusters,
                mean=mean_cp,
                res=res,
                mask=mask_cp,
                g=g,
                n_iter=len(interactions_),
                n_inter_clust=len(interaction_clusters),
                ncls=n_cls,
                stream=cp.cuda.get_current_stream().ptr,
            )
    else:
        for _i in range(n_perms):
            cp.random.shuffle(clustering_use)
            g = cp.zeros((data_cp.shape[1], n_cls), dtype=cp.float32, order="C")
            _lc.mean_dense(
                data_cp,
                clusters=clustering_use,
                g=g,
                rows=data_cp.shape[0],
                cols=data_cp.shape[1],
                ncls=n_cls,
                stream=cp.cuda.get_current_stream().ptr,
            )
            _lc.elementwise_diff(
                g,
                total_counts=total_counts,
                n_genes=data_cp.shape[1],
                n_clusters=n_cls,
                stream=cp.cuda.get_current_stream().ptr,
            )
            _lc.interaction(
                interactions_,
                interaction_clusters=interaction_clusters,
                mean=mean_cp,
                res=res,
                mask=mask_cp,
                g=g,
                n_iter=len(interactions_),
                n_inter_clust=len(interaction_clusters),
                ncls=n_cls,
                stream=cp.cuda.get_current_stream().ptr,
            )

    res_mean = cp.zeros(
        (len(interactions_), len(interaction_clusters)), dtype=np.float32, order="C"
    )

    _lc.res_mean(
        interactions_,
        interaction_clusters=interaction_clusters,
        mean=mean_cp,
        res_mean=res_mean,
        n_inter=len(interactions_),
        n_inter_clust=len(interaction_clusters),
        ncls=n_cls,
        stream=cp.cuda.get_current_stream().ptr,
    )

    res_mean = res_mean.get()
    pvalues = (res / n_perms).get()

    res = {
        "means": _create_sparse_df(
            res_mean,
            index=pd.MultiIndex.from_frame(interactions, names=[SOURCE, TARGET]),
            columns=pd.MultiIndex.from_tuples(
                clusters_named, names=["cluster_1", "cluster_2"]
            ),
            fill_value=0,
        ),
        "pvalues": _create_sparse_df(
            pvalues,
            index=pd.MultiIndex.from_frame(interactions, names=[SOURCE, TARGET]),
            columns=pd.MultiIndex.from_tuples(
                clusters_named, names=["cluster_1", "cluster_2"]
            ),
            fill_value=np.nan,
        ),
        "metadata": interactions_all[
            interactions_all.columns.difference([SOURCE, TARGET])
        ],
    }
    res["metadata"].index = res["means"].index.copy()

    if corr_method is not None:
        res["pvalues"] = _fdr_correct(
            res["pvalues"], corr_method, corr_axis, alpha=alpha
        )

    if copy:
        return res

    if key_added is None:
        key_added = f"{cluster_key}_ligrec"
    adata.uns[key_added] = res
