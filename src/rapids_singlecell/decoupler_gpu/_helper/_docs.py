from __future__ import annotations

from docrep import DocstringProcessor

_features = """\
features
    Column names of `mat`."""

_net = """\
net
    Dataframe in long format. Must include `source` and `target` columns, and optionally a `weight` column."""

_tmin = """\
tmin
    Minimum number of targets per source. Sources with fewer targets will be removed."""

_bsize = """\
bsize
    For large datasets in sparse format, this parameter controls how many observations are processed at once.
    Increasing this value speeds up computation but uses more memory.
    If omitted, uses the method's default batch size."""

_verbose = """\
verbose
    Whether to display progress messages and additional execution details."""

_data = """\
data
    AnnData instance, DataFrame or tuple of [matrix, samples, features]."""

_layer = """\
layer
    Layer key name of an `anndata.AnnData` instance."""

_raw = """\
raw
    Whether to use the `.raw` attribute of `anndata.AnnData`."""

_empty = """\
empty
    Whether to remove empty observations (rows) or features (columns)."""

_adata = """\
adata
    Annotated data matrix with observations (rows) and features (columns)."""

_times = """\
times
    Number of random permutations to do."""

_seed = """\
seed
    Random seed to use."""

_inplace = """\
inplace
    Whether to perform the operation in the same object."""

_min_cells = """\
min_cells
    Minimum number of cells per sample."""

_min_counts = """\
min_counts
    Minimum number of counts per sample."""

_key = """\
key
    `adata.obsm` key to use."""

_yestest = """\
Finally, the obtained :math:`p_{value}` are adjusted by Benjamini-Hochberg correction."""

_notest = """\
This method does not perform statistical testing on :math:`ES` and therefore does not return :math:`p_{value}`."""

_pre_load = """\
pre_load
    Whether to pre-load the data into memory. If `True`, the data will be pre-loaded into memory before processing."""

_returns = """\
Returns
-------

Enrichment scores :math:`ES` and, if applicable, adjusted :math:`p_{value}` by Benjamini-Hochberg.
"""

_tval = """\
tval
    Whether to return the t-value (`tval=True`) the coefficient of the fitted model (`tval=False`)."""

_adj_pv_gpu = """\
adj_pv_gpu
    Whether to use GPU for adjusting p-values."""

_params = f"""\
Parameters
----------
{_data}
{_net}
{_tmin}
{_layer}
{_raw}
{_empty}
{_bsize}
{_verbose}
{_pre_load}
{_adj_pv_gpu}"""

docs = DocstringProcessor(
    net=_net,
    tmin=_tmin,
    bsize=_bsize,
    verbose=_verbose,
    features=_features,
    data=_data,
    layer=_layer,
    raw=_raw,
    empty=_empty,
    adata=_adata,
    times=_times,
    seed=_seed,
    inplace=_inplace,
    min_cells=_min_cells,
    min_counts=_min_counts,
    key=_key,
    params=_params,
    yestest=_yestest,
    notest=_notest,
    returns=_returns,
    tval=_tval,
    pre_load=_pre_load,
)
