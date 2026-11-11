from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np
from pandas.api.types import infer_dtype
from scverse_misc import deprecated_arg

from rapids_singlecell._keys import _EMBEDDINGS, _existing_preset_keys
from rapids_singlecell._settings import Default, Preset, settings
from rapids_singlecell.get._out import DEPR_KEY_ADDED, DEPR_USE, _obs_col, _out_key, _resolve_obs
from rapids_singlecell.get._utils import MultiAcc, _resolve_matrix_acc
from rapids_singlecell.preprocessing._utils import _sanitize_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData

    from rapids_singlecell.get._utils import AdRef


def _basis_obsm_key(adata: AnnData, basis: str) -> str | None:
    """The `.obsm` key holding `basis`, under either preset's naming."""
    if basis in adata.obsm:
        return basis
    embedding, layout = (
        ("draw_graph", basis.removeprefix("draw_graph_")) if basis.startswith("draw_graph_") else (basis, "")
    )
    if embedding in _EMBEDDINGS:
        keys = _existing_preset_keys(adata, embedding, layout=layout)
        if keys is not None:
            return keys.obsm
    return next((key for key in (basis, f"X_{basis}") if key in adata.obsm), None)


@deprecated_arg("key_added", DEPR_KEY_ADDED)
@deprecated_arg("basis", DEPR_USE)
def embedding_density(
    adata: AnnData,
    use: MultiAcc | str | Default = Default(repr="'umap'"),
    *,
    groupby: AdRef | str | None = None,
    components: str | Sequence[str] = None,
    out: AdRef | str | bool = False,
    # deprecated
    basis: str | None = None,
    key_added: str | None = None,
) -> np.ndarray | None:
    """Calculate the density of cells in an embedding (per condition).

    Gaussian kernel density estimation is used to calculate the density of
    cells in an embedded space. This can be performed per category over a
    categorical cell annotation. The cell density can be plotted using the
    `pl.embedding_density` function.
    Note that density values are scaled to be between 0 and 1. Thus, the
    density value at each cell is only comparable to densities in
    the same category.
    This function was written by Sophie Tritschler and implemented into
    Scanpy by Malte Luecken.

    Parameters
    ----------
    adata
        The annotated data matrix.
    use
        The embedding over which the density will be calculated, e.g. `A.obsm['X_umap']`,
        or the name of a basis found in ``adata.obsm[basis]`` or ``adata.obsm[f"X_{basis}"]``.
        Strings are basis names, unless the scanpy v2 preset is active,
        where they are resolved by :meth:`anndata.acc.AdAcc.resolve`, e.g. `'obsm.X_umap'`.
    groupby
        Categorical `.obs` column for which densities are calculated per category,
        e.g. `A.obs['batch']` or its name.
    components
        The embedding dimensions over which the density should be calculated.
        This is limited to two components.
    out
        Where to write the densities, e.g. `A.obs['k']` (or `'obs.k'`);
        the parameters then go to `.uns['k_params']`.
        If `True`, the densities are returned instead of written.
        If `False`, they are written to `.obs['[basis]_density_[groupby]']`.

    Returns
    -------
    The densities if `out=True`, else `None`.
    Sets `adata.obs[key]` and `adata.uns[f'{key}_params']`, where `key` defaults to
    `[basis]_density_[groupby]`, `[basis]` being one of `umap`, `diffmap`, `pca`, `tsne`,
    or `draw_graph_fa`.
    """
    if basis is not None:
        if not isinstance(use, Default):
            raise TypeError("Pass either `use` or `basis`, not both.")
        use = basis
    elif isinstance(use, str) and settings.preset is Preset.ScanpyV2Preview:
        use = _resolve_matrix_acc(use)
    basis = "umap" if isinstance(use, Default) else use
    groupby = _obs_col(groupby, param="groupby")
    if not isinstance(basis, str):
        basis = _resolve_obs(basis, kinds=(MultiAcc,))
        basis_key = basis.k if basis.k in adata.obsm else None
        basis = basis.k.removeprefix("X_")
    else:
        basis = basis.lower()
        if basis == "fa":
            basis = "draw_graph_fa"
        basis_key = _basis_obsm_key(adata, basis)
    key = _out_key(out, key_added, kind="obs", default=f"{basis}_density" + ("" if groupby is None else f"_{groupby}"))
    # ensure that newly created covariates are categorical
    if groupby is not None and key is not True:
        _sanitize_column(adata, groupby)
    if basis_key is None:
        raise ValueError(
            "Cannot find the embedded representation "
            f"`adata.obsm[{basis!r}]` or `adata.obsm['X_{basis}']`. "
            "Compute the embedding first."
        )

    if components is None:
        components = "1,2"
    if isinstance(components, str):
        components = components.split(",")
    components = np.array(components).astype(int) - 1

    if len(components) != 2:
        raise ValueError("Please specify exactly 2 components, or `None`.")

    if basis == "diffmap":
        components += 1

    if groupby is not None:
        groups = adata.obs[groupby]
        if key is True and infer_dtype(groups) == "string":
            groups = groups.astype("category")  # like `_sanitize_column`, but local
        if groups.dtype.name != "category":
            raise ValueError(f"{groupby!r} column does not contain categorical data")

    # Calculate the densities over each category in the groupby column
    if groupby is not None:
        density_values = np.zeros(adata.n_obs)

        for cat in groups.cat.categories:
            cat_mask = (groups == cat).to_numpy()
            embed_x = adata.obsm[basis_key][cat_mask, components[0]]
            embed_y = adata.obsm[basis_key][cat_mask, components[1]]

            dens_embed = _calc_density(cp.array(embed_x), cp.array(embed_y))
            density_values[cat_mask] = dens_embed
    else:  # if groupby is None
        # Calculate the density over the whole embedding without subsetting
        embed_x = cp.asarray(adata.obsm[basis_key][:, components[0]])
        embed_y = cp.asarray(adata.obsm[basis_key][:, components[1]])
        density_values = _calc_density(embed_x, embed_y)

    if key is True:
        return density_values
    adata.obs[key] = density_values

    # Reduce diffmap components for labeling
    # Note: plot_scatter takes care of correcting diffmap components
    #       for plotting automatically
    if basis != "diffmap":
        components += 1

    adata.uns[f"{key}_params"] = {
        "covariate": groupby,
        "components": components.tolist(),
    }


def _calc_density(x: cp.ndarray, y: cp.ndarray) -> np.ndarray:
    """Calculates the density of points in 2 dimensions using a Gaussian KDE kernel.

    Uses a covariance-aware bandwidth (Scott's rule) matching
    :class:`scipy.stats.gaussian_kde`, and min-max scales the PDF.
    Each GPU thread computes the log-density for one query point via an
    in-thread streaming logsumexp over all training points.  No intermediate
    distance matrix is ever materialised.
    """
    from rapids_singlecell._cuda import _kde_cuda

    xy = cp.stack([x, y], axis=1)  # (n, 2), C-contiguous
    n = xy.shape[0]
    dtype = xy.dtype

    # Covariance-aware bandwidth matching scipy.stats.gaussian_kde
    scotts_factor = n ** (-1.0 / 6.0)
    data_cov = cp.cov(xy.T)  # (2, 2)
    inv_cov = cp.linalg.inv(scotts_factor**2 * data_cov)

    # Pre-multiply so the kernel just computes a·dx² + b·dx·dy + c·dy²
    a = -0.5 * float(inv_cov[0, 0])
    b = -float(inv_cov[0, 1])
    c = -0.5 * float(inv_cov[1, 1])

    z = cp.empty(n, dtype=dtype)

    _kde_cuda.gaussian_kde_2d(
        xy,
        out=z,
        n=n,
        a=a,
        b=b,
        c=c,
        stream=cp.cuda.get_current_stream().ptr,
    )

    # Min-max scale PDF (not log-PDF) to match scipy/scanpy
    pdf = cp.exp(z)
    min_pdf = pdf.min()
    scaled = (pdf - min_pdf) / (pdf.max() - min_pdf)

    return scaled.get()
