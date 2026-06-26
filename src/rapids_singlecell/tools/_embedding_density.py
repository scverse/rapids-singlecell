from __future__ import annotations

from typing import TYPE_CHECKING

import cupy as cp
import numpy as np

from rapids_singlecell.preprocessing._utils import _sanitize_column

if TYPE_CHECKING:
    from collections.abc import Sequence

    from anndata import AnnData


def embedding_density(
    adata: AnnData,
    basis: str = "umap",
    *,
    groupby: str | None = None,
    key_added: str | None = None,
    components: str | Sequence[str] = None,
) -> None:
    """\
    Calculate the density of cells in an embedding (per condition).
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
        basis
            The embedding over which the density will be calculated. This embedded
            representation should be found in `adata.obsm['X_[basis]']``.
        groupby
            Key for categorical observation/cell annotation for which densities
            are calculated per category.
        key_added
            Name of the `.obs` covariate that will be added with the density
            estimates.
        components
            The embedding dimensions over which the density should be calculated.
            This is limited to two components.

    Returns
    -------
    Updates `adata.obs` with an additional field specified by the `key_added` \
    parameter. This parameter defaults to `[basis]_density_[groupby]`, where \
    `[basis]` is one of `umap`, `diffmap`, `pca`, `tsne`, or `draw_graph_fa` \
    and `[groupby]` denotes the parameter input.

    Updates `adata.uns` with an additional field `[key_added]_params`.
    """
    # to ensure that newly created covariates are categorical
    # to test for category numbers
    if groupby is not None:
        _sanitize_column(adata, groupby)
    # Test user inputs
    basis = basis.lower()

    if basis == "fa":
        basis = "draw_graph_fa"

    if f"X_{basis}" not in adata.obsm:
        raise ValueError(
            "Cannot find the embedded representation "
            f"`adata.obsm['X_{basis}']`. Compute the embedding first."
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
        if adata.obs[groupby].dtype.name != "category":
            raise ValueError(f"{groupby!r} column does not contain categorical data")

    # Define new covariate name
    if key_added is not None:
        density_covariate = key_added
    elif groupby is not None:
        density_covariate = f"{basis}_density_{groupby}"
    else:
        density_covariate = f"{basis}_density"

    # Calculate the densities over each category in the groupby column
    if groupby is not None:
        categories = adata.obs[groupby].cat.categories

        density_values = np.zeros(adata.n_obs)

        for cat in categories:
            cat_mask = adata.obs[groupby] == cat
            embed_x = adata.obsm[f"X_{basis}"][cat_mask, components[0]]
            embed_y = adata.obsm[f"X_{basis}"][cat_mask, components[1]]

            dens_embed = _calc_density(cp.array(embed_x), cp.array(embed_y))
            density_values[cat_mask] = dens_embed

        adata.obs[density_covariate] = density_values
    else:  # if groupby is None
        # Calculate the density over the whole embedding without subsetting
        embed_x = cp.asarray(adata.obsm[f"X_{basis}"][:, components[0]])
        embed_y = cp.asarray(adata.obsm[f"X_{basis}"][:, components[1]])

        adata.obs[density_covariate] = _calc_density(embed_x, embed_y)

    # Reduce diffmap components for labeling
    # Note: plot_scatter takes care of correcting diffmap components
    #       for plotting automatically
    if basis != "diffmap":
        components += 1

    adata.uns[f"{density_covariate}_params"] = {
        "covariate": groupby,
        "components": components.tolist(),
    }


def _calc_density(x: cp.ndarray, y: cp.ndarray) -> np.ndarray:
    """\
    Calculates the density of points in 2 dimensions using a Gaussian KDE kernel.

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
