from __future__ import annotations

# -- Path setup --------------------------------------------------------------
import sys
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path
from typing import TYPE_CHECKING
import anndata  # noqa
import fast_array_utils  # noqa
from docutils import nodes
from docutils.parsers.rst import roles
from scverse_doc.config import DEFAULTS
from scverse_doc.registry import intersphinx

if TYPE_CHECKING:
    from sphinx.application import Sphinx


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src/"))

# -- Project information -----------------------------------------------------

info = metadata("rapids_singlecell")
project_name = "rapids-singlecell"
project = "rapids-singlecell"
title = "GPU accelerated single cell analysis"
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
source_repository = "https://github.com/scverse/rapids-singlecell"

# The full version, including alpha/beta/rc tags
release = info["Version"]

templates_path = ["_templates"]
bibtex_bibfiles = ["references.bib"]
suppress_warnings = [
    "ref.citation",
    "myst.header",  # https://github.com/executablebooks/MyST-Parser/issues/262
]

# -- General configuration ---------------------------------------------------

extensions = [
    "scverse_doc",  # sets up napoleon, so needs to be before sphinx_autodoc_typehints
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.extlinks",
    "sphinx.ext.imgconverter",
    "nbsphinx",
    "scverse_misc.sphinx_ext",
    "sphinx.ext.linkcode",
    "sphinxcontrib.bibtex",
]

autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "cudf",
    "cuml",
    "cugraph",
    "cupy",
    "cupyx",
    "pylibraft",
    "cuvs",
]
napoleon_use_rtype = False  # having a separate entry generally helps readability
api_dir = HERE / "api"
myst_url_schemes = ("http", "https", "mailto", "ftp")
typehints_defaults = "braces"

# scverse core packages, Python, NumPy, SciPy, pandas and Matplotlib come from the registry
intersphinx_mapping = intersphinx("squidpy", "spatialdata", "pertpy", "decoupler") | {
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "cuml": ("https://docs.rapids.ai/api/cuml/stable/", None),
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "cugraph": ("https://docs.rapids.ai/api/cugraph/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "rmm": ("https://docs.rapids.ai/api/rmm/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "omnipath": ("https://omnipath.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    *DEFAULTS.exclude_patterns,
    "._*",
    "release-notes/blank.md",
]

# These DOI records are valid, but their publishers reject automated link checks.
linkcheck_ignore = [
    r"https://doi\.org/10\.1002/spe\.4380211102",
    r"https://doi\.org/10\.1073/pnas\.0500334102",
    r"https://doi\.org/10\.1093/bioinformatics/btv325",
    r"https://doi\.org/10\.1093/bioinformatics/btw777",
    r"https://doi\.org/10\.1093/bioadv/vbac016",
    r"https://doi\.org/10\.1093/bioinformatics/bty1044",
    r"https://doi\.org/10\.1093/bioinformatics/btz625",
    r"https://doi\.org/10\.1177/10943420231179699",
]


# -- Options for HTML output -------------------------------------------------

html_theme_options = {
    # TODO: drop once scverse-doc is released with the website accents (scverse/scverse.github.io#377)
    "accent": "#76b900",  # NVIDIA green
    "secondary_accent": "#7400ff",  # violet, used for the sidebar captions
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_show_sphinx = False
html_logo = "_static/logo_RTD.svg"
html_static_path = ["_static"]
html_css_files = ["css/override.css"]

# OpenGraph metadata for social media previews
ogp_site_url = "https://rapids-singlecell.readthedocs.io/"
ogp_site_name = "rapids-singlecell"
ogp_image = "_static/logo_RTD.svg"

xref_overrides = {
    "spatialdata._core.spatialdata.SpatialData": "spatialdata.SpatialData",
}

nitpick_ignore = [
    ("py:class", "scipy.sparse.base.spmatrix"),
    ("py:meth", "pandas.DataFrame.iloc"),
    ("py:meth", "pandas.DataFrame.loc"),
    ("py:class", "pandas.core.series.Series"),
    ("py:class", "anndata._core.views.ArrayView"),
    ("py:class", "anndata._core.raw.Raw"),
    ("py:class", "scanpy._utils.Empty"),
    ("py:class", "rapids_singlecell._settings.Default"),
    ("py:class", "rapids_singlecell.pertpy_gpu._distance.MeanVar"),
    *[
        ("py:class", f"rapids_singlecell._settings.{cls}Preset")
        for cls in ["HVG", "BasicEmbedding", "RankGenesGroups", "Scale", "ScoreGenes"]
    ],
    ("py:data", "typing.Union"),
    ("py:class", "cuml.linear_model.LogisticRegression"),
    *[
        ("py:class", f"anndata._core.aligned_mapping.{cls}{kind}")
        for cls in "Layers AxisArrays PairwiseArrays".split()
        for kind in ["", "View"]
    ],
]


def setup(app: Sphinx) -> None:
    """App setup hook."""
    app.add_role("small", roles.GenericRole("small", nodes.inline))
    app.add_role("smaller", roles.GenericRole("smaller", nodes.inline))


# extlinks config
extlinks = {
    "issue": ("https://github.com/scverse/rapids-singlecell/issues/%s", "issue%s"),
    "pr": ("https://github.com/scverse/rapids-singlecell/pull/%s", "pr%s"),
}
