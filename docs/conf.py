from __future__ import annotations

# -- Path setup --------------------------------------------------------------
import sys
import os
from datetime import datetime
from importlib.metadata import metadata
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
import anndata  # noqa
import fast_array_utils  # noqa

if TYPE_CHECKING:
    from sphinx.application import Sphinx


HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "src/"))

on_rtd = os.environ.get("READTHEDOCS") == "True"
rtd_links_prefix = PurePosixPath("src")

# -- Project information -----------------------------------------------------

info = metadata("rapids_singlecell")
project_name = "rapids-singlecell"
project = "rapids-singlecell"
title = "GPU accelerated single cell analysis"
author = info["Author"]
copyright = f"{datetime.now():%Y}, {author}"
version = info["Version"]
repository_url = "https://github.com/scverse/rapids_singlecell"

# The full version, including alpha/beta/rc tags
release = info["Version"]

templates_path = ["_templates"]
bibtex_bibfiles = ["references.bib"]
nitpicky = True  # Warn about broken links
needs_sphinx = "4.5"
suppress_warnings = [
    "ref.citation",
    "myst.header",  # https://github.com/executablebooks/MyST-Parser/issues/262
]

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_nb",
    "sphinx_design",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.githubpages",
    "sphinx_autodoc_typehints",
    "sphinx.ext.extlinks",
    "sphinx.ext.imgconverter",
    "sphinx_copybutton",
    "nbsphinx",
    "scanpydoc",
    "sphinx.ext.linkcode",
    "sphinx_tabs.tabs",
    "sphinxext.opengraph",
    "sphinxcontrib.bibtex",
]

autosummary_generate = True
autodoc_member_order = "bysource"
autodoc_mock_imports = [
    "cudf",
    "cuml",
    "cugraph",
    "cupy",
    "cupyx",
    "pylibraft",
    "dask",
    "cuvs",
]
default_role = "literal"
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = False  # having a separate entry generally helps readability
napoleon_use_param = True
api_dir = HERE / "api"
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "html_image",
    "html_admonition",
]
myst_url_schemes = ("http", "https", "mailto", "ftp")
nb_output_stderr = "remove"
nb_execution_mode = "off"
nb_merge_streams = True
typehints_defaults = "braces"

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
    ".myst": "myst-nb",
}

intersphinx_mapping = {
    "anndata": ("https://anndata.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "cupy": ("https://docs.cupy.dev/en/stable/", None),
    "python": ("https://docs.python.org/3", None),
    "cuml": ("https://docs.rapids.ai/api/cuml/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "cudf": ("https://docs.rapids.ai/api/cudf/stable/", None),
    "cugraph": ("https://docs.rapids.ai/api/cugraph/stable/", None),
    "scanpy": ("https://scanpy.readthedocs.io/en/stable/", None),
    "squidpy": ("https://squidpy.readthedocs.io/en/stable/", None),
    "pertpy": ("https://pertpy.readthedocs.io/en/stable/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "decoupler": ("https://decoupler.readthedocs.io/en/latest/", None),
    "rmm": ("https://docs.rapids.ai/api/rmm/stable/", None),
    "statsmodels": ("https://www.statsmodels.org/stable/", None),
    "omnipath": ("https://omnipath.readthedocs.io/en/latest/", None),
    "dask": ("https://docs.dask.org/en/stable/", None),
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "._*",
    "*.ipynb_checkpoints",
    "release-notes/blank.md",
]


# -- Options for HTML output -------------------------------------------------

html_theme = "scanpydoc"
html_theme_options = {
    "repository_url": repository_url,
    "repository_branch": os.environ.get("READTHEDOCS_GIT_IDENTIFIER", "main"),
    "use_repository_button": True,
    "navigation_with_keys": False,
    "show_toc_level": 2,
}
html_show_sphinx = False
html_logo = "_static/logo_RTD.svg"
html_static_path = ["_static"]
html_css_files = ["css/override.css"]
html_title = "rapids-singlecell"

# OpenGraph metadata for social media previews
ogp_site_url = "https://rapids-singlecell.readthedocs.io/"
ogp_site_name = "rapids-singlecell"
ogp_image = "_static/logo_RTD.svg"

qualname_overrides = {
    "numpy.bool_": "numpy.bool",  # Since numpy 2, numpy.bool is the canonical dtype
}

nitpick_ignore = [
    ("py:class", "scipy.sparse.base.spmatrix"),
    ("py:meth", "pandas.DataFrame.iloc"),
    ("py:meth", "pandas.DataFrame.loc"),
    ("py:class", "pandas.core.series.Series"),
    ("py:class", "anndata._core.views.ArrayView"),
    ("py:class", "anndata._core.raw.Raw"),
    ("py:class", "scanpy._utils.Empty"),
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
    app.warningiserror = True
    app.add_config_value(
        "recommonmark_config",
        default={
            "auto_toc_tree_section": "Contents",
            "enable_auto_toc_tree": True,
            "enable_math": True,
            "enable_inline_math": False,
            "enable_eval_rst": True,
        },
        rebuild=True,
    )


# extlinks config
extlinks = {
    "issue": ("https://github.com/scverse/rapids_singlecell/issues/%s", "issue%s"),
    "pr": ("https://github.com/scverse/rapids_singlecell/pull/%s", "pr%s"),
}
