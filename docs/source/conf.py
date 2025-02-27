# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

project = "xradio"
copyright = "Associated Universities, Inc."
author = " "

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "xradio.sphinx",
    "nbsphinx",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.coverage",
    "sphinx.ext.githubpages",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = []

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
sys.path.insert(0, os.path.abspath("."))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# import sphinx_adc_theme
import sphinx_rtd_theme

# html_theme = "sphinx_adc_theme"
# html_theme_path = [sphinx_adc_theme.get_html_theme_path()]
html_theme = "sphinx_rtd_theme"
# html_theme_path = [sphinx_rtd_theme.get_html_theme_path()] # depricated
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

html_logo = "_images/xradio_logo.png"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
html_theme_options = {
    "collapse_navigation": True,
    "navigation_depth": 3,
    "style_nav_header_background": "white",
    "logo_only": True,
}

autodoc_class_signature = "mixed"

autodoc_type_aliases = {
    "Time": "xradio.vis.model.Time",
    "BaselineId": "BaselineId",
    "Channel": "Channel",
    "Polarization": "Polarization",
    "UvwLabel": "UvwLabel",
    "Data": "Data",
    "Attr": "Attr",
    "Dataof": "Dataof",
    "Attrof": "Attrof",
}

# nitpicky = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "astropy": ("https://docs.astropy.org/en/stable", None),
    "xarray": ("https://docs.xarray.dev/en/stable", None),
    "dask": ("https://docs.dask.org/en/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

add_module_names = False

nbsphinx_allow_errors = True

# Enable syntax highlighting
pygments_style = "sphinx"
