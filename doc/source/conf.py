# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

project = "xradio"
copyright = "2023, Jan-Willem Steeb"
author = "Jan-Willem Steeb"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "xradio_sphinx",
    "nbsphinx",
]

templates_path = ["_templates"]
exclude_patterns = []

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))
sys.path.insert(0, os.path.abspath("."))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

import sphinx_adc_theme

html_theme = "sphinx_adc_theme"
html_theme_path = [sphinx_adc_theme.get_html_theme_path()]
html_static_path = ["_static"]
html_css_files = [
    "custom.css",
]

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
