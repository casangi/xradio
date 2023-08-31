# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import sys
import os

project = 'xradio'
copyright = '2023, Jan-Willem Steeb'
author = 'Jan-Willem Steeb'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosectionlabel'
]

templates_path = ['_templates']
exclude_patterns = []

sys.path.insert(0, os.path.abspath(os.path.join("..", "..", "src")))

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

autodoc_class_signature = "mixed"
html_theme = 'groundwork'
autodoc_type_aliases = {
    'Time': 'xradio.vis.model.Time',
    'BaselineId': 'BaselineId',
    'Channel': 'Channel',
    'Polarization': 'Polarization',
    'UvwLabel': 'UvwLabel',
    'Data': 'Data',
    'Attr': 'Attr',
    'Dataof': 'Dataof',
    'Attrof': 'Attrof',
}

#nitpicky = True
 
