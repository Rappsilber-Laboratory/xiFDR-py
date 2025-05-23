from setuptools_scm import get_version

release = get_version(root='..', relative_to=__file__)
version = ".".join(release.split(".")[:2])

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'xiFDR'
copyright = '2025, Rappsilber Laboratory'
author = 'Rappsilber Laboratory'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
]

typehints_fully_qualified = True
autodoc_type_aliases = {
    "polars.dataframe.frame.DataFrame": "pl.DataFrame",
    "polars.lazyframe.frame.LazyFrame": "pl.LazyFrame",
    "pandas.core.frame.DataFrame": "pl.DataFrame",
}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "polars": ("https://pola-rs.github.io/polars/py-polars/html", None),  # works if the Polars docs are built with intersphinx
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
