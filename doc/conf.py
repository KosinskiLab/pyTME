import os
import sys

sys.path.insert(0, os.path.abspath("../tme"))
sys.path.insert(1, os.path.abspath("../scripts"))

from tme import __version__

project = "pytme"
copyright = "2023-2024, European Molecular Biology Laboratory"
author = "Valentin J. Maurer"
release = __version__

language = "en"

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
    "sphinx.ext.intersphinx",
    "matplotlib.sphinxext.plot_directive",
]

copybutton_prompt_text = ">>> "
copybutton_prompt_is_regexp = False

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
add_module_names = False

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

autodoc_default_options = {
    "show-inheritance": True,
    "inherited_members": True,
}

autodoc_inherit_docstrings = True
autodoc_typehints_format = "short"
autodoc_typehints = "none"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
html_css_files = [
    "css/general.css",
]
html_context = {
    "github_user": "maurerv",
    "github_repo": "https://github.com/KosinskiLab/pyTME",
    "github_version": "master",
    "doc_path": "docs",
}

html_theme_options = {
    # "logo": {
    #     "text": "pytme",
    #     "image_light": "_static/index_api.svg",
    #     "image_dark": "_static/index_api.svg",
    # },
    "header_links_before_dropdown": 4,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/KosinskiLab/pyTME",
            "icon": "fa-brands fa-github",
        },
    ],
    "use_edit_page_button": False,
    "navigation_depth": 3,
    "show_toc_level": 1,
}

intersphinx_mapping = {
    "numpy": ("https://numpy.org/doc/stable/", None),
    "skimage": ("https://scikit-image.org/docs/stable/", None),
    "scipy": ("http://docs.scipy.org/doc/scipy/", None),
}

plot_html_show_source_link = True
plot_html_show_formats = True

plot_rcparams = {
    "figure.facecolor": "none",
    "axes.facecolor": "none",
    "savefig.transparent": True,
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.left": False,
    "axes.spines.bottom": False,
    "axes.spines.right": False,
    "figure.autolayout": True,
    "xtick.bottom": False,
    "xtick.labelbottom": False,
    "ytick.left": False,
    "ytick.labelleft": False,
}
plot_apply_rcparams = True
