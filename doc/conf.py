import os
import sys

sys.path.insert(0, os.path.abspath("../tme"))
sys.path.insert(1, os.path.abspath("../scripts"))

project = "pytme"
copyright = "2023 European Molecular Biology Laboratory"
author = "Valentin Maurer"
release = "0.1"

extensions = [
    "sphinx.ext.autodoc",
    "numpydoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx_tabs.tabs",
]
copybutton_prompt_text = ">>> "
copybutton_prompt_is_regexp = False

autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True
add_module_names = False

numpydoc_show_class_members = False
numpydoc_show_inherited_class_members = False

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
