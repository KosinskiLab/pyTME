# Python Template Matching Engine (PyTME)

[![Build Status](https://img.shields.io/github/actions/workflow/status/KosinskiLab/pyTME/main.yml?label=CI)](https://github.com/KosinskiLab/pyTME/actions)
[![PyPI](https://img.shields.io/pypi/v/pytme.svg)](https://pypi.org/project/pytme/)

**[Documentation](https://kosinskilab.github.io/pyTME/)** | **[Installation](https://kosinskilab.github.io/pyTME/quickstart/installation.html)** | **[API](https://kosinskilab.github.io/pyTME/reference/index.html)**

PyTME is a Python library for data-intensive n-dimensional template matching using CPUs and GPUs.

With its [backend-agnostic design](https://kosinskilab.github.io/pyTME/reference/backends.html), the same code can be run on diverse hardware platforms using a best-of-breed approach. The underyling abstract backend specification allows for adding new backends to benefit from gains in performance and capabilities without modifying the library's core routines. The implementation of template matching scores is modular and provides developers with a flexible framework for rapid prototyping. Furthermore, pyTME supports a unique callback capability through [analyzers](https://kosinskilab.github.io/pyTME/reference/analyzer/base.html), which allows for injection of custom code, enabling real-time processing and manipulation of results.

PyTME includes a [graphical user interface](https://kosinskilab.github.io/pyTME/quickstart/preprocessing/gui_example.html) that provides simplified mask creation, interactive filter exploration, result visualization, and manual refinement capabilities. This GUI serves as an accessible entry point to the library's core functionalities, allowing users to efficiently interact with and analyze their data.

Finally, pyTME offers specialized tools for cryogenic electron microscopy data, such as wedge masks, CTF correction, as well as [means for handling structural data](https://kosinskilab.github.io/pyTME/reference/data_structures/density.html).

Running into bugs or missing a feature? Help us improve the project by opening an [issue](https://github.com/KosinskiLab/pyTME/issues).

## Installation

We recommend installation using one of the following methods

| Method   | Command                                                 |
|----------|---------------------------------------------------------|
| PyPi     | `pip install pytme`                                     |
| Source   | `pip install git+https://github.com/KosinskiLab/pyTME`  |
| Docker   | `docker build -t pytme -f docker/Dockerfile_GPU .`      |

You can find alternative installation methods in the [documentation](https://kosinskilab.github.io/pyTME/quickstart/installation.html).


## User Guide

Learn how to get started with

- [Installation:](https://kosinskilab.github.io/pyTME/quickstart/installation.html).
- [Template matching:](https://kosinskilab.github.io/pyTME/quickstart/matching/particle_picking.html) Find your template of interest.
- [Postprocessing](https://kosinskilab.github.io/pyTME/quickstart/postprocessing/motivation.html) Analyze template matching results and downstream integrations.

## How to Cite

If you used PyTME in your research, please cite the corresponding [publication](https://www.sciencedirect.com/science/article/pii/S2352711024000074).

```bibtex
@article{Maurer:2024aa,
    author = {Maurer, Valentin J. and Siggel, Marc and Kosinski, Jan},
    journal = {SoftwareX},
    pages = {101636},
    title = {PyTME (Python Template Matching Engine): A fast, flexible, and multi-purpose template matching library for cryogenic electron microscopy data},
    volume = {25},
    year = {2024}
}
```
