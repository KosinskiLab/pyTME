.. include:: ../substitutions.rst

Installation
------------

This section provides instructions on how to install the |project| library. The available options are outlined in the tabs below.

.. _installation-section:

.. tab-set::

   .. tab-item:: Conda

      Conda provides an isolated environment, helps avoid conflicts with other packages, and installs necessary dependencies.

      .. code-block:: bash

         conda create \
            --name pytme \
            -c conda-forge \
            python=3.11 \
            pyfftw \
            napari \
            magicgui \
            pyqt

      The following will download and install the newest version of |project| into the Conda environment..

      .. code-block:: bash

         conda activate pytme
         pip install git+https://github.com/KosinskiLab/pyTME.git


   .. tab-item:: PyPI

      Pip will fetch the required packages from PyPI and install them on your system.

      .. code-block:: bash

         pip install pytme

      .. note::

         The Python Package Index (PyPI) provides a simple and convenient way to install |project|. However, installing from Conda or source fetches the most recent version of the code base.


   .. tab-item:: Source

      Installing from Source provides you with the latest unreleased changes.

      .. code-block:: bash

            pip install git+https://github.com/KosinskiLab/pyTME.git

   .. tab-item:: Docker

      Docker provides a consistent and isolated environment for running |project|.

      To build the Docker image locally

      .. code-block:: bash

         docker build -t pytme -f docker/Dockerfile_GPU .

      Alternatively, you can pull the latest version from Docker Hub

      .. code-block:: bash

         docker pull dquz/pytme:latest

      .. note::

         Latest corresponds to the current version of the main branch. You can also use a release version by specifiying the corresponding tag.


CPU/GPU/TPU Support
-------------------

|project|'s `backend agnostic design <https://kosinskilab.github.io/pyTME/reference/backends.html>`_ enables the same code to be run on practically any hardware platform using a best-of-breed approach. To enable |project| to utilize compute devices other than CPUs, install one of the libraries below. |project| defaults to CuPy for GPU applications, but expert users can decide freely between their desired backend.

.. tab-set::

   .. tab-item:: CuPy (Recommended)

      The following will install the CuPy dependencies of |project|

      .. code-block:: bash

         pip install "pytme[cupy]"

      Alternatively, you can install CuPy directly

      .. code-block:: bash

         pip install cupy-cuda12x

      If your CUDA version is lower than 12 or you encounter any issues, please refer to CuPy's official `installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for a version tailored to your system and detailed instructions.

   .. tab-item:: PyTorch

      The following will install the PyTorch dependencies of |project|

      .. code-block:: bash

         pip install "pytme[torch]"

      Alternatively, you can install PyTorch directly

      .. code-block:: bash

         pip install torch torchvision

      PyTorch's installation might vary based on your system and the specific GPU in use. Consult the official `PyTorch website <https://pytorch.org/>`_ for detailed installation options tailored for your environment.


   .. tab-item:: JAX

      The following will install the JAX dependencies of |project|

      .. code-block:: bash

         pip install "pytme[jax]"

      Alternatively, you can install JAX directly

      .. code-block:: bash

         pip install "jax[cuda12]"

      Setting up JAX might require additional attention on certain platforms. Consult the `JAX documentation <https://jax.readthedocs.io/en/latest/installation.html>`_ for tailored options.

   .. tab-item:: MLX

      The following will install the MLX dependencies of |project|

      .. code-block:: bash

         pip install mlx

      The MLX library is only available for Apple silicon chips.

.. _gui-installation:


GUI Setup
---------

If you would like to perform interactive preprocessing and analysis of your data using the GUI shipped with |project|, you need to install the following dependency

.. code-block:: bash

   pip install git+https://github.com/maurerv/napari-density-io.git

If you have installed |project| using Conda, thats it. Otherwise, you have to install the remaining dependencies

.. code-block:: bash

   pip install napari magicgui PyQt5



Testing the Installation
------------------------

To verify that |project| has been installed correctly, you can run the test suite provided with the project as follows:

.. code-block:: bash

   git clone https://github.com/KosinskiLab/pyTME.git
   cd pyTME
   ulimit -n 4096
   pytest

If the tests pass without any errors, |project| has been successfully installed.


Troubleshooting
---------------

The installation of `pyFFTW <https://github.com/pyFFTW/pyFFTW>`_ via pip has been troublesome in the past. Consider using the :ref:`installation method <installation-section>` Conda for a smoother experience. Alternatively, pyFFTW can be installed from source. To compile it on my M1 MacBook running homebrew, I had to modify pyFFTW's setup.py variable self.library_dirs to include the homebrew paths in the EnvironmentSniffer class's __init__ method as follows:

.. code-block:: python

   self.library_dirs = get_library_dirs()
   self.library_dirs.extend(["/opt/homebrew/lib", "/opt/homebrew/opt/fftw/lib"]) # Patch


Support
-------

For issues, questions, or contributions, please open an issue or pull request in the |project| `repository <https://github.com/KosinskiLab/pyTME.git>`_.
