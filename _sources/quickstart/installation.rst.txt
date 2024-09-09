.. include:: ../substitutions.rst

Installation
------------

.. _installation-section:

This section provides instructions on how to install |project|. We recommend creating an installation enviroment for a clean and isolated setup. Available options for different use cases are outlined in the tabs below.

.. tab-set::

   .. tab-item:: Conda

      Conda is a powerful package manager that creates isolated environments with both Python and non-Python dependencies. Conda is a good choice for more complex setups and cross-platform compatibility.

      .. code-block:: bash

         conda create \
            --name pytme \
            -c conda-forge \
            python=3.11 \
            pyfftw \
            napari \
            magicgui \
            pyqt

   .. tab-item:: Venv

      Venv is Python's built-in virtual environment module. Venv is a good choice for simpler setups, but does not handle non-Python dependencies.

      .. code-block:: bash

         python3 -m venv pytme
         source pytme/bin/activate
         pip install pyfftw napari magicgui PyQt5

   .. tab-item:: Docker

      Docker containerizes the entire application and its dependencies, ensuring consistency across different systems. Docker is a good choice for deployment scenarios and provides the highest degree of reproducibility.

      To build the Docker image locally

      .. code-block:: bash

         docker build -t pytme -f docker/Dockerfile_GPU .

      Alternatively, you can pull the latest version from Docker Hub

      .. code-block:: bash

         docker pull dquz/pytme:latest

      .. tip::

         Latest corresponds to the current version of the main branch. You can also use a release version by specifiying the corresponding tag.

After setting up your environment, |project| can be installed from PyPi

.. code-block:: bash

   pip install pytme

Alternatively, you can install the development version with the latest changes

.. code-block:: bash

   pip install git+https://github.com/KosinskiLab/pyTME.git


.. note::

   When using Docker, |project| will already be installed in the container, so these installation steps are not necessary.


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


Troubleshooting
---------------

The following presents known issues encountered during installation and outlines solutions to them.


pyFFTW
^^^^^^

The installation of `pyFFTW <https://github.com/pyFFTW/pyFFTW>`_ via pip has been troublesome in the past. Consider using Conda for a smoother experience. Alternatively, pyFFTW can be installed from source. To compile it on my M1 MacBook running homebrew, I had to modify pyFFTW's setup.py variable ``self.library_dirs`` to include the homebrew paths in the EnvironmentSniffer class's ``__init__`` method as follows

.. code-block:: python

   self.library_dirs = get_library_dirs()
   self.library_dirs.extend(["/opt/homebrew/lib", "/opt/homebrew/opt/fftw/lib"]) # Patch


CuPy
^^^^

GPU backends often require a correct setup of CUDA libraries. CuPy expects the corresponding libraries to be in a set of standard locations and will raise Runtime/Import errors should that not be the case. Possible errors include ``RuntimeError: CuPy failed to load libnvrtc.so``, ``ImportError: libcudart.so: cannot open shared object file`` and ``cupy.cuda.compiler.CompileException``.

Solving this issue typically requires setting as set of environment variables and is outlined in the `cupy installation faq <https://docs.cupy.dev/en/stable/install.html#faq>`_.


Testing
^^^^^^^

The code of |project| is automatically tested before release. Should you run into issues that are not outlined above, you can optionally verify your local installation via the provided test suite as follows

.. code-block:: bash

   git clone https://github.com/KosinskiLab/pyTME.git
   cd pyTME
   ulimit -n 4096
   pytest

If the tests pass without any errors, |project| has been successfully installed.

.. note::

   Running the code above may fail when using Conda or Venv. A possible solution is to install |project| in editable mode

   .. code-block:: bash

      pip uninstall pytme
      pip install -e .
      python3 -m pytest tests/


Support
-------

For issues, questions, or contributions, please open an issue or pull request in the |project| `repository <https://github.com/KosinskiLab/pyTME.git>`_.
