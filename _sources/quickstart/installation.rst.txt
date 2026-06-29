.. include:: ../substitutions.rst

Installation
------------

.. _installation-section:

We recommend creating a virtual environment for a clean and isolated setup.

.. tab-set::

   .. tab-item:: Venv

      Python's built-in option, suitable for simpler setups

      .. code-block:: bash

         python3 -m venv pytme
         source pytme/bin/activate

   .. tab-item:: Conda

      Best for cross-platform compatibility and managing complex dependencies

      .. code-block:: bash

         conda create \
            --name pytme \
            -c conda-forge \
            python=3.11 \
            pyfftw \
            pyqt

   .. tab-item:: Docker

      Docker is a good choice for deployment scenarios and provides the highest degree of reproducibility

      To build the Docker image

      .. code-block:: bash

         docker build -t pytme -f docker/Dockerfile_GPU .

      Alternatively, you can pull an image from Docker Hub

      .. code-block:: bash

         docker pull dquz/pytme:latest

      .. tip::

         Latest corresponds to the current version of the main branch. Releases are tagged accordingly.


After setting up your environment, |project| and GUI dependencies can be installed from PyPi

.. _gui-installation:

.. code-block:: bash

   pip install -U \
      pytme \
      "napari==0.4.19.post1" \
      magicgui \
      git+https://github.com/maurerv/napari-density-io.git


If you would like to run |project| with GPU acceleration, install one of the options below

.. tab-set::

   .. tab-item:: CuPy (Recommended)

      .. code-block:: bash

         pip install "pytme[cupy]"

      See the `CuPy documentation <https://docs.cupy.dev/en/stable/install.html>`_ for system-specific installation instructions.

   .. tab-item:: JAX (Fastest)

      .. code-block:: bash

         pip install "pytme[jax]"

      See the `JAX documentation <https://jax.readthedocs.io/en/latest/installation.html>`_ for system-specific installation instructions.

   .. tab-item:: PyTorch

      .. code-block:: bash

         pip install "pytme[torch]"

      See the `PyTorch website <https://pytorch.org/>`_ for system-specific installation instructions.

   .. tab-item:: MLX

      .. code-block:: bash

         pip install "pytme[mlx]"

      MLX is only available for Apple Silicon chips. See the `MLX documentation <https://ml-explore.github.io/mlx/build/html/install.html>`_ for installation instructions.


Troubleshooting
---------------

The following outlines issues encountered during installation and solutions to them.


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


Testing (For Developers)
^^^^^^^^^^^^^^^^^^^^^^^^

The code of |project| is automatically tested before release. If you are contributing to |project| or experiencing issues, you can verify your local installation via the test suite as follows

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
