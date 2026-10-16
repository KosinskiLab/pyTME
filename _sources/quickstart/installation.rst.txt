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


After setting up your environment, |project| can be installed from PyPi

.. code-block:: bash

   pip install -U pytme

|project| ships with a base CPU install. Additional features and accelerators are available as optional extras:

.. _gui-installation:

.. list-table:: Optional extras
   :widths: 14 60 26
   :header-rows: 1

   * - **Extra**
     - **What it adds**
     - **Install**
   * - ``gui``
     - ``napari`` gui for mask creation and template matching analysis.
     - ``pip install git+https://github.com/maurerv/napari-density-io.git 'pytme[gui]'``
   * - ``mesh``
     - Mesh handling via ``open3d``, used for constrained template matching.
     - ``pip install 'pytme[mesh]'``
   * - ``cupy``
     - GPU acceleration on NVIDIA hardware via CuPy. See the `CuPy installation guide <https://docs.cupy.dev/en/stable/install.html>`_.
     - ``pip install 'pytme[cupy]'``
   * - ``jax``
     - Fastest GPU/TPU backend for aggregation workflows. See the `JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.
     - ``pip install 'pytme[jax]'``
   * - ``pytorch``
     - PyTorch backend (CPU or GPU); general-purpose alternative to JAX or CuPy. See `PyTorch <https://pytorch.org/>`_.
     - ``pip install 'pytme[pytorch]'``

Extras can be combined, e.g. ``pip install 'pytme[gui,cupy]'`` for the GUI plus CUDA acceleration.


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
