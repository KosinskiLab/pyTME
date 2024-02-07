.. include:: ../substitutions.rst

Installation
============

This section provides instructions on how to install the |project| library. Depending on your preferred method or system setup, you can choose between the installation with Conda, PyPI, or from source. Click on the tabs below to view the instructions for each method.

.. _installation-section:

.. tab-set::

   .. tab-item:: Conda

      Conda provides an isolated environment, helps avoid conflicts with other packages, and installs necessary dependencies.

      1. **Create a New Conda Environment and Install Dependencies**:

         .. code-block:: bash

            conda create \
               --name pytme \
               -c conda-forge \
               python=3.11 \
               pyfftw \
               napari \
               magicgui \
               pyqt

      2. **Activate the Environment**:

         .. code-block:: bash

            conda activate pytme

      3. **Install from Source**:

         The following will download and install the newest version of |project|.

         .. code-block:: bash

            pip install git+https://github.com/KosinskiLab/pyTME.git

   .. tab-item:: PyPI

      Pip will fetch the required packages from PyPI and install them on your system.

      1. **Prerequisite**:

      Ensure you have Python 3.11 or higher installed on your system:

      .. code-block:: bash

         python --version

      2. **Install from PyPI**:

         Once available on PyPI, the installation is a single command.

         .. code-block:: bash

            pip install pytme

      .. note::

         The Python Package Index (PyPI) provides a simple and convenient way to install |project|. However, installing from Conda or source fetches the most recent version of the code base.


   .. tab-item:: Source

      Installing from source is ideal for developers and contributors who need access to the latest, unreleased changes and plan to modify the code. This method involves cloning the repository and setting up a local development environment.

      For detailed information on contributing to the project, see our :doc:`contribution`.

      1. **Prerequisite**:

         Ensure you have Python 3.11 or higher and Git installed on your system:

         .. code-block:: bash

            python --version
            git --version

      2. **Clone the Repository**:

         First, clone the |project| repository and navigate into it:

         .. code-block:: bash

            git clone https://github.com/KosinskiLab/pyTME.git
            cd pyTME

      3. **Set Up a Development Environment**:

         It's recommended to create a virtual environment:

         .. code-block:: bash

            python -m venv pytme
            source pytme/bin/activate

      4. **Install the Package in Editable Mode**:

         Install the library in editable mode with `pip`. This allows you to modify the source code and see changes immediately:

         .. code-block:: bash

            pip install -e .

         This will automatically install all required dependencies for |project|. Remember to navigate out of the source directory before using |project|. This ensures the built extensions are properly loaded from the installed library, avoiding potential issues.

      .. note::

         The above method is suited for development purposes. If you simply want to install the latest version of the package without cloning, you can use:

         .. code-block:: bash

            pip install git+https://github.com/KosinskiLab/pyTME.git

         This method installs the package directly from the repository without the need for cloning and is suitable for users who don't plan to modify the source code.


.. _gui-installation:

Optional GUI Setup
------------------

If you installed |project| using Conda, you will only need to execute the following to use the preprocessing GUI:

.. code-block:: bash

   pip install git+https://github.com/maurerv/napari-density-io.git

Otherwise, you also have to install the remaining GUI dependencies:

.. code-block:: bash

   pip install napari magicgui PyQt5


GPU Support
-----------

To enable GPU support in |project|, install one of the following GPU-accelerated libraries: PyTorch or CuPy. While both are supported, CuPy is the recommended choice.

**Check Your CUDA Version**:

Use the `nvidia-smi` command to get information about your NVIDIA driver and GPU:

.. code-block:: bash

   nvidia-smi

Look for the CUDA version in the output; it's usually displayed in the top right corner. After determining your CUDA version, you can proceed to install one of the supported GPU-accelerated libraries.

.. tab-set::

   .. tab-item:: CuPy (Recommended)

      If you have CUDA version 12 or upwards, install CuPy using:

      .. code-block:: bash

         pip install cupy-cuda12x

      If your CUDA version is different or you encounter any issues, please refer to CuPy's official `installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for a version tailored to your system and detailed instructions.

   .. tab-item:: PyTorch

      Alternatively, you can use PyTorch for GPU computations:

      .. code-block:: bash

         pip install torch torchvision

      PyTorch's installation might vary based on your system and the specific GPU in use. Consult the official `PyTorch website <https://pytorch.org/>`_ for detailed installation options tailored for your environment.



Testing the Installation
------------------------

To verify that |project| has been installed correctly, you can run the test suite provided with the project as follows:

.. code-block:: bash

   git clone https://github.com/KosinskiLab/pyTME.git
   cd pytme
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
