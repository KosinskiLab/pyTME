.. include:: ../substitutions.rst

Installation
============

This section provides instructions on how to install the |project| library. Depending on your preferred method or system setup, you can choose between the installation with Conda, Pip, or from source. Click on the tabs below to view the instructions for each method.

.. _installation-section:

.. tabs::

   .. tab:: Conda

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

         .. code-block:: bash

            pip install git+https://github.com/KosinskiLab/pyTME.git

   .. tab:: Pip

      Pip will fetch the required packages from an online repository and install them on your system.

      **Prerequisite**:

      Ensure you have Python 3.11 or higher installed on your system:

      .. code-block:: bash

         python --version

      1. **Install from Source**:

         .. code-block:: bash

            pip install git+https://github.com/KosinskiLab/pyTME.git

      2. **Install from PyPI**:

         Once available on PyPI, the installation is a single command.

         .. code-block:: bash

            pip install pytme

      .. note::

         The Python Package Index (PyPI) provides a simple and convenient way to install |project|. However, installing from github fetches the most recent version of the code base.


   .. tab:: Source

      Installing from source ensures you get the latest, unreleased changes. You'll be compiling and setting up the project directly from its source code.

      **Prerequisite**:

      Ensure you have Python 3.11 or higher installed on your system:

      .. code-block:: bash

         python --version

      1. **Clone the Repository**:

         First, clone the |project| repository:

         .. code-block:: bash

            git clone https://github.com/KosinskiLab/pyTME.git

         Navigate to the cloned repository's directory:

         .. code-block:: bash

            cd pytme

      2. **Install the Package**:

         Use `pip` to install the library:

         .. code-block:: bash

            pip install .

         This will automatically install all required dependencies for |project|.

      .. note::

         After installing from source, navigate out of the source directory before using |project|. This ensures the built extensions are properly loaded from the installed library, avoiding potential issues.


Optional GUI Setup
------------------

To utilize the optional preprocessing GUI provided by |project|, you'll need to install several additional librarys:

.. code-block:: bash

   pip install napari magicgui pyqt5
   pip install git+https://github.com/maurerv/napari-density-io.git


GPU Support
-----------

To enable GPU support in |project|, install one of the following GPU-accelerated libraries: PyTorch or CuPy. While both are supported, CuPy is the recommended choice.

**Check Your CUDA Version**:

Use the `nvidia-smi` command to get information about your NVIDIA driver and GPU:

.. code-block:: bash

   nvidia-smi

Look for the CUDA version in the output; it's usually displayed in the top right corner.

After determining your CUDA version, you can proceed to install one of the following GPU-accelerated libraries: PyTorch or CuPy. While both are supported, CuPy is the recommended choice.


.. tabs::

   .. tab:: CuPy (Recommended)

      If you have CUDA version 12 or upwards, install CuPy using:

      .. code-block:: bash

         pip install cupy-cuda12x

      If your CUDA version is different or you encounter any issues, please refer to CuPy's official `installation guide <https://docs.cupy.dev/en/stable/install.html>`_ for a version tailored to your system and detailed instructions.

   .. tab:: PyTorch

      Alternatively, you can use PyTorch for GPU computations:

      .. code-block:: bash

         pip install torch torchvision

      PyTorch's installation might vary based on your system and the specific GPU in use. Consult the official `PyTorch website <https://pytorch.org/>`_ for detailed installation options tailored for your environment.



Testing the Installation
------------------------

To verify that |project| has been installed correctly, you can run the test suite provided with the project:

.. code-block:: bash

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

For issues, questions, or contributions, please refer to the |project| repository `repository <https://github.com/KosinskiLab/pyTME.git>`_.
