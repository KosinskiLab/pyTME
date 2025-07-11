.. include:: ../substitutions.rst

========
Backends
========

.. currentmodule:: tme.backends

Backends allows users to effortlessly switch between different array and FFT implementations and to define custom ones. Backends enable the same code to be run on CPU or GPU, and to quickly integrate novel solutions and benefit from their performance gains. Having backends makes |project| flexible and highly adaptive to available compute infrastructure.

.. note::

   Currently flexible backends are only implemented for exhaustive template matching operations. Keep an eye on the documentation for extension to other parts.


.. list-table:: Comparison of supported backends
   :widths: 20 16 16 16 16
   :header-rows: 1

   * - Property
     - NumPy FFTW
     - CuPy
     - PyTorch
     - JAX
   * - **Hardware**
     - CPU
     - GPU (CUDA)
     - CPU/GPU
     - CPU/GPU/TPU
   * - **Performance**
     - Fastest CPU
     - Good GPU allrounder
     - Best for peak calling
     - Fastest GPU (aggregation)
   * - **Analyzer Support**
     - All
     - All
     - All
     - MaxScoreOverRotations only
   * - **When to Use**
     - CPU-only systems
     - New users with NVIDIA GPU
     - Peak calling workflows
     - Maximum performance


Backend Manager
~~~~~~~~~~~~~~~

Modules that support flexible backends import a global instance of :py:class:`BackendManager <tme.backends.BackendManager>` like so: :code:`from tme.backends import backend`. The import is quasi-equivalent to :code:`import numpy as np`, apart from the difference in naming. Notably, backend agnostic code is written without instance specific methods such as the ones defined by ``np.ndarray``. Instance-specific functions are wrapped instead, e.g., :py:meth:`NumpyFFTWBackend.fill`. For additional information please refer to the class documentation below.

.. autosummary::
   :toctree: api/

   BackendManager


Supported Backends
~~~~~~~~~~~~~~~~~~

|project| implements the following backends, each offering unique advantages for different computational environments. Users can easily switch between these backends to optimize performance based on their available hardware and specific use cases. For information on implementing custom backends to suit specialized needs, refer to section below.

.. autosummary::
   :toctree: api/

   NumpyFFTWBackend
   PytorchBackend
   CupyBackend
   MLXBackend
   JaxBackend


Abstract Base Backend
~~~~~~~~~~~~~~~~~~~~~

:py:class:`MatchingBackend <tme.backends.MatchingBackend>` serves as specification for new backends. Generally the aim is to create a structure that is syntactically similar to numpy.

.. autosummary::
   :toctree: api/

   MatchingBackend

Schematically, each backend requires an array implementation, such as the ones provided by ``numpy``, ``pytorch`` or ``mars``. This backend should provide array data structures and define operations on it. Since the syntax of many backends is already similar to numpy, wrapping them is fairly straightfoward. In addition to array operations, each backend is requried to define a range of template matching specific operations.

With array and template matching specific operations defined, practically any backend can be used to perform template matching with |project|. Below is an overview of methods that are required from each backend.


Array operations
^^^^^^^^^^^^^^^^

.. autosummary::

   MatchingBackend.add
   MatchingBackend.subtract
   MatchingBackend.multiply
   MatchingBackend.divide
   MatchingBackend.mod
   MatchingBackend.sum
   MatchingBackend.einsum
   MatchingBackend.mean
   MatchingBackend.std
   MatchingBackend.max
   MatchingBackend.min
   MatchingBackend.maximum
   MatchingBackend.minimum
   MatchingBackend.sqrt
   MatchingBackend.square
   MatchingBackend.abs
   MatchingBackend.transpose
   MatchingBackend.tobytes
   MatchingBackend.size
   MatchingBackend.fill
   MatchingBackend.clip
   MatchingBackend.roll
   MatchingBackend.stack
   MatchingBackend.indices
   MatchingBackend.power
   MatchingBackend.astype
   MatchingBackend.concatenate
   MatchingBackend.repeat
   MatchingBackend.unique
   MatchingBackend.topk_indices
   MatchingBackend.argsort
   MatchingBackend.unravel_index
   MatchingBackend.tril_indices


Array initialization
^^^^^^^^^^^^^^^^^^^^

.. autosummary::

   MatchingBackend.full
   MatchingBackend.zeros
   MatchingBackend.arange


Template matching
^^^^^^^^^^^^^^^^^

.. autosummary::

   MatchingBackend.build_fft
   MatchingBackend.topleft_pad
   MatchingBackend.to_sharedarr
   MatchingBackend.from_sharedarr
   MatchingBackend.extract_center
   MatchingBackend.rigid_transform
   MatchingBackend.max_filter_coordinates
   MatchingBackend.compute_convolution_shapes


Conversion
^^^^^^^^^^

.. autosummary::

   MatchingBackend.to_cpu_array
   MatchingBackend.to_numpy_array
   MatchingBackend.to_backend_array


Auxiliary
^^^^^^^^^^

.. autosummary::

   MatchingBackend.eps
   MatchingBackend.free_cache
   MatchingBackend.datatype_bytes
   MatchingBackend.get_available_memory

.. note::

   Notably, `autoray <https://github.com/jcmgray/autoray>`_ provides a framework that allows for writing backend agnostic python code, similar to the approach taken here. In the future, ``tme.backends`` might be deprecated in favor of ``autoray``.
