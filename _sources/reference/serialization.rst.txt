.. include:: ../substitutions.rst

Serialization
=============

.. currentmodule:: tme.utils.serialization

|project| provides utilities for serializing and deserializing template matching results using pickle and HDF5 formats. The serialization module offers a unified interface that automatically handles format selection based on file extensions, supports compression, and efficiently manages numpy memmaps.

.. code-block:: python

    import numpy as np
    from tme.serialization import serialize, deserialize

    # Serialize data to pickle format
    data = [np.random.rand(100, 100), {"scores": [0.9, 0.8, 0.7]}]
    serialize(data, "results.pickle")

    # Deserialize from pickle
    loaded_data = deserialize("results.pickle")

    # Serialize to pickle with compression
    serialize(data, "results.pickle.gz")

    # Serialize to HDF5 with compression (default is lzf)
    serialize(data, "results.h5", compression="lzf")

    # Lazy loading for HDF5 (returns HDF5Loader instance)
    loader = deserialize("results.h5")
    first_item = loader[0]


Both formats are accessible by changing the extension of ``--output-file`` in ``match_template``.

Which format is better depends on your workflow priorities. If you want to explore different parameters in `postprocess`, use HDF5 as its lazy loading capabilities speeds up postprocessing. Use pickle or gzipped pickles for maximum compression and smaller file sizes.


Core Functions
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../api/
    :nosignatures:

    serialize
    deserialize

HDF5 Utilities
~~~~~~~~~~~~~~

.. autosummary::
    :toctree: ../api/
    :nosignatures:

    HDF5Loader