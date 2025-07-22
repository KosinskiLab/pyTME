.. _cli-match_template:

==============
Moving Forward
==============

Having worked through the examples, you are now ready to analyze your own data. This page summarizes the previous sections and lists further things to watch out for.

Summary
-------

``match_template.py`` facilitates template matching using various scoring metrics.

.. code-block:: bash

    match_template.py --help

``preprocess.py`` enables template generation, masks can be created interactively using ``preprocessor_gui.py`` or the API.


.. _scoring-table:

Scoring Functions
-----------------

Different scoring functions are optimized for specific mask geometries and data conditions. You can instruct ``match_template.py`` to use a suitable scoring function from the table below using the ``--score`` flag.

.. list-table:: Scoring Function Selection
   :widths: 25 30 25 10 10
   :header-rows: 1

   * - **Score**
     - **Best For**
     - **Mask Type**
     - **Speed**
     - **Application**
   * - ``flcSphericalMask`` (default)
     - Ribosomes, virus particles, globular proteins
     - Spherical/elliptical
     - Fast
     - Particle picking
   * - ``FLC``
     - Membrane proteins, filaments, asymmetric structures
     - Cylindrical, tube, custom shapes
     - Medium
     - Particle picking
   * - ``CC``
     - Quick screening
     - None required
     - Fast
     - EM map fitting
   * - ``LCC``
     - Edge-enhanced correlation, High S/N
     - None required
     - Fast
     - EM map fitting
   * - ``CORR``
     - General normalized correlation
     - Spherical/elliptical
     - Fast
     - EM map fitting
   * - ``MCC``
     - Partial volume matching
     - Any (requires target mask)
     - Slow
     - EM map fitting

.. warning::

    Match your scoring function to your mask geometry to avoid artifacts.


Compute Backend
---------------

Depending on available hardware and use case, there might be a more ideal compute backend for template matching. You can select the computational backend using ``--backend`` in ``match_template.py``.

.. list-table:: Comparison of supported backends
   :widths: 20 16 16 16 16
   :header-rows: 1

   * - ****
     - **numpyfftw**
     - **cupy**
     - **pytorch**
     - **jax**
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
