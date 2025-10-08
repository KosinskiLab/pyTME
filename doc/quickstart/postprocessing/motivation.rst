.. include:: ../../substitutions.rst

======
Basics
======

Template matching produces similarity estimates (scores) between the target and a set of rotations of the template. Local maxima in those scores correspond to putative occurences of the template in the target. Identifying such local maxima is challenging because of

- **False positives**: Other cellular features or sample preparation
- **Wide peaks**: High scores spread over multiple voxels around each particle
- **Edge artifacts**: Inflated scores near tomogram boundaries
- **Variable backgrounds**: Different score distributions across the tomogram


Peak Calling Strategies
-----------------------

|project| implements different peak calling algorithms for specific scenarios.

.. list-table:: Peak Calling Algorithms
   :widths: 25 35 40
   :header-rows: 1

   * - **Algorithm**
     - **Best For**
     - **Key Features**
   * - ``PeakCallerMaximumFilter`` (default)
     - General use
     - Fast, reliable local maxima detection
   * - ``PeakCallerRecursiveMasking``
     - Crowded environments
     - Uses template mask to restrict potential matches
   * - ``PeakCallerScipy``
     - Well-separated broad peaks
     - Robust, but may miss overlapping peaks

**Key Parameters**

- ``--min-distance``: Minimum separation between peaks in voxels (prevents multiple picks per particle)
- ``--mask-edges``: Automatically exclude tomogram boundaries
- ``--min-boundary-distance``: Minimum distance from tomogram boundaries in voxels.
- ``--num-peaks``: Maximum number of peaks to identify


Number of Peaks
---------------

The number of peaks is generally unknown, but we can determine suitable score cutoffs based on statistical properties of the cross-correlation. This approach automatically sets thresholds to limit false positives

.. code-block:: bash

   postprocess.py \
       --input-file results.pickle \
       --n-false-positives 5 \
       --output-format orientations

Alternatively, score cutoffs can be set using ``--min-score`` and ``--max-score``. The maximum number of peaks is always given by ``--num-peaks``.


Background Correction
---------------------

Cellular environments contain complex backgrounds that can yield high scores for regions that do not represent the template. We can avoid picking such regions by background correction. Conceptually, we account for that by computing template matching scores between the target and a different template, to exclude regions where both the template of interest and alternative template score highly.

Single Background
^^^^^^^^^^^^^^^^^

In the simplest case, a control template (for instance from running ``match_template.py`` with ``--scramble-phases``) can be used to model the score background

.. code-block:: bash

   postprocess.py \
       --input-file signal.pickle \
       --background-file noise.pickle \
       --output-format orientations


Multiple Background
^^^^^^^^^^^^^^^^^^^

We can also account for different cellular components simultaneously

.. code-block:: bash

   postprocess.py \
       --input-file ribosome.pickle \
       --background-file noise.pickle membrane.pickle \
       --output-format orientations


Multiple Entities
^^^^^^^^^^^^^^^^^

When analyzing multiple templates simultaneously, we can distinguish between different macromolecular species

.. code-block:: bash

   postprocess.py \
       --input-file ribosome.pickle proteasome.pickle \
       --background-file noise.pickle \
       --output-format orientations

In all cases, the tool will report statistics for foreground, background, and normalized scores

.. code-block:: text

   > Foreground mean 0.125, std 0.087, max 0.445
   > Background mean 0.089, std 0.023, max 0.234
   > Normalized mean 0.067, std 0.078, max 0.298

Since the background of the individual entities may differ, we can also compare SNR-like cross-correlations instead, using ``--snr``. This is also useful when comparing the scores across an entire dataset.


Local Optimization and Refinement
---------------------------------

For high-precision applications (e.g., fitting atomic structures), |project| offers local optimization

.. code-block:: bash

   postprocess.py \
       --input-file results.pickle \
       --local-optimization \
       --peak-oversampling 2 \
       --output-format alignment

**Peak Oversampling**
Achieves sub-voxel precision by interpolating score maxima. Factor of 2 provides half-voxel precision.

**Local Optimization**
Uses basin-hopping optimization to refine translation and rotation parameters around initial peaks. Most useful when analyzing small numbers of high-quality candidates.

.. _coordinate-system:

Coordinate System
-----------------

Our convention follows the schematics outlined in [1]_. We use a right-handed coordinate system with orthogonal X, Y and Z axes. Euler angles are expressed using intrinsic ZYZ convention, with the first rotation around the Z-axis, the second around the new Y-axis and the third around the new Z-axis (see :py:meth:`euler_to_rotationmatrix <tme.rotations.euler_to_rotationmatrix>`). The default orientation is the z-unit vector (0, 0, 1).

Details for Developers
----------------------

The output of ``match_template.py`` is a `pickle <https://docs.python.org/3/library/pickle.html>`_ file. All but the last element will correspond to the return value of a given :doc:`analyzer </reference/analyzer/base>`'s merge method. The file can be read using :py:meth:`load_pickle <tme.matching_utils.load_pickle>`. For the default analyzer :py:class:`MaxScoreOverRotations <tme.analyzer.MaxScoreOverRotations>` the pickle file contains

- **Scores**: An array with scores mapped to translations.
- **Offset**: Offset informing about shifts in coordinate sytems.
- **Rotations**: An array of optimal rotation indices for each translation.
- **Rotation Dictionary**: Mapping of rotation indices to rotation matrices.
- **Sum of Squares**: Sum of squares of scores for statistics.
- **Metadata**: Coordinate system information and parameters for reproducibility.

However, when you use the `-p` flag the output structure differs

- **Translations**: A numpy array containing translations of peaks.
- **Rotations**: A numpy array containing rotations of peaks.
- **Scores**: Score of each peak.
- **Details**: Additional information regarding each peak.
- **Metadata**: Coordinate system information and parameters for reproducibility.


References
----------

.. [1] Heymann, J.B.; Chagoyen, M.; Belnap, D.M. Common conventions for interchange and archiving of three-dimensional electron microscopy information in structural biology. J Struct Biol 2005, 151, 196-207.