.. include:: ../../substitutions.rst

Exhaustive
==========

.. currentmodule:: tme.matching_exhaustive

Exhaustive template matching evaluates similarity along a provided set of rotations and all possible translations are sampled using Fast Fourier Transform (FFT) operations. Therefore, the algorithm is guaranteed to evaluate a provided set of configurations and to find a global optimum with a sufficiently high angular sampling rate.

Within |project|, exhaustive template matching is modularized into three primary stages: setup, scoring, and callback.

1. **Setup:**
   Creates a context where template matching is poised to be carried out efficiently. :ref:`Setup functions <setup-functions>` involve data preparation, configuring FFT operations and pre-computing shared parameters.

2. **Scoring:**
   Sample :ref:`scoring function <scoring-functions>` across translational and rotational degrees of freedom.

3. **Callback:**
   Custom on the fly processing of template matching results using :doc:`analyzers <../analyzer/base>`.

This concept is embodied in the :py:class:`scan <tme.matching_exhaustive.scan>` and :py:class:`scan_subsets <tme.matching_exhaustive.scan_subsets>` methods below.

If you wish to integrate custom template matching methods into |project|, please refer to the :ref:`custom-methods` section.


Methods
~~~~~~~

:py:class:`scan_subsets <tme.matching_exhaustive.scan_subsets>` orchestrates the matching process, supporting parallel processing and post-scoring operations. :py:class:`scan_subsets <tme.matching_exhaustive.scan_subsets>` is a wrapper around :py:class:`scan <tme.matching_exhaustive.scan>` that enables template matching on data subsets, making it particularly useful for handling large datasets or targeting specific regions.

.. autosummary::
   :toctree: ../api/

   scan
   scan_subsets


.. _setup-functions:

.. currentmodule:: tme.matching_scores

Setup functions
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../api/

   cc_setup
   lcc_setup
   corr_setup
   cam_setup
   flc_setup
   flcSphericalMask_setup
   mcc_setup


.. _scoring-functions:

Scoring functions
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../api/

   corr_scoring
   flc_scoring
   mcc_scoring


.. _custom-methods:

.. currentmodule:: tme.matching_exhaustive

Adding Custom Methods
~~~~~~~~~~~~~~~~~~~~~

For a method to be considered by |project|'s template matching engine, it needs to be registered via :py:meth:`register_matching_exhaustive`. This enables developers to specify a unique name, setup function, scoring function, and a custom memory estimation class for their method. This ensures the modular and extensible design of |project|, allowing developers to continuously expand tme’s capabilities.

Adding a new template matching methods requires defining the following parameters:

- ``matching``: Name of the matching method.
- ``matching_setup``: The setup function associated with the name.
- ``matching_scoring``: The scoring function associated with the name.
- ``memory_class``: A custom memory estimation class, which inherits from :py:class:`MatchingMemoryUsage <tme.memory.MatchingMemoryUsage>`

.. autosummary::
   :toctree: ../api/

   register_matching_exhaustive
