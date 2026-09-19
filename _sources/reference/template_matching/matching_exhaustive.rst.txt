:tocdepth: 3

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

If you wish to integrate custom template matching methods into |project|, please refer to the :ref:`custom-methods` section.


Methods
~~~~~~~

:py:class:`match_exhaustive <tme.matching_exhaustive.match_exhaustive>` orchestrates the matching process, supporting parallel processing and analysis operations. Depending on user specification, parallelization can be performed by splitting the search region into subsets, and/or by distributing the angular search.

.. autosummary::
   :toctree: ../api/

   match_exhaustive

Concrete implementations are outlined below.

.. _setup-functions:

.. currentmodule:: tme.matching_scores

Setup functions
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../api/

   cc_setup
   lcc_setup
   ncc_setup
   cam_setup
   flc_setup
   flcSphericalMask_setup
   mcc_setup


.. _scoring-functions:

Scoring functions
^^^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../api/

   ncc_scoring
   flc_scoring
   mcc_scoring


.. _custom-methods:

.. currentmodule:: tme.matching_exhaustive

Adding Custom Methods
~~~~~~~~~~~~~~~~~~~~~

New scoring methods are registered via :py:meth:`register_matching_exhaustive`.

.. autosummary::
   :toctree: ../api/

   register_matching_exhaustive

Adding a new template matching method requires defining the following:

- Name of the matching method
- Setup function associated with the name
- Scoring function associated with the name
- Custom memory estimation class inheriting from :py:class:`MatchingMemoryUsage <tme.memory.MatchingMemoryUsage>`


The following outlines an example implementation.

.. code-block:: python

   from tme.memory import MemoryProfile, register_memory
   from tme.matching_exhaustive import register_matching_exhaustive

   @register_memory("CustomMethod")
   class CustomMethodMemoryUsage(MemoryProfile):
       """Memory estimator for CustomMethod."""
       base_float = 2
       base_complex = 1
       fork_float = 1
       fork_complex = 1

   def custom_setup(target, template, **kwargs):
       """
       Prepare data structures for matching.

       Returns context dictionary with shared parameters.
       """
       # Setup implementation
       return context

   def custom_scoring(score_space, rotated_template, **kwargs):
       """
       Compute similarity scores.
       """
       # Scoring implementation
       pass

   register_matching_exhaustive("CustomMethod", custom_setup, custom_scoring)
