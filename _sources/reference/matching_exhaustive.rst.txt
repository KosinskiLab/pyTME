.. include:: ../substitutions.rst

Exhaustive
==========

.. currentmodule:: tme.matching_exhaustive

Within |project|, exhaustive template matching is modularized into three primary stages: setup, scoring, and callback.

1. **Setup**
    - The first step is to prepare the data and set the stage for the matching operation. This step typically involves configuring FFT operations, selecting the appropriate backend for the computations, and defining parameters for the matching. The outcome of this step is a context where template matching is poised to be carried out efficiently. See the :ref:`setup-functions` section for available setup methods.

2. **Scoring**
    - Once the environment is set, the actual template matching is done using the scoring method. This method essentially computes how well the template fits at every possible location within the target. Depending on the application, various scoring methods might be employed (e.g., correlation, convolution). See the :ref:`scoring-functions` section for available scoring methods.

3. **Callback**
    - After scoring, it's often essential to post-process the results, extract valuable metrics, or make decisions based on the scores. The callback is a mechanism that allows for such post-scoring operations. It could be visualizing the results, finding the maximum score, etc. Further details and callback options can be found in :doc:`analyzer`.

This concept is embodied in the :py:meth:`tme.matching_exhaustive.scan` and :py:meth:`tme.matching_exhaustive.scan_subsets` methods below.

If you wish to integrate custom template matching methods into |project|, please refer to the :ref:`custom-methods` section.


Methods
~~~~~~~
:py:meth:`tme.matching_exhaustive.scan` orchestrates the matching process, supporting parallel processing and post-scoring operations. :py:meth:`tme.matching_exhaustive.scan_subsets` is a wrapper around :py:meth:`tme.matching_exhaustive.scan` that enables template matching on data subsets, making it particularly useful for handling large datasets or targeting specific regions.

.. autosummary::
   :toctree: api/

   scan
   scan_subsets


.. _setup-functions:

.. currentmodule:: tme.matching_scores

Setup functions
~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

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
   :toctree: api/

   corr_scoring
   flc_scoring
   mcc_scoring


.. _custom-methods:

.. currentmodule:: tme.matching_exhaustive

Adding Custom Methods
~~~~~~~~~~~~~~~~~~~~~
For a method to be considered by |project|'s template matching engine, it needs to be registered. The :py:meth:`register_matching_exhaustive` function is designed for this purpose. It enables developers to specify a unique name, setup function, scoring function, and a custom memory estimation class for their method. This ensures the modular and extensible design of |project|, allowing developers to continuously expand tme’s capabilities.

Adding a new template matching methods requires defining the following parameters:

- ``matching``: Name of the matching method.
- ``matching_setup``: The setup function associated with the name.
- ``matching_scoring``: The scoring function associated with the name.
- ``memory_class``: A custom memory estimation class, which should extend :py:class:`tme.memory.MatchingMemoryUsage`.

.. autosummary::
   :toctree: api/

   register_matching_exhaustive