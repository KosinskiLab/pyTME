.. _analyzer-label:

.. currentmodule:: tme.analyzer

Analyzers
=========

Analyzers are internal callbacks that are passed to :py:meth:`tme.matching_exhaustive.scan_subsets` and provide custom processing of exhaustive template matching results. This flexibility enables on the fly analysis, logging, or additional processing at the level of individual rotations. The exact workflow can be adapted to individual requirements.


Score aggregation
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MemmapHandler
   MaxScoreOverRotations


Peak calling
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   PeakCaller
   PeakCallerSort
   PeakCallerMaximumFilter
   PeakCallerFast
   PeakCallerRecursiveMasking
   PeakCallerScipy
