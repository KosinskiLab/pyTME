.. _analyzer-label:

.. currentmodule:: tme.analyzer

Analyzers
=========

Analyzers are internal callbacks that are passed to :py:class:`scan_subsets <tme.matching_exhaustive.scan_subsets>` and provide custom processing of exhaustive template matching results. This flexibility enables on the fly analysis, logging, or additional processing at the level of individual rotations. The exact workflow can be adapted to individual requirements.

Specification
~~~~~~~~~~~~~

:py:class:`AbstractAnalyzer <tme.analyzer.AbstractAnalyzer>` serves as specification for new analyzers.

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   AbstractAnalyzer


Score aggregation
~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   MaxScoreOverRotations
   MaxScoreOverTranslations
   MaxScoreOverRotationsConstrained


Peak calling
~~~~~~~~~~~~
.. autosummary::
   :toctree: ../api/

   PeakCaller
   PeakCallerSort
   PeakCallerMaximumFilter
   PeakCallerFast
   PeakCallerRecursiveMasking
   PeakCallerScipy
