Non-exhaustive
==============

.. currentmodule:: tme.matching_optimization

Constructor
~~~~~~~~~~~
.. autosummary::
   :toctree: api/
   :nosignatures:

   FitRefinement

Computations
~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   FitRefinement.refine
   FitRefinement.array_from_coordinates
   FitRefinement.map_coordinates_to_array

Available scores
~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   CrossCorrelation
   LaplaceCrossCorrelation
   NormalizedCrossCorrelation
   NormalizedCrossCorrelationMean
   MaskedCrossCorrelation
   PartialLeastSquareDifference
   Chamfer
   MutualInformation
   Envelope
   NormalVectorScore

Helper classes
~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatchCoordinatesToDensity
