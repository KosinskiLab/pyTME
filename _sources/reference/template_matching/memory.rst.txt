.. include:: ../../substitutions.rst
.. currentmodule:: tme.memory

Memory
======

|project| is aware of the memory required for template matching operations. This enables us to map template matching problems to the available hardware and compute a splitting strategy ahead of time.


Memory usage helpers
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   estimate_ram_usage


Abstract Base
~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatchingMemoryUsage


Memory usage classes
~~~~~~~~~~~~~~~~~~~~
.. autosummary::
   :toctree: api/

   MatchingMemoryUsage
   CCMemoryUsage
   LCCMemoryUsage
   CORRMemoryUsage
   CAMMemoryUsage
   FLCSphericalMaskMemoryUsage
   FLCMemoryUsage
   MCCMemoryUsage
   MaxScoreOverRotationsMemoryUsage
   PeakCallerMaximumFilterMemoryUsage
   CupyBackendMemoryUsage
