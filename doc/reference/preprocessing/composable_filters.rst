.. include:: ../../substitutions.rst

Composable Filters
==================

Composable filters in |project| are inspired by the similarly named `Compose <https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html>`_ operation used in deep learning frameworks. Composable filters are a very explicit solution to defining complex filtering procedures and can be lazily evaluated.


Specification
~~~~~~~~~~~~~

.. currentmodule:: tme.preprocessing.composable_filter

.. autosummary::
   :toctree: api/
   :nosignatures:

   ComposableFilter


Aggregator
~~~~~~~~~~

.. currentmodule:: tme.preprocessing.compose

.. autosummary::
   :toctree: api/
   :nosignatures:

   Compose


Frequency Filters
~~~~~~~~~~~~~~~~~

.. currentmodule:: tme.preprocessing.frequency_filters

.. autosummary::
   :toctree: api/
   :nosignatures:

   BandPassFilter
   LinearWhiteningFilter


CryoEM Filters
~~~~~~~~~~~~~~

.. currentmodule:: tme.preprocessing.tilt_series

.. autosummary::
   :toctree: api/
   :nosignatures:

   CTF
   Wedge
   WedgeReconstructed

Reconstruction
~~~~~~~~~~~~~~

.. currentmodule:: tme.preprocessing.tilt_series

.. autosummary::
   :toctree: api/
   :nosignatures:

   ReconstructFromTilt
