.. include:: ../../substitutions.rst
.. currentmodule:: tme.filters

Composable Filters
==================

Composable filters in |project| are inspired by the similarly named `Compose <https://pytorch.org/vision/main/generated/torchvision.transforms.Compose.html>`_ operation used in deep learning frameworks. Composable filters are an explicit solution to defining complex filtering procedures and can be lazily evaluated.

To demonstrate the use of composable filters, let's walk through a basic example showcasing how to combine :py:class:`BandPassFilter <tme.filters.bandpass.BandPassFilter>` and :py:class:`WedgeReconstructed <tme.filters.wedge.WedgeReconstructed>`

.. code-block:: python

   import numpy as np
   from tme.filters import Compose, BandPassFilter, WedgeReconstructed

   sampling_rate = (5,5,5)           # Sampling rate, typically Å/voxel

   # Setup BandPassFilter
   bandpass = BandPassFilter(
       lowpass=30,                   # Resolution to lowpass filter to
       highpass=5,                   # Resolution to highpass filter to
       sampling_rate=sampling_rate
   )

   # Setup WedgeReconstructed
   wedge = WedgeReconstructed(
       angles=[60, 60],              # Start, stop tilt angle
       opening_axis=0,               # Wedge is open in z-axis
       tilt_axis=1,                  # Sample is tilted over y-axis
       create_continuous_wedge=True, # Include all angles from -60 to 60
       sampling_rate=sampling_rate,
   )

   # Combine bandpass and wedge mask into a single filter
   composed_filter = Compose([bandpass, wedge])

   data_shape = (50,50,50)
   filter_mask = composed_filter(
      shape=data_shape, return_real_fourier=False
   )["data"]

   # Apply the filter mask
   data = np.random.rand(*data_shape)
   data_filtered = np.fft.ifftn(np.fft.fftn(data * filter_mask)).real


Specification
~~~~~~~~~~~~~

:py:class:`ComposableFilter <tme.filters.compose.ComposableFilter>` serves as specification for new composable filters.

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   ComposableFilter


Aggregator
~~~~~~~~~~

:py:class:`Compose <tme.filters.compose.Compose>` allows for combining the operations described by multiple objects of type :py:class:`ComposableFilter <tme.filters.compose.ComposableFilter>`.

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   Compose


Frequency Filters
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   BandPassFilter
   LinearWhiteningFilter


Cryo-EM Filters
~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   CTF
   Wedge
   WedgeReconstructed

Reconstruction
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   ReconstructFromTilt
