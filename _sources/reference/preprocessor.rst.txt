Preprocessor
============

.. currentmodule:: tme.preprocessor

Constructor
~~~~~~~~~~~

.. autosummary::
   :toctree: api/
   :nosignatures:

   Preprocessor

Filters
~~~~~~~
.. autosummary::
   :toctree: api/

   Preprocessor.wedge_mask
   Preprocessor.gaussian_filter
   Preprocessor.difference_of_gaussian_filter
   Preprocessor.bandpass_filter
   Preprocessor.local_gaussian_alignment_filter
   Preprocessor.local_gaussian_filter
   Preprocessor.edge_gaussian_filter
   Preprocessor.ntree_filter
   Preprocessor.mean_filter
   Preprocessor.rank_filter
   Preprocessor.mipmap_filter
   Preprocessor.wavelet_filter
   Preprocessor.kaiserb_filter
   Preprocessor.blob_filter
   Preprocessor.hamming_filter
   Preprocessor.molmap

Utilities
~~~~~~~~~
.. autosummary::
   :toctree: api/

   Preprocessor.fftfreqn
   Preprocessor.fourier_crop
   Preprocessor.fourier_uncrop
   Preprocessor.interpolate_box
