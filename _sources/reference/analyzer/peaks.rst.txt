.. currentmodule:: tme.analyzer

Peak Calling
============

Peak calling analyzers identify and extract local maxima from correlation score maps using various detection algorithms. These analyzers can handle distance constraints, boundary conditions, and score thresholding to produce filtered lists of candidate template locations with their corresponding orientations.

.. autosummary::
   :toctree: ../api/

   PeakCaller
   PeakCallerSort
   PeakCallerMaximumFilter
   PeakCallerFast
   PeakCallerRecursiveMasking
   PeakCallerScipy


**PeakCaller** provides the base class for peak detection algorithms with configurable distance and boundary constraints.

**PeakCallerSort** selects the highest scoring peaks by sorting and taking the top N candidates.

**PeakCallerMaximumFilter** finds local maxima using maximum filtering with distance constraints, similar to skimage's peak_local_max.

**PeakCallerFast** subdivides the score space into grids and identifies maxima within each subdivision for efficient processing.

**PeakCallerRecursiveMasking** iteratively identifies peaks by masking regions around detected maxima to prevent duplicate detections.

**PeakCallerScipy** uses scikit-image's peak_local_max implementation for local maxima detection of well defined peaks.
