.. currentmodule:: tme.analyzer

Peak Calling
============

Peak calling analyzers identify and extract local maxima from correlation score maps using various detection algorithms. These analyzers can handle distance constraints, boundary conditions, and score thresholding to produce filtered lists of candidate template locations with their corresponding orientations.

Since there exist a multitude of approaches to identify local maxima, we define an abstract interface that peak callers must implement

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   PeakCaller

Built-in Peak Callers
~~~~~~~~~~~~~~~~~~~~~

Subclasses of :class:`PeakCaller` implement their individual peak calling schemes through :func:`PeakCaller.call_peaks`. A list of concrete implementations can be found below.

.. autosummary::
   :toctree: ../api/

   PeakCallerMaximumFilter
   PeakCallerRecursiveMasking
   PeakCallerScipy
   PeakCallerSort
   PeakCallerFast

Comparison
~~~~~~~~~~

See :doc:`/quickstart/overview` for a visual comparison of all peak callers
and guidance on which to use.

For most particle picking tasks you will use ``PeakCallerMaximumFilter`` or
``PeakCallerRecursiveMasking``. ``PeakCallerScipy`` is preferable for broad,
well-separated peaks. ``PeakCallerSort`` and ``PeakCallerFast`` are internal
tools that are computationally efficient but not suitable for particle
picking directly.
