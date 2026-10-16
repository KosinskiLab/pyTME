.. include:: ../../substitutions.rst

=============
Using the API
=============

The Python API exposes :py:class:`Density <tme.density.Density>` for loading and manipulating density maps and :py:meth:`create_mask <tme.matching_utils.create_mask>` for building masks. Filtering operations such as Gaussian blurring, bandpass, or median filtering are done with :py:mod:`scipy.ndimage`. The following constructs a sphere mask and applies a Gaussian smoothing:

.. code-block:: python

    from tme import Density
    from tme.matching_utils import create_mask
    from scipy.ndimage import gaussian_filter

    density = Density.from_file("example.mrc")
    mask = create_mask(
        mask_type="ellipse",
        shape=density.shape,
        center=(d // 2 for d in density.shape),
        radius=10,
        soft_edge_width=2,
    )
    smoothed = gaussian_filter(density.data, sigma=2)
