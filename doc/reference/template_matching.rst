.. include:: ../substitutions.rst

=================
Template matching
=================

Template matching is a signal processing technique, often employed in image analysis, to find parts of a larger dataset (often called the target) that match a smaller sample (referred to as the template). This is commonly applied in scenarios like searching an image for a smaller sub-image.

Within |project| we distinguish between exhaustive and non-exhaustive template mathcing. In exhaustive matching, the template is rotated across all possible angles, and all translations within the target are sampled using Fast Fourier Transform (FFT) operations. This contrasts with non-exhaustive template matching, where only selected rotations or translations are sampled subject to numerical optimization.

.. toctree::

    matching_optimization
    matching_exhaustive
    matching_utils
    matching_memory
