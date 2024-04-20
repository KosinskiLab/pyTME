.. include:: ../substitutions.rst

=================
Template matching
=================

Template matching is a signal processing technique, often employed in image analysis, to find parts of a larger dataset (often called the target) that match a smaller sample (referred to as the template). Within |project| we distinguish between exhaustive and non-exhaustive template matching.

Exhaustive template matching evaluates similarity along a provided set of rotations and all possible translations are sampled using Fast Fourier Transform (FFT) operations. Therefore, the algorithm is guaranteed to evaluate a provided set of configurations and to find a global optimum with a sufficiently high angular sampling rate.

Non-exhaustive template aims to identify a global optimum without evaluating all possible configurations, by using mathematical optimization techniques. However, whether non-exhaustive template matching is able to find a global optimum is highly dependent on the topology of the score space and available constraints. Typically, non-exhaustive template matching is useful when the rough orientation is known.

.. toctree::

    matching_optimization
    matching_exhaustive
    matching_utils
    matching_memory
