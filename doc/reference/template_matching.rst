.. include:: ../substitutions.rst

=================
Template matching
=================

Template matching is a signal processing technique, often employed in image analysis, to find parts of a larger dataset (often called the target) that match a smaller sample (referred to as the template). Within |project| we distinguish between exhaustive and non-exhaustive template matching.

.. toctree::
    :maxdepth: 1

    matching_exhaustive

Exhaustive template matching evaluates similarity along a provided set of rotations and all possible translations are sampled using Fast Fourier Transform (FFT) operations. Therefore, the algorithm is guaranteed to evaluate a provided set of configurations and to find a global optimum with a sufficiently high angular sampling rate.


.. toctree::
    :maxdepth: 1

    matching_optimization

Non-exhaustive template aims to identify a global optimum without evaluating all possible configurations, e.g. via gradient-descent. However, whether non-exhaustive template matching is able to find a global optimum is highly dependent on the topology of the score space and available constraints. Typically, non-exhaustive template matching is useful when the rough orientation is known.


.. toctree::
    :maxdepth: 1

    memory

|project| is aware of the memory required for template matching operations. This enables us to map template matching problems to the available hardware and compute a splitting strategy ahead of time.


.. toctree::
    :maxdepth: 1

    matching_utils

Functions that are used in exhaustive and non-exhaustive template matching or are specific to template matching in general are grouped in the utilities module.
