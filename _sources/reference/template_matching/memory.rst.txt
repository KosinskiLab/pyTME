:tocdepth: 3

.. include:: ../../substitutions.rst
.. currentmodule:: tme.memory

Memory
======

|project| computes memory requirements for template matching operations ahead of time, enabling automatic workload splitting across available hardware resources.

Scheduling occurs in three phases:

1. Predict peak memory consumption for the operation
2. Determine optimal box decomposition within hardware constraints
3. Process independent boxes sequentially or in parallel

Core Functions
~~~~~~~~~~~~~~

.. autosummary::
   :toctree: ../api/

   compute_schedule
   estimate_memory_usage

:func:`compute_schedule` determines the box decomposition based on available memory and operation requirements. :func:`estimate_memory_usage` predicts peak memory consumption for a given configuration.


Decomposition Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

The scheduler decomposes the search space into boxes that cover the target volume while minimizing computational overhead. Two modes are available:

- **Uniform**: Regular grid decomposition across the entire volume
- **Masked**: Uses binary segmentation to place boxes only where needed, minimizing the number of computations

.. plot::

   import numpy as np
   import matplotlib.pyplot as plt
   from matplotlib import patches
   from tme.memory import compute_schedule

   mask = np.zeros((256, 256), dtype=bool)
   mask[32:96, 48:112] = True      # 64×64 region
   mask[128:192, 176:240] = True   # 64×64 region
   mask[96:128, 128:192] = True    # 32×64 region
   mask[192:224, 64:128] = True    # 32×64 region

   y, x = np.ogrid[:256, :256]
   circle1 = (x - 224)**2 + (y - 64)**2 <= 22**2
   circle2 = (x - 160)**2 + (y - 208)**2 <= 18**2
   mask[circle1] = True
   mask[circle2] = True

   # Compute schedules for both modes
   boxes_uniform, schedule = compute_schedule(
      shape=mask.shape,
      mode="uniform",
      max_memory=1e10,
      max_workers=4,
      matching_method="CC",
   )

   boxes_masked, schedule = compute_schedule(
      shape=mask.shape,
      mode="subdivide",
      mask=mask,
      min_box_size=16,
      padding=(8, 8),
      max_memory=1e10,
      max_workers=1,
      matching_method="CC",
      verbose=True,
      min_improvement=1.0,
      n_sat=64,
   )

   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
   colors_uniform = plt.cm.Set3(np.linspace(0, 1, len(boxes_uniform)))
   colors_masked = plt.cm.Set3(np.linspace(0, 1, len(boxes_masked)))

   # Uniform mode
   ax1.imshow(mask, cmap="gray", alpha=0.3, origin="upper",
       extent=[-0.5, mask.shape[1]-0.5, mask.shape[0]-0.5, -0.5])
   for i, box in enumerate(boxes_uniform):
      y, x = box
      rect = patches.Rectangle(
          (x.start -0.5, y.start - 0.5),
          x.stop - x.start, y.stop - y.start,
          linewidth=2.5, edgecolor=colors_uniform[i],
          facecolor=colors_uniform[i], alpha=0.25
      )
      ax1.add_patch(rect)
   ax1.set_title("Uniform", fontsize=14)
   ax1.axis("off")

   # Masked mode
   ax2.imshow(mask, cmap="gray", alpha=0.3, origin="upper",
       extent=[-0.5, mask.shape[1]-0.5, mask.shape[0]-0.5, -0.5])
   for i, box in enumerate(boxes_masked):
      y, x = box
      rect = patches.Rectangle(
          (x.start - 0.5, y.start - 0.5),
          x.stop - x.start, y.stop - y.start,
          linewidth=2.5, edgecolor=colors_masked[i],
          facecolor=colors_masked[i], alpha=0.25
      )
      ax2.add_patch(rect)
   ax2.set_title("Masked", fontsize=14)
   ax2.axis("off")

   plt.tight_layout()
   plt.show()


Defining Memory Requirements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Custom operations register their memory footprint by subclassing :class:`MatchingMemoryUsage` and implementing the required methods.

.. autosummary::
   :toctree: ../api/

   MatchingMemoryUsage

A concrete implementation for methods with uniform array requirements is :class:`MemoryProfile`.

.. autosummary::
   :toctree: ../api/

   MemoryProfile


Built-in Memory Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~

Memory profiles are registered for scoring methods, analyzers, and backends. Some profiles serve multiple method identifiers (e.g., :class:`CORRMemoryUsage` handles CORR, NCC, CAM, and related variants).

Scoring Methods
^^^^^^^^^^^^^^^

.. autosummary::
   :toctree: ../api/

   CCMemoryUsage
   CORRMemoryUsage
   FLCMemoryUsage
   MCCMemoryUsage

Analyzers
^^^^^^^^^

.. autosummary::
   :toctree: ../api/

   MaxScoreOverRotationsMemoryUsage
   MaxScoreOverRotationsConstrainedMemoryUsage
   PeakCallerMaximumFilterMemoryUsage

Backends
^^^^^^^^

.. autosummary::
   :toctree: ../api/

   CupyBackendMemoryUsage
   NumpyBackendMemoryUsage


Registering Custom Profiles
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Use the :func:`register_memory` decorator to associate memory profiles with custom methods.

.. autosummary::
   :toctree: ../api/

   register_memory

The decorator accepts multiple method identifiers, allowing a single profile to serve multiple operations:

.. code-block:: python

   from tme.memory import MemoryProfile, register_memory

   @register_memory("CustomMethod", "CustomMethodVariant")
   class CustomMethodMemoryUsage(MemoryProfile):
       """Custom memory estimator."""

       #: Number of shared real arrays
       base_float = 2
       #: Number of shared complex arrays
       base_complex = 1
       #: Number of real arrays per fork
       fork_float = 1
       #: Number of complex arrays per fork
       fork_complex = 1

The :class:`MemoryProfile` class simplifies registration for methods with uniform array requirements. For more complex memory patterns, subclass :class:`MatchingMemoryUsage` directly:

.. code-block:: python

   from tme.memory import MatchingMemoryUsage, register_memory

   @register_memory("ComplexMethod")
   class ComplexMethodMemoryUsage(MatchingMemoryUsage):
       """Memory estimator for methods with non-uniform requirements."""

       def base_usage(self):
           # Custom implementation for base usage

       def per_fork(self):
           # Custom implementation per fork

Once registered, the memory profile is automatically used by :func:`estimate_memory_usage` and :func:`compute_schedule` for the specified method identifiers.
