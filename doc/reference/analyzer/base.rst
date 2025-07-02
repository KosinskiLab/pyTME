.. _analyzer-label:

.. currentmodule:: tme.analyzer

Specification
=============

Analyzers are callback objects designed to process template matching results in real-time during exhaustive searches. Rather than storing all intermediate results in memory, analyzers accumulate only the most relevant data as each rotation is evaluated, enabling adaptation of template matching analysis routines to individual requirements.

Key characteristics of analyzers

- Called individually for each rotation in the search space
- Support distributed computation with result merging capabilities
- Maintain internal state that evolves throughout the matching process

Analyzers are passed to :py:class:`scan_subsets <tme.matching_exhaustive.scan_subsets>` and similar matching functions to enable custom processing workflows tailored to specific analysis requirements.

:py:class:`AbstractAnalyzer <tme.analyzer.AbstractAnalyzer>` defines the interface specification that all analyzers must implement.

.. autosummary::
   :toctree: ../api/
   :nosignatures:

   AbstractAnalyzer

How Analyzers Work
==================

Analyzers follow a stateful, three-phase pattern

**1. Initialization Phase**
   The analyzer creates an initial state using :py:meth:`AbstractAnalyzer.init_state`. This state contains empty data structures that will accumulate results across all rotations. For example, an analyzer might initialize empty arrays for storing top scores, coordinates, and rotation matrices.

**2. Accumulation Phase**
   For each rotation tested during template matching, the analyzer's :py:meth:`AbstractAnalyzer.__call__` method is invoked with

   - Current state from previous iterations
   - New scores array from the current rotation
   - Rotation matrix that generated these scores

   The analyzer updates its state by incorporating the new data according to its specific logic (e.g., keeping only top N scores, filtering by threshold, etc.) and returns the updated state for the next iteration.

**3. Finalization Phase**
   After all rotations are processed, :py:meth:`AbstractAnalyzer.result` converts the accumulated state into the final output format. This may include post-processing operations like coordinate transformations or convolution mode corrections.


Template Matching Integration
=============================

During exhaustive template matching, the analyzer is called for each rotation::

    # For each rotation in the search space
    for rotation in rotations:
        # Compute correlation scores for this rotation
        scores = compute_scores(template, target, rotation)

        # Update analyzer state with new results
        analyzer_state = analyzer(analyzer_state, scores, rotation)

This design enables real-time analysis during the matching process rather than requiring separate post-processing steps.


State Management
================

Analyzers use a stateful accumulation pattern that ensures data ownership remains external to analyzer instances. This design maintains compatibility with just-in-time compilation features, particularly for the JAX backend.

Parallel Processing
===================

Analyzers support parallel processing through

- **Shared Memory**: The :py:attr:`AbstractAnalyzer.shareable` property enables multi-process operation using shared memory buffers, avoiding expensive data copying.

- **Result Merging**: The :py:meth:`AbstractAnalyzer.merge` class method combines results from different processes or data splits into unified output.


Custom Analyzer Implementation
==============================

To implement a custom analyzer:

1. **Inherit from AbstractAnalyzer**::

    class MyAnalyzer(AbstractAnalyzer):
        pass

2. **Define the state structure** in :py:meth:`init_state`
3. **Implement accumulation logic** in :py:meth:`__call__`
4. **Handle finalization** in :py:meth:`result`
5. **Support merging** in :py:meth:`merge`
6. **Set shareability** via the :py:attr:`shareable` property

This flexible architecture allows for diverse analysis strategies including peak detection, clustering, statistical analysis, and custom filtering without modifying the core template matching algorithms.
