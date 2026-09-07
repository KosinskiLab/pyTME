"""
Compute memory consumption of template matching components.

Copyright (c) 2023-2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from functools import partial
from itertools import permutations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Callable, Set

import numpy as np

from .types import NDArray
from .backends import backend as be
from .utils.subdivide import solve_subdivide

__all__ = [
    "compute_schedule",
    "estimate_memory_usage",
    "register_memory",
    "MatchingMemoryUsage",
    "MemoryProfile",
    "CCMemoryUsage",
    "CORRMemoryUsage",
    "FLCMemoryUsage",
    "MCCMemoryUsage",
    "MaxScoreOverRotationsMemoryUsage",
    "MaxScoreOverRotationsConstrainedMemoryUsage",
    "PeakCallerMaximumFilterMemoryUsage",
    "NumpyBackendMemoryUsage",
    "CupyBackendMemoryUsage",
]

MATCHING_MEMORY_REGISTRY = {}

# Approximation derived from testing a range of GPU types
DEFAULT_N_SAT = 256**3


def register_memory(*names: str):
    """
    Register a :class:`MatchingMemoryUsage` subclass under one or more names.

    Parameters
    ----------
    *names : str
        Lookup keys (matching method, analyzer, or backend name).
    """

    def decorator(cls):
        for name in names:
            MATCHING_MEMORY_REGISTRY[name] = cls
        return cls

    return decorator


class MatchingMemoryUsage(ABC):
    """
    Strategy class for estimating memory requirements.

    Parameters
    ----------
    fast_shape : tuple of int
        Shape of the real array.
    ft_shape : tuple of int
        Shape of the complex array.
    float_nbytes : int
        Number of bytes of the used float, e.g. 4 for float32.
    complex_nbytes : int
        Number of bytes of the used complex, e.g. 8 for complex64.
    integer_nbytes : int
        Number of bytes of the used integer, e.g. 4 for int32.
    """

    def __init__(
        self,
        fast_shape: Tuple[int, ...],
        ft_shape: Tuple[int, ...],
        float_nbytes: int,
        complex_nbytes: int,
        integer_nbytes: int,
    ):
        self.real_array_size = int(np.prod(fast_shape))
        self.complex_array_size = int(np.prod(ft_shape))
        self.float_nbytes = float_nbytes
        self.complex_nbytes = complex_nbytes
        self.integer_nbytes = integer_nbytes

    @abstractmethod
    def base_usage(self) -> int:
        """Return the base memory usage in bytes."""

    @abstractmethod
    def per_fork(self) -> int:
        """Return the memory usage per fork in bytes."""


class MemoryProfile(MatchingMemoryUsage):
    """Memory estimator for methods with uniform array requirements."""

    #: Number of shared real arrays
    base_float: int = 0
    #: Number of shared complex arrays
    base_complex: int = 0
    #: Number of real arrays per fork
    fork_float: int = 0
    #: Number of complex arrays per fork
    fork_complex: int = 0

    def base_usage(self) -> int:
        return (
            self.base_float * self.real_array_size * self.float_nbytes
            + self.base_complex * self.complex_array_size * self.complex_nbytes
        )

    def per_fork(self) -> int:
        return (
            self.fork_float * self.real_array_size * self.float_nbytes
            + self.fork_complex * self.complex_array_size * self.complex_nbytes
        )


@register_memory("CC", "LCC")
class CCMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.cc_setup` memory estimator."""

    base_float, base_complex = 1, 1
    fork_float, fork_complex = 1, 1


@register_memory("CORR", "NCC", "CAM", "FLCSphericalMask", "batchFLCSphericalMask")
class CORRMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.corr_setup` memory estimator."""

    base_float, base_complex = 4, 1
    fork_float, fork_complex = 1, 1


@register_memory("FLC", "batchFLC")
class FLCMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.flc_setup` memory estimator."""

    base_float, base_complex = 2, 2
    fork_float, fork_complex = 3, 2


@register_memory("MCC")
class MCCMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.mcc_setup` memory estimator."""

    base_float, base_complex = 2, 3
    fork_float, fork_complex = 6, 1


@register_memory("MaxScoreOverRotations")
class MaxScoreOverRotationsMemoryUsage(MemoryProfile):
    """:py:class:`tme.analyzer.MaxScoreOverRotations` memory estimator."""

    base_float = 2


@register_memory("MaxScoreOverRotationsConstrained")
class MaxScoreOverRotationsConstrainedMemoryUsage(MemoryProfile):
    """:py:class:`tme.analyzer.MaxScoreOverRotationsConstrained` memory estimator."""

    # This ultimately depends on the number of seed points and mask size.
    # Ideally we would use that in the memory estimation, but for now we
    # approximate by reqesting memory for another real array
    base_float = 3


@register_memory("PeakCallerMaximumFilter")
class PeakCallerMaximumFilterMemoryUsage(MemoryProfile):
    """:py:class:`tme.analyzer.peaks.PeakCallerMaximumFilter` memory estimator."""

    base_float, fork_float = 1, 1


@register_memory("numpyfftw", "jax", "mlx", "KernelFit")
class NumpyBackendMemoryUsage(MemoryProfile):
    """:py:class:`tme.backends.NumpyFFTWBackend` memory estimator."""

    # We assume no overhead for these backends


@register_memory("cupy", "pytorch")
class CupyBackendMemoryUsage(MemoryProfile):
    """:py:class:`tme.backends.CupyBackend` memory estimator."""

    # FFT plans, overhead from assigning FFT result, rotation interpolation
    base_complex, base_float = 3, 2


def estimate_memory_usage(
    shape1: Tuple[int],
    shape2: Tuple[int],
    matching_method: str,
    ncores: int,
    analyzer_method: Optional[str] = None,
    backend: Optional[str] = None,
    float_nbytes: int = 4,
    complex_nbytes: int = 8,
    integer_nbytes: int = 4,
) -> int:
    """
    Estimate the memory usage of a given template matching run.

    Parameters
    ----------
    shape1 : tuple
        Shape of the target array.
    shape2 : tuple
        Shape of the template array.
    matching_method : str
        Matching method used to compute scores.
    analyzer_method : str, optional
        Analyzer used for score analysis.
    backend : str, optional
        Backend used for computation.
    ncores : int
        The number of operations running in parallel.
    float_nbytes : int
        Byte size of used float, defaults to 4 (float32).
    complex_nbytes : int
        Byte size of used complex, defaults to 8 (complex64).
    integer_nbytes : int
        Byte size of used integer, defaults to 4 (int32).

    Returns
    -------
    int
        The estimated memory usage for the operation in bytes.

    Raises
    ------
    ValueError
        If matching_method, analyzer_method or backend are unsupported and not None.
    """
    _, fast_shape, ft_shape = be.compute_convolution_shapes(shape1, shape2)

    kwargs = {
        "fast_shape": fast_shape,
        "ft_shape": ft_shape,
        "float_nbytes": float_nbytes,
        "complex_nbytes": complex_nbytes,
        "integer_nbytes": integer_nbytes,
    }

    nbytes = 0
    for method in (matching_method, analyzer_method, backend):
        if method is None:
            continue
        elif method not in MATCHING_MEMORY_REGISTRY:
            _supported = ", ".join(f"'{k}'" for k in MATCHING_MEMORY_REGISTRY.keys())
            raise ValueError(f"Supported are {_supported}, got {method}.")

        instance = MATCHING_MEMORY_REGISTRY[method](**kwargs)
        nbytes += instance.base_usage() + instance.per_fork() * ncores
    return nbytes


def compute_schedule(
    shape: Tuple[int],
    max_memory: int,
    max_workers: int,
    matching_method: str,
    mode: str = "uniform",
    padding: Optional[Tuple[int]] = None,
    analyzer_method: Optional[str] = None,
    backend: Optional[str] = None,
    float_nbytes: int = 4,
    complex_nbytes: int = 8,
    integer_nbytes: int = 4,
    verbose: bool = True,
    target_subset: Optional[Tuple[slice, ...]] = None,
    n_sat: int = DEFAULT_N_SAT,
    min_improvement: float = 1.5,
    **mode_kwargs,
) -> Tuple[Tuple[Tuple[slice, ...]], Tuple[int, int]]:
    """
    Plan a parallelization schedule that fits ``max_memory`` and ``max_workers``.

    Parameters
    ----------
    shape : tuple of int
        Shape of the target array.
    max_memory : int
        Maximum memory usage allowed in bytes.
    max_workers : int
        Maximum number of concurrent workers.
    matching_method : str
        Scoring metric for template matching (e.g., 'CC', 'NCC', 'FLC').
    mode : {'uniform', 'subdivide'}
        Scheduling strategy:

        - ``uniform`` : Regular-grid split of ``shape`` via integer factorization.

        - ``subdivide`` : Recursive bisection of ``mask`` to find tighter bounding
            boxes around regions of interest. Falls back to ``uniform`` when the
            improvement is below threshold.

    padding : tuple of int, optional
        Padding applied to target in each dimension.
    analyzer_method : str, optional
        Analyzer class name (e.g., 'MaxScoreOverRotations').
    backend : str, optional
        Computation backend (e.g., 'cupy', 'pytorch').
    float_nbytes : int
        Bytes per float element (4 for float32).
    complex_nbytes : int
        Bytes per complex element (8 for complex64).
    integer_nbytes : int
        Bytes per integer element (4 for int32).
    verbose : bool
        Print scheduling statistics and diagnostic information.
    target_subset : tuple of slice, optional
        Restrict scheduling to a subregion of ``shape``. When combined with
        ``mask``, the mask is first cropped to this subset, then further refined
        to its tight bounding box. Returned boxes are in the original ``shape``
        coordinates.
    n_sat : int, optional
        FFT saturation voxel count: boxes below this size are scored as if their
        volume were ``n_sat`` (overhead-bound regime). Default :data:`DEFAULT_N_SAT`.
    min_improvement: float, optional
        Minimum fractional improvement over uniform fallback.
    **mode_kwargs
        Additional mode-specific parameters.

    Other Parameters
    ----------------
    For mode 'uniform'

    split_axes : tuple of int, optional
        Axes along which splitting is allowed. Default is all axes.
    split_only_outer : bool, default False
        If True, parallelize only the outer loop (all workers process the same chunk
        sequentially). If False, explore nested parallelization strategies.
    max_splits : int, default 256
        Maximum number of boxes to create.

    For mode 'subdivide'

    mask : NDArray
        Binary mask indicating regions of interest.
    mask_spacing : float or tuple of float, optional
        Voxel spacing of ``mask`` relative to ``shape`` (scalar or per-axis).
        E.g., shape at 4 and mask at 8 Angstrom per voxel gives ``2.0``.
    min_box_size : int, optional
        Minimum box dimension along any axis, defaults to 32. ``None``
        disables the check.

    Returns
    -------
    tuple of tuple of slice
        Per-box slices in the coordinates of ``shape``.
    tuple of int, int
        ``(n_outer_jobs, n_inner_workers)``.

    Raises
    ------
    ValueError
        If no valid schedule fits the constraints, or ``mode`` is unsupported.

    Examples
    --------
    >>> boxes, (n_outer, n_inner) = compute_schedule(
    >>>     shape=(512, 512, 512),
    >>>     padding=(64, 64, 64),
    >>>     max_memory=8e9,  # 8 GB
    >>>     max_workers=4,
    >>>     matching_method='NCC',
    >>>     mode='uniform'
    >>> )
    """
    if mode not in ("uniform", "subdivide"):
        raise ValueError(f"Modes 'uniform', 'subdivide' are supported, got '{mode}'.")

    shape = tuple(int(x) for x in shape)
    if padding is None:
        padding = (0,) * len(shape)
    padding = tuple(int(x) for x in padding)

    initial_shape = shape
    if target_subset is None:
        target_subset = tuple(slice(0, x) for x in shape)

    offsets = tuple(x.start for x in target_subset)
    shape = tuple(x.stop - x.start for x in target_subset)

    mask = mode_kwargs.get("mask", None)
    mask_spacing = mode_kwargs.get("mask_spacing", 1.0)
    if not isinstance(mask_spacing, tuple):
        mask_spacing = (mask_spacing,) * len(shape)

    if isinstance(mask, np.ndarray) and mode == "subdivide":
        scaled_subset = tuple(
            slice(max(int(x.start / s), 0), min(int(np.ceil(x.stop / s)), m))
            for x, s, m in zip(target_subset, mask_spacing, mask.shape)
        )

        mask = mask[scaled_subset]
        mask_subset = _bounding_box(mask)
        mode_kwargs["mask"] = mask[mask_subset]

        shape = mode_kwargs["mask"].shape
        offsets = tuple(
            (x.start + y.start) * s
            for x, y, s in zip(mask_subset, scaled_subset, mask_spacing)
        )

    estimator = partial(
        estimate_memory_usage,
        matching_method=matching_method,
        analyzer_method=analyzer_method,
        backend=backend,
        float_nbytes=float_nbytes,
        complex_nbytes=complex_nbytes,
        integer_nbytes=integer_nbytes,
    )
    cost_fn = partial(_box_cost, padding=padding, n_sat=n_sat)
    kwargs = {
        "shape": shape,
        "max_workers": max_workers,
        "max_memory": max_memory,
        "padding": padding,
        "memory_estimator": estimator,
        "verbose": verbose,
        "cost_fn": cost_fn,
    } | mode_kwargs

    boxes, schedule = _schedule_uniform(**kwargs)
    score = _select_schedule(boxes, np.ceil(len(boxes) / max_workers), cost_fn)
    if verbose:
        n = int(np.prod(np.add(initial_shape, padding)))
        print("\n> Box decomposition")
        print(f"  - none: {len(boxes)} box(es), {n:,} voxels")

        n = sum(int(np.prod(_slice_to_shape(box, padding))) for box in boxes)
        print(f"  - uniform: {len(boxes)} box(es), {n:,} voxels, score {score:.3e}")

    if mode == "subdivide":
        bxs, schedule_mask = _schedule_subdivide(**kwargs)
        sc_mask = _select_schedule(bxs, np.ceil(len(bxs) / max_workers), cost_fn)

        ratio = score / sc_mask
        if verbose:
            n = sum(int(np.prod(_slice_to_shape(box, padding))) for box in bxs)
            print(
                f"  - subdivide: {len(bxs)} box(es), {n:,} voxels, score {sc_mask:.3e}"
            )

        if ratio < min_improvement:
            print(
                f"Subdivide is only {ratio:.2f}x cheaper than uniform "
                f"(threshold: {min_improvement:.2f}x). Falling back to uniform."
            )
        else:
            boxes, schedule = bxs, schedule_mask

    if not len(boxes):
        raise ValueError("No viable schedule. Increase memory or decrease workers.")

    boxes = tuple(
        tuple(
            slice(int(b.start * s + o), int(np.ceil(b.stop * s + o)))
            for o, b, s in zip(offsets, box, mask_spacing)
        )
        for box in boxes
    )
    return boxes, schedule


def _factorize(x: int, n: int, min_factor: int = 1) -> Set[Tuple[int, ...]]:
    """
    Factorize an integer into a set of integers with given cardinality.

    Parameters
    ----------
    x : int
        Integer to factorize.
    n : int
        Cardinality of factor set
    min_factor : int
        Minimal factor to consider.

    Returns
    -------
    set of tuple
        Possible factorizations.
    """
    if x < 1 or n < 1:
        raise ValueError("Both x and n must be >= 1")
    if n == 1:
        return {(x,)} if x >= min_factor else set()
    result = set()

    # Only try divisors from min_factor up to x^(1/n)
    max_d = int(x ** (1 / n)) + 1
    for d in range(min_factor, min(max_d + 1, x + 1)):
        if x % d == 0:
            for sub_factorization in _factorize(x // d, n - 1, d):
                result.add((d,) + sub_factorization)
    return {x for factorization in result for x in permutations(factorization)}


def _slice_to_shape(
    slices: Tuple[slice, ...], padding: Optional[Tuple[int, ...]] = None
) -> Tuple[int, ...]:
    """Per-axis extents of a slice tuple, optionally with padding added."""
    if padding is None:
        return tuple(s.stop - s.start for s in slices)
    return tuple(s.stop - s.start + p for s, p in zip(slices, padding))


def _box_cost(shapes, padding=None, n_sat: int = DEFAULT_N_SAT):
    """
    Computes max(N_sat, V) * log1p(max(N_sat, V)) with V = prod(shape + padding).

    Below N_sat voxels the FFT is overhead-bound and runtime is roughly constant
    per box; above, runtime grows with N*log(N). Accepts a single shape or batched
    shapes ``(n_boxes, ndim)``.
    """
    shapes = np.asarray(shapes, dtype=np.float64)
    if padding is not None:
        shapes = shapes + np.asarray(padding, dtype=np.float64)

    volumes = np.prod(shapes, axis=-1)
    effective = np.maximum(n_sat, volumes)
    return effective * np.log1p(effective)


def _schedule_uniform(
    shape: Tuple[int],
    max_workers: int,
    max_memory: int,
    padding: Tuple[int],
    memory_estimator: Callable,
    split_axes: Optional[Tuple[int]] = None,
    split_only_outer: bool = False,
    max_splits: int = 512,
    cost_fn: Callable = _box_cost,
    **kwargs,
) -> Tuple[Tuple[Tuple[slice, ...]], Tuple[int, int]]:
    """
    Search regular-grid splits of ``shape`` that fit the memory budget.

    Splits are explored by factorizing ``max_workers`` into outer/inner pairs
    and distributing the outer factor across ``split_axes``. The split with the
    lowest cost (and most balanced box shape on ties) is returned.

    Parameters
    ----------
    memory_estimator : Callable
        Partial of :py:meth:`estimate_memory_usage` with static parameters frozen.
    split_axes : tuple of int, optional
        Axes eligible for splitting. Defaults to all axes.
    split_only_outer : bool, default False
        If True, only consider ``(outer, inner) = (max_workers, 1)``.
    max_splits : int, default 512
        Upper bound on the number of boxes produced.

    Returns
    -------
    tuple of tuple of slice
        Per-box slices in the coordinates of ``shape``.
    tuple of int, int
        ``(n_outer_jobs, n_inner_workers)``.
    """
    from .matching_utils import split_shape

    core_assignments = [(1, max_workers)]
    if not split_only_outer:
        core_assignments = _factorize(max_workers, 2)

    if split_axes is None:
        split_axes = tuple(range(len(shape)))
    split_axes = sorted(split_axes, key=lambda x: shape[x], reverse=True)

    min_balance, min_score, min_param = float("inf"), float("inf"), ((), None)
    for inner_cores, outer_cores in core_assignments:

        # Create possible splits given the current factorization
        split_factors = []
        for base_factor in _factorize(outer_cores, len(shape)):
            multipliers = [1] * len(shape)
            base_n_splits = np.prod(base_factor)

            while (base_n_splits * np.prod(multipliers)) <= max_splits:
                new_factor = [x * y for x, y in zip(base_factor, multipliers)]

                # Split the largest axis
                _, ax = max([(shape[ax] / new_factor[ax], ax) for ax in split_axes])
                multipliers[ax] += 1
                split_factors.append(new_factor)

        for split_factor in split_factors:
            n_splits = np.prod(split_factor)
            assignment, split_factor = {}, sorted(split_factor, reverse=True)
            for index, axis in enumerate(split_axes):
                assignment[axis] = split_factor[index]

            if np.prod(list(assignment.values())) != n_splits:
                continue

            splits = split_shape(shape=shape, splits=assignment)
            widths = [tuple(x.stop - x.start for x in split) for split in splits]
            mem_usage = [
                memory_estimator(
                    shape1=tuple(sum(x) for x in zip(shp, padding)),
                    shape2=padding,
                    ncores=inner_cores,
                )
                for shp in widths
            ]
            max_usage = max(
                np.sum(mem_usage[i : i + outer_cores])
                for i in range(0, len(mem_usage), outer_cores)
            )
            if max_usage > max_memory:
                continue

            # Prefer boxes with more unifom dimensions
            mean_box_dim = np.mean(widths[0]) if widths else 0
            balance = np.sum((np.array(widths[0]) - mean_box_dim) ** 2)

            score = _select_schedule(splits, n_splits // outer_cores, cost_fn=cost_fn)
            if score < min_score or (score == min_score and balance < min_balance):
                min_score = score
                min_balance = balance
                min_param = (splits, (outer_cores, inner_cores))

    return min_param


def _bounding_box(segmentation: NDArray, threshold: float = 0) -> Tuple[slice, ...]:
    """Compute tight bounding box around nonzero regions."""
    mask = segmentation > threshold
    ndims = list(range(mask.ndim))
    mask = [mask.any(axis=tuple(j for j in ndims if j != i)) for i in ndims]

    # Handle empty masks
    if not mask[0].any():
        return (slice(0, 0),) * len(mask)

    starts = np.array([np.argmax(x) for x in mask], dtype=np.int32)
    stops = np.array([x.size - np.argmax(x[::-1]) for x in mask], dtype=np.int32)
    return tuple(slice(int(x), int(y)) for x, y in zip(starts, stops))


def _schedule_subdivide(
    max_workers: int,
    mask: NDArray,
    min_box_size: Optional[int] = 32,
    equal_shape: bool = False,
    verbose: bool = False,
    cost_fn: Callable = _box_cost,
    **kwargs,
) -> Tuple[Tuple[Tuple[slice, ...]], Tuple[int, int]]:
    """
    Cover ``mask`` with bounding boxes via recursive bisection.

    Parameters
    ----------
    max_workers : int
        Reported back as the outer-parallelism count in the returned schedule.
    mask : NDArray
        Binary mask indicating regions of interest.
    min_box_size : int, optional
        Minimum extent along any axis for a candidate box, defaults to 32.
        ``None`` disables the check.
    equal_shape : bool, default False
        If True, force every output box to share one shape (see
        :func:`tme.utils.subdivide.solve_subdivide`).
    verbose : bool, default False
        Print solver statistics.
    cost_fn : Callable
        Per-box cost function passed through to the solver.

    Returns
    -------
    tuple of tuple of slice
        Boxes covering ``mask`` in its own coordinates.
    tuple of int, int
        ``(max_workers, 1)`` (all parallelism is outer).
    """
    boxes, cost = solve_subdivide(
        mask=mask,
        min_box_size=min_box_size,
        equal_shape=equal_shape,
        return_slices=True,
        verbose=verbose,
        cost_fn=cost_fn,
    )
    return boxes, (max_workers, 1)


def _evaluate_config(solver: Callable, valid_box_sizes, max_workers, cost_fn):
    best_boxes, _ = solver(valid_box_sizes=valid_box_sizes)
    n_inits = int(np.ceil(len(best_boxes) / max_workers))
    score = _select_schedule(best_boxes, n_inits, cost_fn=cost_fn)
    return best_boxes, score


def _select_schedule(
    boxes: Tuple[Tuple[slice, ...]],
    n_inits: int,
    cost_fn: Callable = _box_cost,
) -> float:
    """
    Computes the computational complexity for a given schedule configuration.

    Parameters
    ----------
    boxes : tuple of tuple of slice
        Tuple of box slices for each split.
    n_inits : int
        Number of sequential initializations (rate rotations are recomputed).
    cost_fn : callable, optional
        Function mapping per-box shapes to cost. Defaults to :func:`_box_cost`.

    Returns
    -------
    float
        Computational complexity score (lower is better).
    """
    if len(boxes) == 0:
        return float("inf")

    ndim = len(boxes[0])
    overhead_complexity = float(cost_fn((0,) * ndim)) * int(n_inits)
    shapes = np.array([_slice_to_shape(box) for box in boxes], dtype=np.float64)
    return float(np.sum(cost_fn(shapes))) + overhead_complexity
