"""
Heuristic solver for the set cover problem.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from time import time
from typing import List, Tuple, Dict, Optional, Union

import numpy as np
from scipy.sparse import coo_matrix
from scipy.optimize import milp, LinearConstraint, Bounds

from ..types import NDArray

__all__ = ["solve", "solve_hierarchical"]

CandidateType = Tuple[Tuple[int, ...], Tuple[int, ...]]
Candidates = List[CandidateType]


_ACTIVE_LEAVES_CACHE = {}
_CACHE_ENABLED = False


def _set_cache(enabled: bool = False):
    """
    Enable or disable caching for :py:meth:`compute_active_leaves`.

    Parameters
    ----------
    enabled : bool, default False
        If True, enables caching. If False, disables caching and clears cache.
    """
    global _CACHE_ENABLED
    if enabled:
        _CACHE_ENABLED = True
    else:
        _CACHE_ENABLED = False
        _ACTIVE_LEAVES_CACHE.clear()


def compute_active_leaves(data, leaf_size: int) -> set:
    """
    Compute active leaves with optional caching.

    Parameters
    ----------
    data : NDArray
        Binary mask array
    leaf_size : int
        Size of each leaf cell

    Returns
    -------
    tuple
        Tuple of active leaf coordinates
    """
    # I promise
    cache_key = (id(data), leaf_size)
    if _CACHE_ENABLED and cache_key in _ACTIVE_LEAVES_CACHE:
        return _ACTIVE_LEAVES_CACHE[cache_key]

    shape = data.shape
    new_shape = tuple(int(np.ceil(dim / leaf_size)) for dim in shape)
    dim_slices = [
        [slice(i * leaf_size, min((i + 1) * leaf_size, dim)) for i in range(n)]
        for n, dim in zip(new_shape, shape)
    ]

    active_leaves = set()
    for leaf_idx in np.ndindex(new_shape):
        slices = tuple(dim_slices[axis][idx] for axis, idx in enumerate(leaf_idx))
        if data[slices].any():
            active_leaves.add(leaf_idx)

    ret = tuple(active_leaves)
    if _CACHE_ENABLED:
        _ACTIVE_LEAVES_CACHE[cache_key] = ret
    return ret


def _scale(box: CandidateType, scale_factor: float) -> CandidateType:
    return (
        tuple(int(x * scale_factor) for x in box[0]),
        tuple(int(x * scale_factor) for x in box[1]),
    )


def _to_slice(box: CandidateType) -> Tuple[slice, ...]:
    shape, pos = box
    stop = tuple(sum(x) for x in zip(shape, pos))
    return tuple(slice(*x) for x in zip(pos, stop))


def _solve_ilp(
    active_leaves: NDArray,
    candidates: Candidates,
    padding: Tuple[float, ...],
    milp_options: Dict = {},
    leaf_size: int = 1,
) -> Tuple[Candidates, float]:
    """
    Solve the minimum volume box covering problem using integer linear programming.

    Parameters
    ----------
    active_leaves : NDArray
        Octree leaf coordinates (N, ndim) that must be covered.
    candidates : list of tuple
        Candidate box placements, each as (box_size, box_position) in leaf coordinates.
    padding : tuple of float
        Padding to apply in each dimension in leaf units.
    milp_options : dict, optional
        Options passed to :func:`scipy.optimize.milp`.
    leaf_size : int, optional
        Leaf size for scaling the objective to enable comparison across granularities.

    Returns
    -------
    list of tuple
        Boxes selected by solver, each as (shape, position) in leaf coordinates.
    float
        Value of the objective function for the solution.

    Notes
    -----
    The objective function per box is (V + P) · log(1 + V + P) where V is the
    box volume in voxels and P is the padding volume. The log term encourages
    solutions with fewer boxes.
    """
    from ..extensions import setup_vertex_cover

    # Sparse constraint matrix A[leaf_idx, box_idx] = 1 if box covers leaf
    n_leaves, n_candidates = len(active_leaves), len(candidates)
    rows, cols, objective = setup_vertex_cover(
        candidates, active_leaves, padding, leaf_size
    )

    A = coo_matrix(
        (np.ones(len(rows), dtype=np.int8), (rows, cols)),
        shape=(n_leaves, n_candidates),
        dtype=np.int8,
    ).tocsr()

    # Each active leaf must be covered by at least one box
    b_lower = np.ones(n_leaves)
    b_upper = np.full(n_leaves, np.inf)
    constraints = LinearConstraint(A, lb=b_lower, ub=b_upper)

    result = milp(
        c=objective,
        constraints=constraints,
        integrality=np.ones(n_candidates),
        bounds=Bounds(lb=0, ub=1),
        options=milp_options,
    )
    return [candidates[i] for i, val in enumerate(result.x) if val > 0.5], result.fun


def solve(
    mask: NDArray,
    leaf_size: int,
    valid_box_sizes: Tuple[Tuple[int, ...]],
    return_slices: bool = True,
    tile_candidates: bool = False,
    padding: Optional[Tuple[int, ...]] = None,
    candidates: Optional[Candidates] = None,
    jitter_margin: int = 1,
    jitter_stride: int = 1,
) -> Union[Candidates, List[slice]]:
    """
    Cover a binary mask with minimum-volume boxes using ILP.

    Divides mask into leaf_size^n voxel cells. Each occupied cell must be
    covered by at least one box. Smaller leaf_size gives tighter fits but slower
    optimization.

    Parameters
    ----------
    mask : NDArray
        Binary mask mask.
    leaf_size : int
        Octree leaf size in voxels (8, 16, 32, or 64 typical).
    valid_box_sizes : tuple of tuple of int
        Allowed box dimensions in voxels (e.g., FFT-friendly sizes).
    return_slices : bool
        Return slices (True) or (shape, position) tuples (False), default True.
    tile_candidates : bool
        Put boxes at every leaf (False) or using box-sized steps (True), default False.
    padding : tuple of int, optional
        Voxel padding per dimension.
    candidates : list of tuple, optional
        Pre-computed boxes as (shape, position) in leaf coordinates.
    jitter_margin : int
        Search radius in leaf units, default 1.
    jitter_stride : int
        Search step size in leaf units. Larger values = fewer candidates,
        faster solve, default 1.

    Returns
    -------
    list of slice or list of tuple
        Selected boxes as slices (if return_slices=True) or (shape, position) tuples.
    float
        Objective function value.

    Examples
    --------
    >>> mask = np.zeros((400, 600), dtype=bool)
    >>> mask[50:150, 100:200] = True
    >>> boxes, score = solve(mask, leaf_size=16, valid_box_sizes=[(64, 64), (96, 96)])
    """
    from ..extensions import generate_candidates, jitter_candidates

    options = {
        "time_limit": 300.0,
    }

    if padding is None:
        padding = (0,) * mask.ndim
    padding = tuple(x / leaf_size for x in padding)

    active_leaves = compute_active_leaves(mask, leaf_size)
    if len(active_leaves) == 0:
        return [], None

    active_leaves = np.asarray(active_leaves, dtype=np.int32)
    if candidates is None:
        unique_boxes, filtered_boxes = set(), []
        for box in valid_box_sizes:
            new_box = tuple(int(x / leaf_size) for x in box)
            if new_box not in unique_boxes and all(x >= 1 for x in new_box):
                unique_boxes.add(new_box)
                filtered_boxes.append(box)

        # Place boxes at every grid position with leaf size spacing
        candidates = generate_candidates(
            mask.shape,
            valid_box_sizes=tuple(filtered_boxes),
            leaf_size=leaf_size,
            tile_candidates=tile_candidates,
        )

    if len(candidates) == 0:
        return [], None

    candidates = jitter_candidates(
        candidates, active_leaves, jitter_margin, max(jitter_stride, 1)
    )

    boxes, score = _solve_ilp(active_leaves, candidates, padding, options, leaf_size)

    boxes = [_scale(box, leaf_size) for box in boxes]
    if return_slices:
        boxes = [_to_slice(box) for box in boxes]
    return boxes, score


def solve_hierarchical(
    mask: NDArray,
    valid_box_sizes: Tuple[Tuple[int, ...]],
    padding: Optional[Tuple[int, ...]] = None,
    return_slices: bool = True,
    leaf_size: int = 64,
    k: int = 2,
    epsilon: float = 0.1,
    verbose: bool = False,
) -> Union[Candidates, List[slice]]:
    """
    Coarse-to-fine hierarchical solver that refines solutions across multiple scales.

    Parameters
    ----------
    mask : NDArray
        Binary mask mask.
    valid_box_sizes : tuple of tuple of int
        Allowed box dimensions in voxels.
    padding : tuple of int, optional
        Voxel padding per dimension.
    return_slices : bool, default True
        Return slices (True) or (shape, position) tuples (False), default True.
    leaf_size : int, default 32
        Octree voxel resolution. Will be halved iteratively until improvement falls
        below the epsilon threshold.
    k : int, default 2
        Box size multiplier for decomposition filtering.
    epsilon : float, default 0.1
        Relative improvement threshold for early stopping. Stops if
        (prev_score - score) / prev_score < epsilon.
    verbose : bool
        Print iteration statistics, default False.

    Returns
    -------
    list of slice or list of tuple
        Selected boxes as slices (if return_slices=True) or (shape, position) tuples.
    float
        Objective function value.
    """
    from ..extensions import decompose_boxes

    while leaf_size > min(mask.shape):
        leaf_size = leaf_size // 2

    # Required for active leaf cache
    valid_box_sizes = tuple(tuple(x) for x in valid_box_sizes)
    if verbose:
        print(f"{'Leaf Size':<16} {'Score':<16} {'Time (s)':<8} {'Delta':<8}")
        print("-" * 51)

    candidates, prev_score = None, float("nan")
    min_size = max(min(min(x) for x in valid_box_sizes) // 2, 1)
    while leaf_size >= min_size:
        start = time()
        if candidates is not None:
            relevant_boxes = _filter_boxes(valid_box_sizes, leaf_size, k=k)
            candidates = decompose_boxes(
                candidates,
                relevant_boxes,
                leaf_size=leaf_size,
                overlapping=False,
                nearest=False,
            )

        candidates, score = solve(
            mask,
            leaf_size=leaf_size,
            valid_box_sizes=valid_box_sizes,
            candidates=candidates,
            padding=padding,
            return_slices=False,
            tile_candidates=False,
        )
        if len(candidates) == 0:
            return candidates, None

        delta = (prev_score - score) / prev_score if prev_score == prev_score else 1
        if verbose:
            elapsed = time() - start
            print(f"{leaf_size:<16} {score:<16.2f} {elapsed:<8.2f} {delta:<8.2%}")

        if delta < epsilon:
            if verbose:
                print(f"\nSolution converged ({delta:.2%} < {epsilon:.2%}).\n")
            break
        prev_score, leaf_size = score, leaf_size // 2

    if return_slices:
        candidates = [_to_slice(box) for box in candidates]
    return candidates, prev_score


def _filter_boxes(
    valid_box_sizes: Tuple[Tuple[int, ...]], target_leaf_size: int, k: int = 1
) -> Tuple[Tuple[int, ...]]:
    """
    Return box sizes closest to the target leaf size.
    Use boxes in range [target_leaf_size, target_leaf_size * K]
    """
    min_size = target_leaf_size
    max_size = target_leaf_size * k

    filtered = tuple(
        size for size in valid_box_sizes if min_size <= min(size) <= max_size
    )
    if not filtered:
        filtered = (min(valid_box_sizes, key=lambda s: min(s)),)
    return filtered
