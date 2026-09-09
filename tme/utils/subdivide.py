"""
Greedy top-down subdivision solver for mask coverage in voxel space.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import List, Tuple, Optional, Union, Callable

import numpy as np

from ..types import NDArray

__all__ = ["solve_subdivide"]

CandidateType = Tuple[Tuple[int, ...], Tuple[int, ...]]
Candidates = List[CandidateType]


def _default_cost(shapes):
    """``V * log1p(V)`` over per-box shapes; rough proxy for FFT cost."""
    shapes = np.asarray(shapes)
    volumes = np.prod(shapes, axis=-1)
    return volumes * np.log1p(volumes)


def _to_slice(box: CandidateType) -> Tuple[slice, ...]:
    """Convert a ``(shape, pos)`` box to per-axis ``slice`` objects."""
    shape, pos = box
    return tuple(slice(p, p + s) for p, s in zip(pos, shape))


def _bbox_of_coords(coords: np.ndarray) -> Optional[CandidateType]:
    """Tight ``(shape, pos)`` bbox of an ``(N, ndim)`` coord array, or ``None`` if empty."""
    if coords.shape[0] == 0:
        return None

    starts = coords.min(axis=0)
    stops = coords.max(axis=0) + 1
    pos = tuple(int(x) for x in starts)
    shape = tuple(int(y - x) for y, x in zip(stops, starts))
    return (shape, pos)


def _scalar_cost(box: CandidateType, cost_fn: Callable) -> float:
    """``cost_fn`` evaluated on a single box, returned as a Python float."""
    shape = np.asarray(box[0]).reshape(1, -1)
    return float(np.asarray(cost_fn(shape)).reshape(-1)[0])


def _best_axis_split(
    box_coords: np.ndarray,
    cost_fn: Callable,
    min_box_size: Optional[int],
    equal_shape: bool = False,
) -> Tuple[
    float,
    Optional[Tuple[Tuple[CandidateType, np.ndarray], Tuple[CandidateType, np.ndarray]]],
]:
    """
    Best bipartition of ``box_coords`` minimizing the combined child cost.

    Parameters
    ----------
    box_coords : np.ndarray
        ``(N, ndim)`` integer coords of active voxels in the parent box.
    cost_fn : callable
        Maps an ``(n_boxes, ndim)`` array of shapes to a per-box cost vector.
    min_box_size : int, optional
        Minimum extent along any axis for a candidate child.
    equal_shape : bool, default False
        Force candidates to share the per-axis max of their tight boxes.

    Returns
    -------
    best_sum : float
        Combined cost of the best split, or ``inf`` if none is valid.
    best_children : tuple, optional
        ``((L_box, L_coords), (R_box, R_coords))`` for the optimal split,
        or ``None`` if no valid split exists.
    """
    M, ndim = box_coords.shape
    best_sum, best_children = float("inf"), None

    if M < 2:
        return best_sum, best_children

    for axis in range(ndim):
        # Sort so prefix/suffix slices line up with the split plane along axis
        order = np.argsort(box_coords[:, axis])
        sorted_coords = box_coords[order]
        coord_a = sorted_coords[:, axis]

        # Forward sweep cum_*[i] = bbox of sorted_coords[:i+1] (the L side)
        cum_min = np.minimum.accumulate(sorted_coords, axis=0)
        cum_max = np.maximum.accumulate(sorted_coords, axis=0)

        # Reverse sweep rev_cum_*[i] = bbox of sorted_coords[i:] (the R side)
        rev = sorted_coords[::-1]
        rev_cum_min = np.minimum.accumulate(rev, axis=0)[::-1]
        rev_cum_max = np.maximum.accumulate(rev, axis=0)[::-1]

        # Split "after index i" with i in [0..M-2]: L = sorted[:i+1],
        # R = sorted[i+1:]. The [:-1] / [1:] slicing excludes the degenerate
        # "everything on one side" positions; both halves are always non-empty
        L_shapes = cum_max[:-1] - cum_min[:-1] + 1
        R_shapes = rev_cum_max[1:] - rev_cum_min[1:] + 1

        # Disallow splits between voxels sharing coord_a, as the L max and R min
        # would coincide on the split axis
        valid = coord_a[:-1] < coord_a[1:]
        if equal_shape:
            child_shapes = np.maximum(L_shapes, R_shapes)
            if min_box_size is not None:
                valid = valid & (child_shapes.min(axis=1) >= min_box_size)
            if not valid.any():
                continue
            costs = np.asarray(cost_fn(child_shapes), dtype=np.float32)
            totals = 2.0 * costs
        else:
            if min_box_size is not None:
                valid = (
                    valid
                    & (L_shapes.min(axis=1) >= min_box_size)
                    & (R_shapes.min(axis=1) >= min_box_size)
                )
            if not valid.any():
                continue
            L_costs = np.asarray(cost_fn(L_shapes), dtype=np.float32)
            R_costs = np.asarray(cost_fn(R_shapes), dtype=np.float32)
            totals = L_costs + R_costs

        # Mask invalid candidates to +inf rather than filtering, so the argmin
        # index stays aligned with cum_*/L_shapes/R_shapes for reconstruction.
        totals = np.where(valid, totals, np.inf)
        i_best = int(np.argmin(totals))
        if totals[i_best] < best_sum:
            best_sum = float(totals[i_best])
            # Each child is rebuilt from cum_min/cum_max, i.e. the bbox of its
            # active mass, making parent and child minimal for their voxel sets
            L_pos = tuple(int(x) for x in cum_min[i_best])
            R_pos = tuple(int(x) for x in rev_cum_min[i_best + 1])
            if equal_shape:
                shape_tuple = tuple(int(x) for x in child_shapes[i_best])
                L_box = (shape_tuple, L_pos)
                R_box = (shape_tuple, R_pos)
            else:
                L_box = (tuple(int(x) for x in L_shapes[i_best]), L_pos)
                R_box = (tuple(int(x) for x in R_shapes[i_best]), R_pos)
            best_children = (
                (L_box, sorted_coords[: i_best + 1]),
                (R_box, sorted_coords[i_best + 1 :]),
            )
    return best_sum, best_children


def _pad_boxes_to_global_shape(
    boxes: Candidates, mask_shape: Tuple[int, ...]
) -> Candidates:
    """
    Pad every box to the per-axis max shape across ``boxes`` (clipped to
    ``mask_shape``), shifting each anchor to fit inside the mask.

    Padded boxes may overlap neighbors but always stay inside ``mask_shape``
    and still contain the original box's content.
    """
    if not boxes:
        return boxes

    shapes_arr = np.array([b[0] for b in boxes], dtype=np.int32)
    global_shape = tuple(
        int(min(int(s), int(m))) for s, m in zip(shapes_arr.max(axis=0), mask_shape)
    )
    padded: Candidates = []
    for _, pos in boxes:
        new_pos = tuple(
            int(max(0, min(int(p), int(m) - int(g))))
            for p, g, m in zip(pos, global_shape, mask_shape)
        )
        padded.append((global_shape, new_pos))
    return padded


def solve_subdivide(
    mask: NDArray,
    return_slices: bool = True,
    min_box_size: Optional[int] = None,
    equal_shape: bool = False,
    verbose: bool = False,
    cost_fn: Callable = _default_cost,
    **kwargs,
) -> Tuple[Union[Candidates, List[Tuple[slice, ...]]], Optional[float]]:
    """
    Cover a binary mask by recursive bisection in voxel space.

    Parameters
    ----------
    mask : NDArray
        Binary mask of regions to cover.
    return_slices : bool, default True
        If True, return boxes as tuples of ``slice`` objects; otherwise as
        ``(shape, position)`` tuples.
    min_box_size : int, optional
        Minimum extent along any axis for a candidate child. ``None``
        (default) disables the check.
    equal_shape : bool, default False
        If True, force every output box to share one shape (see
        :func:`_pad_boxes_to_global_shape`).
    verbose : bool, default False
        Unused; kept for API symmetry with other solvers.
    cost_fn : callable, optional
        Maps an ``(n_boxes, ndim)`` array of shapes to a per-box cost
        vector. Defaults to :func:`_default_cost`.

    Returns
    -------
    boxes : list of slice tuples or list of ``(shape, pos)`` tuples
        Boxes covering ``mask``. Empty tuple for an empty mask.
    total_cost : float or None
        Sum of ``cost_fn`` over the boxes, or ``None`` for an empty mask.
    """
    coords = np.argwhere(np.asarray(mask)).astype(np.int32)
    if coords.shape[0] == 0:
        return (), None

    root = _bbox_of_coords(coords)
    if root is None:
        raise ValueError("Could not determine bounding box from coordinates.")

    output: Candidates = []
    queue: List[Tuple[CandidateType, np.ndarray]] = [(root, coords)]
    while queue:
        box, box_coords = queue.pop()
        parent_cost = _scalar_cost(box, cost_fn)

        best_sum, best_children = _best_axis_split(
            box_coords, cost_fn, min_box_size, equal_shape=equal_shape
        )
        if best_children is None or best_sum >= parent_cost:
            output.append(box)
            continue

        queue.extend(best_children)

    if equal_shape:
        output = _pad_boxes_to_global_shape(output, tuple(mask.shape))

    total_cost = sum(_scalar_cost(b, cost_fn) for b in output)
    if return_slices:
        return [_to_slice(b) for b in output], total_cost
    return output, total_cost
