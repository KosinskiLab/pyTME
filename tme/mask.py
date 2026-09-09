"""
Utility functions for generating template matching masks.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import numpy as np
from typing import Tuple, Optional

from .types import NDArray
from .matching_utils import _rigid_transform

__all__ = [
    "soft_edge",
    "elliptical_mask",
    "tube_mask",
    "box_mask",
    "membrane_mask",
    "threshold_mask",
]


def soft_edge(
    mask: NDArray,
    soft_edge_width: float = 0,
    extend: float = 0,
    method: str = "gaussian",
    cutoff_sigma: float = 3.0,
    **kwargs,
) -> NDArray:
    """
    Optionally dilate a binary mask, then apply a distance-based soft edge.

    Parameters
    ----------
    mask : NDArray
        Binary (or near-binary) N-dimensional array. Values > 0.5 are
        treated as interior.
    soft_edge_width : float, optional
        Falloff extent in voxels. For ``"gaussian"`` this is the sigma;
        for ``"cosine"`` it is the full width of the cosine bell.
        If <= 0, no soft edge is applied.
    extend : float, optional
        Isotropic dilation in voxels applied before the soft edge.
    method : str, optional
        ``"gaussian"`` (default) or ``"cosine"`` (RELION-style).
    cutoff_sigma : float, optional
        Number of standard deviations at which the Gaussian is truncated
        to zero.  Ignored when *method* is ``"cosine"``.

    Returns
    -------
    NDArray
        Float array with the same shape as *mask*. Interior voxels are
        1.0; exterior voxels decay from 1 to 0 as a function of their
        Euclidean distance to the nearest interior voxel.
    """
    from scipy.ndimage import distance_transform_edt

    binary = np.asarray(mask) > 0.5

    if extend > 0:
        dist_out = distance_transform_edt(~binary)
        binary = dist_out <= extend

    if soft_edge_width <= 0:
        return binary.astype(np.float32)

    dist = distance_transform_edt(~binary)

    if method == "cosine":
        max_dist = soft_edge_width
    elif method == "gaussian":
        max_dist = cutoff_sigma * soft_edge_width
    else:
        raise ValueError(f"method must be 'gaussian' or 'cosine', got '{method}'")

    out = binary.astype(np.float32)
    transition = (~binary) & (dist <= max_dist)

    if method == "cosine":
        out[transition] = 0.5 * (
            1.0 + np.cos(np.pi * dist[transition] / soft_edge_width)
        )
    else:
        out[transition] = np.exp(-(dist[transition] ** 2) / (2.0 * soft_edge_width**2))

    return out


def threshold_mask(
    data: NDArray,
    threshold: float,
    **kwargs,
) -> NDArray:
    """
    Create a mask by binarising a density map at *threshold*.

    Dilation and soft edge are handled by :func:`soft_edge` via
    ``extend`` and ``soft_edge_width`` in *kwargs*.

    Parameters
    ----------
    data : NDArray
        Input density map.
    threshold : float
        Voxels with values >= *threshold* are set to 1.
    **kwargs
        Forwarded to :func:`soft_edge` (``soft_edge_width``, ``extend``,
        ``method``, ``cutoff_sigma``).

    Returns
    -------
    NDArray
        The created mask.
    """
    return soft_edge(np.asarray(data) >= threshold, **kwargs)


def elliptical_mask(
    shape: Tuple[int],
    radius: Tuple[float],
    center: Optional[Tuple[float]] = None,
    orientation: Optional[NDArray] = None,
    **kwargs,
) -> NDArray:
    """
    Creates an ellipsoidal mask.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the mask to be created.
    radius : tuple of floats
        Radius of the mask.
    center : tuple of floats, optional
        Center of the mask, default to shape // 2.
    orientation : NDArray, optional.
        Orientation of the mask as rotation matrix with shape (d,d).
    **kwargs
        Forwarded to :func:`soft_edge` (``soft_edge_width``, ``method``,
        ``cutoff_sigma``).

    Returns
    -------
    NDArray
        The created ellipsoidal mask.

    Raises
    ------
    ValueError
        If the length of center and radius is not one or the same as shape.

    Examples
    --------
    >>> from tme.matching_utils import elliptical_mask
    >>> mask = elliptical_mask(shape=(20,20), radius=(5,5), center=(10,10))
    """
    shape, radius = np.asarray(shape), np.asarray(radius)

    shape = shape.astype(int)
    if center is None:
        center = np.divide(shape, 2).astype(int)

    center = np.asarray(center, dtype=np.float32)
    radius = np.repeat(radius, shape.size // radius.size)
    center = np.repeat(center, shape.size // center.size)
    if radius.size != shape.size:
        raise ValueError("Length of radius has to be either one or match shape.")
    if center.size != shape.size:
        raise ValueError("Length of center has to be either one or match shape.")

    n = shape.size
    center = center.reshape((-1,) + (1,) * n)
    radius = radius.reshape((-1,) + (1,) * n)

    indices = np.indices(shape, dtype=np.float32) - center
    if orientation is not None:
        return_shape = indices.shape
        indices = indices.reshape(n, -1)
        _rigid_transform(
            coordinates=indices,
            rotation_matrix=np.asarray(orientation),
            out=indices,
            translation=np.zeros(n),
            use_geometric_center=False,
        )
        indices = indices.reshape(*return_shape)

    dist = np.linalg.norm(indices / radius, axis=0)
    mask = (dist <= 1).astype(np.float32)

    return soft_edge(mask, **kwargs)


def box_mask(
    shape: Tuple[int],
    center: Tuple[int],
    size: Tuple[int],
    **kwargs,
) -> np.ndarray:
    """
    Creates a box mask centered around the provided center point.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the output array.
    center : tuple of ints
        Center point coordinates of the box.
    size : tuple of ints
        Side length of the box along each axis.
    **kwargs
        Forwarded to :func:`soft_edge` (``soft_edge_width``, ``method``,
        ``cutoff_sigma``).

    Returns
    -------
    NDArray
        The created box mask.

    Raises
    ------
    ValueError
        If ``shape`` and ``center`` do not have the same length.
        If ``center`` and ``height`` do not have the same length.
    """
    if len(shape) != len(center) or len(center) != len(size):
        raise ValueError("The length of shape, center, and height must be consistent.")

    shape = tuple(int(x) for x in shape)
    center, size = np.array(center, dtype=int), np.array(size, dtype=int)

    half_heights = size // 2
    starts = np.maximum(center - half_heights, 0)
    stops = np.minimum(center + half_heights + np.mod(size, 2) + 1, shape)
    slice_indices = tuple(slice(*coord) for coord in zip(starts, stops))

    out = np.zeros(shape, dtype=np.float32)
    out[slice_indices] = 1

    return soft_edge(out, **kwargs)


def tube_mask(
    shape: Tuple[int],
    symmetry_axis: int,
    center: Tuple[int],
    inner_radius: float,
    outer_radius: float,
    height: int,
    **kwargs,
) -> NDArray:
    """
    Creates a tube mask.

    Parameters
    ----------
    shape : tuple
        Shape of the mask to be created.
    symmetry_axis : int
        The axis of symmetry for the tube.
    base_center : tuple
        Center of the tube.
    inner_radius : float
        Inner radius of the tube.
    outer_radius : float
        Outer radius of the tube.
    height : int
        Height of the tube.
    **kwargs
        Forwarded to :func:`soft_edge` (``soft_edge_width``, ``method``,
        ``cutoff_sigma``).

    Returns
    -------
    NDArray
        The created tube mask.

    Raises
    ------
    ValueError
        If ``inner_radius`` is larger than ``outer_radius``.
        If ``height`` is larger than the symmetry axis.
        If ``base_center`` and ``shape`` do not have the same length.
    """
    if inner_radius > outer_radius:
        raise ValueError("inner_radius should be smaller than outer_radius.")

    if height > shape[symmetry_axis]:
        raise ValueError(f"Height can be no larger than {shape[symmetry_axis]}.")

    if symmetry_axis > len(shape):
        raise ValueError(f"symmetry_axis can be not larger than {len(shape)}.")

    if len(center) != len(shape):
        raise ValueError("shape and base_center need to have the same length.")

    shape = tuple(int(x) for x in shape)
    circle_shape = tuple(b for ix, b in enumerate(shape) if ix != symmetry_axis)
    circle_center = tuple(b for ix, b in enumerate(center) if ix != symmetry_axis)

    inner_circle = np.zeros(circle_shape)
    outer_circle = np.zeros_like(inner_circle)
    if inner_radius > 0:
        inner_circle = elliptical_mask(
            shape=circle_shape,
            radius=inner_radius,
            center=circle_center,
        )
    if outer_radius > 0:
        outer_circle = elliptical_mask(
            shape=circle_shape,
            radius=outer_radius,
            center=circle_center,
        )
    circle = outer_circle - inner_circle
    circle = np.expand_dims(circle, axis=symmetry_axis)

    center = center[symmetry_axis]
    start_idx = int(center - height // 2)
    stop_idx = int(center + height // 2 + height % 2)
    start_idx, stop_idx = max(start_idx, 0), min(stop_idx, shape[symmetry_axis])

    height_profile = np.zeros(shape[symmetry_axis], dtype=np.float32)
    height_profile[start_idx:stop_idx] = 1.0
    rshape = tuple(shape[i] if i == symmetry_axis else 1 for i in range(len(shape)))

    mask = circle * height_profile.reshape(rshape)

    return soft_edge(mask, **kwargs)


def membrane_mask(
    shape: Tuple[int],
    radius: float,
    thickness: float,
    separation: float,
    symmetry_axis: int = 2,
    center: Optional[Tuple[float]] = None,
    cutoff_sigma: float = 3,
    soft_edge_width: float = 0.5,
    method: str = "gaussian",
    **kwargs,
) -> NDArray:
    """
    Creates a membrane mask consisting of two parallel disks with
    Gaussian leaflet intensity profiles.

    The disk boundary uses :func:`soft_edge` (via ``soft_edge_width``
    and ``method``). The axial leaflet profiles are physical Gaussians
    controlled by *thickness* and are not affected by the soft-edge
    parameters. *cutoff_sigma* applies to both the disk edge and the
    leaflet profile truncation.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the mask to be created.
    radius : float
        Radius of the membrane disks.
    thickness : float
        Thickness of each disk in the membrane.
    separation : float
        Distance between the centers of the two disks.
    symmetry_axis : int, optional
        The axis perpendicular to the membrane disks, defaults to 2.
    center : tuple of floats, optional
        Center of the membrane (midpoint between the two disks),
        defaults to shape // 2.
    cutoff_sigma : float, optional
        Truncation threshold in standard deviations, applied to both
        the disk soft edge and the leaflet height profile, defaults to 3.
    soft_edge_width : float, optional
        Soft-edge width in voxels for the disk boundary, defaults to 0.5.
    method : str, optional
        Soft-edge method for disk boundary: ``"gaussian"`` (default) or
        ``"cosine"``.

    Returns
    -------
    NDArray
        The created membrane mask.

    Raises
    ------
    ValueError
        If ``thickness`` is negative.
        If ``separation`` is negative.
        If ``center`` and ``shape`` do not have the same length.
        If ``symmetry_axis`` is out of bounds.

    Examples
    --------
    >>> from tme.matching_utils import membrane_mask
    >>> mask = membrane_mask(shape=(50,50,50), radius=10, thickness=2, separation=15)
    """
    shape = np.asarray(shape, dtype=int)

    if center is None:
        center = np.divide(shape, 2).astype(float)

    center = np.asarray(center, dtype=np.float32)
    center = np.repeat(center, shape.size // center.size)

    if thickness < 0:
        raise ValueError("thickness must be non-negative.")
    if separation < 0:
        raise ValueError("separation must be non-negative.")
    if symmetry_axis >= len(shape):
        raise ValueError(f"symmetry_axis must be less than {len(shape)}.")
    if center.size != shape.size:
        raise ValueError("Length of center has to be either one or match shape.")

    disk_mask = elliptical_mask(
        shape=[x for i, x in enumerate(shape) if i != symmetry_axis],
        radius=radius,
        soft_edge_width=soft_edge_width,
        cutoff_sigma=cutoff_sigma,
        method=method,
    )

    axial_coord = np.arange(shape[symmetry_axis]) - center[symmetry_axis]
    height_profile = np.zeros((shape[symmetry_axis],), dtype=np.float32)
    for leaflet_pos in [-separation / 2, separation / 2]:
        leaflet_profile = np.exp(
            -((axial_coord - leaflet_pos) ** 2) / (2 * (thickness / 3) ** 2)
        )
        cutoff_threshold = np.exp(-(cutoff_sigma**2) / 2)
        leaflet_profile *= leaflet_profile > cutoff_threshold

        height_profile = np.maximum(height_profile, leaflet_profile)

    disk_mask = disk_mask.reshape(
        [x if i != symmetry_axis else 1 for i, x in enumerate(shape)]
    )
    height_profile = height_profile.reshape(
        [1 if i != symmetry_axis else x for i, x in enumerate(shape)]
    )

    return disk_mask * height_profile
