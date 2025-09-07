"""
Utility functions for generating template matching masks.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import numpy as np
from typing import Tuple, Optional

from .types import NDArray
from scipy.ndimage import gaussian_filter
from .matching_utils import _rigid_transform

__all__ = ["elliptical_mask", "tube_mask", "box_mask", "membrane_mask"]


def elliptical_mask(
    shape: Tuple[int],
    radius: Tuple[float],
    center: Optional[Tuple[float]] = None,
    orientation: Optional[NDArray] = None,
    sigma_decay: float = 0.0,
    cutoff_sigma: float = 3,
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
    if sigma_decay > 0:
        sigma_decay = 2 * (sigma_decay / np.mean(radius)) ** 2
        mask = np.maximum(0, dist - 1)
        mask = np.exp(-(mask**2) / sigma_decay)
        mask *= mask > np.exp(-(cutoff_sigma**2) / 2)
    else:
        mask = (dist <= 1).astype(int)
    return mask


def box_mask(
    shape: Tuple[int],
    center: Tuple[int],
    size: Tuple[int],
    sigma_decay: float = 0.0,
    cutoff_sigma: float = 3.0,
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

    out = np.zeros(shape)
    out[slice_indices] = 1

    if sigma_decay > 0:
        mask_filter = gaussian_filter(
            out.astype(np.float32), sigma=sigma_decay, truncate=cutoff_sigma
        )
        out = np.add(out, (1 - out) * mask_filter)
        out *= out > np.exp(-(cutoff_sigma**2) / 2)
    return out


def tube_mask(
    shape: Tuple[int],
    symmetry_axis: int,
    center: Tuple[int],
    inner_radius: float,
    outer_radius: float,
    height: int,
    sigma_decay: float = 0.0,
    cutoff_sigma: float = 3.0,
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
            sigma_decay=sigma_decay,
            cutoff_sigma=cutoff_sigma,
        )
    if outer_radius > 0:
        outer_circle = elliptical_mask(
            shape=circle_shape,
            radius=outer_radius,
            center=circle_center,
            sigma_decay=sigma_decay,
            cutoff_sigma=cutoff_sigma,
        )
    circle = outer_circle - inner_circle
    circle = np.expand_dims(circle, axis=symmetry_axis)

    center = center[symmetry_axis]
    start_idx = int(center - height // 2)
    stop_idx = int(center + height // 2 + height % 2)

    start_idx, stop_idx = max(start_idx, 0), min(stop_idx, shape[symmetry_axis])

    slice_indices = tuple(
        slice(None) if i != symmetry_axis else slice(start_idx, stop_idx)
        for i in range(len(shape))
    )
    tube = np.zeros(shape)
    tube[slice_indices] = circle

    return tube


def membrane_mask(
    shape: Tuple[int],
    radius: float,
    thickness: float,
    separation: float,
    symmetry_axis: int = 2,
    center: Optional[Tuple[float]] = None,
    sigma_decay: float = 0.5,
    cutoff_sigma: float = 3,
    **kwargs,
) -> NDArray:
    """
    Creates a membrane mask consisting of two parallel disks with Gaussian intensity profile.
    Uses efficient broadcasting approach: flat disk mask × height profile.

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
        Center of the membrane (midpoint between the two disks), defaults to shape // 2.
    sigma_decay : float, optional
        Controls edge sharpness relative to radius, defaults to 0.5.
    cutoff_sigma : float, optional
        Cutoff for height profile in standard deviations, defaults to 3.

    Returns
    -------
    NDArray
        The created membrane mask with Gaussian intensity profile.

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
        sigma_decay=sigma_decay,
        cutoff_sigma=cutoff_sigma,
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
