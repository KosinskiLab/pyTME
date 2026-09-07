"""
Implements class ReconstructFromTilt and ShiftFourier.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict, Optional, Literal
from dataclasses import dataclass

import numpy as np

from ..types import BackendArray
from ..backends import backend as be

from .compose import ComposableFilter
from ..rotations import euler_to_rotationmatrix
from ._utils import shift_fourier, create_reconstruction_filter, fftfreqn

__all__ = ["ReconstructFromTilt"]


@dataclass
class ReconstructFromTilt(ComposableFilter):
    """
    Tomographic reconstruction of 3D Fourier volume from 2D Fourier slices.
    """

    #: Angle of each individual tilt in degrees.
    angles: Optional[Tuple[float, ...]] = None
    #: Projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: Tilt axis, defaults to 0 (x).
    tilt_axis: int = 0
    #: Interpolation order used for method 'rotation'.
    interpolation_order: int = 1
    #: Filter window applied during reconstruction.
    reconstruction_filter: Optional[str] = None
    #: Reconstruction method: "rotation" or "gridding".
    method: Literal["rotation", "gridding"] = "gridding"

    @staticmethod
    def _evaluate(
        data: BackendArray,
        shape: Tuple[int, ...],
        angles: Tuple[float, ...],
        data_weights: Optional[BackendArray] = None,
        opening_axis: int = 2,
        tilt_axis: int = 0,
        interpolation_order: int = 1,
        reconstruction_filter: Optional[str] = None,
        method: Literal["rotation", "gridding"] = "gridding",
        **kwargs,
    ) -> Dict:
        """
        Reconstruct a 3D array from 2D inputs.

        Parameters
        ----------
        data : BackendArray
            D-dimensional image stack with shape (n, ...). The data is assumed to be
            the Fourier transform of the stack with DC component at the origin.
        shape : tuple of int
            The shape of the reconstruction volume.
        angles : tuple of float
            Angle to place individual slices at in degrees.
        reconstruction_filter : str, optional
            Filter window applied during reconstruction.
            See :py:meth:`create_reconstruction_filter` for available options.
        tilt_axis : int
            Axis the plane is tilted over, defaults to 0 (x).
        opening_axis : int
            The projection axis, defaults to 2 (z).
        interpolation_order : int
            Interpolation order used for method 'rotation'.
        method : str
            Reconstruction method: "rotation", "gridding", "binary", or "sinc".

        Returns
        -------
        Dict
            Dictionary with reconstructed data and metadata.
        """
        valid_methods = ("rotation", "gridding", "binary", "sinc", "nufft")
        if method not in valid_methods:
            raise ValueError(f"Unknown method '{method}'. Use one of {valid_methods}.")

        # Correction term for non-cubical volumes
        aspect_ratio = shape[opening_axis] / shape[tilt_axis]
        angles = np.degrees(np.arctan(np.tan(np.radians(angles)) * aspect_ratio))

        # Composable filters use frequency grids centered at the origin
        # Here we require them to be centered at subset.shape // 2
        for i in range(data.shape[0]):
            shift = shift_fourier(data[i], shape_is_real_fourier=False, ifftshift=False)
            data = be.at(data, i, shift)

            if data_weights is not None and data.shape == data_weights.shape:
                shift = shift_fourier(
                    data_weights[i], shape_is_real_fourier=False, ifftshift=False
                )
                data_weights = be.at(data_weights, i, shift)

        # reconstruct_gridding normalizes based on trilinear interpolation
        # weights, these filters are primarily useful for reconstruct_rotation.
        if reconstruction_filter is not None:
            rec_filter = create_reconstruction_filter(
                filter_type=reconstruction_filter,
                filter_shape=(shape[tilt_axis],),
                tilt_angles=angles,
                fftshift=True,
            )
            index = tilt_axis - (1 if tilt_axis > opening_axis else 0)
            rec_shape = (
                1,
                *tuple(1 if i != index else -1 for i, x in enumerate(data.shape[1:])),
            )
            rec_filter = be.to_backend_array(rec_filter.reshape(rec_shape))
            data = data * rec_filter

        common_kwargs = {
            "data": data,
            "shape": shape,
            "angles": angles,
            "opening_axis": opening_axis,
            "tilt_axis": tilt_axis,
        }

        if method == "gridding":
            rec = reconstruct_gridding(**common_kwargs, data_weights=data_weights)
        else:
            rec = reconstruct_rotation(
                **common_kwargs, interpolation_order=interpolation_order
            )

        freq = fftfreqn(
            shape=shape,
            sampling_rate=1,
            compute_euclidean_norm=True,
            shape_is_real_fourier=False,
            fftshift=True,
        )
        mask = be.to_backend_array(freq <= 0.5)
        rec = be.multiply(rec, mask, out=rec)

        # Shift DC component back to origin
        rec = shift_fourier(rec, shape_is_real_fourier=False, ifftshift=True)
        return {"data": rec, "shape": shape, "is_multiplicative_filter": False}


def reconstruct_rotation(
    data: BackendArray,
    shape: Tuple[int, ...],
    angles: np.ndarray,
    opening_axis: int,
    tilt_axis: int,
    interpolation_order: int = 1,
) -> np.ndarray:
    """
    Reconstruct by placing 2D slices in 3D and rotating the volume.

    Each 2D Fourier slice is placed at the central plane of a 3D volume
    (perpendicular to the opening axis), then the entire volume is rotated
    by the tilt angle. The rotated volumes are summed to form the reconstruction.

    Parameters
    ----------
    data : BackendArray
        Stack of 2D Fourier slices with shape (n_tilts, *slice_shape)
        and DC component at the center of each slice.
    shape : tuple of int
        Shape of the output 3D reconstruction volume.
    angles : np.ndarray
        Tilt angles in degrees.
    opening_axis : int
        The projection/beam axis (perpendicular to the detector plane).
    tilt_axis : int
        The axis around which the sample is tilted.
    interpolation_order : int
        Spline interpolation order for the 3D rotation (0-5).

    Returns
    -------
    np.ndarray
        Reconstructed 3D Fourier volume.
    """
    volume_temp = be.zeros(shape, dtype=data.dtype)
    rec = be.zeros(shape, dtype=data.dtype)

    slices = tuple(slice(a // 2, (a // 2) + 1) for a in shape)
    subset = tuple(
        slice(None) if i != opening_axis else x for i, x in enumerate(slices)
    )
    wedge_dim = [x for x in data.shape]
    wedge_dim.insert(1 + opening_axis, 1)
    wedges = be.reshape(data, wedge_dim)

    rot_axis = min(i for i in range(len(shape)) if i not in (tilt_axis, opening_axis))
    for index, angle in enumerate(angles):
        volume_temp = be.fill(volume_temp, 0)
        volume_temp = be.at(volume_temp, subset, wedges[index])

        # We want a push rotation but rigid transform assumes pull
        rotation_matrix = _rotation_matrix_around_axis(rot_axis, angle).T

        volume_temp, _ = be.rigid_transform(
            arr=volume_temp,
            rotation_matrix=be.to_backend_array(rotation_matrix),
            center="fourier",
            order=interpolation_order,
        )
        rec = be.add(rec, volume_temp, out=rec)

    return rec


def reconstruct_gridding(
    data: BackendArray,
    shape: Tuple[int, ...],
    angles: np.ndarray,
    opening_axis: int,
    tilt_axis: int,
    data_weights: Optional[BackendArray] = None,
) -> np.ndarray:
    """
    Reconstruct by directly inserting 2D slices into 3D Fourier space.

    For each 2D Fourier slice, computes the corresponding 3D Fourier coordinates
    based on the tilt angle, then distributes the values onto the 3D grid using
    trilinear interpolation. A weight array tracks the sampling density at each
    voxel for proper normalization.

    Parameters
    ----------
    data : BackendArray
        Stack of 2D Fourier slices with shape (n_tilts, *slice_shape)
        and DC component at the center of each slice.
    shape : tuple of int
        Shape of the output 3D reconstruction volume.
    angles : np.ndarray
        Tilt angles in degrees.
    opening_axis : int
        The projection/beam axis (perpendicular to the detector plane).
    tilt_axis : int
        The axis around which the sample is tilted.
    data_weights : BackendArray, optional
        Weights for the individual data points.

    Returns
    -------
    np.ndarray
        Reconstructed 3D Fourier volume.
    """
    grid_shape = list(data.shape[1:])
    grid_shape.insert(opening_axis, 1)

    coords = fftfreqn(shape=grid_shape, sampling_rate=None, fftshift=True)
    coords = be.to_backend_array(coords.reshape(3, -1))
    offset = be.to_backend_array(shape)[:, None] // 2

    rec = be.zeros(shape, dtype=data.dtype)
    weights = be.zeros(shape, dtype=be._float)
    interpweight = be.zeros(shape, dtype=be._float)

    rot_axis = min(i for i in range(len(shape)) if i not in (tilt_axis, opening_axis))
    for idx, angle in enumerate(angles):
        rmat = be.to_backend_array(_rotation_matrix_around_axis(rot_axis, angle))
        grid_coords = rmat @ coords + offset
        rec, weights, interpweight = _trilinear_insert(
            rec,
            weights,
            interpweight,
            grid_coords,
            data[idx].ravel(),
            1.0 if data_weights is None else data_weights[idx].ravel(),
        )

    interpweight = be.minimum(interpweight, 1, out=interpweight)
    rec = be.multiply(rec, interpweight, out=rec)
    with np.errstate(divide="ignore", invalid="ignore"):
        return be.where(weights > 1e-6, rec / weights, 0)


def _rotation_matrix_around_axis(axis: int, angle_deg: float) -> np.ndarray:
    """Create a 3x3 rotation matrix for rotation around the given axis."""
    angles = tuple(0 if i != axis else angle_deg for i in range(3))
    return euler_to_rotationmatrix(angles, seq="xyz")


def _trilinear_insert(
    vol: BackendArray,
    weights: BackendArray,
    interpweight: BackendArray,
    coords: BackendArray,
    sample: BackendArray,
    sample_weights: Optional[BackendArray] = None,
) -> Tuple[BackendArray, BackendArray]:
    """
    Insert values into a 3D volume using trilinear interpolation.

    Parameters
    ----------
    vol : BackendArray
        Output volume to accumulate values into.
    weights : BackendArray
        Weight accumulator for normalization.
    interpweight : BackendArray
        Interpolatoin weight accumulator for normalization.
    coords : BackendArray
        3D coordinates with shape (3, n_points).
    sample : BackendArray
        Complex Fourier samples to insert with shape (n_points,).
    sample_weights : BackendArray, optional
        Per-sample weights (e.g., CTF). Defaults to 1.

    Returns
    -------
    Tuple[BackendArray, BackendArray, BackendArray]
        Output volume and accumulated sample and interpolation weights.
    """
    if sample_weights is None:
        sample_weights = 1.0

    coords_floor = be.astype(be.floor(coords), be._int)

    frac = coords - coords_floor
    one_minus_frac = 1.0 - frac

    for dz in range(2):
        iz = be.mod(coords_floor[2] + dz, vol.shape[2])
        wz = frac[2] if dz else one_minus_frac[2]

        for dy in range(2):
            iy = be.mod(coords_floor[1] + dy, vol.shape[1])
            wy = wz * (frac[1] if dy else one_minus_frac[1])

            for dx in range(2):
                ix = be.mod(coords_floor[0] + dx, vol.shape[0])
                w = wy * (frac[0] if dx else one_minus_frac[0])

                vol = be.addat(vol, (ix, iy, iz), w * sample)
                weights = be.addat(weights, (ix, iy, iz), w * sample_weights)
                interpweight = be.addat(interpweight, (ix, iy, iz), w)

    return vol, weights, interpweight
