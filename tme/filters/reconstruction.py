"""
Defines filters on tomographic tilt series.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
from dataclasses import dataclass

import numpy as np

from ..backends import backend as be
from ..types import NDArray, BackendArray

from .compose import ComposableFilter
from ..rotations import euler_to_rotationmatrix
from ._utils import crop_real_fourier, shift_fourier, create_reconstruction_filter

__all__ = ["ReconstructFromTilt", "ShiftFourier"]


@dataclass
class ReconstructFromTilt(ComposableFilter):
    """
    Reconstruct a d+1 array from a d-dimensional input projection using weighted
    backprojection (WBP).
    """

    #: Shape of the reconstruction.
    shape: Tuple[int] = None
    #: Angle of each individual tilt.
    angles: Tuple[float] = None
    #: Projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: Tilt axis, defaults to 0 (x).
    tilt_axis: int = 0
    #: Whether to return a share compliant with rfftn.
    return_real_fourier: bool = True
    #: Interpolation order used for rotation
    interpolation_order: int = 1
    #: Filter window applied during reconstruction.
    reconstruction_filter: str = None

    def __call__(self, return_real_fourier: bool = False, **kwargs):
        """
        Reconstruct a  d+1 array from a d-dimensional input using WBP.

        Parameters
        ----------
        shape : tuple of int
            The shape of the reconstruction volume.
        data : BackendArray
            D-dimensional image stack with shape (n, ...). The data is assumed to be
            a Fourier transform of the stack you are trying to reconstruct with
            DC component in the center.
        angles : tuple of float
            Angle of each individual tilt.
        return_real_fourier : bool, optional
            Return a shape compliant
        return_real_fourier : tuple of int
            Return a shape compliant with rfft, i.e., omit the negative frequencies
            terms resulting in a return shape (*shape[:-1], shape[-1]//2+1). Defaults
            to False.
        reconstruction_filter : bool, optional
           Filter window applied during reconstruction.
           See :py:meth:`create_reconstruction_filter` for available options.
        tilt_axis : int
            Axis the plane is tilted over, defaults to 0 (x).
        opening_axis : int
            The projection axis, defaults to 2 (z).

        Returns
        -------
        dict
            data: BackendArray
                The filter mask.
            shape: tuple of ints
                The requested filter shape
            return_real_fourier: bool
                Whether data is compliant with rfftn.
            is_multiplicative_filter: bool
                Whether the filter is multiplicative in Fourier space.
        """

        func_args = vars(self).copy()
        func_args.update(kwargs)

        ret = self.reconstruct(**func_args)

        ret = shift_fourier(data=ret, shape_is_real_fourier=False)
        if return_real_fourier:
            ret = crop_real_fourier(ret)

        return {
            "data": ret,
            "shape": func_args["shape"],
            "return_real_fourier": return_real_fourier,
            "is_multiplicative_filter": False,
        }

    @staticmethod
    def reconstruct(
        data: NDArray,
        shape: Tuple[int],
        angles: Tuple[float],
        opening_axis: int,
        tilt_axis: int,
        interpolation_order: int = 1,
        reconstruction_filter: str = None,
        **kwargs,
    ):
        """
        Reconstruct a volume from a tilt series.

        Parameters
        ----------
        data : NDArray
            The Fourier transform of tilt series data.
        shape : tuple of int
            Shape of the reconstruction.
        angles : tuple of float
            Angle of each individual tilt.
        opening_axis : int
            The axis around which the volume is opened.
        tilt_axis : int
            Axis the plane is tilted over.
        interpolation_order : int, optional
            Interpolation order used for rotation, defaults to 1.
        reconstruction_filter : bool, optional
           Filter window applied during reconstruction.
           See :py:meth:`create_reconstruction_filter` for available options.

        Returns
        -------
        NDArray
            The reconstructed volume.
        """
        if data.shape == shape:
            return data

        data = be.to_backend_array(data)
        volume_temp = be.zeros(shape, dtype=data.dtype)
        volume_temp_rotated = be.zeros(shape, dtype=data.dtype)
        volume = be.zeros(shape, dtype=data.dtype)

        slices = tuple(slice(a // 2, (a // 2) + 1) for a in shape)
        subset = tuple(
            slice(None) if i != opening_axis else x for i, x in enumerate(slices)
        )
        angles_loop = be.zeros(len(shape))
        wedge_dim = [x for x in data.shape]
        wedge_dim.insert(1 + opening_axis, 1)
        wedges = be.reshape(data, wedge_dim)

        rec_filter = 1
        aspect_ratio = shape[opening_axis] / shape[tilt_axis]
        angles = np.degrees(np.arctan(np.tan(np.radians(angles)) * aspect_ratio))
        if reconstruction_filter is not None:
            rec_filter = create_reconstruction_filter(
                filter_type=reconstruction_filter,
                filter_shape=(shape[tilt_axis],),
                tilt_angles=angles,
            )
            rec_shape = tuple(1 if i != tilt_axis else x for i, x in enumerate(shape))
            rec_filter = be.to_backend_array(rec_filter)
            rec_filter = be.reshape(rec_filter, rec_shape)

        angles = be.to_backend_array(angles)
        for index in range(len(angles)):
            angles_loop = be.fill(angles_loop, 0)
            volume_temp = be.fill(volume_temp, 0)
            volume_temp_rotated = be.fill(volume_temp_rotated, 0)

            # Jax compatibility
            volume_temp = be.at(volume_temp, subset, wedges[index] * rec_filter)
            angles_loop = be.at(angles_loop, tilt_axis, angles[index])

            angles_loop = be.roll(angles_loop, (opening_axis - 1,), axis=0)
            rotation_matrix = euler_to_rotationmatrix(be.to_numpy_array(angles_loop))
            rotation_matrix = be.to_backend_array(rotation_matrix)

            volume_temp_rotated, _ = be.rigid_transform(
                arr=volume_temp,
                rotation_matrix=rotation_matrix,
                out=volume_temp_rotated,
                use_geometric_center=True,
                order=interpolation_order,
            )
            volume = be.add(volume, volume_temp_rotated, out=volume)

        return volume


class ShiftFourier(ComposableFilter):
    def __call__(
        self,
        data: BackendArray,
        shape_is_real_fourier: bool = False,
        return_real_fourier: bool = True,
        **kwargs,
    ):
        ret = []
        for index in range(data.shape[0]):
            mask = be.to_numpy_array(data[index])

            mask = shift_fourier(data=mask, shape_is_real_fourier=shape_is_real_fourier)
            if return_real_fourier:
                mask = crop_real_fourier(mask)
            ret.append(mask[None])
        ret = np.concatenate(ret, axis=0)

        return {
            "data": ret,
            "shape": kwargs.get("shape"),
            "return_real_fourier": return_real_fourier,
            "is_multiplicative_filter": False,
        }
