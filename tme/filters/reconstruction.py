"""
Implements class ReconstructFromTilt and ShiftFourier.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ..types import BackendArray
from ..backends import backend as be

from .compose import ComposableFilter
from ..rotations import euler_to_rotationmatrix
from ._utils import shift_fourier, create_reconstruction_filter

__all__ = ["ReconstructFromTilt", "ShiftFourier"]


@dataclass
class ReconstructFromTilt(ComposableFilter):
    """
    Place Fourier transforms of d-dimensional inputs into a d+1-dimensional array
    aking of weighted backprojection using direct fourier inversion.

    This class is used to reconstruct the output of ComposableFilter instances for
    individual tilts to be applied to query templates.

    See Also
    --------
    :py:class:`tme.filters.CTF`
    :py:class:`tme.filters.Wedge`
    :py:class:`tme.filters.BandPass`

    """

    #: Angle of each individual tilt in degrees.
    angles: Tuple[float] = None
    #: Projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: Tilt axis, defaults to 0 (x).
    tilt_axis: int = 0
    #: Interpolation order used for rotation
    interpolation_order: int = 1
    #: Filter window applied during reconstruction.
    reconstruction_filter: str = None

    @staticmethod
    def _evaluate(
        data: BackendArray,
        shape: Tuple[int, ...],
        angles: Tuple[float],
        opening_axis: int = 2,
        tilt_axis: int = 0,
        interpolation_order: int = 1,
        reconstruction_filter: str = None,
        **kwargs,
    ) -> Dict:
        """
        Reconstruct a 3-dimensional array from n 2-dimensional inputs using WBP.

        Parameters
        ----------
        data : BackendArray
            D-dimensional image stack with shape (n, ...). The data is assumed to be
            the Fourier transform of the stack you are trying to reconstruct with
            DC component at the origin. Notably, the data needs to be the output of
            np.fft.fftn not the reduced np.fft.rffn.
        shape : tuple of int
            The shape of the reconstruction volume.
        angles : tuple of float
            Angle to place individual slices at in degrees.
        reconstruction_filter : bool, optional
           Filter window applied during reconstruction.
           See :py:meth:`create_reconstruction_filter` for available options.
        tilt_axis : int
            Axis the plane is tilted over, defaults to 0 (x).
        opening_axis : int
            The projection axis, defaults to 2 (z).
        """

        if data.shape == shape:
            return data

        # Composable filters use frequency grids centered at the origin
        # Here we require them to be centered at subset.shape // 2
        for i in range(data.shape[0]):
            data_shifted = shift_fourier(
                data[i], shape_is_real_fourier=False, ifftshift=False
            )
            data = be.at(data, i, data_shifted)

        volume_temp = be.zeros(shape, dtype=data.dtype)
        rec = be.zeros(shape, dtype=data.dtype)

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
                fftshift=True,
            )
            rec_shape = tuple(1 if i != tilt_axis else x for i, x in enumerate(shape))
            rec_filter = be.to_backend_array(rec_filter)
            rec_filter = be.reshape(rec_filter, rec_shape)

        angles = be.to_backend_array(angles)
        axis_index = min(
            tuple(i for i in range(len(shape)) if i not in (tilt_axis, opening_axis))
        )
        for index in range(len(angles)):
            angles_loop = be.fill(angles_loop, 0)
            volume_temp = be.fill(volume_temp, 0)

            # Jax compatibility
            volume_temp = be.at(volume_temp, subset, wedges[index] * rec_filter)
            angles_loop = be.at(angles_loop, axis_index, angles[index])

            # We want a push rotation but rigid transform assumes pull
            rotation_matrix = euler_to_rotationmatrix(
                be.to_numpy_array(angles_loop), seq="xyz"
            ).T

            volume_temp, _ = be.rigid_transform(
                arr=volume_temp,
                rotation_matrix=be.to_backend_array(rotation_matrix),
                use_geometric_center=True,
                order=interpolation_order,
            )
            rec = be.add(rec, volume_temp, out=rec)

        # Shift DC component back to origin
        rec = shift_fourier(rec, shape_is_real_fourier=False, ifftshift=True)
        return {"data": rec, "shape": shape, "is_multiplicative_filter": False}


class ShiftFourier(ComposableFilter):
    def _evaluate(self, shape: Tuple[int, ...], data: BackendArray, **kwargs) -> Dict:
        ret = []
        for index in range(data.shape[0]):
            mask = shift_fourier(
                data=data[index],
                shape_is_real_fourier=kwargs.get("return_real_fourier", False),
            )
            ret.append(mask[None])

        ret = be.concatenate(ret, axis=0)
        return {"data": ret, "shape": shape, "is_multiplicative_filter": False}
