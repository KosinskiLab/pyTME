"""
Fourier space operations for composable filters.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from dataclasses import dataclass

from ..types import BackendArray
from ..backends import backend as be

from .compose import ComposableFilter
from ._utils import shift_fourier, gridding_correction


__all__ = ["Oversample", "Undersample", "ShiftFourier"]


@dataclass
class Oversample(ComposableFilter):
    """Zero-pad data Fourier data in real space for oversampled gridding"""

    factor: int = 2

    @staticmethod
    def _evaluate(
        shape: Tuple[int, ...],
        factor: int = 2,
        data: BackendArray = None,
        **kwargs,
    ) -> Dict:
        new_shape = tuple(s * factor for s in shape)

        if data is None:
            return {"data": None, "shape": new_shape, "is_multiplicative_filter": False}

        slice_shape = shape[1:]
        new_slice_shape = new_shape[1:]
        center = tuple(ns // 2 for ns in new_slice_shape)
        start = tuple(c - s // 2 for c, s in zip(center, slice_shape))
        slices = tuple(slice(st, st + s) for st, s in zip(start, slice_shape))

        ret = []
        for i in range(data.shape[0]):
            real_data = be.fft.ifftn(data[i])
            real_data = shift_fourier(
                real_data, shape_is_real_fourier=False, ifftshift=False
            )

            padded = be.zeros(new_slice_shape, dtype=real_data.dtype)
            padded = be.at(padded, slices, real_data)

            padded = shift_fourier(padded, shape_is_real_fourier=False, ifftshift=True)
            ret.append(be.fft.fftn(padded)[None].real)

        ret = be.concatenate(ret, axis=0)
        return {"data": ret, "shape": new_shape, "is_multiplicative_filter": False}


@dataclass
class Undersample(ComposableFilter):
    """Real space crop Fourier data from oversampled gridding to original space"""

    factor: int = 2

    @staticmethod
    def _evaluate(
        data: BackendArray,
        shape: Tuple[int, ...],
        factor: int = 2,
        **kwargs,
    ) -> Dict:
        original_shape = tuple(s // factor for s in shape)

        data_real = be.fft.ifftn(data)

        # Shift real space data to center
        data_real = shift_fourier(
            data_real, shape_is_real_fourier=False, ifftshift=False
        )

        mask = be.to_backend_array(
            gridding_correction(shape, fftshift=True, padding_factor=factor)
        )
        data_real = be.divide(data_real, mask, out=data_real)

        center = tuple(s // 2 for s in shape)
        start = tuple(c - os // 2 for c, os in zip(center, original_shape))
        slices = tuple(slice(st, st + os) for st, os in zip(start, original_shape))
        data_real = data_real[slices]

        # Shift real space data back to origin
        data_real = shift_fourier(
            data_real, shape_is_real_fourier=False, ifftshift=True
        )
        data = be.fft.fftn(data_real).real

        return {
            "data": data,
            "shape": original_shape,
            "is_multiplicative_filter": False,
        }


class ShiftFourier(ComposableFilter):
    """Shift DC component between origin and center of array."""

    @staticmethod
    def _evaluate(
        data: BackendArray,
        shape: Tuple[int, ...],
        return_real_fourier: bool = False,
        **kwargs,
    ) -> Dict:
        ret = []
        for index in range(data.shape[0]):
            mask = shift_fourier(
                data=data[index],
                shape_is_real_fourier=return_real_fourier,
            )
            ret.append(mask[None])

        ret = be.concatenate(ret, axis=0)
        return {"data": ret, "shape": shape, "is_multiplicative_filter": False}
