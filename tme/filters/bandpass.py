"""
Implements class BandPassFilter to create Fourier filter representations.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
from math import log, sqrt
from dataclasses import dataclass

import numpy as np

from ..types import BackendArray
from ..backends import backend as be
from .compose import ComposableFilter
from ._utils import (
    crop_real_fourier,
    shift_fourier,
    pad_to_length,
    frequency_grid_at_angle,
    fftfreqn,
)

__all__ = ["BandPass", "BandPassReconstructed"]


@dataclass
class BandPass(ComposableFilter):
    """
    Generate per-slice Fourier Bandpass filter
    """

    #: The tilt angles.
    angles: Tuple[float]
    #: The lowpass cutoffs. Either one or one per angle, defaults to None.
    lowpass: Tuple[float] = None
    #: The highpass cutoffs. Either one or one per angle, defaults to None.
    highpass: Tuple[float] = None
    #: The shape of the to-be created mask.
    shape: Tuple[int] = None
    #: Axis the plane is tilted over, defaults to 0 (x).
    tilt_axis: int = 0
    #: The projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Tuple[float] = 1
    #: Whether to use Gaussian bandpass filter, defaults to True.
    use_gaussian: bool = True
    #: Whether to return a mask for rfft
    return_real_fourier: bool = False

    def __call__(self, **kwargs):
        """
        Returns a Bandpass stack of chosen parameters with DC component in the center.
        """
        func_args = vars(self).copy()
        func_args.update(kwargs)

        func = discrete_bandpass
        if func_args.get("use_gaussian"):
            func = gaussian_bandpass

        return_real_fourier = kwargs.get("return_real_fourier", True)
        shape_is_real_fourier = kwargs.get("shape_is_real_fourier", False)
        if shape_is_real_fourier:
            return_real_fourier = False

        angles = np.atleast_1d(func_args["angles"])
        _lowpass = pad_to_length(func_args["lowpass"], angles.size)
        _highpass = pad_to_length(func_args["highpass"], angles.size)

        masks = []
        for index, angle in enumerate(angles):
            frequency_grid = frequency_grid_at_angle(
                shape=func_args["shape"],
                tilt_axis=func_args["tilt_axis"],
                opening_axis=func_args["opening_axis"],
                angle=angle,
                sampling_rate=1,
            )
            func_args["lowpass"] = _lowpass[index]
            func_args["highpass"] = _highpass[index]
            mask = func(grid=frequency_grid, **func_args)

            mask = shift_fourier(data=mask, shape_is_real_fourier=shape_is_real_fourier)
            if return_real_fourier:
                mask = crop_real_fourier(mask)
            masks.append(mask[None])

        masks = be.concatenate(masks, axis=0)
        return {
            "data": be.to_backend_array(masks),
            "shape": func_args["shape"],
            "return_real_fourier": return_real_fourier,
            "is_multiplicative_filter": True,
        }


@dataclass
class BandPassReconstructed(ComposableFilter):
    """
    Generate reconstructed bandpass filters in Fourier space.
    """

    #: The lowpass cutoff, defaults to None.
    lowpass: float = None
    #: The highpass cutoff, defaults to None.
    highpass: float = None
    #: The shape of the to-be created mask.
    shape: Tuple[int] = None
    #: Axis the plane is tilted over, defaults to 0 (x).
    tilt_axis: int = 0
    #: The projection axis, defaults to 2 (z).
    opening_axis: int = 2
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Tuple[float] = 1
    #: Whether to use Gaussian bandpass filter, defaults to True.
    use_gaussian: bool = True
    #: Whether to return a mask for rfft
    return_real_fourier: bool = False

    def __call__(self, **kwargs):
        func_args = vars(self).copy()
        func_args.update(kwargs)

        func = discrete_bandpass
        if func_args.get("use_gaussian"):
            func = gaussian_bandpass

        return_real_fourier = func_args.get("return_real_fourier", True)
        shape_is_real_fourier = func_args.get("shape_is_real_fourier", False)
        if shape_is_real_fourier:
            return_real_fourier = False

        grid = fftfreqn(
            shape=func_args["shape"],
            sampling_rate=0.5,
            shape_is_real_fourier=shape_is_real_fourier,
            compute_euclidean_norm=True,
        )
        mask = func(grid=grid, **func_args)

        mask = shift_fourier(data=mask, shape_is_real_fourier=shape_is_real_fourier)
        if return_real_fourier:
            mask = crop_real_fourier(mask)

        return {
            "data": be.to_backend_array(mask),
            "shape": func_args["shape"],
            "return_real_fourier": return_real_fourier,
            "is_multiplicative_filter": True,
        }


def discrete_bandpass(
    grid: BackendArray,
    lowpass: float,
    highpass: float,
    sampling_rate: Tuple[float],
    **kwargs,
) -> BackendArray:
    """
    Generate a bandpass filter using discrete frequency cutoffs.

    Parameters
    ----------
    grid : BackendArray
        Frequencies in Fourier space.
    lowpass : float
        The lowpass cutoff in units of sampling rate.
    highpass : float
        The highpass cutoff in units of sampling rate.
    return_real_fourier : bool, optional
        Whether to return only the real Fourier space, defaults to False.
    sampling_rate : float
        The sampling rate in Fourier space.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    BackendArray
        The bandpass filter in Fourier space.
    """
    grid = be.astype(be.to_backend_array(grid), be._float_dtype)
    sampling_rate = be.to_backend_array(sampling_rate)

    highcut = grid.max()
    if lowpass is not None:
        highcut = be.max(2 * sampling_rate / lowpass)

    lowcut = 0
    if highpass is not None:
        lowcut = be.max(2 * sampling_rate / highpass)

    bandpass_filter = ((grid <= highcut) & (grid >= lowcut)) * 1.0
    return bandpass_filter


def gaussian_bandpass(
    grid: BackendArray,
    lowpass: float = None,
    highpass: float = None,
    sampling_rate: float = 1,
    **kwargs,
) -> BackendArray:
    """
    Generate a bandpass filter using Gaussians.

    Parameters
    ----------
    grid : BackendArray
        Frequency grid in Fourier space.
    lowpass : float, optional
        The lowpass cutoff in units of sampling rate, defaults to None.
    highpass : float, optional
        The highpass cutoff in units of sampling rate, defaults to None.
    sampling_rate : float, optional
        The sampling rate in Fourier space, defaults to one.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    BackendArray
        The bandpass filter in Fourier space.
    """
    grid = be.astype(be.to_backend_array(grid), be._float_dtype)
    grid = -be.square(grid, out=grid)

    has_lowpass, has_highpass = False, False
    norm = float(sqrt(2 * log(2)))
    upper_sampling = float(be.max(be.multiply(2, be.to_backend_array(sampling_rate))))

    if lowpass is not None:
        lowpass, has_lowpass = float(lowpass), True
        lowpass = be.maximum(lowpass, be.eps(be._float_dtype))
    if highpass is not None:
        highpass, has_highpass = float(highpass), True
        highpass = be.maximum(highpass, be.eps(be._float_dtype))

    if has_lowpass:
        lowpass = upper_sampling / (lowpass * norm)
        lowpass = be.multiply(2, be.square(lowpass))
        if not has_highpass:
            lowpass_filter = be.divide(grid, lowpass, out=grid)
        else:
            lowpass_filter = be.divide(grid, lowpass)
        lowpass_filter = be.exp(lowpass_filter, out=lowpass_filter)

    if has_highpass:
        highpass = upper_sampling / (highpass * norm)
        highpass = be.multiply(2, be.square(highpass))
        highpass_filter = be.divide(grid, highpass, out=grid)
        highpass_filter = be.exp(highpass_filter, out=highpass_filter)
        highpass_filter = be.subtract(1, highpass_filter, out=highpass_filter)

    if has_lowpass and not has_highpass:
        bandpass_filter = lowpass_filter
    elif not has_lowpass and has_highpass:
        bandpass_filter = highpass_filter
    elif has_lowpass and has_highpass:
        bandpass_filter = be.multiply(
            lowpass_filter, highpass_filter, out=lowpass_filter
        )
    else:
        bandpass_filter = be.full(grid.shape, fill_value=1, dtype=be._float_dtype)

    return bandpass_filter
