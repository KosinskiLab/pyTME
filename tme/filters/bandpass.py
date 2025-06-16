"""
Implements class BandPassFilter to create Fourier filter representations.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
from math import log, sqrt

from ..types import BackendArray
from ..backends import backend as be
from .compose import ComposableFilter
from ._utils import fftfreqn, crop_real_fourier, shift_fourier

__all__ = ["BandPassFilter"]


class BandPassFilter(ComposableFilter):
    """
    Generate bandpass filters in Fourier space.

    Parameters
    ----------
    lowpass : float, optional
        The lowpass cutoff, defaults to None.
    highpass : float, optional
        The highpass cutoff, defaults to None.
    sampling_rate : Tuple[float], optional
        The sampling r_position_to_molmapate in Fourier space, defaults to 1.
    use_gaussian : bool, optional
        Whether to use Gaussian bandpass filter, defaults to True.
    return_real_fourier : bool, optional
        Whether to return only the real Fourier space, defaults to False.
    shape_is_real_fourier : bool, optional
        Whether the shape represents the real Fourier space, defaults to False.
    """

    def __init__(
        self,
        lowpass: float = None,
        highpass: float = None,
        sampling_rate: Tuple[float] = 1,
        use_gaussian: bool = True,
        return_real_fourier: bool = False,
        shape_is_real_fourier: bool = False,
    ):
        self.lowpass = lowpass
        self.highpass = highpass
        self.use_gaussian = use_gaussian
        self.return_real_fourier = return_real_fourier
        self.shape_is_real_fourier = shape_is_real_fourier
        self.sampling_rate = sampling_rate

    @staticmethod
    def discrete_bandpass(
        shape: Tuple[int],
        lowpass: float,
        highpass: float,
        sampling_rate: Tuple[float],
        return_real_fourier: bool = False,
        shape_is_real_fourier: bool = False,
        **kwargs,
    ) -> BackendArray:
        """
        Generate a bandpass filter using discrete frequency cutoffs.

        Parameters
        ----------
        shape : tuple of int
            The shape of the bandpass filter.
        lowpass : float
            The lowpass cutoff in units of sampling rate.
        highpass : float
            The highpass cutoff in units of sampling rate.
        return_real_fourier : bool, optional
            Whether to return only the real Fourier space, defaults to False.
        sampling_rate : float
            The sampling rate in Fourier space.
        shape_is_real_fourier : bool, optional
            Whether the shape represents the real Fourier space, defaults to False.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        BackendArray
            The bandpass filter in Fourier space.
        """
        if shape_is_real_fourier:
            return_real_fourier = False

        grid = fftfreqn(
            shape=shape,
            sampling_rate=0.5,
            shape_is_real_fourier=shape_is_real_fourier,
            compute_euclidean_norm=True,
        )
        grid = be.astype(be.to_backend_array(grid), be._float_dtype)
        sampling_rate = be.to_backend_array(sampling_rate)

        highcut = grid.max()
        if lowpass is not None:
            highcut = be.max(2 * sampling_rate / lowpass)

        lowcut = 0
        if highpass is not None:
            lowcut = be.max(2 * sampling_rate / highpass)

        bandpass_filter = ((grid <= highcut) & (grid >= lowcut)) * 1.0
        bandpass_filter = shift_fourier(
            data=bandpass_filter, shape_is_real_fourier=shape_is_real_fourier
        )

        if return_real_fourier:
            bandpass_filter = crop_real_fourier(bandpass_filter)

        return bandpass_filter

    @staticmethod
    def gaussian_bandpass(
        shape: Tuple[int],
        lowpass: float,
        highpass: float,
        sampling_rate: float,
        return_real_fourier: bool = False,
        shape_is_real_fourier: bool = False,
        **kwargs,
    ) -> BackendArray:
        """
        Generate a bandpass filter using Gaussians.

        Parameters
        ----------
        shape : tuple of int
            The shape of the bandpass filter.
        lowpass : float
            The lowpass cutoff in units of sampling rate.
        highpass : float
            The highpass cutoff in units of sampling rate.
        sampling_rate : float
            The sampling rate in Fourier space.
        return_real_fourier : bool, optional
            Whether to return only the real Fourier space, defaults to False.
        shape_is_real_fourier : bool, optional
            Whether the shape represents the real Fourier space, defaults to False.
        **kwargs : dict
            Additional keyword arguments.

        Returns
        -------
        BackendArray
            The bandpass filter in Fourier space.
        """
        if shape_is_real_fourier:
            return_real_fourier = False

        grid = fftfreqn(
            shape=shape,
            sampling_rate=0.5,
            shape_is_real_fourier=shape_is_real_fourier,
            compute_euclidean_norm=True,
        )
        grid = be.astype(be.to_backend_array(grid), be._float_dtype)
        grid = -be.square(grid, out=grid)

        has_lowpass, has_highpass = False, False
        norm = float(sqrt(2 * log(2)))
        upper_sampling = float(
            be.max(be.multiply(2, be.to_backend_array(sampling_rate)))
        )

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
            bandpass_filter = be.full(shape, fill_value=1, dtype=be._float_dtype)

        bandpass_filter = shift_fourier(
            data=bandpass_filter, shape_is_real_fourier=shape_is_real_fourier
        )

        if return_real_fourier:
            bandpass_filter = crop_real_fourier(bandpass_filter)

        return bandpass_filter

    def __call__(self, **kwargs):
        func_args = vars(self)
        func_args.update(kwargs)

        func = self.discrete_bandpass
        if func_args.get("use_gaussian"):
            func = self.gaussian_bandpass

        mask = func(**func_args)

        return {
            "data": be.to_backend_array(mask),
            "shape": func_args["shape"],
            "return_real_fourier": func_args.get("return_real_fourier"),
            "is_multiplicative_filter": True,
        }
