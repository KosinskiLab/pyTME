""" Defines Fourier frequency filters.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from math import log, sqrt
from typing import Tuple, Dict

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import mean as ndimean

from ._utils import fftfreqn, crop_real_fourier, shift_fourier
from ..backends import backend


class BandPassFilter:
    """
    This class provides methods to generate bandpass filters in Fourier space,
    either by directly specifying the frequency cutoffs (discrete_bandpass) or
    by using Gaussian functions (gaussian_bandpass).

    Parameters:
    -----------
    lowpass : float, optional
        The lowpass cutoff, defaults to None.
    highpass : float, optional
        The highpass cutoff, defaults to None.
    sampling_rate : Tuple[float], optional
        The sampling rate in Fourier space, defaults to 1.
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
    ) -> NDArray:
        """
        Generate a bandpass filter using discrete frequency cutoffs.

        Parameters:
        -----------
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

        Returns:
        --------
        NDArray
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

        lowpass = 0 if lowpass is None else lowpass
        highpass = 1e10 if highpass is None else highpass

        highcut = grid.max()
        if lowpass > 0:
            highcut = np.max(2 * sampling_rate / lowpass)
        lowcut = np.max(2 * sampling_rate / highpass)

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
    ) -> NDArray:
        """
        Generate a bandpass filter using Gaussian functions.

        Parameters:
        -----------
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

        Returns:
        --------
        NDArray
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
        grid = -backend.square(grid)

        lowpass_filter, highpass_filter = 1, 1
        norm = float(sqrt(2 * log(2)))
        upper_sampling = float(backend.max(backend.multiply(2, sampling_rate)))

        if lowpass is not None:
            lowpass = float(lowpass)
            lowpass = backend.maximum(lowpass, backend.eps(backend._float_dtype))
        if highpass is not None:
            highpass = float(highpass)
            highpass = backend.maximum(highpass, backend.eps(backend._float_dtype))

        if lowpass is not None:
            lowpass = upper_sampling / (lowpass * norm)
            lowpass = backend.multiply(2, backend.square(lowpass))
            lowpass_filter = backend.exp(backend.divide(grid, lowpass))
        if highpass is not None:
            highpass = upper_sampling / (highpass * norm)
            highpass = backend.multiply(2, backend.square(highpass))
            highpass_filter = 1 - backend.exp(backend.divide(grid, highpass))

        bandpass_filter = backend.multiply(lowpass_filter, highpass_filter)
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
            "data": backend.to_backend_array(mask),
            "sampling_rate": func_args.get("sampling_rate", 1),
            "is_multiplicative_filter": True,
        }


class LinearWhiteningFilter:
    """
    This class provides methods to compute the spectrum of the input data and
    apply linear whitening to the Fourier coefficients.

    Parameters:
    -----------
    **kwargs : Dict, optional
        Additional keyword arguments.
    """

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _compute_spectrum(
        data_rfft: NDArray, n_bins: int = None, batch_dimension: int = None
    ) -> Tuple[NDArray, NDArray]:
        """
        Compute the spectrum of the input data.

        Parameters:
        -----------
        data_rfft : NDArray
            The Fourier transform of the input data.
        n_bins : int, optional
            The number of bins for computing the spectrum, defaults to None.
        batch_dimension : int, optional
            Batch dimension to average over.

        Returns:
        --------
        bins : NDArray
            Array containing the bin indices for the spectrum.
        radial_averages : NDArray
            Array containing the radial averages of the spectrum.
        """
        shape = tuple(x for i, x in enumerate(data_rfft.shape) if i != batch_dimension)

        max_bins = max(max(shape[:-1]) // 2 + 1, shape[-1])
        n_bins = max_bins if n_bins is None else n_bins
        n_bins = int(min(n_bins, max_bins))

        grid = fftfreqn(
            shape=shape,
            sampling_rate=None,
            shape_is_real_fourier=True,
            compute_euclidean_norm=True,
        )
        grid = backend.to_numpy_array(grid)
        _, bin_edges = np.histogram(grid, bins=n_bins - 1)
        bins = np.digitize(grid, bins=bin_edges, right=True)

        fft_shift_axes = tuple(
            i for i in range(data_rfft.ndim - 1) if i != batch_dimension
        )
        fourier_spectrum = np.fft.fftshift(data_rfft, axes=fft_shift_axes)
        fourier_spectrum = np.abs(fourier_spectrum)
        np.square(fourier_spectrum, out=fourier_spectrum)
        radial_averages = ndimean(fourier_spectrum, labels=bins, index=np.unique(bins))

        np.sqrt(radial_averages, out=radial_averages)
        np.reciprocal(radial_averages, out=radial_averages)
        np.divide(radial_averages, radial_averages.max(), out=radial_averages)

        return bins, radial_averages

    def __call__(
        self,
        data: NDArray = None,
        data_rfft: NDArray = None,
        n_bins: int = None,
        batch_dimension: int = None,
        **kwargs: Dict,
    ) -> Dict:
        """
        Apply linear whitening to the data and return the result.

        Parameters:
        -----------
        data : NDArray, optional
            The input data, defaults to None.
        data_rfft : NDArray, optional
            The Fourier transform of the input data, defaults to None.
        n_bins : int, optional
            The number of bins for computing the spectrum, defaults to None.
        batch_dimension : int, optional
            Batch dimension to average over.
        **kwargs : Dict
            Additional keyword arguments.

        Returns:
        --------
        Dict
            A dictionary containing the whitened data and information
            about the filter being a multiplicative filter.
        """
        if data_rfft is None:
            data_rfft = np.fft.rfftn(backend.to_numpy_array(data))

        data_rfft = backend.to_numpy_array(data_rfft)

        bins, radial_averages = self._compute_spectrum(
            data_rfft, n_bins, batch_dimension
        )

        radial_averages = np.fft.ifftshift(
            radial_averages[bins], axes=tuple(range(data_rfft.ndim - 1))
        )

        return {
            "data": backend.to_backend_array(radial_averages),
            "is_multiplicative_filter": True,
        }
