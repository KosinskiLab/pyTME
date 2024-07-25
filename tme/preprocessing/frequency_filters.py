""" Defines Fourier frequency filters.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from math import log, sqrt
from typing import Tuple, Dict

import numpy as np
from scipy.ndimage import mean as ndimean
from scipy.ndimage import map_coordinates

from ..types import BackendArray
from ..backends import backend as be
from ._utils import fftfreqn, crop_real_fourier, shift_fourier, compute_fourier_shape


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
    ) -> BackendArray:
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
        grid = be.to_backend_array(grid)
        grid = -be.square(grid)

        lowpass_filter, highpass_filter = 1, 1
        norm = float(sqrt(2 * log(2)))
        upper_sampling = float(
            be.max(be.multiply(2, be.to_backend_array(sampling_rate)))
        )

        if lowpass is not None:
            lowpass = float(lowpass)
            lowpass = be.maximum(lowpass, be.eps(be._float_dtype))
        if highpass is not None:
            highpass = float(highpass)
            highpass = be.maximum(highpass, be.eps(be._float_dtype))

        if lowpass is not None:
            lowpass = upper_sampling / (lowpass * norm)
            lowpass = be.multiply(2, be.square(lowpass))
            lowpass_filter = be.exp(be.divide(grid, lowpass))
        if highpass is not None:
            highpass = upper_sampling / (highpass * norm)
            highpass = be.multiply(2, be.square(highpass))
            highpass_filter = 1 - be.exp(be.divide(grid, highpass))

        bandpass_filter = be.multiply(lowpass_filter, highpass_filter)
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


    References
    ----------
    .. [1] de Teresa-Trueba, I.; Goetz, S. K.; Mattausch, A.; Stojanovska, F.; Zimmerli, C. E.;
        Toro-Nahuelpan, M.; Cheng, D. W. C.; Tollervey, F.; Pape, C.; Beck, M.; Diz-Munoz,
        A.; Kreshuk, A.; Mahamid, J.; Zaugg, J. B. Nat. Methods 2023, 20, 284–294.
    .. [2]  M. L. Chaillet, G. van der Schot, I. Gubins, S. Roet,
        R. C. Veltkamp, and F. Förster, Int. J. Mol. Sci. 24,
        13375 (2023)
    """

    def __init__(self, **kwargs):
        pass

    @staticmethod
    def _compute_spectrum(
        data_rfft: BackendArray, n_bins: int = None, batch_dimension: int = None
    ) -> Tuple[BackendArray, BackendArray]:
        """
        Compute the spectrum of the input data.

        Parameters:
        -----------
        data_rfft : BackendArray
            The Fourier transform of the input data.
        n_bins : int, optional
            The number of bins for computing the spectrum, defaults to None.
        batch_dimension : int, optional
            Batch dimension to average over.

        Returns:
        --------
        bins : BackendArray
            Array containing the bin indices for the spectrum.
        radial_averages : BackendArray
            Array containing the radial averages of the spectrum.
        """
        shape = tuple(x for i, x in enumerate(data_rfft.shape) if i != batch_dimension)

        max_bins = max(max(shape[:-1]) // 2 + 1, shape[-1])
        n_bins = max_bins if n_bins is None else n_bins
        n_bins = int(min(n_bins, max_bins))

        bins = fftfreqn(
            shape=shape,
            sampling_rate=0.5,
            shape_is_real_fourier=True,
            compute_euclidean_norm=True,
        )
        bins = be.to_numpy_array(bins)

        # Implicit lowpass to nyquist
        bins = np.floor(bins * (n_bins - 1) + 0.5).astype(int)
        fft_shift_axes = tuple(
            i for i in range(data_rfft.ndim - 1) if i != batch_dimension
        )
        fourier_spectrum = np.fft.fftshift(data_rfft, axes=fft_shift_axes)
        fourier_spectrum = np.abs(fourier_spectrum)
        np.square(fourier_spectrum, out=fourier_spectrum)

        radial_averages = ndimean(
            fourier_spectrum, labels=bins, index=np.arange(n_bins)
        )
        np.sqrt(radial_averages, out=radial_averages)
        np.reciprocal(radial_averages, out=radial_averages)
        np.divide(radial_averages, radial_averages.max(), out=radial_averages)

        return bins, radial_averages

    @staticmethod
    def _interpolate_spectrum(
        spectrum: BackendArray,
        shape: Tuple[int],
        shape_is_real_fourier: bool = True,
        order: int = 1,
    ) -> BackendArray:
        """
        References
        ----------
        .. [1]  M. L. Chaillet, G. van der Schot, I. Gubins, S. Roet,
            R. C. Veltkamp, and F. Förster, Int. J. Mol. Sci. 24,
            13375 (2023)
        """
        grid = fftfreqn(
            shape=shape,
            sampling_rate=0.5,
            shape_is_real_fourier=shape_is_real_fourier,
            compute_euclidean_norm=True,
        )
        grid = be.to_numpy_array(grid)
        np.multiply(grid, (spectrum.shape[0] - 1), out=grid) + 0.5
        spectrum = map_coordinates(spectrum, grid.reshape(1, -1), order=order)
        return spectrum.reshape(grid.shape)

    def __call__(
        self,
        data: BackendArray = None,
        data_rfft: BackendArray = None,
        n_bins: int = None,
        batch_dimension: int = None,
        order: int = 1,
        **kwargs: Dict,
    ) -> Dict:
        """
        Apply linear whitening to the data and return the result.

        Parameters:
        -----------
        data : BackendArray, optional
            The input data, defaults to None.
        data_rfft : BackendArray, optional
            The Fourier transform of the input data, defaults to None.
        n_bins : int, optional
            The number of bins for computing the spectrum, defaults to None.
        batch_dimension : int, optional
            Batch dimension to average over.
        order : int, optional
            Interpolation order to use.
        **kwargs : Dict
            Additional keyword arguments.

        Returns:
        --------
        Dict
           Filter data and associated parameters.
        """
        if data_rfft is None:
            data_rfft = np.fft.rfftn(be.to_numpy_array(data))

        data_rfft = be.to_numpy_array(data_rfft)

        bins, radial_averages = self._compute_spectrum(
            data_rfft, n_bins, batch_dimension
        )

        if order is None:
            cutoff = bins < radial_averages.size
            filter_mask = np.zeros(bins.shape, radial_averages.dtype)
            filter_mask[cutoff] = radial_averages[bins[cutoff]]
        else:
            shape = bins.shape
            if kwargs.get("shape", False):
                shape = compute_fourier_shape(
                    shape=kwargs.get("shape"),
                    shape_is_real_fourier=kwargs.get("shape_is_real_fourier", False),
                )

            filter_mask = self._interpolate_spectrum(
                spectrum=radial_averages,
                shape=shape,
                shape_is_real_fourier=True,
            )

        filter_mask = np.fft.ifftshift(
            filter_mask,
            axes=tuple(i for i in range(data_rfft.ndim - 1) if i != batch_dimension),
        )

        return {
            "data": be.to_backend_array(filter_mask),
            "is_multiplicative_filter": True,
        }
