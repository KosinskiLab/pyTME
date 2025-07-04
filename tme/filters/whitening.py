"""
Implements class BandPassFilter to create Fourier filter representations.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict

import numpy as np
from scipy.ndimage import mean as ndimean
from scipy.ndimage import map_coordinates

from ..types import BackendArray
from ..backends import backend as be
from .compose import ComposableFilter
from ._utils import fftfreqn, compute_fourier_shape, shift_fourier

__all__ = ["LinearWhiteningFilter"]


class LinearWhiteningFilter(ComposableFilter):
    """
    Compute Fourier power spectrums and perform whitening.

    References
    ----------
    .. [1] de Teresa-Trueba, I.; Goetz, S. K.; Mattausch, A.; Stojanovska, F.; Zimmerli, C. E.;
        Toro-Nahuelpan, M.; Cheng, D. W. C.; Tollervey, F.; Pape, C.; Beck, M.; Diz-Munoz,
        A.; Kreshuk, A.; Mahamid, J.; Zaugg, J. B. Nat. Methods 2023, 20, 284–294.
    .. [2]  M. L. Chaillet, G. van der Schot, I. Gubins, S. Roet,
        R. C. Veltkamp, and F. Förster, Int. J. Mol. Sci. 24,
        13375 (2023)
    """

    def __init__(self, *args, **kwargs):
        pass

    @staticmethod
    def _compute_spectrum(
        data_rfft: BackendArray, n_bins: int = None, batch_dimension: int = None
    ) -> Tuple[BackendArray, BackendArray]:
        """
        Compute the power spectrum of the input data.

        Parameters
        ----------
        data_rfft : BackendArray
            The Fourier transform of the input data.
        n_bins : int, optional
            The number of bins for computing the spectrum, defaults to None.
        batch_dimension : int, optional
            Batch dimension to average over.

        Returns
        -------
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
        shape: Tuple[int],
        data: BackendArray = None,
        data_rfft: BackendArray = None,
        n_bins: int = None,
        batch_dimension: int = None,
        order: int = 1,
        return_real_fourier: bool = True,
        **kwargs: Dict,
    ) -> Dict:
        """
        Apply linear whitening to the data and return the result.

        Parameters
        ----------
        shape : tuple of ints
            Shape of the returned whitening filter.
        data : BackendArray, optional
            The input data, defaults to None.
        data_rfft : BackendArray, optional
            The Fourier transform of the input data, defaults to None.
        n_bins : int, optional
            The number of bins for computing the spectrum, defaults to None.
        batch_dimension : int, optional
            Batch dimension to average over.
        return_real_fourier : tuple of int
            Return a shape compliant with rfft, i.e., omit the negative frequencies
            terms resulting in a return shape (*shape[:-1], shape[-1]//2+1)
        **kwargs : Dict
            Additional keyword arguments.

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
        if data_rfft is None:
            data_rfft = be.rfftn(data)

        data_rfft = be.to_numpy_array(data_rfft)
        bins, radial_averages = self._compute_spectrum(
            data_rfft, n_bins, batch_dimension
        )
        shape = tuple(int(x) for i, x in enumerate(shape) if i != batch_dimension)

        shape_filter = shape
        if return_real_fourier:
            shape_filter = compute_fourier_shape(
                shape=shape,
                shape_is_real_fourier=False,
            )

        ret = self._interpolate_spectrum(
            spectrum=radial_averages,
            shape=shape_filter,
            shape_is_real_fourier=return_real_fourier,
        )

        ret = shift_fourier(data=ret, shape_is_real_fourier=return_real_fourier)

        return {
            "data": be.to_backend_array(ret),
            "shape": shape,
            "return_real_fourier": return_real_fourier,
            "is_multiplicative_filter": True,
        }
