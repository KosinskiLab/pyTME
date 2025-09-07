"""
Implements class LinearWhiteningFilter

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import mean as ndimean
from scipy.ndimage import map_coordinates

from ._utils import fftfreqn
from ..types import BackendArray
from ..analyzer.peaks import batchify
from ..backends import backend as be
from .compose import ComposableFilter


__all__ = ["LinearWhiteningFilter"]


@dataclass
class LinearWhiteningFilter(ComposableFilter):
    """
    Generate Fourier whitening filters.

    References
    ----------
    .. [1] de Teresa-Trueba, I.; Goetz, S. K.; Mattausch, A.; Stojanovska, F.; Zimmerli, C. E.;
        Toro-Nahuelpan, M.; Cheng, D. W. C.; Tollervey, F.; Pape, C.; Beck, M.; Diz-Munoz,
        A.; Kreshuk, A.; Mahamid, J.; Zaugg, J. B. Nat. Methods 2023, 20, 284–294.
    .. [2]  M. L. Chaillet, G. van der Schot, I. Gubins, S. Roet,
        R. C. Veltkamp, and F. Förster, Int. J. Mol. Sci. 24,
        13375 (2023)
    """

    @staticmethod
    def _compute_spectrum(
        data_rfft: BackendArray, n_bins: int = None
    ) -> Tuple[BackendArray, BackendArray]:
        """
        Compute the power spectrum of the input data.

        Parameters
        ----------
        data_rfft : BackendArray
            The Fourier transform of the input data.
        n_bins : int, optional
            The number of bins for computing the spectrum, defaults to None.

        Returns
        -------
        bins : BackendArray
            Array containing the bin indices for the spectrum.
        radial_averages : BackendArray
            Array containing the radial averages of the spectrum.
        """
        shape = data_rfft.shape

        max_bins = max(max(shape[:-1]) // 2 + 1, shape[-1])
        n_bins = max_bins if n_bins is None else n_bins
        n_bins = int(min(n_bins, max_bins))

        bins = fftfreqn(
            shape=shape,
            sampling_rate=0.5,
            shape_is_real_fourier=True,
            compute_euclidean_norm=True,
            fftshift=False,
        )
        bins = be.to_numpy_array(bins)
        bins = np.floor(bins * (n_bins - 1) + 0.5).astype(int)

        fourier_spectrum = np.abs(data_rfft)
        fourier_spectrum = np.square(fourier_spectrum, out=fourier_spectrum)

        radial_averages = ndimean(
            fourier_spectrum, labels=bins, index=np.arange(n_bins)
        )
        radial_averages = np.sqrt(radial_averages, out=radial_averages)
        radial_averages = np.where(radial_averages != 0, 1 / radial_averages, 0)
        norm_factor = radial_averages.max()
        if norm_factor != 0:
            radial_averages = np.divide(radial_averages, norm_factor)
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
            fftshift=False,
        )
        grid = be.to_numpy_array(grid)
        grid = np.floor(np.multiply(grid, spectrum.shape[0] - 1) + 0.5)
        spectrum = map_coordinates(spectrum, grid.reshape(1, -1), order=order)
        return spectrum.reshape(grid.shape)

    def _evaluate(
        self,
        shape: Tuple[int, ...],
        data_rfft: BackendArray,
        axes: Tuple[int] = (),
        order: int = 1,
        **kwargs: Dict,
    ) -> Dict:
        """
        Apply linear whitening to the data and return the result.

        Parameters
        ----------
        shape : tuple of ints
            Shape of the returned whitening filter.
        data_rfft : BackendArray, optional
            The Fourier transform of the input data, defaults to None.
        axes : tuple of ints, optional
            Axes to compute spectrum for independently.
        **kwargs : Dict
            Additional keyword arguments.
        """
        if isinstance(axes, int):
            axes = (axes,)

        stack = []
        data_rfft = be.to_numpy_array(data_rfft)
        for subset, _ in batchify(data_rfft.shape, axes):
            _, radial_avg = self._compute_spectrum(np.squeeze(data_rfft[subset]))
            ret = self._interpolate_spectrum(
                spectrum=radial_avg,
                shape=shape,
                shape_is_real_fourier=False,
                order=order,
            )
            stack.append(ret)

        ret = np.array(stack)
        if not len(axes):
            ret = np.squeeze(ret)
        return {"data": be.to_backend_array(ret), "shape": shape}
