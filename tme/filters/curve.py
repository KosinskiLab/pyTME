"""
Curve maps a precomputed 1D radial profile onto an Fourier grid.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from dataclasses import dataclass

import numpy as np

from ._utils import fftfreqn
from ..types import BackendArray
from ..backends import backend as be
from .compose import ComposableFilter

__all__ = ["Curve"]


@dataclass
class Curve(ComposableFilter):
    """Interpolate a 1D radial profile onto an nD Fourier grid."""

    #: 1D radial profile from DC to Nyquist.
    spectrum: BackendArray = None

    def _evaluate(self, shape: Tuple[int, ...], **kwargs: Dict) -> Dict:
        spectrum = be.to_numpy_array(self.spectrum)
        bin_centers = fftfreqn(
            shape=(spectrum.shape[0],),
            sampling_rate=0.5,
            compute_euclidean_norm=True,
            shape_is_real_fourier=True,
            fftshift=False,
        )
        ret = _interpolate_spectrum(
            bin_centers=bin_centers,
            spectrum=spectrum,
            shape=shape,
            shape_is_real_fourier=False,
        )
        return {"data": be.to_backend_array(ret), "shape": shape}


def _interpolate_spectrum(
    bin_centers: BackendArray,
    spectrum: BackendArray,
    shape: Tuple[int],
    shape_is_real_fourier: bool = True,
) -> BackendArray:
    """
    Interpolate a 1D radial spectrum onto an nD frequency grid.

    Parameters
    ----------
    bin_centers : BackendArray
        Frequency values corresponding to each spectrum entry.
    spectrum : BackendArray
        1D radial spectrum values.
    shape : Tuple[int]
        Shape of the output grid.
    shape_is_real_fourier : bool, optional
        Whether shape corresponds to a real Fourier transform.

    Returns
    -------
    BackendArray
        The spectrum mapped onto the nD frequency grid.
    """
    from scipy.interpolate import PchipInterpolator

    grid = fftfreqn(
        shape=shape,
        sampling_rate=0.5,
        compute_euclidean_norm=True,
        shape_is_real_fourier=shape_is_real_fourier,
        fftshift=False,
    )

    # Shape-preserving interpolation with hard Nyquist cutoff
    interpolator = PchipInterpolator(bin_centers, spectrum, extrapolate=False)
    ret = interpolator(grid.ravel()).reshape(grid.shape)
    ret[np.isnan(ret)] = 0
    return np.fmax(ret, 0, out=ret)
