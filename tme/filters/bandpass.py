"""
Implements class BandPass and BandPassReconstructed.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from math import log
from typing import Tuple, Union, Optional
from pydantic.dataclasses import dataclass

import numpy as np

from ..types import BackendArray
from ..backends import backend as be
from .compose import ComposableFilter
from ._utils import pad_to_length, frequency_grid_at_angle, fftfreqn

__all__ = ["BandPass", "BandPassReconstructed"]


@dataclass(config=dict(extra="allow"))
class BandPass(ComposableFilter):
    """
    Generate per-tilt Bandpass filter.

    Examples
    --------

    The following creates an instance of :py:class:`BandPass`

    >>> from tme.filters import BandPass
    >>> bpf_instance = BandPass(
    >>>     angles=(-70, 0, 30),
    >>>     lowpass=30,
    >>>     sampling_rate=10
    >>> )

    Differently from :py:class:`tme.filters.BandPassReconstructed`, the filter
    masks are intended to be used in subsequent reconstruction using
    :py:class:`tme.filters.ReconstructFromTilt`.

    The ``opening_axis``, ``tilt_axis`` and ``angles`` parameter are used
    to determine the correct frequencies for non-cubical input shapes. The
    ``shape`` argument contains the shape of the reconstruction.

    >>> ret = bpf_instance(shape=(50,50,25), return_real_fourier=False)
    >>> mask = ret["data"]
    >>> mask.shape # 3, 50, 50
    >>> import matplotlib.pyplot as plt
    >>> fix, ax = plt.subplots(nrows=1, ncols=3)
    >>> _ = [ax[i].imshow(mask[i]) for i in range(mask.shape[0])]
    >>> plt.show()

    """

    #: The lowpass cutoff, defaults to None.
    lowpass: Optional[float] = None
    #: The highpass cutoff, defaults to None.
    highpass: Optional[float] = None
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Union[Tuple[float, ...], float] = 1
    #: Whether to use Gaussian bandpass filter, defaults to True.
    use_gaussian: bool = True
    #: The tilt angles in degrees.
    angles: Tuple[float, ...] = None
    #: Axis the plane is tilted over, defaults to 0 (x).
    tilt_axis: int = 0
    #: The projection axis, defaults to 2 (z).
    opening_axis: int = 2

    def _evaluate(self, shape: Tuple[int, ...], **kwargs):
        """
        Returns a Bandpass stack of chosen parameters with DC component in the center.
        """
        func = discrete_bandpass
        if kwargs.get("use_gaussian"):
            func = gaussian_bandpass

        angles = np.atleast_1d(kwargs["angles"])
        _lowpass = pad_to_length(kwargs["lowpass"], angles.size)
        _highpass = pad_to_length(kwargs["highpass"], angles.size)

        masks = []
        for index, angle in enumerate(angles):
            frequency_grid = frequency_grid_at_angle(
                shape=shape,
                tilt_axis=kwargs["tilt_axis"],
                opening_axis=kwargs["opening_axis"],
                angle=angle,
                sampling_rate=1,
            )
            kwargs["lowpass"] = _lowpass[index]
            kwargs["highpass"] = _highpass[index]
            mask = func(grid=frequency_grid, **kwargs)
            masks.append(be.to_backend_array(mask[None]))
        return {"data": be.concatenate(masks, axis=0), "shape": shape}


@dataclass(config=dict(extra="allow"))
class BandPassReconstructed(ComposableFilter):
    """
    Generate Bandpass filter for reconstructions.

    Examples
    --------
    The following creates an instance of :py:class:`BandPassReconstructed`

    >>> from tme.filters import BandPassReconstructed
    >>> bpf_instance = BandPassReconstructed(
    >>>     lowpass=30,
    >>>     sampling_rate=10
    >>> )

    We can use its call method to create filters of given shape

    >>> import matplotlib.pyplot as plt
    >>> ret = bpf_instance(shape=(50,50))

    The ``data`` key of the returned dictionary contains the corresponding
    Fourier filter mask. The DC component is located at the origin.

    >>> plt.imshow(ret["data"])
    >>> plt.show()
    """

    #: The lowpass cutoff, defaults to None.
    lowpass: Optional[float] = None
    #: The highpass cutoff, defaults to None.
    highpass: Optional[float] = None
    #: The sampling rate, defaults to 1 Ångstrom / voxel.
    sampling_rate: Union[Tuple[float, ...], float] = 1
    #: Whether to use Gaussian bandpass filter, defaults to True.
    use_gaussian: bool = True

    def _evaluate(self, shape: Tuple[int, ...], **kwargs):
        func = discrete_bandpass
        if kwargs.get("use_gaussian"):
            func = gaussian_bandpass

        grid = fftfreqn(
            shape=shape,
            sampling_rate=1,
            shape_is_real_fourier=False,
            compute_euclidean_norm=True,
            fftshift=False,
        )
        ret = be.to_backend_array(func(grid=grid, **kwargs))
        return {"data": ret, "shape": shape}


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
        The lowpass cutoff in spatial units of sampling rate.
    highpass : float
        The highpass cutoff in spatial units of sampling rate.
    sampling_rate : float
        The sampling rate in Fourier space.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    BackendArray
        The bandpass filter in Fourier space.
    """
    grid = be.to_backend_array(grid, be._float)
    sampling_rate = be.to_backend_array(sampling_rate)

    highcut = grid.max()
    if lowpass is not None:
        highcut = be.max(sampling_rate / lowpass)

    lowcut = 0
    if highpass is not None:
        lowcut = be.max(sampling_rate / highpass)
    return (grid <= highcut) & (grid >= lowcut)


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
    grid = be.to_backend_array(grid, be._float)
    grid = -be.square(grid, out=grid)

    upper_sampling = float(np.max(sampling_rate))

    filters = []
    for cutoff, invert in ((lowpass, False), (highpass, True)):
        if cutoff is None:
            continue
        denom = (upper_sampling / float(cutoff)) ** 2 / log(2)
        f = be.exp(grid / denom)
        filters.append(be.subtract(1, f, out=f) if invert else f)

    if not filters:
        return be.full(grid.shape, fill_value=1, dtype=be._float)

    result = filters[0]
    for f in filters[1:]:
        result = be.multiply(result, f, out=result)
    return result
