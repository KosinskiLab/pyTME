"""
Utilities for the creation of composable filters.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, List, Dict

import numpy as np

from ..backends import backend as be
from ..backends import NumpyFFTWBackend
from ..types import BackendArray, NDArray
from ..rotations import euler_to_rotationmatrix

__all__ = [
    "compute_tilt_shape",
    "frequency_grid_at_angle",
    "fftfreqn",
    "crop_real_fourier",
    "compute_fourier_shape",
    "shift_fourier",
    "create_reconstruction_filter",
    "pad_to_length",
    "gridding_correction",
    "radial_average",
    "radial_bins",
    "power_at_tilt",
]


def compute_tilt_shape(shape: Tuple[int], opening_axis: int, reduce_dim: bool = False):
    """
    Given an opening_axis, computes the shape of the remaining dimensions.

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the input array.
    opening_axis : int
        The axis along which the array will be tilted.
    reduce_dim : bool, optional (default=False)
        Whether to reduce the dimensionality after tilting.

    Returns
    -------
    Tuple[int]
        The shape of the array after tilting.
    """
    tilt_shape = tuple(x if i != opening_axis else 1 for i, x in enumerate(shape))
    if reduce_dim:
        tilt_shape = tuple(x for i, x in enumerate(shape) if i != opening_axis)

    return tilt_shape


def frequency_grid_at_angle(
    shape: Tuple[int],
    angle: float,
    sampling_rate: Tuple[float],
    opening_axis: int = None,
    tilt_axis: int = None,
    fftshift: bool = False,
) -> NDArray:
    """
    Generate a frequency grid from 0 to 1/(2 * sampling_rate) in each axis.

    Conceptually, this function generates accurate frequency grid of tilted
    projections. Given a non-cubical shape, it no longer accurate to compute
    frequences as Euclidean distances from a centered index grid. This function
    solves this issue, and makes it possible to create complex filters on
    non-cubical input shapes.

    Parameters
    ----------
    shape : tuple of int
        The shape of the grid.
    angle : float
        The angle at which to generate the grid in degrees.
    sampling_rate : tuple of float
        The sampling rate for each dimension.
    opening_axis : int, optional
        The projection axis, defaults to None.
    tilt_axis : int, optional
        The axis along which the grid is tilted, defaults to None.
    fftshift : bool, optional
        Whether to return a grid centered at shape // 2. Default is grid centered around
        origin, which is compliant with the rfftn definitions used in this project

    Returns
    -------
    NDArray
        The frequency grid.
    """
    sampling_rate = np.array(sampling_rate)
    sampling_rate = np.repeat(sampling_rate, len(shape) // sampling_rate.size)

    tilt_shape = compute_tilt_shape(
        shape=shape, opening_axis=opening_axis, reduce_dim=False
    )

    missing_axes = opening_axis is None or tilt_axis is None
    if angle == 0 or missing_axes or len(set(shape)) == 1:
        # Crop the sampling rate to tilt shape
        sampling_rate = compute_tilt_shape(
            shape=sampling_rate, opening_axis=opening_axis, reduce_dim=True
        )
        return fftfreqn(
            tuple(x for x in tilt_shape if x != 1),
            sampling_rate=sampling_rate,
            compute_euclidean_norm=True,
            fftshift=fftshift,
        )

    if angle != 0:
        aspect_ratio = shape[opening_axis] / shape[tilt_axis]
        angle = np.degrees(np.arctan(np.tan(np.radians(angle)) * aspect_ratio))

        angles = np.zeros(len(shape))
        angles[tilt_axis] = angle
        rotation_matrix = euler_to_rotationmatrix(
            np.roll(angles, opening_axis - 1), seq="zyz"
        )

        index_grid = fftfreqn(tilt_shape, sampling_rate=None, fftshift=fftshift)
        index_grid = np.einsum("ij,j...->i...", rotation_matrix, index_grid)
        norm = np.multiply(sampling_rate, shape).astype(int)

        index_grid = np.divide(index_grid.T, norm).T
        index_grid = np.squeeze(index_grid)
        index_grid = np.linalg.norm(index_grid, axis=(0))

    return index_grid


def fftfreqn(
    shape: Tuple[int],
    sampling_rate: Tuple[float],
    fftshift: bool = False,
    compute_euclidean_norm: bool = False,
    shape_is_real_fourier: bool = False,
    return_sparse_grid: bool = False,
) -> NDArray:
    """
    Generate n-dimensional (frequency) grids.

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the data.
    sampling_rate : float or Tuple[float]
        Sets the maximum value along each axis in shape to x=1/(2*sampling_rate), e.g.,
        a sampling_rate of 1 yields a grid from -n/x * 1/n to (n)/x -1 * 1/n. A sampling
        rate of None returns a grid from -n/2 to n/2 - 1
    fftshift : bool, optional
        Whether to return a grid centered at shape // 2. Default is grid centered around
        origin, which is compliant with the rfftn definitions used in this project.
    compute_euclidean_norm : bool, optional
        Whether to compute the Euclidean norm, defaults to False.
    shape_is_real_fourier : bool, optional
        Whether the shape corresponds to a real Fourier transform, defaults to False.

    Returns:
    --------
    NDArray
        The sample frequencies.
    """
    # There is no real need to have these operations on GPU right now
    np_be = NumpyFFTWBackend()
    norm = np_be.full(len(shape), fill_value=1, dtype=np_be._float)
    center = np_be.astype(np_be.divide(shape, 2), np_be._int)
    if sampling_rate is not None:
        norm = np_be.astype(np_be.multiply(shape, sampling_rate), int)

    if shape_is_real_fourier:
        center[-1], norm[-1] = 0, 1
        if sampling_rate is not None:
            norm[-1] = (shape[-1] - 1) * 2 * sampling_rate

    ndim, grids = len(shape), []
    for i, x in enumerate(shape):
        baseline_dims = tuple(1 if i != t else x for t in range(len(shape)))
        grid = (np_be.arange(x, dtype=np_be._int) - center[i]) / norm[i]

        # We have to invert because we build the grid centered around shape // 2
        if not fftshift:
            if shape_is_real_fourier and i == (ndim - 1):
                pass
            else:
                grid = np.fft.ifftshift(grid)

        grid = np_be.astype(grid, np_be._float)
        grids.append(np_be.reshape(grid, baseline_dims))

    if compute_euclidean_norm:
        grids = sum(np_be.square(x) for x in grids)
        grids = np_be.sqrt(grids, out=grids)
        return grids

    if return_sparse_grid:
        return grids

    grid_flesh = np_be.full(shape, fill_value=1, dtype=np_be._float)
    return np_be.stack(tuple(grid * grid_flesh for grid in grids))


def crop_real_fourier(data: BackendArray) -> BackendArray:
    """
    Crop the real part of a Fourier transform.

    Parameters
    ----------
    data : BackendArray
        The Fourier transformed data.

    Returns:
    --------
    BackendArray
        The cropped data.
    """
    stop = 1 + (data.shape[-1] // 2)
    return data[..., :stop]


def compute_fourier_shape(
    shape: Tuple[int], shape_is_real_fourier: bool = False
) -> List[int]:
    if shape_is_real_fourier:
        return shape
    shape = [int(x) for x in shape]
    shape[-1] = 1 + shape[-1] // 2
    return shape


def shift_fourier(
    data: BackendArray, shape_is_real_fourier: bool = False, ifftshift: bool = True
) -> BackendArray:
    comp = be
    if isinstance(data, np.ndarray):
        comp = NumpyFFTWBackend()

    shape = comp.to_backend_array(data.shape)
    shift = comp.divide(shape, 2)
    if ifftshift:
        shift = comp.add(shift, comp.mod(shape, 2))

    shift = [int(x) for x in shift]
    if shape_is_real_fourier:
        shift[-1] = 0
    return comp.roll(data, shift, tuple(i for i in range(len(shift))))


def create_reconstruction_filter(
    filter_shape: Tuple[int], filter_type: str, fftshift: bool = True, **kwargs: Dict
):
    """
    Create a reconstruction filter of given filter_type.

    Parameters
    ----------
    filter_shape : tuple of int
        Shape of the returned filter.
    filter_type: str
        The type of created filter, available options are:

        +---------------+----------------------------------------------------+
        | ram-lak       | Returns |w|                                        |
        +---------------+----------------------------------------------------+
        | ramp-cont     | Principles of Computerized Tomographic Imaging Avin|
        |               | ash C. Kak and Malcolm Slaney Chap 3 Eq. 61 [1]_   |
        +---------------+----------------------------------------------------+
        | ramp          | Like ramp-cont but considering tilt angles         |
        +---------------+----------------------------------------------------+
        | shepp-logan   | |w| * sinc(|w| / 2) [2]_                           |
        +---------------+----------------------------------------------------+
        | cosine        | |w| * cos(|w| * pi / 2) [2]_                       |
        +---------------+----------------------------------------------------+
        | hamming       | |w| * (.54 + .46 ( cos(|w| * pi))) [2]_            |
        +---------------+----------------------------------------------------+
    fftshift : bool, optional
        Should the DC component be located at the center, default is True.
    kwargs: Dict
        Keyword arguments for particular filter_types.

    Returns
    -------
    NDArray
        Reconstruction filter

    References
    ----------
    .. [1]  Principles of Computerized Tomographic Imaging Avinash C. Kak and Malcolm Slaney Chap 3 Eq. 61
    .. [2]  https://odlgroup.github.io/odl/index.html
    """
    filter_type = str(filter_type).lower()
    freq = fftfreqn(
        filter_shape, sampling_rate=0.5, compute_euclidean_norm=True, fftshift=fftshift
    )

    if filter_type == "ram-lak":
        ret = np.copy(freq)
    elif filter_type == "ramp-cont":
        ret, ndim = None, len(filter_shape)
        for dim, size in enumerate(filter_shape):
            n = np.concatenate(
                (
                    np.arange(1, size // 2 + 1, 2, dtype=int),
                    np.arange(size // 2 - 1, 0, -2, dtype=int),
                )
            )
            ret1d = np.zeros(size)
            ret1d[0] = 0.25
            ret1d[1::2] = -1 / (np.pi * n) ** 2
            ret1d_shape = tuple(size if i == dim else 1 for i in range(ndim))
            ret1d = ret1d.reshape(ret1d_shape)
            if ret is None:
                ret = ret1d
            else:
                ret = ret * ret1d
        ret = 2 * np.fft.fftshift(np.real(np.fft.fftn(ret)))
    elif filter_type == "ramp":
        tilt_angles = kwargs.get("tilt_angles", False)
        if tilt_angles is False:
            raise ValueError("'ramp' filter requires specifying tilt angles.")
        size = filter_shape[0]
        ret = fftfreqn(
            (size,), sampling_rate=1, compute_euclidean_norm=True, fftshift=fftshift
        )
        min_increment = np.radians(np.min(np.abs(np.diff(np.sort(tilt_angles)))))
        ret *= min_increment * size
        ret = np.fmin(ret, 1, out=ret)
    elif filter_type == "shepp-logan":
        ret = freq * np.sinc(freq / 2)
    elif filter_type == "cosine":
        ret = freq * np.cos(freq * np.pi / 2)
    elif filter_type == "hamming":
        ret = freq * (0.54 + 0.46 * np.cos(freq * np.pi))
    else:
        raise ValueError("Unsupported filter type")

    return ret


def pad_to_length(arr, length: int):
    ret = np.atleast_1d(arr)
    return np.repeat(ret, length // ret.size)


def gridding_correction(
    shape: Tuple[int, ...], padding_factor: int = 1, fftshift: bool = False
) -> NDArray:
    """
    Compute separable sinc^2 gridding correction for trilinear interpolation.

    Trilinear interpolation is separable, so the correction is the product
    of per-axis sinc^2 terms rather than a single radial sinc^2.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the data.
    padding_factor : int
        The oversampling/padding factor used during reconstruction.
    fftshift : bool, optional
        Whether to return a grid centered at shape // 2. Default is grid centered around
        origin, which is compliant with the rfftn definitions used in this project.

    Returns
    -------
    NDArray
        Separable sinc^2 correction of given shape.
    """
    grids = fftfreqn(
        shape, sampling_rate=None, fftshift=fftshift, return_sparse_grid=True
    )
    correction = np.ones(shape)
    for i, grid in enumerate(grids):
        normalized = grid / (shape[i] * padding_factor)
        sinc_val = np.sinc(normalized)
        correction *= sinc_val * sinc_val
    return correction


def radial_average(
    data_fft: BackendArray, n_bins: int = None, shape_is_real_fourier: bool = True
) -> Tuple[BackendArray, BackendArray]:
    """
    Compute the radial power spectrum of the input data.

    Parameters
    ----------
    data_fft : BackendArray
        The Fourier transform of the input data with DC at origin.
    n_bins : int, optional
        The number of bins for computing the spectrum, defaults to None.
    shape_is_real_fourier : bool
        Whether the input it the rfftn or fftn of the data.

    Returns
    -------
    bin_centers : BackendArray
        Frequency values at the center of each bin (range 0 to 1).
    radial_averages : BackendArray
        Normalized inverse-amplitude spectrum per bin.
    """
    from scipy.ndimage import mean as ndimean

    if not shape_is_real_fourier:
        data_fft = crop_real_fourier(data_fft)

    bin_indices, bin_centers = radial_bins(data_fft.shape, n_bins)

    fourier_spectrum = np.abs(data_fft)
    fourier_spectrum = np.square(fourier_spectrum, out=fourier_spectrum)

    radial_averages = ndimean(
        fourier_spectrum, labels=bin_indices, index=np.arange(bin_centers.shape[0])
    )
    radial_averages = np.sqrt(radial_averages, out=radial_averages)

    radial_averages = np.where(radial_averages != 0, 1 / radial_averages, 0)
    norm_factor = radial_averages.max()
    if norm_factor != 0:
        radial_averages = np.divide(radial_averages, norm_factor)

    return bin_centers, radial_averages


def radial_bins(shape: Tuple[int, ...], n_bins: int = None) -> Tuple[NDArray, NDArray]:
    """Assign the voxels of an rfft-shaped grid to radial shells.

    Parameters
    ----------
    shape : tuple of int
        Shape of the real Fourier (rfft) grid to bin.
    n_bins : int, optional
        Number of radial bins, capped at the largest meaningful value for the
        shape. Defaults to that maximum.

    Returns
    -------
    bin_indices : NDArray
        Per voxel shell index over ``shape``.
    bin_centers : NDArray
        Fractional frequency (0 to 1 at Nyquist) at each shell center, of length
        ``n_bins``.
    """
    max_bins = shape[-1]
    if len(shape) > 1:
        max_bins = max(max(shape[:-1]) // 2 + 1, max_bins)

    n_bins = max_bins if n_bins is None else n_bins
    n_bins = int(min(n_bins, max_bins))

    freqs = fftfreqn(
        shape=shape,
        sampling_rate=0.5,
        compute_euclidean_norm=True,
        shape_is_real_fourier=True,
        fftshift=False,
    )
    bin_indices = np.floor(freqs * (n_bins - 1) + 0.5).astype(int)
    bin_centers = fftfreqn(
        shape=(n_bins,),
        sampling_rate=0.5,
        compute_euclidean_norm=True,
        shape_is_real_fourier=True,
        fftshift=False,
    )
    return bin_indices, bin_centers


def power_at_tilt(
    data_fft: BackendArray,
    n_bins: int = 90,
    shape_is_real_fourier: bool = True,
    mask: NDArray = None,
) -> Tuple[NDArray, NDArray]:
    """
    Compute average Fourier power as a function of tilt angle.

    For each angle, samples the 2D Fourier amplitude along a ray from the
    origin at that angle using linear interpolation, and returns the mean
    squared amplitude.

    Parameters
    ----------
    data_fft : BackendArray
        2D Fourier transform with DC at origin.
    n_bins : int, optional
        Number of angles from 0 to 90 degrees, defaults to 90.
    shape_is_real_fourier : bool, optional
        Whether data_fft is from rfftn (True) or fftn (False).
    mask : NDArray, optional
        Binary mask matching data_fft. When provided, averages only over
        non-zero elements along each ray.

    Returns
    -------
    bin_centers : NDArray
        Angles in degrees (0 to 90).
    powers : NDArray
        Mean squared Fourier amplitude per angle.
    """
    from scipy.ndimage import map_coordinates

    if not shape_is_real_fourier:
        data_fft = crop_real_fourier(data_fft)
        if mask is not None:
            mask = crop_real_fourier(mask)

    shape = data_fft.shape

    fourier_power = np.abs(data_fft)
    np.square(fourier_power, out=fourier_power)

    n_samples = max(shape)
    t = np.linspace(0, 1, n_samples)

    angles = np.linspace(0, 90, n_bins)
    angles_rad = np.radians(angles)

    # (n_bins, n_samples) coordinates along each ray
    freq_row = np.outer(np.sin(angles_rad), t)
    freq_col = np.outer(np.cos(angles_rad), t)

    row_idx = freq_row * shape[0] * 0.5
    col_idx = freq_col * (shape[1] - 1)

    coords = np.array([row_idx.ravel(), col_idx.ravel()])

    sampled = map_coordinates(fourier_power, coords, order=1).reshape(n_bins, n_samples)

    if mask is not None:
        mask_sampled = map_coordinates(
            mask.astype(np.float64), coords, order=1
        ).reshape(n_bins, n_samples)
        counts = np.maximum(mask_sampled.sum(axis=1), 1)
        powers = (sampled * mask_sampled).sum(axis=1) / counts
    else:
        powers = sampled.mean(axis=1)

    return angles, powers
