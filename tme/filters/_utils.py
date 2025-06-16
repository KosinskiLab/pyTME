"""
Utilities for the generation of frequency grids.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, List, Dict

import numpy as np

from ..backends import backend as be
from ..backends import NumpyFFTWBackend
from ..types import BackendArray, NDArray
from ..rotations import euler_to_rotationmatrix


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


def centered_grid(shape: Tuple[int]) -> NDArray:
    """
    Generate an integer valued grid centered around size // 2

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the grid.

    Returns
    -------
    NDArray
        The centered grid.
    """
    index_grid = np.array(
        np.meshgrid(*[np.arange(size) - size // 2 for size in shape], indexing="ij")
    )
    return index_grid


def frequency_grid_at_angle(
    shape: Tuple[int],
    angle: float,
    sampling_rate: Tuple[float],
    opening_axis: int = None,
    tilt_axis: int = None,
) -> NDArray:
    """
    Generate a frequency grid from 0 to 1/(2 * sampling_rate) in each axis.

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the grid.
    angle : float
        The angle at which to generate the grid.
    sampling_rate : Tuple[float]
        The sampling rate for each dimension.
    opening_axis : int, optional
        The axis to be opened, defaults to None.
    tilt_axis : int, optional
        The axis along which the grid is tilted, defaults to None.

    Returns:
    --------
    NDArray
        The frequency grid.
    """
    sampling_rate = np.array(sampling_rate)
    sampling_rate = np.repeat(sampling_rate, len(shape) // sampling_rate.size)

    tilt_shape = compute_tilt_shape(
        shape=shape, opening_axis=opening_axis, reduce_dim=False
    )

    if angle == 0:
        sampling_rate = compute_tilt_shape(
            shape=sampling_rate, opening_axis=opening_axis, reduce_dim=True
        )
        index_grid = fftfreqn(
            tuple(x for x in tilt_shape if x != 1),
            sampling_rate=sampling_rate,
            compute_euclidean_norm=True,
        )

    if angle != 0:
        aspect_ratio = shape[opening_axis] / shape[tilt_axis]
        angle = np.degrees(np.arctan(np.tan(np.radians(angle)) * aspect_ratio))

        angles = np.zeros(len(shape))
        angles[tilt_axis] = angle
        rotation_matrix = euler_to_rotationmatrix(np.roll(angles, opening_axis - 1))

        index_grid = fftfreqn(tilt_shape, sampling_rate=None)
        index_grid = np.einsum("ij,j...->i...", rotation_matrix, index_grid)
        norm = np.multiply(sampling_rate, shape).astype(int)

        index_grid = np.divide(index_grid.T, norm).T
        index_grid = np.squeeze(index_grid)
        index_grid = np.linalg.norm(index_grid, axis=(0))

    return index_grid


def fftfreqn(
    shape: Tuple[int],
    sampling_rate: Tuple[float],
    compute_euclidean_norm: bool = False,
    shape_is_real_fourier: bool = False,
    return_sparse_grid: bool = False,
) -> NDArray:
    """
    Generate the n-dimensional discrete Fourier transform sample frequencies.

    Parameters
    ----------
    shape : Tuple[int]
        The shape of the data.
    sampling_rate : float or Tuple[float]
        The sampling rate.
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
    norm = np_be.full(len(shape), fill_value=1, dtype=np_be._float_dtype)
    center = np_be.astype(np_be.divide(shape, 2), np_be._int_dtype)
    if sampling_rate is not None:
        norm = np_be.astype(np_be.multiply(shape, sampling_rate), int)

    if shape_is_real_fourier:
        center[-1], norm[-1] = 0, 1
        if sampling_rate is not None:
            norm[-1] = (shape[-1] - 1) * 2 * sampling_rate

    grids = []
    for i, x in enumerate(shape):
        baseline_dims = tuple(1 if i != t else x for t in range(len(shape)))
        grid = (np_be.arange(x, dtype=np_be._int_dtype) - center[i]) / norm[i]
        grid = np_be.astype(grid, np_be._float_dtype)
        grids.append(np_be.reshape(grid, baseline_dims))

    if compute_euclidean_norm:
        grids = sum(np_be.square(x) for x in grids)
        grids = np_be.sqrt(grids, out=grids)
        return grids

    if return_sparse_grid:
        return grids

    grid_flesh = np_be.full(shape, fill_value=1, dtype=np_be._float_dtype)
    grids = np_be.stack(tuple(grid * grid_flesh for grid in grids))

    return grids


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
    data: BackendArray, shape_is_real_fourier: bool = False
) -> BackendArray:
    comp = be
    if isinstance(data, np.ndarray):
        comp = NumpyFFTWBackend()
    shape = comp.to_backend_array(data.shape)
    shift = comp.add(comp.divide(shape, 2), comp.mod(shape, 2))
    shift = [int(x) for x in shift]
    if shape_is_real_fourier:
        shift[-1] = 0

    data = comp.roll(data, shift, tuple(i for i in range(len(shift))))
    return data


def create_reconstruction_filter(
    filter_shape: Tuple[int], filter_type: str, **kwargs: Dict
):
    """Create a reconstruction filter of given filter_type.

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
    freq = fftfreqn(filter_shape, sampling_rate=0.5, compute_euclidean_norm=True)

    if filter_type == "ram-lak":
        ret = np.copy(freq)
    elif filter_type == "ramp-cont":
        ret, ndim = None, len(filter_shape)
        for dim, size in enumerate(filter_shape):
            n = np.concatenate(
                (
                    np.arange(1, size / 2 + 1, 2, dtype=int),
                    np.arange(size / 2 - 1, 0, -2, dtype=int),
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
        ret = fftfreqn((size,), sampling_rate=1, compute_euclidean_norm=True)
        min_increment = np.radians(np.min(np.abs(np.diff(np.sort(tilt_angles)))))
        ret *= min_increment * size
        np.fmin(ret, 1, out=ret)
    elif filter_type == "shepp-logan":
        ret = freq * np.sinc(freq / 2)
    elif filter_type == "cosine":
        ret = freq * np.cos(freq * np.pi / 2)
    elif filter_type == "hamming":
        ret = freq * (0.54 + 0.46 * np.cos(freq * np.pi))
    else:
        raise ValueError("Unsupported filter type")

    return ret
