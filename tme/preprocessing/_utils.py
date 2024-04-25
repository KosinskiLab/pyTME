""" Utilities for the generation of frequency grids.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple

import numpy as np
from numpy.typing import NDArray

from ..backends import backend
from ..matching_utils import euler_to_rotationmatrix


def compute_tilt_shape(shape: Tuple[int], opening_axis: int, reduce_dim: bool = False):
    """
    Given an opening_axis, computes the shape of the remaining dimensions.

    Parameters:
    -----------
    shape : Tuple[int]
        The shape of the input array.
    opening_axis : int
        The axis along which the array will be tilted.
    reduce_dim : bool, optional (default=False)
        Whether to reduce the dimensionality after tilting.

    Returns:
    --------
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

    Parameters:
    -----------
    shape : Tuple[int]
        The shape of the grid.

    Returns:
    --------
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

    Parameters:
    -----------
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
    index_grid = centered_grid(shape=tilt_shape)
    if angle != 0:
        angles = np.zeros(len(shape))
        angles[tilt_axis] = angle
        rotation_matrix = euler_to_rotationmatrix(np.roll(angles, opening_axis - 1))
        index_grid = np.einsum("ij,j...->i...", rotation_matrix, index_grid)

    norm = np.divide(1, 2 * sampling_rate * np.divide(shape, 2).astype(int))

    index_grid = np.multiply(index_grid.T, norm).T
    index_grid = np.squeeze(index_grid)
    index_grid = np.linalg.norm(index_grid, axis=(0))
    return index_grid


def fftfreqn(
    shape: Tuple[int],
    sampling_rate: Tuple[float],
    compute_euclidean_norm: bool = False,
    shape_is_real_fourier: bool = False,
) -> NDArray:
    """
    Generate the n-dimensional discrete Fourier Transform sample frequencies.

    Parameters:
    -----------
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
    center = backend.astype(backend.divide(shape, 2), backend._default_dtype_int)

    norm = np.ones(3)
    if sampling_rate is not None:
        norm = backend.multiply(shape, sampling_rate).astype(int)

    if shape_is_real_fourier:
        center[-1] = 0
        norm[-1] = 1
        if sampling_rate is not None:
            norm[-1] = (shape[-1] - 1) * 2 * sampling_rate

    indices = backend.transpose(backend.indices(shape))
    indices -= center
    indices = backend.divide(indices, norm)
    indices = backend.transpose(indices)

    if compute_euclidean_norm:
        backend.square(indices, indices)
        indices = backend.sum(indices, axis=0)
        indices = backend.sqrt(indices)

    return indices


def crop_real_fourier(data: NDArray) -> NDArray:
    """
    Crop the real part of a Fourier transform.

    Parameters:
    -----------
    data : NDArray
        The Fourier transformed data.

    Returns:
    --------
    NDArray
        The cropped data.
    """
    stop = 1 + (data.shape[-1] // 2)
    return data[..., :stop]
