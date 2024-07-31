""" Utilities for the generation of frequency grids.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, List

import numpy as np

from ..backends import backend as be
from ..backends import NumpyFFTWBackend
from ..types import BackendArray, NDArray
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

    if angle == 0:
        index_grid = fftfreqn(
            tuple(x for x in tilt_shape if x != 1),
            sampling_rate=1,
            compute_euclidean_norm=True,
        )

    if angle != 0:
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
    # There is no real need to have these operations on GPU right now
    temp_backend = NumpyFFTWBackend()
    norm = temp_backend.full(len(shape), fill_value=1)
    center = temp_backend.astype(temp_backend.divide(shape, 2), temp_backend._int_dtype)
    if sampling_rate is not None:
        norm = temp_backend.astype(temp_backend.multiply(shape, sampling_rate), int)

    if shape_is_real_fourier:
        center[-1], norm[-1] = 0, 1
        if sampling_rate is not None:
            norm[-1] = (shape[-1] - 1) * 2 * sampling_rate

    grids = []
    for i, x in enumerate(shape):
        baseline_dims = tuple(1 if i != t else x for t in range(len(shape)))
        grid = (temp_backend.arange(x) - center[i]) / norm[i]
        grids.append(temp_backend.reshape(grid, baseline_dims))

    if compute_euclidean_norm:
        grids = sum(temp_backend.square(x) for x in grids)
        grids = temp_backend.sqrt(grids, out=grids)
        return grids

    if return_sparse_grid:
        return grids

    grid_flesh = temp_backend.full(shape, fill_value=1)
    grids = temp_backend.stack(tuple(grid * grid_flesh for grid in grids))

    return grids


def crop_real_fourier(data: BackendArray) -> BackendArray:
    """
    Crop the real part of a Fourier transform.

    Parameters:
    -----------
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
    shape = be.to_backend_array(data.shape)
    shift = be.add(be.divide(shape, 2), be.mod(shape, 2))
    shift = [int(x) for x in shift]
    if shape_is_real_fourier:
        shift[-1] = 0

    data = be.roll(data, shift, tuple(i for i in range(len(shift))))
    return data
