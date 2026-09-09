"""
Analyzer utility functions.

Copyright (c) 2023-2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Union

import numpy as np

from ..backends import backend as be
from ..types import BackendArray, NDArray

__all__ = ["cart_to_score", "score_to_cart", "upsampled_dft"]


def upsampled_dft(data, upsampled_region_size, upsample_factor=1, axis_offsets=None):
    """
    Upsampled DFT by matrix multiplication.

    Computes the DFT of ``data`` at a set of finely spaced output points
    without zero-padding, by evaluating the Fourier basis at fractional
    frequencies via matrix multiplication. This is much faster than
    zero-padding when ``upsampled_region_size`` is small relative to
    ``data.size * upsample_factor``.

    Parameters
    ----------
    data : ndarray
        The input data array (DFT of original data) to upsample.
    upsampled_region_size : int or tuple of int
        The size of the region to be sampled. If one integer is provided, it
        is duplicated up to the dimensionality of ``data``.
    upsample_factor : int, optional
        The upsampling factor. Defaults to 1.
    axis_offsets : tuple of int, optional
        The offsets of the region to be sampled. Defaults to None (uses
        image center).

    Returns
    -------
    ndarray
        The upsampled DFT of the specified region.

    References
    ----------
    .. [1]  Manuel Guizar-Sicairos, Samuel T. Thurman, and James R. Fienup,
            "Efficient subpixel image registration algorithms,"
            Optics Letters 33, 156-158 (2008). DOI:10.1364/OL.33.000156
    """
    if not hasattr(upsampled_region_size, "__iter__"):
        upsampled_region_size = [upsampled_region_size] * data.ndim
    elif len(upsampled_region_size) != data.ndim:
        raise ValueError(
            "shape of upsampled region sizes must be equal "
            "to input data's number of dimensions."
        )

    if axis_offsets is None:
        axis_offsets = [0] * data.ndim
    elif len(axis_offsets) != data.ndim:
        raise ValueError(
            "number of axis offsets must be equal to input "
            "data's number of dimensions."
        )

    im2pi = 1j * 2 * np.pi

    for n_items, ups_size, ax_offset in list(
        zip(data.shape, upsampled_region_size, axis_offsets)
    )[::-1]:
        kernel = (np.arange(ups_size) - ax_offset)[:, None] * np.fft.fftfreq(
            n_items, upsample_factor
        )
        kernel = np.exp(-im2pi * kernel).astype(data.dtype, copy=False)
        data = np.tensordot(kernel, data, axes=(1, -1))

    return data


def _convmode_to_shape(
    convolution_mode: str,
    targetshape: Tuple[int],
    templateshape: Tuple[int],
    **kwargs,
) -> Tuple[int]:
    """
    Calculate convolution shape based on convolution mode.

    Parameters
    ----------
    convolution_mode : str
        Mode of convolution. Supported values are:
        - 'same': Output shape will match target shape.
        - 'valid': All elements that do not rely on zero padding.
        - 'full': Full discrete linear cross-correlation of inputs.
    targetshape : tuple of int
        Shape of the target array.
    templateshape : tuple of int
        Shape of the template array.

    Returns
    -------
    tuple of int
        Convolution shape.
    """
    if convolution_mode == "same":
        output_shape = targetshape
    elif convolution_mode == "valid":
        output_shape = tuple(x - y + 1 for x, y in zip(targetshape, templateshape))
    elif convolution_mode == "full":
        output_shape = tuple(x + y - 1 for x, y in zip(targetshape, templateshape))
    else:
        raise ValueError("Supported convolution modes are 'same', 'valid', and 'full'.")
    return tuple(int(x) for x in output_shape)


def cart_to_score(
    positions: Union[BackendArray, NDArray],
    fast_shape: Tuple[int],
    targetshape: Tuple[int],
    templateshape: Tuple[int],
    convolution_shape: Tuple[int] = None,
    fourier_shift: Tuple[int] = None,
    convolution_mode: str = None,
    **kwargs,
) -> Tuple[Union[BackendArray, NDArray]]:
    """Maps peak positions from cartesian to padded score space coordinates."""
    xp = be
    to_array = xp.to_backend_array
    if isinstance(positions, np.ndarray):
        from ..backends import NumpyFFTWBackend

        xp = NumpyFFTWBackend()
        to_array = np.asarray

    positions = to_array(positions)
    fast_shape = to_array(fast_shape)
    if convolution_mode is None:
        valid_positions = (positions >= 0) & (positions < fast_shape)
        valid_positions = xp.sum(valid_positions, axis=1) == positions.shape[1]
        return positions, valid_positions

    # Offset from padding to Fourier friendly shapes
    output_shape = _convmode_to_shape(
        convolution_mode=convolution_mode,
        targetshape=targetshape,
        templateshape=templateshape,
    )
    output_shape = to_array(output_shape)
    convolution_shape = to_array(convolution_shape)

    # Offset from padding the target
    starts = xp.astype(
        xp.divide(xp.subtract(convolution_shape, output_shape), 2),
        xp._int,
    )
    valid_positions = (positions >= -starts) & (positions <= output_shape)
    valid_positions = xp.sum(valid_positions, axis=1) == positions.shape[1]

    positions = xp.add(positions, starts)
    if fourier_shift is not None:
        positions = xp.subtract(positions, to_array(fourier_shift))
        positions = xp.mod(positions, fast_shape)
    return positions, valid_positions


def score_to_cart(
    positions,
    fast_shape: Tuple[int] = None,
    targetshape: Tuple[int] = None,
    templateshape: Tuple[int] = None,
    convolution_shape: Tuple[int] = None,
    fourier_shift: Tuple[int] = None,
    convolution_mode: str = None,
    **kwargs,
) -> Tuple[BackendArray]:
    """
    Maps peak positions from padded score to cartesian coordinates.

    Parameters
    ----------
    positions : BackendArray
        Positions in padded Fourier space system.
    fast_shape : tuple of int
        Shape of the score space padded to efficient Fourier shape.
    targetshape : tuple of int
        Shape of the target array.
    templateshape : tuple of int
        Shape of the template array.
    convolution_shape : tuple of int, optional
        Non-padded convolution_shape of template and target.
    fourier_shift : tuple of int, optional
        Translation offset of coordinates.
    convolution_mode : str, optional
        Mode of convolution ('same', 'valid', or 'full')

    Returns
    -------
    Tuple of BackendArray
        Adjusted positions. and boolean array indicating whether corresponding
        positions are valid positions w.r.t. to supplied bounds.
    """
    positions = be.to_backend_array(positions)
    convolution_shape = be.to_backend_array(convolution_shape)
    fast_shape = be.to_backend_array(fast_shape)
    targetshape = be.to_backend_array(targetshape)
    templateshape = be.to_backend_array(templateshape)

    valid_positions = be.ones((positions.shape[0],)) == 1

    # Wrap peaks around score space
    if fourier_shift is not None:
        fourier_shift = be.to_backend_array(fourier_shift)
        positions = be.add(positions, fourier_shift)
        positions = be.mod(positions, fast_shape)

    if convolution_mode is not None:
        output_shape = _convmode_to_shape(
            convolution_mode=convolution_mode,
            targetshape=targetshape,
            templateshape=templateshape,
            convolution_shape=convolution_shape,
        )
        output_shape = be.to_backend_array(output_shape)
        starts = be.astype(
            be.divide(be.subtract(convolution_shape, output_shape), 2),
            be._int,
        )
        stops = be.add(starts, output_shape)

        valid_positions = be.multiply(positions >= starts, positions < stops)
        valid_positions = be.sum(valid_positions, axis=1) == positions.shape[1]
        positions = be.subtract(positions, starts)

    # Get rid of -1 position peaks used to keep GPU happy
    valid_positions = be.multiply(
        be.sum(positions >= 0, axis=1) == positions.shape[1], valid_positions
    )
    return positions, valid_positions
