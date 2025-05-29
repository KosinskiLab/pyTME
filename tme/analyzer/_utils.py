""" Analyzer utility functions.

    Copyright (c) 2023-2025 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple

from ..types import BackendArray
from ..backends import backend as be

__all__ = ["cart_to_score", "score_to_cart"]


def _convmode_to_shape(
    convolution_mode: str,
    targetshape: BackendArray,
    templateshape: BackendArray,
    convolution_shape: BackendArray,
) -> BackendArray:
    """
    Calculate convolution shape based on convolution mode.

    Parameters
    ----------
    convolution_mode : str
        Mode of convolution. Supported values are:
        - 'same': Output shape will match target shape
        - 'valid': Output shape will be target shape minus template shape plus
                   template shape modulo 2
        - Other: Output shape will be equal to convolution_shape
    targetshape : BackendArray
        Shape of the target array.
    templateshape : BackendArray
        Shape of the template array.
    convolution_shape : BackendArray
        Shape of the convolution output.

    Returns
    -------
    BackendArray
        Convolution shape.
    """
    output_shape = convolution_shape
    if convolution_mode == "same":
        output_shape = targetshape
    elif convolution_mode == "valid":
        output_shape = be.add(
            be.subtract(targetshape, templateshape),
            be.mod(templateshape, 2),
        )
    return be.to_backend_array(output_shape)


def cart_to_score(
    positions: BackendArray,
    fast_shape: Tuple[int],
    targetshape: Tuple[int],
    templateshape: Tuple[int],
    convolution_shape: Tuple[int] = None,
    fourier_shift: Tuple[int] = None,
    convolution_mode: str = None,
    **kwargs,
) -> Tuple[BackendArray]:
    """
    Maps peak positions from cartesian to padded score space coordinates.

    Parameters
    ----------
    positions : BackendArray
        Positions in cartesian coordinates.
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
    fast_shape = be.to_backend_array(fast_shape)
    targetshape = be.to_backend_array(targetshape)
    templateshape = be.to_backend_array(templateshape)
    convolution_shape = be.to_backend_array(convolution_shape)

    # Compute removed padding
    output_shape = _convmode_to_shape(
        convolution_mode=convolution_mode,
        targetshape=targetshape,
        templateshape=templateshape,
        convolution_shape=convolution_shape,
    )
    valid_positions = be.multiply(positions >= 0, positions < output_shape)
    valid_positions = be.sum(valid_positions, axis=1) == positions.shape[1]

    starts = be.astype(
        be.divide(be.subtract(convolution_shape, output_shape), 2),
        be._int_dtype,
    )

    positions = be.add(positions, starts)
    if fourier_shift is not None:
        fourier_shift = be.to_backend_array(fourier_shift)
        positions = be.subtract(positions, fourier_shift)
        positions = be.mod(positions, fast_shape)

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

    # Wrap peaks around score space
    if fourier_shift is not None:
        fourier_shift = be.to_backend_array(fourier_shift)
        positions = be.add(positions, fourier_shift)
        positions = be.mod(positions, fast_shape)

    output_shape = _convmode_to_shape(
        convolution_mode=convolution_mode,
        targetshape=targetshape,
        templateshape=templateshape,
        convolution_shape=convolution_shape,
    )
    starts = be.astype(
        be.divide(be.subtract(convolution_shape, output_shape), 2),
        be._int_dtype,
    )
    stops = be.add(starts, output_shape)

    valid_positions = be.multiply(positions >= starts, positions < stops)
    valid_positions = be.sum(valid_positions, axis=1) == positions.shape[1]
    positions = be.subtract(positions, starts)

    return positions, valid_positions
