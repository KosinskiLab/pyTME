"""
Compute memory consumption of template matching components.

Copyright (c) 2023-2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional

import numpy as np

from .backends import backend as be

MATCHING_MEMORY_REGISTRY = {}


def register_memory(*names: str):
    """
    Decorator to auto-register memory estimators.

    Parameters
    ----------
    *names : str
        Names to register this memory estimator under.
    """

    def decorator(cls):
        for name in names:
            MATCHING_MEMORY_REGISTRY[name] = cls
        return cls

    return decorator


class MatchingMemoryUsage(ABC):
    """
    Strategy class for estimating memory requirements.

    Parameters
    ----------
    fast_shape : tuple of int
        Shape of the real array.
    ft_shape : tuple of int
        Shape of the complex array.
    float_nbytes : int
        Number of bytes of the used float, e.g. 4 for float32.
    complex_nbytes : int
        Number of bytes of the used complex, e.g. 8 for complex64.
    integer_nbytes : int
        Number of bytes of the used integer, e.g. 4 for int32.
    """

    def __init__(
        self,
        fast_shape: Tuple[int, ...],
        ft_shape: Tuple[int, ...],
        float_nbytes: int,
        complex_nbytes: int,
        integer_nbytes: int,
    ):
        self.real_array_size = np.prod(fast_shape)
        self.complex_array_size = np.prod(ft_shape)
        self.float_nbytes = float_nbytes
        self.complex_nbytes = complex_nbytes
        self.integer_nbytes = integer_nbytes

    @abstractmethod
    def base_usage(self) -> int:
        """Return the base memory usage in bytes."""

    @abstractmethod
    def per_fork(self) -> int:
        """Return the memory usage per fork in bytes."""


class MemoryProfile(MatchingMemoryUsage):
    """Memory estimator for methods with uniform array requirements."""

    #: Number of shared real arrays
    base_float: int = 0
    #: Number of shared complex arrays
    base_complex: int = 0
    #: Number of real arrays per fork
    fork_float: int = 0
    #: Number of complex arrays per fork
    fork_complex: int = 0

    def base_usage(self) -> int:
        return (
            self.base_float * self.real_array_size * self.float_nbytes
            + self.base_complex * self.complex_array_size * self.complex_nbytes
        )

    def per_fork(self) -> int:
        return (
            self.fork_float * self.real_array_size * self.float_nbytes
            + self.fork_complex * self.complex_array_size * self.complex_nbytes
        )


@register_memory("CC", "LCC")
class CCMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.cc_setup` memory estimator."""

    base_float, base_complex = 1, 1
    fork_float, fork_complex = 1, 1


@register_memory("CORR", "NCC", "CAM", "FLCSphericalMask", "batchFLCSphericalMask")
class CORRMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.corr_setup` memory estimator."""

    base_float, base_complex = 4, 1
    fork_float, fork_complex = 1, 1


@register_memory("FLC", "batchFLC")
class FLCMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.flc_setup` memory estimator."""

    base_float, base_complex = 2, 2
    fork_float, fork_complex = 3, 2


@register_memory("MCC")
class MCCMemoryUsage(MemoryProfile):
    """:py:meth:`tme.matching_scores.mcc_setup` memory estimator."""

    base_float, base_complex = 2, 3
    fork_float, fork_complex = 6, 1


@register_memory("MaxScoreOverRotations")
class MaxScoreOverRotationsMemoryUsage(MemoryProfile):
    """:py:class:`tme.analyzer.MaxScoreOverRotations` memory estimator."""

    base_float = 2


@register_memory("MaxScoreOverRotationsConstrained")
class MaxScoreOverRotationsConstrainedMemoryUsage(MemoryProfile):
    """:py:class:`tme.analyzer.MaxScoreOverRotationsConstrained` memory estimator."""

    # This ultimately depends on the number of seed points and mask size.
    # Ideally we would use that in the memory estimation, but for now we
    # approximate by reqesting memory for another real array
    base_float = 3


@register_memory("PeakCallerMaximumFilter")
class PeakCallerMaximumFilterMemoryUsage(MemoryProfile):
    """:py:class:`tme.analyzer.peaks.PeakCallerMaximumFilter` memory estimator."""

    base_float, fork_float = 1, 1


@register_memory("cupy", "pytorch")
class CupyBackendMemoryUsage(MemoryProfile):
    """:py:class:`tme.backends.CupyBackend` memory estimator."""

    # FFT plans, overhead from assigning FFT result, rotation interpolation
    base_complex, base_float = 3, 2


def estimate_memory_usage(
    shape1: Tuple[int, ...],
    shape2: Tuple[int, ...],
    matching_method: str,
    ncores: int,
    analyzer_method: Optional[str] = None,
    backend: Optional[str] = None,
    float_nbytes: int = 4,
    complex_nbytes: int = 8,
    integer_nbytes: int = 4,
) -> int:
    """
    Estimate the memory usage for a given template matching run.

    Parameters
    ----------
    shape1 : tuple
        Shape of the target array.
    shape2 : tuple
        Shape of the template array.
    matching_method : str
        Matching method to estimate memory usage for.
    analyzer_method : str, optional
        The method used for score analysis.
    backend : str, optional
        Backend used for computation.
    ncores : int
        The number of CPU cores used for the operation.
    float_nbytes : int
        Number of bytes of the used float, defaults to 4 (float32).
    complex_nbytes : int
        Number of bytes of the used complex, defaults to 8 (complex64).
    integer_nbytes : int
        Number of bytes of the used integer, defaults to 4 (int32).

    Returns
    -------
    int
        The estimated memory usage for the operation in bytes.

    Raises
    ------
    ValueError
        If an unsupported matching_method is provided.
    """
    if matching_method not in MATCHING_MEMORY_REGISTRY:
        raise ValueError(
            f"Supported options are {','.join(MATCHING_MEMORY_REGISTRY.keys())}"
        )

    _, fast_shape, ft_shape = be.compute_convolution_shapes(shape1, shape2)
    memory_instance = MATCHING_MEMORY_REGISTRY[matching_method](
        fast_shape=fast_shape,
        ft_shape=ft_shape,
        float_nbytes=float_nbytes,
        complex_nbytes=complex_nbytes,
        integer_nbytes=integer_nbytes,
    )

    nbytes = memory_instance.base_usage() + memory_instance.per_fork() * ncores

    if analyzer_method in MATCHING_MEMORY_REGISTRY:
        analyzer_instance = MATCHING_MEMORY_REGISTRY[analyzer_method](
            fast_shape=fast_shape,
            ft_shape=ft_shape,
            float_nbytes=float_nbytes,
            complex_nbytes=complex_nbytes,
            integer_nbytes=integer_nbytes,
        )
        nbytes += analyzer_instance.base_usage() + analyzer_instance.per_fork() * ncores

    if backend in MATCHING_MEMORY_REGISTRY:
        backend_instance = MATCHING_MEMORY_REGISTRY[backend](
            fast_shape=fast_shape,
            ft_shape=ft_shape,
            float_nbytes=float_nbytes,
            complex_nbytes=complex_nbytes,
            integer_nbytes=integer_nbytes,
        )
        nbytes += backend_instance.base_usage() + backend_instance.per_fork() * ncores

    return nbytes
