""" Compute memory consumption of template matching components.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np

from .backends import backend as be


class MatchingMemoryUsage(ABC):
    """
    Class specification for estimating the memory requirements of template matching.

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

    Attributes
    ----------
    real_array_size : int
        Number of elements in real array.
    complex_array_size : int
        Number of elements in complex array.
    float_nbytes : int
        Number of bytes of the used float, e.g. 4 for float32.
    complex_nbytes : int
        Number of bytes of the used complex, e.g. 8 for complex64.
    integer_nbytes : int
        Number of bytes of the used integer, e.g. 4 for int32.
    """

    def __init__(
        self,
        fast_shape: Tuple[int],
        ft_shape: Tuple[int],
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


class CCMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation for CC scoring.

    See Also
    --------
    :py:meth:`tme.matching_scores.cc_setup`.
    """

    def base_usage(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes
        complex_arrays = self.complex_array_size * self.complex_nbytes
        return float_arrays + complex_arrays

    def per_fork(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes
        complex_arrays = self.complex_array_size * self.complex_nbytes
        return float_arrays + complex_arrays


class LCCMemoryUsage(CCMemoryUsage):
    """
    Memory usage estimation for LCC scoring.

    See Also
    --------
    :py:meth:`tme.matching_scores.lcc_setup`.
    """


class CORRMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation for CORR scoring.

    See Also
    --------
    :py:meth:`tme.matching_scores.corr_setup`.
    """

    def base_usage(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes * 4
        complex_arrays = self.complex_array_size * self.complex_nbytes
        return float_arrays + complex_arrays

    def per_fork(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes
        complex_arrays = self.complex_array_size * self.complex_nbytes
        return float_arrays + complex_arrays


class CAMMemoryUsage(CORRMemoryUsage):
    """
    Memory usage estimation for CAM scoring.

    See Also
    --------
    :py:meth:`tme.matching_scores.cam_setup`.
    """


class FLCSphericalMaskMemoryUsage(CORRMemoryUsage):
    """
    Memory usage estimation for FLCMSphericalMask scoring.

    See Also
    --------
    :py:meth:`tme.matching_scores.flcSphericalMask_setup`.
    """


class FLCMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation for FLC scoring.

    See Also
    --------
    :py:meth:`tme.matching_scores.flc_setup`.
    """

    def base_usage(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes * 2
        complex_arrays = self.complex_array_size * self.complex_nbytes * 2
        return float_arrays + complex_arrays

    def per_fork(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes * 3
        complex_arrays = self.complex_array_size * self.complex_nbytes * 2
        return float_arrays + complex_arrays


class MCCMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation for MCC scoring.

    See Also
    --------
    :py:meth:`tme.matching_scores.mcc_setup`.
    """

    def base_usage(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes * 2
        complex_arrays = self.complex_array_size * self.complex_nbytes * 3
        return float_arrays + complex_arrays

    def per_fork(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes * 6
        complex_arrays = self.complex_array_size * self.complex_nbytes
        return float_arrays + complex_arrays


class MaxScoreOverRotationsMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation MaxScoreOverRotations Analyzer.

    See Also
    --------
    :py:class:`tme.analyzer.MaxScoreOverRotations`.
    """

    def base_usage(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes * 2
        return float_arrays

    def per_fork(self) -> int:
        return 0


class PeakCallerMaximumFilterMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation MaxScoreOverRotations Analyzer.

    See Also
    --------
    :py:class:`tme.analyzer.PeakCallerMaximumFilter`.
    """

    def base_usage(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes
        return float_arrays

    def per_fork(self) -> int:
        float_arrays = self.real_array_size * self.float_nbytes
        return float_arrays


class CupyBackendMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation for CupyBackend.

    See Also
    --------
    :py:class:`tme.backends.CupyBackend`.
    """

    def base_usage(self) -> int:
        # FFT plans, overhead from assigning FFT result, rotation interpolation
        complex_arrays = self.real_array_size * self.complex_nbytes * 3
        float_arrays = self.complex_array_size * self.float_nbytes * 2
        return float_arrays + complex_arrays

    def per_fork(self) -> int:
        return 0


MATCHING_MEMORY_REGISTRY = {
    "CC": CCMemoryUsage,
    "LCC": LCCMemoryUsage,
    "CORR": CORRMemoryUsage,
    "CAM": CAMMemoryUsage,
    "MCC": MCCMemoryUsage,
    "FLCSphericalMask": FLCSphericalMaskMemoryUsage,
    "FLC": FLCMemoryUsage,
    "MaxScoreOverRotations": MaxScoreOverRotationsMemoryUsage,
    "PeakCallerMaximumFilter": PeakCallerMaximumFilterMemoryUsage,
    "cupy": CupyBackendMemoryUsage,
    "pytorch": CupyBackendMemoryUsage,
    "batchFLCSpherical": FLCSphericalMaskMemoryUsage,
    "batchFLC": FLCMemoryUsage,
}


def estimate_memory_usage(
    shape1: Tuple[int],
    shape2: Tuple[int],
    matching_method: str,
    ncores: int,
    analyzer_method: str = None,
    backend: str = None,
    float_nbytes: int = 4,
    complex_nbytes: int = 8,
    integer_nbytes: int = 4,
) -> int:
    """
    Estimate the memory usage for a given template matching operation.

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

    convolution_shape, fast_shape, ft_shape = be.compute_convolution_shapes(
        shape1, shape2
    )

    memory_instance = MATCHING_MEMORY_REGISTRY[matching_method](
        fast_shape=fast_shape,
        ft_shape=ft_shape,
        float_nbytes=float_nbytes,
        complex_nbytes=complex_nbytes,
        integer_nbytes=integer_nbytes,
    )

    nbytes = memory_instance.base_usage() + memory_instance.per_fork() * ncores

    analyzer_instance = MATCHING_MEMORY_REGISTRY.get(analyzer_method, None)
    if analyzer_instance is not None:
        analyzer_instance = analyzer_instance(
            fast_shape=fast_shape,
            ft_shape=ft_shape,
            float_nbytes=float_nbytes,
            complex_nbytes=complex_nbytes,
            integer_nbytes=integer_nbytes,
        )
        nbytes += analyzer_instance.base_usage() + analyzer_instance.per_fork() * ncores

    backend_instance = MATCHING_MEMORY_REGISTRY.get(backend, None)
    if backend_instance is not None:
        backend_instance = backend_instance(
            fast_shape=fast_shape,
            ft_shape=ft_shape,
            float_nbytes=float_nbytes,
            complex_nbytes=complex_nbytes,
            integer_nbytes=integer_nbytes,
        )
        nbytes += backend_instance.base_usage() + backend_instance.per_fork() * ncores

    return nbytes
