""" Compute memory consumption of template matching components.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np
from pyfftw import next_fast_len


class MatchingMemoryUsage(ABC):
    """
    Base class for estimating the memory usage of template matching.

    This class provides a template for estimating memory usage for
    different matching methods. Users should subclass it and implement the
    `base_usage` and `per_fork` methods to specify custom memory usage
    estimates.

    Parameters
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

    Methods
    -------
    base_usage():
        Returns the base memory usage in bytes.
    per_fork():
        Returns the memory usage in bytes per fork.
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
    Memory usage estimation for the CC fitter.

    See Also
    --------
    :py:meth:`tme.matching_exhaustive.cc_setup`.
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
    :py:meth:`tme.matching_exhaustive.lcc_setup`.
    """


class CORRMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation for CORR scoring.

    See Also
    --------
    :py:meth:`tme.matching_exhaustive.corr_setup`.
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
    :py:meth:`tme.matching_exhaustive.cam_setup`.
    """


class FLCSphericalMaskMemoryUsage(CORRMemoryUsage):
    """
    Memory usage estimation for FLCMSphericalMask scoring.

    See Also
    --------
    :py:meth:`tme.matching_exhaustive.flcSphericalMask_setup`.
    """


class FLCMemoryUsage(MatchingMemoryUsage):
    """
    Memory usage estimation for FLC scoring.

    See Also
    --------
    :py:meth:`tme.matching_exhaustive.flc_setup`.
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
    :py:meth:`tme.matching_exhaustive.mcc_setup`.
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


def _compute_convolution_shapes(
    arr1_shape: Tuple[int], arr2_shape: Tuple[int]
) -> Tuple[Tuple[int], Tuple[int], Tuple[int]]:
    """
    Computes regular, optimized and fourier convolution shape.

    Parameters
    ----------
    arr1_shape : tuple
        Tuple of integers corresponding to array1 shape.
    arr2_shape : tuple
        Tuple of integers corresponding to array2 shape.

    Returns
    -------
    tuple
        Tuple with regular convolution shape, convolution shape optimized for faster
        fourier transform, shape of the forward fourier transform
        (see :py:meth:`build_fft`).
    """
    convolution_shape = np.add(arr1_shape, arr2_shape) - 1
    fast_shape = [next_fast_len(x) for x in convolution_shape]
    fast_ft_shape = list(fast_shape[:-1]) + [fast_shape[-1] // 2 + 1]

    return convolution_shape, fast_shape, fast_ft_shape


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
}


def estimate_ram_usage(
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
    Estimate the RAM usage for a given convolution operation based on input shapes,
    matching_method, and number of cores.

    Parameters
    ----------
    shape1 : tuple
        The shape of the input target.
    shape2 : tuple
        The shape of the input template.
    matching_method : str
        The method used for the operation.
    is_gpu : bool, optional
        Whether the computation is performed on GPU. This factors in FFT
        plan caching.
    analyzer_method : str, optional
        The method used for score analysis.
    backend : str, optional
        Backend used for computation.
    ncores : int
        The number of CPU cores used for the operation.
    float_nbytes : int
        Number of bytes of the used float, e.g. 4 for float32.
    complex_nbytes : int
        Number of bytes of the used complex, e.g. 8 for complex64.
    integer_nbytes : int
        Number of bytes of the used integer, e.g. 4 for int32.

    Returns
    -------
    int
        The estimated RAM usage for the operation in bytes.

    Notes
    -----
        Residual memory from other objects that may remain allocated during
        template matching, e.g. the full sized target when using splitting,
        are not considered by this function.

    Raises
    ------
    ValueError
        If an unsupported matching_methode is provided.
    """
    if matching_method not in MATCHING_MEMORY_REGISTRY:
        raise ValueError(
            f"Supported fitters are {','.join(MATCHING_MEMORY_REGISTRY.keys())}"
        )

    convolution_shape, fast_shape, ft_shape = _compute_convolution_shapes(
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
