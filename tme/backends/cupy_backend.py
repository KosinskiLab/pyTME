""" Backend using cupy and GPU acceleration for
    template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Tuple, Dict, Callable
from contextlib import contextmanager

import numpy as np
from numpy.typing import NDArray

from .npfftw_backend import NumpyFFTWBackend
from ..types import CupyArray


class CupyBackend(NumpyFFTWBackend):
    """
    A cupy based backend for template matching
    """

    def __init__(
        self, default_dtype=None, complex_dtype=None, default_dtype_int=None, **kwargs
    ):
        import cupy as cp
        from cupyx.scipy.fft import get_fft_plan
        from cupyx.scipy.ndimage import affine_transform
        from cupyx.scipy.ndimage import maximum_filter

        default_dtype = cp.float32 if default_dtype is None else default_dtype
        complex_dtype = cp.complex64 if complex_dtype is None else complex_dtype
        default_dtype_int = cp.int32 if default_dtype_int is None else default_dtype_int

        super().__init__(
            array_backend=cp,
            default_dtype=default_dtype,
            complex_dtype=complex_dtype,
            default_dtype_int=default_dtype_int,
        )
        self.get_fft_plan = get_fft_plan
        self.affine_transform = affine_transform
        self.maximum_filter = maximum_filter

    def to_backend_array(self, arr: NDArray) -> CupyArray:
        if isinstance(arr, self._array_backend.ndarray):
            return arr
        return self._array_backend.asarray(arr)

    def to_numpy_array(self, arr: CupyArray) -> NDArray:
        return self._array_backend.asnumpy(arr)

    def to_cpu_array(self, arr: NDArray) -> NDArray:
        return self.to_numpy_array(arr)

    def sharedarr_to_arr(
        self, shape: Tuple[int], dtype: str, shm: CupyArray
    ) -> CupyArray:
        return shm

    @staticmethod
    def arr_to_sharedarr(
        arr: CupyArray, shared_memory_handler: type = None
    ) -> CupyArray:
        return arr

    def preallocate_array(self, shape: Tuple[int], dtype: type) -> NDArray:
        """
        Returns a byte-aligned array of zeros with specified shape and dtype.

        Parameters
        ----------
        shape : Tuple[int]
            Desired shape for the array.
        dtype : type
            Desired data type for the array.

        Returns
        -------
        NDArray
            Byte-aligned array of zeros with specified shape and dtype.
        """
        arr = self._array_backend.zeros(shape, dtype=dtype)
        return arr

    def unravel_index(self, indices, shape):
        return self._array_backend.unravel_index(indices=indices, dims=shape)

    def unique(self, ar, axis=None, *args, **kwargs):
        if axis is None:
            return self._array_backend.unique(ar=ar, axis=axis, *args, **kwargs)
        warnings.warn("Axis argument not yet supported in CupY, falling back to NumPy.")

        ret = np.unique(ar=self.to_numpy_array(ar), axis=axis, *args, **kwargs)
        if type(ret) != tuple:
            return self.to_backend_array(ret)
        return tuple(self.to_backend_array(k) for k in ret)

    def build_fft(
        self,
        fast_shape: Tuple[int],
        fast_ft_shape: Tuple[int],
        real_dtype: type,
        complex_dtype: type,
        fftargs: Dict = {},
        temp_real: NDArray = None,
        temp_fft: NDArray = None,
    ) -> Tuple[Callable, Callable]:
        """
        Build pyFFTW builder functions.

        Parameters
        ----------
        fast_shape : tuple
            Tuple of integers corresponding to fast convolution shape
            (see `compute_convolution_shapes`).
        fast_ft_shape : tuple
            Tuple of integers corresponding to the shape of the fourier
            transform array (see `compute_convolution_shapes`).
        real_dtype : dtype
            Numpy dtype of the inverse fourier transform.
        complex_dtype : dtype
            Numpy dtype of the fourier transform.
        fftargs : dict, optional
            Dictionary passed to pyFFTW builders.
        temp_real : NDArray, optional
            Temporary real numpy array, by default None.
        temp_fft : NDArray, optional
            Temporary fft numpy array, by default None.

        Returns
        -------
        tuple
            Tupple containing callable rfft and irfft object.
        """

        if temp_real is None:
            temp_real = self.preallocate_array(fast_shape, real_dtype)
        if temp_fft is None:
            temp_fft = self.preallocate_array(fast_ft_shape, complex_dtype)

        cache = self._array_backend.fft.config.get_plan_cache()
        cache.set_size(2)

        def rfftn(arr: CupyArray, out: CupyArray) -> None:
            out[:] = self.fft.rfftn(arr)[:]

        def irfftn(arr: CupyArray, out: CupyArray) -> None:
            out[:] = self.fft.irfftn(arr)[:]

        return rfftn, irfftn

    def compute_convolution_shapes(
        self, arr1_shape: Tuple[int], arr2_shape: Tuple[int]
    ) -> Tuple[Tuple[int], Tuple[int], Tuple[int]]:
        ret = super().compute_convolution_shapes(arr1_shape, arr2_shape)
        convolution_shape, fast_shape, fast_ft_shape = ret

        # cuFFT plans do not support automatic padding yet.
        is_odd = fast_shape[-1] % 2
        fast_shape[-1] += is_odd
        fast_ft_shape[-1] += is_odd

        return convolution_shape, fast_shape, fast_ft_shape

    def max_filter_coordinates(self, score_space, min_distance: Tuple[int]):
        score_box = tuple(min_distance for _ in range(score_space.ndim))
        max_filter = self.maximum_filter(score_space, size=score_box, mode="constant")
        max_filter = max_filter == score_space

        peaks = self._array_backend.array(self._array_backend.nonzero(max_filter)).T
        return peaks

    def rotate_array(
        self,
        arr: CupyArray,
        rotation_matrix: CupyArray,
        arr_mask: CupyArray = None,
        translation: CupyArray = None,
        use_geometric_center: bool = False,
        out: CupyArray = None,
        out_mask: CupyArray = None,
        order: int = 3,
    ) -> None:
        """
        Rotates coordinates of arr according to rotation_matrix.

        If no output array is provided, this method will compute an array with
        sufficient space to hold all elements. If both `arr` and `arr_mask`
        are provided, `arr_mask` will be centered according to arr.

        Parameters
        ----------
        arr : CupyArray
            The input array to be rotated.
        arr_mask : CupyArray, optional
            The mask of `arr` that will be equivalently rotated.
        rotation_matrix : CupyArray
            The rotation matrix to apply [d x d].
        translation : CupyArray
            The translation to apply [d].
        use_geometric_center : bool, optional
            Whether the rotation should be centered around the geometric
            or mass center. Default is mass center.
        out : CupyArray, optional
            The output array to write the rotation of `arr` to.
        out_mask : CupyArray, optional
            The output array to write the rotation of `arr_mask` to.
        order : int, optional
            Spline interpolation order. Has to be in the range 0-5.

        Notes
        -----
        Only a box of size arr, arr_mask will be consisdered for interpolation
        in out, out_mask.
        """

        rotate_mask = arr_mask is not None
        return_type = (out is None) + 2 * rotate_mask * (out_mask is None)
        translation = self.zeros(arr.ndim) if translation is None else translation

        center = self.divide(self.to_backend_array(arr.shape), 2)
        if not use_geometric_center:
            center = self.center_of_mass(arr, cutoff=0)

        rotation_matrix_inverted = self.linalg.inv(rotation_matrix)
        transformed_center = rotation_matrix_inverted @ center.reshape(-1, 1)
        transformed_center = transformed_center.reshape(-1)
        base_offset = self.subtract(center, transformed_center)
        offset = self.subtract(base_offset, translation)

        out = self.zeros_like(arr) if out is None else out
        out_slice = tuple(slice(0, stop) for stop in arr.shape)

        # Applying the prefilter leads to the creation of artifacts in the mask.
        self.affine_transform(
            input=arr,
            matrix=rotation_matrix_inverted,
            offset=offset,
            mode="constant",
            output=out[out_slice],
            order=order,
            prefilter=True,
        )

        if rotate_mask:
            out_mask = self.zeros_like(arr_mask) if out_mask is None else out_mask
            out_mask_slice = tuple(slice(0, stop) for stop in arr_mask.shape)
            self.affine_transform(
                input=arr_mask,
                matrix=rotation_matrix_inverted,
                offset=offset,
                mode="constant",
                output=out_mask[out_mask_slice],
                order=order,
                prefilter=False,
            )

        match return_type:
            case 0:
                return None
            case 1:
                return out
            case 2:
                return out_mask
            case 3:
                return out, out_mask

    def get_available_memory(self) -> int:
        with self._array_backend.cuda.Device():
            (
                free_memory,
                available_memory,
            ) = self._array_backend.cuda.runtime.memGetInfo()
        return free_memory

    @contextmanager
    def set_device(self, device_index: int):
        """
        Set the active GPU device as a context.

        This method sets the active GPU device for operations within the context.

        Parameters
        ----------
        device_index : int
            Index of the GPU device to be set as active.

        Yields
        ------
        None
            Operates as a context manager, yielding None and providing
            the set GPU context for enclosed operations.
        """
        with self._array_backend.cuda.Device(device_index):
            yield

    def device_count(self) -> int:
        """
        Return the number of available GPU devices.

        Returns
        -------
        int
            Number of available GPU devices.
        """
        return self._array_backend.cuda.runtime.getDeviceCount()
