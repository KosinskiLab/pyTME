""" Backend Apple's MLX library for template matching.

    Copyright (c) 2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from typing import Tuple, List, Callable

import numpy as np

from .npfftw_backend import NumpyFFTWBackend
from ..types import NDArray, MlxArray, Scalar


class MLXBackend(NumpyFFTWBackend):
    """
    A MLX based backend for template matching.
    """

    def __init__(
        self,
        device="cpu",
        default_dtype=None,
        complex_dtype=None,
        default_dtype_int=None,
        **kwargs,
    ):
        import mlx.core as mx

        device = mx.cpu if device == "cpu" else mx.gpu
        default_dtype = mx.float32 if default_dtype is None else default_dtype
        complex_dtype = mx.complex64 if complex_dtype is None else complex_dtype
        default_dtype_int = mx.int32 if default_dtype_int is None else default_dtype_int

        super().__init__(
            array_backend=mx,
            default_dtype=default_dtype,
            complex_dtype=complex_dtype,
            default_dtype_int=default_dtype_int,
        )

        self.device = device

    def to_backend_array(self, arr: NDArray) -> MlxArray:
        return self._array_backend.array(arr)

    def to_numpy_array(self, arr: MlxArray) -> NDArray:
        return np.array(arr)

    def to_cpu_array(self, arr: MlxArray) -> NDArray:
        return arr

    def free_cache(self):
        pass

    def mod(self, arr1: MlxArray, arr2: MlxArray, out: MlxArray = None) -> MlxArray:
        if out is not None:
            out[:] = arr1 % arr2
            return None
        return arr1 % arr2

    def add(self, x1, x2, out: MlxArray = None, **kwargs) -> MlxArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)

        if out is not None:
            out[:] = self._array_backend.add(x1, x2, **kwargs)
            return None
        return self._array_backend.add(x1, x2, **kwargs)

    def std(self, arr: MlxArray, axis) -> Scalar:
        return self._array_backend.sqrt(arr.var(axis=axis))

    def unique(self, *args, **kwargs):
        ret = np.unique(*args, **kwargs)
        if isinstance(ret, tuple):
            ret = [self.to_backend_array(x) for x in ret]
        return ret

    def tobytes(self, arr):
        return self.to_numpy_array(arr).tobytes()

    def preallocate_array(self, shape: Tuple[int], dtype: type = None) -> NDArray:
        """
        Returns a byte-aligned array of zeros with specified shape and dtype.

        Parameters
        ----------
        shape : Tuple[int]
            Desired shape for the array.
        dtype : type, optional
            Desired data type for the array.

        Returns
        -------
        NDArray
            Byte-aligned array of zeros with specified shape and dtype.
        """
        arr = self._array_backend.zeros(shape, dtype=dtype)
        return arr

    def full(self, shape, fill_value, dtype=None):
        return self._array_backend.full(shape=shape, dtype=dtype, vals=fill_value)

    def fill(self, arr: MlxArray, value: Scalar) -> None:
        arr[:] = value

    def zeros(self, shape: Tuple[int], dtype: type = None) -> MlxArray:
        return self._array_backend.zeros(shape=shape, dtype=dtype)

    def roll(self, a: MlxArray, shift, axis, **kwargs):
        a = self.to_numpy_array(a)
        ret = NumpyFFTWBackend().roll(
            a,
            shift=shift,
            axis=axis,
            **kwargs,
        )
        return self.to_backend_array(ret)

    def extract_center(self, arr: NDArray, newshape: Tuple[int]) -> NDArray:
        """
        Extract the centered portion of an array based on a new shape.

        Parameters
        ----------
        arr : NDArray
            Input array.
        newshape : tuple
            Desired shape for the central portion.

        Returns
        -------
        NDArray
            Central portion of the array with shape `newshape`.

        References
        ----------
        .. [1] https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_signaltools.py
        """
        new_shape = self.to_backend_array(newshape)
        current_shape = self.to_backend_array(arr.shape)
        starts = self.subtract(current_shape, new_shape)
        starts = self.astype(self.divide(starts, 2), self._default_dtype_int)
        stops = self.astype(self.add(starts, newshape), self._default_dtype_int)
        starts, stops = starts.tolist(), stops.tolist()
        box = tuple(slice(start, stop) for start, stop in zip(starts, stops))
        return arr[box]

    def build_fft(
        self, fast_shape: Tuple[int], fast_ft_shape: Tuple[int], **kwargs
    ) -> Tuple[Callable, Callable]:
        """
        Build fft builder functions.

        Parameters
        ----------
        fast_shape : tuple
            Tuple of integers corresponding to fast convolution shape
            (see `compute_convolution_shapes`).
        fast_ft_shape : tuple
            Tuple of integers corresponding to the shape of the fourier
            transform array (see `compute_convolution_shapes`).
        **kwargs : dict, optional
            Additional parameters that are not used for now.

        Returns
        -------
        tuple
            Tupple containing callable rfft and irfft object.
        """

        # Runs on mlx.core.cpu until Metal support is available
        def rfftn(arr: MlxArray, out: MlxArray, shape: Tuple[int] = fast_shape) -> None:
            out[:] = self._array_backend.fft.rfftn(
                arr, s=shape, stream=self._array_backend.cpu
            )

        def irfftn(
            arr: MlxArray, out: MlxArray, shape: Tuple[int] = fast_shape
        ) -> None:
            out[:] = self._array_backend.fft.irfftn(
                arr, s=shape, stream=self._array_backend.cpu
            )

        return rfftn, irfftn

    def sharedarr_to_arr(
        self, shape: Tuple[int], dtype: str, shm: MlxArray
    ) -> MlxArray:
        return shm

    @staticmethod
    def arr_to_sharedarr(arr: MlxArray, shared_memory_handler: type = None) -> MlxArray:
        return arr

    def topk_indices(self, arr: NDArray, k: int):
        arr = self.to_numpy_array(arr)
        ret = NumpyFFTWBackend().topk_indices(arr=arr, k=k)
        ret = [self.to_backend_array(x) for x in ret]
        return ret

    def rotate_array(
        self,
        arr: NDArray,
        rotation_matrix: NDArray,
        arr_mask: NDArray = None,
        translation: NDArray = None,
        use_geometric_center: bool = False,
        out: NDArray = None,
        out_mask: NDArray = None,
        order: int = 3,
    ) -> None:
        rotate_mask = arr_mask is not None
        return_type = (out is None) + 2 * rotate_mask * (out_mask is None)

        arr = self.to_numpy_array(arr)
        rotation_matrix = self.to_numpy_array(rotation_matrix)

        if arr_mask is not None:
            arr_mask = self.to_numpy_array(arr_mask)

        if translation is not None:
            translation = self.to_numpy_array(translation)

        out_pass, out_mask_pass = None, None
        if out is not None:
            out_pass = self.to_numpy_array(out)
        if out_mask is not None:
            out_mask_pass = self.to_numpy_array(out_mask)

        ret = NumpyFFTWBackend().rotate_array(
            arr=arr,
            rotation_matrix=rotation_matrix,
            arr_mask=arr_mask,
            translation=translation,
            use_geometric_center=use_geometric_center,
            out=out_pass,
            out_mask=out_mask_pass,
            order=order,
        )

        if ret is not None:
            if len(ret) == 1 and out is None:
                out_pass = ret
            elif len(ret) == 1 and out_mask is None:
                out_mask_pass = ret
            else:
                out_pass, out_mask_pass = ret

        if out is not None:
            out[:] = self.to_backend_array(out_pass)

        if out_mask is not None:
            out_mask[:] = self.to_backend_array(out_mask_pass)

        match return_type:
            case 0:
                return None
            case 1:
                return out
            case 2:
                return out_mask
            case 3:
                return out, out_mask

    def indices(self, arr: List) -> MlxArray:
        ret = NumpyFFTWBackend().indices(arr)
        return self.to_backend_array(ret)
