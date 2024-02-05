""" Backend using numpy and pyFFTW for template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict, List
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from contextlib import contextmanager

import numpy as np
from psutil import virtual_memory
from numpy.typing import NDArray
from pyfftw import zeros_aligned, simd_alignment, FFTW, next_fast_len
from pyfftw.builders import rfftn as rfftn_builder, irfftn as irfftn_builder
from scipy.ndimage import maximum_filter, affine_transform

from .matching_backend import MatchingBackend
from ..matching_utils import rigid_transform


class NumpyFFTWBackend(MatchingBackend):
    """
    A numpy and pyfftw based backend for template matching.
    """

    def __init__(
        self,
        array_backend=np,
        default_dtype=np.float32,
        complex_dtype=np.complex64,
        default_dtype_int=np.int32,
        **kwargs,
    ):
        super().__init__(
            array_backend=array_backend,
            default_dtype=default_dtype,
            complex_dtype=complex_dtype,
            default_dtype_int=default_dtype_int,
        )
        self.affine_transform = affine_transform

    def to_backend_array(self, arr: NDArray) -> NDArray:
        if isinstance(arr, self._array_backend.ndarray):
            return arr
        return self._array_backend.asarray(arr)

    def to_numpy_array(self, arr: NDArray) -> NDArray:
        return arr

    def to_cpu_array(self, arr: NDArray) -> NDArray:
        return arr

    def free_cache(self):
        pass

    def add(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.add(x1, x2, *args, **kwargs)

    def subtract(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.subtract(x1, x2, *args, **kwargs)

    def multiply(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.multiply(x1, x2, *args, **kwargs)

    def divide(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.divide(x1, x2, *args, **kwargs)

    def mod(self, x1, x2, *args, **kwargs):
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.mod(x1, x2, *args, **kwargs)

    def sum(self, *args, **kwargs) -> NDArray:
        return self._array_backend.sum(*args, **kwargs)

    def einsum(self, *args, **kwargs) -> NDArray:
        return self._array_backend.einsum(*args, **kwargs)

    def mean(self, *args, **kwargs) -> NDArray:
        return self._array_backend.mean(*args, **kwargs)

    def std(self, *args, **kwargs) -> NDArray:
        return self._array_backend.std(*args, **kwargs)

    def max(self, *args, **kwargs) -> NDArray:
        return self._array_backend.max(*args, **kwargs)

    def min(self, *args, **kwargs) -> NDArray:
        return self._array_backend.min(*args, **kwargs)

    def maximum(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.maximum(x1, x2, *args, **kwargs)

    def minimum(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.minimum(x1, x2, *args, **kwargs)

    def sqrt(self, *args, **kwargs) -> NDArray:
        return self._array_backend.sqrt(*args, **kwargs)

    def square(self, *args, **kwargs) -> NDArray:
        return self._array_backend.square(*args, **kwargs)

    def abs(self, *args, **kwargs) -> NDArray:
        return self._array_backend.abs(*args, **kwargs)

    def transpose(self, arr):
        return arr.T

    def power(self, *args, **kwargs):
        return self._array_backend.power(*args, **kwargs)

    def tobytes(self, arr):
        return arr.tobytes()

    def size(self, arr):
        return arr.size

    def fill(self, arr: NDArray, value: float) -> None:
        arr.fill(value)

    def zeros(self, shape, dtype=np.float64) -> NDArray:
        return self._array_backend.zeros(shape=shape, dtype=dtype)

    def full(self, shape, fill_value, dtype=None, **kwargs) -> NDArray:
        return self._array_backend.full(
            shape, dtype=dtype, fill_value=fill_value, **kwargs
        )

    def eps(self, dtype: type) -> NDArray:
        """
        Returns the eps defined as diffeerence between 1.0 and the next
        representable floating point value larger than 1.0.

        Parameters
        ----------
        dtype : type
            Data type for which eps should be returned.

        Returns
        -------
        Scalar
            The eps for the given data type
        """
        return self._array_backend.finfo(dtype).eps

    def datatype_bytes(self, dtype: type) -> NDArray:
        """
        Return the number of bytes occupied by a given datatype.

        Parameters
        ----------
        dtype : type
            Datatype for which the number of bytes is to be determined.

        Returns
        -------
        int
            Number of bytes occupied by the datatype.
        """
        temp = self._array_backend.zeros(1, dtype=dtype)
        return temp.nbytes

    def clip(self, *args, **kwargs) -> NDArray:
        return self._array_backend.clip(*args, **kwargs)

    def flip(self, a, axis, **kwargs):
        return self._array_backend.flip(a, axis, **kwargs)

    @staticmethod
    def astype(arr, dtype):
        return arr.astype(dtype)

    def arange(self, *args, **kwargs):
        return self._array_backend.arange(*args, **kwargs)

    def stack(self, *args, **kwargs):
        return self._array_backend.stack(*args, **kwargs)

    def concatenate(self, *args, **kwargs):
        return self._array_backend.concatenate(*args, **kwargs)

    def repeat(self, *args, **kwargs):
        return self._array_backend.repeat(*args, **kwargs)

    def topk_indices(self, arr: NDArray, k: int):
        temp = arr.reshape(-1)
        indices = self._array_backend.argpartition(temp, -k)[-k:][:k]
        sorted_indices = indices[self._array_backend.argsort(temp[indices])][::-1]
        sorted_indices = self.unravel_index(indices=sorted_indices, shape=arr.shape)
        return sorted_indices

    def indices(self, *args, **kwargs) -> NDArray:
        return self._array_backend.indices(*args, **kwargs)

    def roll(self, a, shift, axis, **kwargs):
        return self._array_backend.roll(
            a,
            shift=shift,
            axis=axis,
            **kwargs,
        )

    def unique(self, *args, **kwargs):
        return self._array_backend.unique(*args, **kwargs)

    def argsort(self, *args, **kwargs):
        return self._array_backend.argsort(*args, **kwargs)

    def unravel_index(self, indices, shape):
        return self._array_backend.unravel_index(indices=indices, shape=shape)

    def tril_indices(self, *args, **kwargs):
        return self._array_backend.tril_indices(*args, **kwargs)

    def max_filter_coordinates(self, score_space, min_distance: Tuple[int]):
        score_box = tuple(min_distance for _ in range(score_space.ndim))
        max_filter = maximum_filter(score_space, size=score_box, mode="constant")
        max_filter = max_filter == score_space

        peaks = np.array(np.nonzero(max_filter)).T
        return peaks

    @staticmethod
    def preallocate_array(shape: Tuple[int], dtype: type) -> NDArray:
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
        arr = zeros_aligned(shape, dtype=dtype, n=simd_alignment)
        return arr

    def sharedarr_to_arr(
        self, shape: Tuple[int], dtype: str, shm: shared_memory.SharedMemory
    ) -> NDArray:
        """
        Returns an array of given shape and dtype from shared memory location.

        Parameters
        ----------
        shape : tuple
            Tuple of integers specifying the shape of the array.
        dtype : str
            String specifying the dtype of the array.
        shm : shared_memory.SharedMemory
            Shared memory object where the array is stored.

        Returns
        -------
        NDArray
            Array of the specified shape and dtype from the shared memory location.
        """
        return self.ndarray(shape, dtype, shm.buf)

    def arr_to_sharedarr(
        self, arr: NDArray, shared_memory_handler: type = None
    ) -> shared_memory.SharedMemory:
        """
        Converts a numpy array to an object shared in memory.

        Parameters
        ----------
        arr : NDArray
            Numpy array to convert.
        shared_memory_handler : type, optional
            The type of shared memory handler. Default is None.

        Returns
        -------
        shared_memory.SharedMemory
            The shared memory object containing the numpy array.
        """
        if type(shared_memory_handler) == SharedMemoryManager:
            shm = shared_memory_handler.SharedMemory(size=arr.nbytes)
        else:
            shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
        np_array = self.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np_array[:] = arr[:].copy()
        return shm

    def topleft_pad(self, arr: NDArray, shape: Tuple[int], padval: int = 0) -> NDArray:
        """
        Returns an array that has been padded to a specified shape with a padding
        value at the top-left corner.

        Parameters
        ----------
        arr : NDArray
            Input array to be padded.
        shape : Tuple[int]
            Desired shape for the output array.
        padval : int, optional
            Value to use for padding, default is 0.

        Returns
        -------
        NDArray
            Array that has been padded to the specified shape.
        """
        b = self.preallocate_array(shape, arr.dtype)
        self.add(b, padval, out=b)
        aind = [slice(None, None)] * arr.ndim
        bind = [slice(None, None)] * arr.ndim
        for i in range(arr.ndim):
            if arr.shape[i] > shape[i]:
                aind[i] = slice(0, shape[i])
            elif arr.shape[i] < shape[i]:
                bind[i] = slice(0, arr.shape[i])
        b[tuple(bind)] = arr[tuple(aind)]
        return b

    def build_fft(
        self,
        fast_shape: Tuple[int],
        fast_ft_shape: Tuple[int],
        real_dtype: type,
        complex_dtype: type,
        fftargs: Dict = {},
        temp_real: NDArray = None,
        temp_fft: NDArray = None,
    ) -> Tuple[FFTW, FFTW]:
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
            Tuple containing callable pyFFTW objects for forward and inverse
            fourier transform.
        """
        if temp_real is None:
            temp_real = self.preallocate_array(fast_shape, real_dtype)
        if temp_fft is None:
            temp_fft = self.preallocate_array(fast_ft_shape, complex_dtype)

        default_values = {
            "planner_effort": "FFTW_MEASURE",
            "auto_align_input": False,
            "auto_contiguous": False,
            "avoid_copy": True,
            "overwrite_input": True,
            "threads": 1,
        }
        for key in default_values:
            if key in fftargs:
                continue
            fftargs[key] = default_values[key]

        rfftn = rfftn_builder(temp_real, s=fast_shape, **fftargs)

        overwrite_input = None
        if "overwrite_input" in fftargs:
            overwrite_input = fftargs.pop("overwrite_input")
        irfftn = irfftn_builder(temp_fft, s=fast_shape, **fftargs)

        if overwrite_input is not None:
            fftargs["overwrite_input"] = overwrite_input
        return rfftn, irfftn

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
        box = tuple(slice(start, stop) for start, stop in zip(starts, stops))
        return arr[box]

    def compute_convolution_shapes(
        self, arr1_shape: Tuple[int], arr2_shape: Tuple[int]
    ) -> Tuple[List[int], List[int], List[int]]:
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
        convolution_shape = [
            int(x) + int(y) - 1 for x, y in zip(arr1_shape, arr2_shape)
        ]
        fast_shape = [next_fast_len(x) for x in convolution_shape]
        fast_ft_shape = list(fast_shape[:-1]) + [fast_shape[-1] // 2 + 1]

        return convolution_shape, fast_shape, fast_ft_shape

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
        """
        Rotates coordinates of arr according to rotation_matrix.

        If no output array is provided, this method will compute an array with
        sufficient space to hold all elements. If both `arr` and `arr_mask`
        are provided, `arr_mask` will be centered according to arr.

        Parameters
        ----------
        arr : NDArray
            The input array to be rotated.
        arr_mask : NDArray, optional
            The mask of `arr` that will be equivalently rotated.
        rotation_matrix : NDArray
            The rotation matrix to apply [d x d].
        translation : NDArray
            The translation to apply [d].
        use_geometric_center : bool, optional
            Whether the rotation should be centered around the geometric
            or mass center. Default is mass center.
        out : NDArray, optional
            The output array to write the rotation of `arr` to.
        out_mask : NDArray, optional
            The output array to write the rotation of `arr_mask` to.
        order : int, optional
            Spline interpolation order. Has to be in the range 0-5. Non-zero
            elements will be converted into a point-cloud and rotated according
            to ``rotation_matrix`` if order is None.
        """

        if order is None:
            mask_coordinates = None
            if arr_mask is not None:
                mask_coordinates = np.array(np.where(arr_mask > 0))
            return self.rotate_array_coordinates(
                arr=arr,
                arr_mask=arr_mask,
                coordinates=np.array(np.where(arr > 0)),
                mask_coordinates=mask_coordinates,
                out=out,
                out_mask=out_mask,
                rotation_matrix=rotation_matrix,
                translation=translation,
                use_geometric_center=use_geometric_center,
            )

        rotate_mask = arr_mask is not None
        return_type = (out is None) + 2 * rotate_mask * (out_mask is None)
        translation = np.zeros(arr.ndim) if translation is None else translation

        center = np.divide(arr.shape, 2)
        if not use_geometric_center:
            center = self.center_of_mass(arr, cutoff=0)

        rotation_matrix_inverted = np.linalg.inv(rotation_matrix)
        transformed_center = rotation_matrix_inverted @ center.reshape(-1, 1)
        transformed_center = transformed_center.reshape(-1)
        base_offset = np.subtract(center, transformed_center)
        offset = np.subtract(base_offset, translation)

        out = np.zeros_like(arr) if out is None else out
        out_slice = tuple(slice(0, stop) for stop in arr.shape)

        # Applying the prefilter can cause artifacts in the mask
        affine_transform(
            input=arr,
            matrix=rotation_matrix_inverted,
            offset=offset,
            mode="constant",
            output=out[out_slice],
            order=order,
            prefilter=True,
        )

        if rotate_mask:
            out_mask = np.zeros_like(arr_mask) if out_mask is None else out_mask
            out_mask_slice = tuple(slice(0, stop) for stop in arr_mask.shape)
            affine_transform(
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

    @staticmethod
    def rotate_array_coordinates(
        arr: NDArray,
        coordinates: NDArray,
        rotation_matrix: NDArray,
        translation: NDArray = None,
        out: NDArray = None,
        use_geometric_center: bool = True,
        arr_mask: NDArray = None,
        mask_coordinates: NDArray = None,
        out_mask: NDArray = None,
    ) -> None:
        """
        Rotates coordinates of arr according to rotation_matrix.

        If no output array is provided, this method will compute an array with
        sufficient space to hold all elements. If both `arr` and `arr_mask`
        are provided, `arr_mask` will be centered according to arr.

        No centering will be performed if the rotation matrix is the identity matrix.

        Parameters
        ----------
        arr : NDArray
            The input array to be rotated.
        coordinates : NDArray
            The pointcloud [d x N] containing elements of `arr` that should be rotated.
            See :py:meth:`Density.to_pointcloud` on how to obtain the coordinates.
        rotation_matrix : NDArray
            The rotation matrix to apply [d x d].
        rotation_matrix : NDArray
            The translation to apply [d].
        out : NDArray, optional
            The output array to write the rotation of `arr` to.
        use_geometric_center : bool, optional
            Whether the rotation should be centered around the geometric
            or mass center.
        arr_mask : NDArray, optional
            The mask of `arr` that will be equivalently rotated.
        mask_coordinates : NDArray, optional
            Equivalent to `coordinates`, but containing elements of `arr_mask`
            that should be rotated.
        out_mask : NDArray, optional
            The output array to write the rotation of `arr_mask` to.
        """
        rotate_mask = arr_mask is not None and mask_coordinates is not None
        return_type = (out is None) + 2 * rotate_mask * (out_mask is None)

        # Otherwise array might be slightly shifted by centering
        if np.allclose(
            rotation_matrix,
            np.eye(rotation_matrix.shape[0], dtype=rotation_matrix.dtype),
        ):
            center_rotation = False

        coordinates_rotated = np.empty(coordinates.shape, dtype=rotation_matrix.dtype)
        mask_rotated = (
            np.empty(mask_coordinates.shape, dtype=rotation_matrix.dtype)
            if rotate_mask
            else None
        )

        center = np.array(arr.shape) // 2 if use_geometric_center else None
        if translation is None:
            translation = np.zeros(coordinates_rotated.shape[0])

        rigid_transform(
            coordinates=coordinates,
            coordinates_mask=mask_coordinates,
            out=coordinates_rotated,
            out_mask=mask_rotated,
            rotation_matrix=rotation_matrix,
            translation=translation,
            use_geometric_center=use_geometric_center,
            center=center,
        )

        coordinates_rotated = coordinates_rotated.astype(int)
        offset = coordinates_rotated.min(axis=1)
        np.multiply(offset, offset < 0, out=offset)
        coordinates_rotated -= offset[:, None]

        out_offset = np.zeros(
            coordinates_rotated.shape[0], dtype=coordinates_rotated.dtype
        )
        if out is None:
            out_offset = coordinates_rotated.min(axis=1)
            coordinates_rotated -= out_offset[:, None]
            out = np.zeros(coordinates_rotated.max(axis=1) + 1, dtype=arr.dtype)

        if rotate_mask:
            mask_rotated = mask_rotated.astype(int)
            if out_mask is None:
                mask_rotated -= out_offset[:, None]
                out_mask = np.zeros(
                    coordinates_rotated.max(axis=1) + 1, dtype=arr.dtype
                )

            in_box = np.logical_and(
                mask_rotated < np.array(out_mask.shape)[:, None],
                mask_rotated >= 0,
            ).min(axis=0)
            out_of_box = np.invert(in_box).sum()
            if out_of_box != 0:
                print(
                    f"{out_of_box} elements out of bounds. Perhaps increase"
                    " *arr_mask* size."
                )

            mask_coordinates = tuple(mask_coordinates[:, in_box])
            mask_rotated = tuple(mask_rotated[:, in_box])
            np.add.at(out_mask, mask_rotated, arr_mask[mask_coordinates])

        # Negative coordinates would be (mis)interpreted as reverse index
        in_box = np.logical_and(
            coordinates_rotated < np.array(out.shape)[:, None], coordinates_rotated >= 0
        ).min(axis=0)
        out_of_box = np.invert(in_box).sum()
        if out_of_box != 0:
            print(f"{out_of_box} elements out of bounds. Perhaps increase *out* size.")

        coordinates = coordinates[:, in_box]
        coordinates_rotated = coordinates_rotated[:, in_box]

        coordinates = tuple(coordinates)
        coordinates_rotated = tuple(coordinates_rotated)
        np.add.at(out, coordinates_rotated, arr[coordinates])

        match return_type:
            case 0:
                return None
            case 1:
                return out
            case 2:
                return out_mask
            case 3:
                return out, out_mask

    def center_of_mass(self, arr: NDArray, cutoff: float = None) -> NDArray:
        """
        Computes the center of mass of a numpy ndarray instance using all available
        elements. For template matching it typically makes sense to only input
        positive densities.

        Parameters
        ----------
        arr : NDArray
            Array to compute the center of mass of.
        cutoff : float, optional
            Densities less than or equal to cutoff are nullified for center
            of mass computation. By default considers all values.

        Returns
        -------
        NDArray
            Center of mass with shape (arr.ndim).
        """
        cutoff = arr.min() - 1 if cutoff is None else cutoff
        arr = self._array_backend.where(arr > cutoff, arr, 0)
        denominator = self.sum(arr)
        grids = self._array_backend.ogrid[tuple(slice(0, i) for i in arr.shape)]
        grids = [grid.astype(self._default_dtype) for grid in grids]

        center_of_mass = self.array(
            [
                self.sum(self.multiply(arr, grids[dim])) / denominator
                for dim in range(arr.ndim)
            ]
        )

        return center_of_mass

    def get_available_memory(self) -> int:
        return virtual_memory().available

    @contextmanager
    def set_device(self, device_index: int):
        yield None

    def device_count(self) -> int:
        return 1

    @staticmethod
    def reverse(arr: NDArray) -> NDArray:
        """
        Reverse the order of elements in an array along all its axes.

        Parameters
        ----------
        arr : NDArray
            Input array.

        Returns
        -------
        NDArray
            Reversed array.
        """
        return arr[(slice(None, None, -1),) * arr.ndim]
