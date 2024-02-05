""" Backend using pytorch and optionally GPU acceleration for
    template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Callable
from contextlib import contextmanager
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from .npfftw_backend import NumpyFFTWBackend
from ..types import NDArray, TorchTensor


class PytorchBackend(NumpyFFTWBackend):
    """
    A pytorch based backend for template matching
    """

    def __init__(
        self,
        device="cuda",
        default_dtype=None,
        complex_dtype=None,
        default_dtype_int=None,
        **kwargs,
    ):
        import torch
        import torch.nn.functional as F

        default_dtype = torch.float32 if default_dtype is None else default_dtype
        complex_dtype = torch.complex64 if complex_dtype is None else complex_dtype
        default_dtype_int = (
            torch.int32 if default_dtype_int is None else default_dtype_int
        )

        super().__init__(
            array_backend=torch,
            default_dtype=default_dtype,
            complex_dtype=complex_dtype,
            default_dtype_int=default_dtype_int,
        )
        self.device = device
        self.F = F

    def to_backend_array(self, arr: NDArray) -> TorchTensor:
        if isinstance(arr, self._array_backend.Tensor):
            if arr.device == self.device:
                return arr
            return arr.to(self.device)
        return self.tensor(arr, device=self.device)

    def to_numpy_array(self, arr: TorchTensor) -> NDArray:
        if isinstance(arr, np.ndarray):
            return arr
        return arr.cpu().numpy()

    def to_cpu_array(self, arr: TorchTensor) -> NDArray:
        return arr.cpu()

    def free_cache(self):
        self._array_backend.cuda.empty_cache()

    def mod(self, x1, x2, *args, **kwargs):
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.remainder(x1, x2, *args, **kwargs)

    def sum(self, *args, **kwargs) -> NDArray:
        return self._array_backend.sum(*args, **kwargs)

    def mean(self, *args, **kwargs) -> NDArray:
        return self._array_backend.mean(*args, **kwargs)

    def std(self, *args, **kwargs) -> NDArray:
        return self._array_backend.std(*args, **kwargs)

    def max(self, *args, **kwargs) -> NDArray:
        ret = self._array_backend.amax(*args, **kwargs)
        if type(ret) == self._array_backend.Tensor:
            return ret
        return ret[0]

    def min(self, *args, **kwargs) -> NDArray:
        ret = self._array_backend.amin(*args, **kwargs)
        if type(ret) == self._array_backend.Tensor:
            return ret
        return ret[0]

    def maximum(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.maximum(input=x1, other=x2, *args, **kwargs)

    def minimum(self, x1, x2, *args, **kwargs) -> NDArray:
        x1 = self.to_backend_array(x1)
        x2 = self.to_backend_array(x2)
        return self._array_backend.minimum(input=x1, other=x2, *args, **kwargs)

    def tobytes(self, arr):
        return arr.cpu().numpy().tobytes()

    def size(self, arr):
        return arr.numel()

    def zeros(self, shape, dtype=None):
        return self._array_backend.zeros(shape, dtype=dtype, device=self.device)

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
        arr = self._array_backend.zeros(shape, dtype=dtype, device=self.device)
        return arr

    def full(self, shape, fill_value, dtype=None):
        return self._array_backend.full(
            size=shape, dtype=dtype, fill_value=fill_value, device=self.device
        )

    def datatype_bytes(self, dtype: type) -> int:
        temp = self.zeros(1, dtype=dtype)
        return temp.element_size()

    def fill(self, arr: TorchTensor, value: float):
        arr.fill_(value)

    def astype(self, arr, dtype):
        return arr.to(dtype)

    def flip(self, a, axis, **kwargs):
        return self._array_backend.flip(input=a, dims=axis, **kwargs)

    def arange(self, *args, **kwargs):
        return self._array_backend.arange(*args, **kwargs, device=self.device)

    def stack(self, *args, **kwargs):
        return self._array_backend.stack(*args, **kwargs)

    def topk_indices(self, arr, k):
        temp = arr.reshape(-1)
        values, indices = self._array_backend.topk(temp, k)
        indices = self.unravel_index(indices=indices, shape=arr.shape)
        return indices

    def indices(self, shape: Tuple[int]) -> TorchTensor:
        grids = [self.arange(x) for x in shape]
        mesh = self._array_backend.meshgrid(*grids, indexing="ij")
        return self._array_backend.stack(mesh)

    def unravel_index(self, indices, shape):
        indices = self.to_backend_array(indices)
        shape = self.to_backend_array(shape)
        strides = self._array_backend.cumprod(shape.flip(0), dim=0).flip(0)
        strides = self._array_backend.cat(
            (strides[1:], self.to_backend_array([1])),
        )
        unraveled_coords = (indices.view(-1, 1) // strides.view(1, -1)) % shape.view(
            1, -1
        )
        if unraveled_coords.size(0) == 1:
            return tuple(unraveled_coords[0, :].tolist())

        else:
            return tuple(unraveled_coords.T)

    def roll(self, a, shift, axis, **kwargs):
        shift = tuple(shift)
        return self._array_backend.roll(input=a, shifts=shift, dims=axis, **kwargs)

    def unique(
        self,
        ar,
        return_index: bool = False,
        return_inverse: bool = False,
        return_counts: bool = False,
        axis: int = None,
        sorted: bool = True,
    ):
        # https://github.com/pytorch/pytorch/issues/36748#issuecomment-1478913448
        unique, inverse, counts = self._array_backend.unique(
            ar, return_inverse=True, return_counts=True, dim=axis, sorted=sorted
        )
        inverse = inverse.reshape(-1)

        if return_index:
            inv_sorted = inverse.argsort(stable=True)
            tot_counts = self._array_backend.cat(
                (counts.new_zeros(1), counts.cumsum(dim=0))
            )[:-1]
            index = inv_sorted[tot_counts]

        ret = unique
        if return_index or return_inverse or return_counts:
            ret = [unique]

        if return_index:
            ret.append(index)
        if return_inverse:
            ret.append(inverse)
        if return_counts:
            ret.append(counts)

        return ret

    def max_filter_coordinates(self, score_space, min_distance: Tuple[int]):
        if score_space.ndim == 3:
            func = self._array_backend.nn.MaxPool3d
        elif score_space.ndim == 2:
            func = self._array_backend.nn.MaxPool2d
        else:
            raise NotImplementedError("Operation only implemented for 2 and 3D inputs.")

        pool = func(kernel_size=min_distance, return_indices=True)
        _, indices = pool(score_space.reshape(1, 1, *score_space.shape))
        coordinates = self.unravel_index(indices.reshape(-1), score_space.shape)
        coordinates = self.transpose(self.stack(coordinates))
        return coordinates

    def repeat(self, *args, **kwargs):
        return self._array_backend.repeat_interleave(*args, **kwargs)

    def sharedarr_to_arr(
        self, shape: Tuple[int], dtype: str, shm: TorchTensor
    ) -> TorchTensor:
        if self.device == "cuda":
            return shm

        required_size = int(self._array_backend.prod(self.to_backend_array(shape)))

        ret = self._array_backend.frombuffer(shm.buf, dtype=dtype)[
            :required_size
        ].reshape(shape)
        return ret

    def arr_to_sharedarr(
        self, arr: TorchTensor, shared_memory_handler: type = None
    ) -> TorchTensor:
        if self.device == "cuda":
            return arr

        nbytes = arr.numel() * arr.element_size()

        if type(shared_memory_handler) == SharedMemoryManager:
            shm = shared_memory_handler.SharedMemory(size=nbytes)
        else:
            shm = shared_memory.SharedMemory(create=True, size=nbytes)

        shm.buf[:nbytes] = arr.numpy().tobytes()

        return shm

    def transpose(self, arr):
        return arr.permute(*self._array_backend.arange(arr.ndim - 1, -1, -1))

    def power(self, *args, **kwargs):
        return self._array_backend.pow(*args, **kwargs)

    def rotate_array(
        self,
        arr: TorchTensor,
        rotation_matrix: TorchTensor,
        arr_mask: TorchTensor = None,
        translation: TorchTensor = None,
        out: TorchTensor = None,
        out_mask: TorchTensor = None,
        order: int = 1,
        **kwargs,
    ):
        """
        Rotates the given tensor `arr` based on the provided `rotation_matrix`.

        This function optionally allows for rotating an accompanying mask
        tensor (`arr_mask`) alongside the main tensor. The rotation is defined
        by the `rotation_matrix` and the optional `translation`.

        Parameters
        ----------
        arr : TorchTensor
            The input tensor to be rotated.
        rotation_matrix : TorchTensor
            The rotation matrix to apply. Must be square and of shape [d x d].
        arr_mask : TorchTensor, optional
            The mask of `arr` to be equivalently rotated.
        translation : TorchTensor, optional
            The translation to apply after rotation. Shape should
            match tensor dimensions [d].
        out : TorchTensor, optional
            The output tensor to hold the rotated `arr`. If not provided, a new
            tensor will be created.
        out_mask : TorchTensor, optional
            The output tensor to hold the rotated `arr_mask`. If not provided and
            `arr_mask` is given, a new tensor will be created.
        order : int, optional
            Spline interpolation order. Supports orders:

            +-------+---------------------------------------------------------------+
            |   0   | Use 'nearest' neighbor interpolation.                         |
            +-------+---------------------------------------------------------------+
            |   1   | Use 'bilinear' interpolation for smoother results.            |
            +-------+---------------------------------------------------------------+
            |   3   | Use 'bicubic' interpolation for higher order smoothness.      |
            +-------+---------------------------------------------------------------+

        Returns
        -------
        out, out_mask : TorchTensor or Tuple[TorchTensor, TorchTensor] or None
            Returns the rotated tensor(s). If `out` and `out_mask` are provided, the
            function will return `None`.
            If only `arr` is provided without `out`, it returns rotated `arr`.
            If both `arr` and `arr_mask` are given without `out` and `out_mask`, it
            returns a tuple of rotated tensors.

        Notes
        -----
        Only a region of the size of `arr` and `arr_mask` is considered for
        interpolation in `out` and `out_mask` respectively.

        Currently bicubic interpolation is not supported for 3D inputs.
        """
        device = arr.device
        mode_mapping = {0: "nearest", 1: "bilinear", 3: "bicubic"}
        mode = mode_mapping.get(order, None)
        if mode is None:
            modes = ", ".join([str(x) for x in mode_mapping.keys()])
            raise ValueError(
                f"Got {order} but supported interpolation orders are: {modes}."
            )
        rotate_mask = arr_mask is not None
        return_type = (out is None) + 2 * rotate_mask * (out_mask is None)

        out = self.zeros_like(arr) if out is None else out
        if translation is None:
            translation = self._array_backend.zeros(arr.ndim, device=device)

        normalized_translation = self.divide(
            -2.0 * translation, self.tensor(arr.shape, device=arr.device)
        )

        rotation_matrix_pull = self.linalg.inv(self.flip(rotation_matrix, [0, 1]))

        out_slice = tuple(slice(0, x) for x in arr.shape)
        out[out_slice] = self._affine_transform(
            arr=arr,
            rotation_matrix=rotation_matrix_pull,
            translation=normalized_translation,
            mode=mode,
        )

        if rotate_mask:
            out_mask_slice = tuple(slice(0, x) for x in arr_mask.shape)
            if out_mask is None:
                out_mask = self._array_backend.zeros_like(arr_mask)
            out_mask[out_mask_slice] = self._affine_transform(
                arr=arr_mask,
                rotation_matrix=rotation_matrix_pull,
                translation=normalized_translation,
                mode=mode,
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

        def rfftn(
            arr: TorchTensor, out: TorchTensor, shape: Tuple[int] = fast_shape
        ) -> None:
            return self._array_backend.fft.rfftn(arr, s=shape, out=out)

        def irfftn(
            arr: TorchTensor, out: TorchTensor, shape: Tuple[int] = fast_shape
        ) -> None:
            return self._array_backend.fft.irfftn(arr, s=shape, out=out)

        return rfftn, irfftn

    def _affine_transform(
        self,
        arr: TorchTensor,
        rotation_matrix: TorchTensor,
        translation: TorchTensor,
        mode,
    ) -> TorchTensor:
        """
        Performs an affine transformation on the given tensor.

        The affine transformation is defined by the provided `rotation_matrix`
        and the `translation` vector. The transformation is applied to the
        input tensor `arr`.

        Parameters
        ----------
        arr : TorchTensor
            The input tensor on which the transformation will be applied.
        rotation_matrix : TorchTensor
            The matrix defining the rotation component of the transformation.
        translation : TorchTensor
            The vector defining the translation to be applied post rotation.
        mode : str
            Interpolation mode to use. Options are: 'nearest', 'bilinear', 'bicubic'.

        Returns
        -------
        TorchTensor
            The tensor after applying the affine transformation.
        """

        transformation_matrix = self._array_backend.zeros(
            arr.ndim, arr.ndim + 1, device=arr.device, dtype=arr.dtype
        )
        transformation_matrix[:, : arr.ndim] = rotation_matrix
        transformation_matrix[:, arr.ndim] = translation

        size = self.Size([1, 1, *arr.shape])
        grid = self.F.affine_grid(
            theta=transformation_matrix.unsqueeze(0), size=size, align_corners=False
        )
        output = self.F.grid_sample(
            input=arr.unsqueeze(0).unsqueeze(0),
            grid=grid,
            mode=mode,
            align_corners=False,
        )

        return output.squeeze()

    def get_available_memory(self) -> int:
        if self.device == "cpu":
            return super().get_available_memory()
        return self._array_backend.cuda.mem_get_info()[0]

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
        if self.device == "cuda":
            with self._array_backend.cuda.device(device_index):
                yield
        else:
            yield None

    def device_count(self) -> int:
        """
        Return the number of available GPU devices.

        Returns
        -------
        int
            Number of available GPU devices.
        """
        return self._array_backend.cuda.device_count()

    def reverse(self, arr: TorchTensor) -> TorchTensor:
        """
        Reverse the order of elements in a tensor along all its axes.

        Parameters
        ----------
        tensor : TorchTensor
            Input tensor.

        Returns
        -------
        TorchTensor
            Reversed tensor.
        """
        return self._array_backend.flip(arr, [i for i in range(arr.ndim)])
