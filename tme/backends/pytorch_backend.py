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
from ..types import NDArray, TorchTensor, shm_type


class PytorchBackend(NumpyFFTWBackend):
    """
    A pytorch-based matching backend.
    """

    def __init__(
        self,
        device="cuda",
        float_dtype=None,
        complex_dtype=None,
        int_dtype=None,
        overflow_safe_dtype=None,
        **kwargs,
    ):
        import torch
        import torch.nn.functional as F

        float_dtype = torch.float32 if float_dtype is None else float_dtype
        complex_dtype = torch.complex64 if complex_dtype is None else complex_dtype
        int_dtype = torch.int32 if int_dtype is None else int_dtype
        if overflow_safe_dtype is None:
            overflow_safe_dtype = torch.float32

        super().__init__(
            array_backend=torch,
            float_dtype=float_dtype,
            complex_dtype=complex_dtype,
            int_dtype=int_dtype,
            overflow_safe_dtype=overflow_safe_dtype,
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
        elif isinstance(arr, self._array_backend.Tensor):
            return arr.cpu().numpy()
        return np.array(arr)

    def to_cpu_array(self, arr: TorchTensor) -> NDArray:
        return arr.cpu()

    def get_fundamental_dtype(self, arr):
        if self._array_backend.is_floating_point(arr):
            return float
        elif self._array_backend.is_complex(arr):
            return complex
        return int

    def free_cache(self):
        self._array_backend.cuda.empty_cache()

    def mod(self, x1, x2, *args, **kwargs):
        return self._array_backend.remainder(x1, x2, *args, **kwargs)

    def max(self, *args, **kwargs) -> NDArray:
        ret = self._array_backend.amax(*args, **kwargs)
        if isinstance(ret, self._array_backend.Tensor):
            return ret
        return ret[0]

    def min(self, *args, **kwargs) -> NDArray:
        ret = self._array_backend.amin(*args, **kwargs)
        if isinstance(ret, self._array_backend.Tensor):
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

    def full(self, shape, fill_value, dtype=None):
        return self._array_backend.full(
            size=shape, dtype=dtype, fill_value=fill_value, device=self.device
        )

    def arange(self, *args, **kwargs):
        return self._array_backend.arange(*args, **kwargs, device=self.device)

    def datatype_bytes(self, dtype: type) -> int:
        temp = self.zeros(1, dtype=dtype)
        return temp.element_size()

    def fill(self, arr: TorchTensor, value: float) -> TorchTensor:
        arr.fill_(value)
        return arr

    def astype(self, arr: TorchTensor, dtype: type) -> TorchTensor:
        return arr.to(dtype)

    def flip(self, a, axis, **kwargs):
        return self._array_backend.flip(input=a, dims=axis, **kwargs)

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

    def from_sharedarr(self, args) -> TorchTensor:
        if self.device == "cuda":
            return args

        shm, shape, dtype = args
        required_size = int(self._array_backend.prod(self.to_backend_array(shape)))

        ret = self._array_backend.frombuffer(shm.buf, dtype=dtype)[
            :required_size
        ].reshape(shape)
        return ret

    def to_sharedarr(
        self, arr: TorchTensor, shared_memory_handler: type = None
    ) -> shm_type:
        if self.device == "cuda":
            return arr

        nbytes = arr.numel() * arr.element_size()

        if isinstance(shared_memory_handler, SharedMemoryManager):
            shm = shared_memory_handler.SharedMemory(size=nbytes)
        else:
            shm = shared_memory.SharedMemory(create=True, size=nbytes)

        shm.buf[:nbytes] = arr.numpy().tobytes()
        return shm, arr.shape, arr.dtype

    def transpose(self, arr):
        return arr.permute(*self._array_backend.arange(arr.ndim - 1, -1, -1))

    def power(self, *args, **kwargs):
        return self._array_backend.pow(*args, **kwargs)

    def rigid_transform(
        self,
        arr: TorchTensor,
        rotation_matrix: TorchTensor,
        arr_mask: TorchTensor = None,
        translation: TorchTensor = None,
        use_geometric_center: bool = False,
        out: TorchTensor = None,
        out_mask: TorchTensor = None,
        order: int = 1,
        cache: bool = False,
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

        if arr_mask is not None:
            out_mask_slice = tuple(slice(0, x) for x in arr_mask.shape)
            if out_mask is None:
                out_mask = self._array_backend.zeros_like(arr_mask)
            out_mask[out_mask_slice] = self._affine_transform(
                arr=arr_mask,
                rotation_matrix=rotation_matrix_pull,
                translation=normalized_translation,
                mode=mode,
            )

        return out, out_mask

    def build_fft(
        self,
        fast_shape: Tuple[int],
        fast_ft_shape: Tuple[int],
        inverse_fast_shape: Tuple[int] = None,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        if inverse_fast_shape is None:
            inverse_fast_shape = fast_shape

        def rfftn(
            arr: TorchTensor, out: TorchTensor, shape: Tuple[int] = fast_shape
        ) -> TorchTensor:
            return self._array_backend.fft.rfftn(arr, s=shape, out=out)

        def irfftn(
            arr: TorchTensor, out: TorchTensor, shape: Tuple[int] = inverse_fast_shape
        ) -> TorchTensor:
            return self._array_backend.fft.irfftn(arr, s=shape, out=out)

        return rfftn, irfftn

    def _affine_transform(
        self,
        arr: TorchTensor,
        rotation_matrix: TorchTensor,
        translation: TorchTensor,
        mode,
    ) -> TorchTensor:
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
        if self.device == "cuda":
            with self._array_backend.cuda.device(device_index):
                yield
        else:
            yield None

    def device_count(self) -> int:
        if self.device == "cpu":
            return 1
        return self._array_backend.cuda.device_count()

    def reverse(self, arr: TorchTensor) -> TorchTensor:
        return self._array_backend.flip(arr, [i for i in range(arr.ndim)])
