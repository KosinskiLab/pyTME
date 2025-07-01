"""
Backend using pytorch and optionally GPU acceleration for
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

    def to_backend_array(self, arr: NDArray, check_device: bool = True) -> TorchTensor:
        if isinstance(arr, self._array_backend.Tensor):
            if arr.device == self.device or not check_device:
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
        x1 = self.to_backend_array(x1, check_device=False)
        x2 = self.to_backend_array(x2, check_device=False).to(x1.device)
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

    def copy(self, arr: TorchTensor) -> TorchTensor:
        return self._array_backend.clone(arr)

    def full(self, shape, fill_value, dtype=None):
        if isinstance(shape, int):
            shape = (shape,)
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

    @staticmethod
    def at(arr, idx, value) -> NDArray:
        arr[idx] = value
        return arr

    @staticmethod
    def addat(arr, indices, *args, **kwargs) -> NDArray:
        return arr.index_put_(indices, *args, accumulate=True, **kwargs)

    def flip(self, a, axis, **kwargs):
        return self._array_backend.flip(input=a, dims=axis, **kwargs)

    def topk_indices(self, arr, k):
        temp = arr.reshape(-1)
        values, indices = self._array_backend.topk(temp, k)
        indices = self.unravel_index(indices=indices, shape=arr.shape)
        return indices

    def indices(self, shape: Tuple[int], dtype: type = int) -> TorchTensor:
        grids = [self.arange(x, dtype=dtype) for x in shape]
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
            return (unraveled_coords[0, :],)

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

        pool = func(
            kernel_size=min_distance, padding=min_distance // 2, return_indices=True
        )
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

    def transpose(self, arr, axes=None):
        if axes is None:
            axes = tuple(range(arr.ndim - 1, -1, -1))
        return arr.permute(axes)

    def power(self, *args, **kwargs):
        return self._array_backend.pow(*args, **kwargs)

    def eye(self, *args, **kwargs):
        if "device" not in kwargs:
            kwargs["device"] = self.device
        return self._array_backend.eye(*args, **kwargs)

    def build_fft(
        self,
        fwd_shape: Tuple[int],
        inv_shape: Tuple[int],
        inv_output_shape: Tuple[int] = None,
        fwd_axes: Tuple[int] = None,
        inv_axes: Tuple[int] = None,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        rfft_shape = self._format_fft_shape(fwd_shape, fwd_axes)
        irfft_shape = fwd_shape if inv_output_shape is None else inv_output_shape
        irfft_shape = self._format_fft_shape(irfft_shape, inv_axes)

        def rfftn(
            arr: TorchTensor, out: TorchTensor, s=rfft_shape, axes=fwd_axes
        ) -> TorchTensor:
            return self._array_backend.fft.rfftn(arr, s=s, out=out, dim=axes)

        def irfftn(
            arr: TorchTensor, out: TorchTensor = None, s=irfft_shape, axes=inv_axes
        ) -> TorchTensor:
            return self._array_backend.fft.irfftn(arr, s=s, out=out, dim=axes)

        return rfftn, irfftn

    def rfftn(self, arr: NDArray, *args, **kwargs) -> NDArray:
        kwargs["dim"] = kwargs.pop("axes", None)
        return self._array_backend.fft.rfftn(arr, **kwargs)

    def irfftn(self, arr: NDArray, *args, **kwargs) -> NDArray:
        kwargs["dim"] = kwargs.pop("axes", None)
        return self._array_backend.fft.irfftn(arr, **kwargs)

    def _rigid_transform_matrix(self, rotation_matrix, *args, **kwargs):
        return rotation_matrix

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
        **kwargs,
    ):
        _mode_mapping = {0: "nearest", 1: "bilinear", 3: "bicubic"}
        mode = _mode_mapping.get(order, None)
        if mode is None:
            modes = ", ".join([str(x) for x in _mode_mapping.keys()])
            raise ValueError(
                f"Got {order} but supported interpolation orders are: {modes}."
            )

        out = self.zeros_like(arr) if out is None else out

        if translation is None:
            translation = self._array_backend.zeros(arr.ndim, device=arr.device)

        normalized_translation = self.divide(
            -2.0 * translation, self.tensor(arr.shape, device=arr.device)
        )
        rotation_matrix_pull = self.linalg.inv(self.flip(rotation_matrix, [0, 1]))

        out_slice = tuple(slice(0, x) for x in arr.shape)
        subset = tuple(slice(None) for _ in range(arr.ndim))
        offset = max(int(arr.ndim - rotation_matrix.shape[0]) - 1, 0)
        if offset > 0:
            normalized_translation = normalized_translation[offset:]
            subset = tuple(0 if i < offset else slice(None) for i in range(arr.ndim))
            out_slice = tuple(
                slice(0, 1) if i < offset else slice(0, x)
                for i, x in enumerate(arr.shape)
            )

        out[out_slice] = self._affine_transform(
            arr=arr[subset],
            rotation_matrix=rotation_matrix_pull,
            translation=normalized_translation,
            mode=mode,
        )

        if arr_mask is not None:
            out_mask_slice = tuple(slice(0, x) for x in arr_mask.shape)
            if out_mask is None:
                out_mask = self._array_backend.zeros_like(arr_mask)
            out_mask[out_mask_slice] = self._affine_transform(
                arr=arr_mask[subset],
                rotation_matrix=rotation_matrix_pull,
                translation=normalized_translation,
                mode=mode,
            )

        return out, out_mask

    def _affine_transform(
        self,
        arr: TorchTensor,
        rotation_matrix: TorchTensor,
        translation: TorchTensor,
        mode,
    ) -> TorchTensor:
        batched = arr.ndim != rotation_matrix.shape[0]

        batch_size, spatial_dims = 1, arr.shape
        if batched:
            translation = translation[1:]
            batch_size, *spatial_dims = arr.shape

        n_dims = len(spatial_dims)
        transformation_matrix = self._array_backend.zeros(
            n_dims, n_dims + 1, device=arr.device, dtype=arr.dtype
        )

        transformation_matrix[:, :n_dims] = rotation_matrix
        transformation_matrix[:, n_dims] = translation
        transformation_matrix = transformation_matrix.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        if not batched:
            arr = arr.unsqueeze(0)

        size = self.Size([batch_size, 1, *spatial_dims])
        grid = self.F.affine_grid(
            theta=transformation_matrix, size=size, align_corners=False
        )
        output = self.F.grid_sample(
            input=arr.unsqueeze(1),
            grid=grid,
            mode=mode,
            align_corners=False,
        )

        if not batched:
            output = output.squeeze(0)

        return output.squeeze(1)

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

    def reverse(self, arr: TorchTensor, axis: Tuple[int] = None) -> TorchTensor:
        if axis is None:
            axis = tuple(range(arr.ndim))
        return self._array_backend.flip(arr, [i for i in range(arr.ndim) if i in axis])

    def triu_indices(self, n: int, k: int = 0, m: int = None) -> TorchTensor:
        if m is None:
            m = n
        return self._array_backend.triu_indices(n, m, k)
