"""
Backend using pytorch and optionally GPU acceleration for
template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
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

    def rfftn(self, arr: NDArray, *args, **kwargs) -> NDArray:
        kwargs["dim"] = kwargs.pop("axes", None)
        return self._array_backend.fft.rfftn(arr, **kwargs)

    def irfftn(self, arr: NDArray, *args, **kwargs) -> NDArray:
        kwargs["dim"] = kwargs.pop("axes", None)
        return self._array_backend.fft.irfftn(arr, **kwargs)

    def _build_transform_matrix(
        self,
        shape: Tuple[int],
        rotation_matrix: TorchTensor,
        translation: TorchTensor = None,
        center: TorchTensor = None,
        **kwargs,
    ) -> TorchTensor:
        """
        Express the transform matrix in normalized coordinates.
        """
        shape = self.to_backend_array(shape) - 1

        scale_factors = 2.0 / shape
        if center is not None:
            center = center - shape / 2
            center = center * scale_factors

        if translation is not None:
            translation = translation * scale_factors

        return super()._build_transform_matrix(
            rotation_matrix=self.flip(rotation_matrix, [0, 1]),
            translation=translation,
            center=center,
        )

    def _rigid_transform(
        self,
        arr: TorchTensor,
        matrix: TorchTensor,
        arr_mask: TorchTensor = None,
        out: TorchTensor = None,
        out_mask: TorchTensor = None,
        order: int = 1,
        batched: bool = False,
        **kwargs,
    ) -> Tuple[TorchTensor, TorchTensor]:
        """Apply rigid transformation using homogeneous transformation matrix."""
        _mode_mapping = {0: "nearest", 1: "bilinear", 3: "bicubic"}
        mode = _mode_mapping.get(order, None)
        if mode is None:
            modes = ", ".join([str(x) for x in _mode_mapping.keys()])
            raise ValueError(
                f"Got {order} but supported interpolation orders are: {modes}."
            )

        batch_size, spatial_dims = 1, arr.shape
        out_slice = tuple(slice(0, x) for x in arr.shape)
        if batched:
            matrix = matrix[1:, 1:]
            batch_size, *spatial_dims = arr.shape

        # Remove homogeneous row and expand for batch processing
        matrix = matrix[:-1, :].to(arr.dtype)
        matrix = matrix.unsqueeze(0).expand(batch_size, -1, -1)

        grid = self.F.affine_grid(
            theta=matrix.to(arr.dtype),
            size=self.Size([batch_size, 1, *spatial_dims]),
            align_corners=False,
        )

        arr = arr.unsqueeze(0) if not batched else arr
        ret = self.F.grid_sample(
            input=arr.unsqueeze(1),
            grid=grid,
            mode=mode,
            align_corners=False,
        ).squeeze(1)

        ret_mask = None
        if arr_mask is not None:
            arr_mask = arr_mask.unsqueeze(0) if not batched else arr_mask
            ret_mask = self.F.grid_sample(
                input=arr_mask.unsqueeze(1),
                grid=grid,
                mode=mode,
                align_corners=False,
            ).squeeze(1)

        if not batched:
            ret = ret.squeeze(0)
            ret_mask = ret_mask.squeeze(0) if arr_mask is not None else None

        if out is not None:
            out[out_slice] = ret
        else:
            out = ret

        if out_mask is not None:
            out_mask[out_slice] = ret_mask
        else:
            out_mask = ret_mask
        return out, out_mask

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
