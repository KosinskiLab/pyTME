"""
Backend using Apple's MLX library for template matching.

Copyright (c) 2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from functools import wraps
from typing import Tuple, List

import numpy as np

from .npfftw_backend import NumpyFFTWBackend
from ..types import NDArray, MlxArray, Scalar, shm_type


def emulate_out(func):
    """Adds an out argument to write output of ``func`` to."""

    @wraps(func)
    def inner(*args, out=None, **kwargs):
        ret = func(*args, **kwargs)
        if out is not None:
            out[:] = ret
            return out
        return ret

    return inner


class MLXBackend(NumpyFFTWBackend):
    """
    A mlx-based matching backend.
    """

    def __init__(
        self,
        float_dtype=None,
        complex_dtype=None,
        int_dtype=None,
        overflow_safe_dtype=None,
        **kwargs,
    ):
        import mlx.core as mx

        float_dtype = mx.float32 if float_dtype is None else float_dtype
        complex_dtype = mx.complex64 if complex_dtype is None else complex_dtype
        int_dtype = mx.int32 if int_dtype is None else int_dtype
        if overflow_safe_dtype is None:
            overflow_safe_dtype = mx.float32

        super().__init__(
            array_backend=mx,
            float_dtype=float_dtype,
            complex_dtype=complex_dtype,
            int_dtype=int_dtype,
            overflow_safe_dtype=overflow_safe_dtype,
            # We omit them on purpose
            float16_dtype=float_dtype,
            uint16_dtype=int_dtype,
        )

        self._create_ufuncs()

    def to_backend_array(self, arr: NDArray, dtype: type = None) -> MlxArray:
        # Older mlx releases reject dtype=None, so branch explicitly.
        if dtype is None:
            return self._array_backend.array(arr)
        return self._array_backend.array(arr, dtype=dtype)

    def to_numpy_array(self, arr: MlxArray) -> NDArray:
        return np.array(arr)

    def to_cpu_array(self, arr: MlxArray) -> NDArray:
        return arr

    def free_cache(self):
        pass

    def _create_ufuncs(self):
        ufuncs = [
            "add",
            "subtract",
            "multiply",
            "divide",
            "square",
            "sqrt",
            "maximum",
            "exp",
        ]
        for ufunc in ufuncs:
            backend_method = emulate_out(getattr(self._array_backend, ufunc))
            setattr(self, ufunc, staticmethod(backend_method))

        backend_method = getattr(self._array_backend, "remainder")
        setattr(self, "mod", staticmethod(backend_method))

        backend_method = getattr(self._array_backend, "tensordot")
        setattr(self, "dot", staticmethod(backend_method))

    def std(self, arr: MlxArray, axis) -> Scalar:
        return self._array_backend.sqrt(arr.var(axis=axis))

    def unique(self, *args, **kwargs):
        ret = np.unique(*args, **kwargs)
        if isinstance(ret, tuple):
            ret = [self.to_backend_array(x) for x in ret]
        return ret

    def tobytes(self, arr):
        return self.to_numpy_array(arr).tobytes()

    def full(self, shape, fill_value, dtype=None):
        if dtype is bool:
            dtype = None
        return self._array_backend.full(shape=shape, dtype=dtype, vals=fill_value)

    def fill(self, arr: MlxArray, value: Scalar) -> MlxArray:
        arr[:] = value
        return arr

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

    def rfftn(self, arr, out=None, *args, **kwargs):
        return self.fft.rfftn(arr, **kwargs)

    def irfftn(self, arr, out=None, *args, **kwargs):
        return self.fft.irfftn(arr, **kwargs)

    def max_score_over_rotations(
        self,
        scores: MlxArray,
        max_scores: MlxArray,
        rotations: MlxArray,
        rotation_index: int,
    ) -> Tuple[MlxArray, MlxArray]:
        update = self.greater(max_scores, scores)
        max_scores = self.where(update, max_scores, scores)
        rotations = self.where(update, rotations, rotation_index)
        return max_scores, rotations

    def from_sharedarr(self, arr: MlxArray) -> MlxArray:
        return arr

    @staticmethod
    def to_sharedarr(arr: MlxArray, shared_memory_handler: type = None) -> shm_type:
        return arr

    def topk_indices(self, arr: NDArray, k: int):
        arr = self.to_numpy_array(arr)
        ret = NumpyFFTWBackend().topk_indices(arr=arr, k=k)
        ret = [self.to_backend_array(x) for x in ret]
        return ret

    def rigid_transform(
        self,
        arr: NDArray,
        rotation_matrix: NDArray,
        arr_mask: NDArray = None,
        translation: NDArray = None,
        use_geometric_center: bool = False,
        out: NDArray = None,
        out_mask: NDArray = None,
        **kwargs,
    ) -> None:
        arr = self.to_numpy_array(arr)
        rotation_matrix = self.to_numpy_array(rotation_matrix)

        if arr_mask is not None:
            arr_mask = self.to_numpy_array(arr_mask)

        if translation is not None:
            translation = self.to_numpy_array(translation)

        if out is None:
            out = self.zeros(arr.shape)
        if out_mask is None and arr_mask is not None:
            out_mask_pass = self.zeros(arr_mask.shape)

        ret = NumpyFFTWBackend().rigid_transform(
            arr=arr,
            rotation_matrix=rotation_matrix,
            arr_mask=arr_mask,
            translation=translation,
            use_geometric_center=use_geometric_center,
            **kwargs,
        )

        out_pass, out_mask_pass = ret
        out[:] = self.to_backend_array(out_pass)

        if out_mask_pass is not None:
            out_mask_pass = self.to_backend_array(out_mask_pass)

        if out_mask is not None:
            out_mask[:] = out_mask_pass
        else:
            out_mask = out_mask_pass

        return out, out_mask

    def indices(self, arr: List) -> MlxArray:
        ret = NumpyFFTWBackend().indices(arr)
        return self.to_backend_array(ret)
