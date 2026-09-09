"""
Backend using cupy for template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, List
from importlib.util import find_spec
from contextlib import contextmanager

from .npfftw_backend import NumpyFFTWBackend
from ..types import CupyArray, NDArray, shm_type

PLAN_CACHE = {}
TEXTURE_CACHE = {}


class CupyBackend(NumpyFFTWBackend):
    """
    A cupy-based matching backend.
    """

    def __init__(
        self,
        float_dtype: type = None,
        complex_dtype: type = None,
        int_dtype: type = None,
        overflow_safe_dtype: type = None,
        **kwargs,
    ):
        import cupy as cp
        import cupyx.scipy.fft as cufft
        from cupyx.scipy.ndimage import (
            affine_transform,
            maximum_filter,
            distance_transform_edt,
        )

        float_dtype = cp.float32 if float_dtype is None else float_dtype
        complex_dtype = cp.complex64 if complex_dtype is None else complex_dtype
        int_dtype = cp.int32 if int_dtype is None else int_dtype
        if overflow_safe_dtype is None:
            overflow_safe_dtype = cp.float32

        super().__init__(
            array_backend=cp,
            float_dtype=float_dtype,
            complex_dtype=complex_dtype,
            int_dtype=int_dtype,
            overflow_safe_dtype=overflow_safe_dtype,
            float16_dtype=cp.float16,
            uint16_dtype=cp.uint16,
        )
        self._cufft = cufft
        self.maximum_filter = maximum_filter
        self.affine_transform = affine_transform
        self.distance_transform_edt = distance_transform_edt

        itype = f"int{self.datatype_bytes(int_dtype) * 8}"
        ftype = f"float{self.datatype_bytes(float_dtype) * 8}"
        self._max_score_over_rotations = self._array_backend.ElementwiseKernel(
            f"{ftype} internal_scores, {ftype} scores, {itype} rot_index",
            f"{ftype} out1, {itype} rotations",
            "if (internal_scores < scores) {out1 = scores; rotations = rot_index;}",
            "max_score_over_rotations",
        )
        self.norm_scores = cp.ElementwiseKernel(
            f"{ftype} arr, {ftype} exp_sq, {ftype} sq_exp, {ftype} n_obs, {ftype} eps",
            f"{ftype} out",
            """
            // tmp1 = E(X)^2; tmp2 = E(X^2)
            float tmp1 = sq_exp / n_obs;
            float tmp2 = exp_sq / n_obs;
            tmp1 *= tmp1;

            tmp2 = sqrt(max(tmp2 - tmp1, 0.0));
            // out = (tmp2 < eps) ? 0.0 : arr / (tmp2 * n_obs);
            tmp1 = arr;
            if (tmp2 < eps){
                tmp1 = 0;
                tmp2 = 1;
            }
            tmp2 *= n_obs;
            out = tmp1 / tmp2;
            """,
            "norm_scores",
        )

        # Sum of square computation similar to the demeaned variance in pytom
        self.ssum = cp.ReductionKernel(
            f"{ftype} arr",
            f"{ftype} ret",
            "arr * arr",
            "a + b",
            "ret = a",
            "0",
            f"ssum_{ftype}",
        )
        self.texture_available = find_spec("voltools") is not None

    def to_backend_array(self, arr: NDArray, dtype: type = None) -> CupyArray:
        current_device = self._array_backend.cuda.device.get_device_id()
        if (
            isinstance(arr, self._array_backend.ndarray)
            and arr.device.id == current_device
        ):
            if dtype is not None and arr.dtype != dtype:
                return arr.astype(dtype)
            return arr
        return self._array_backend.asarray(arr, dtype=dtype)

    def to_numpy_array(self, arr: CupyArray) -> NDArray:
        return self._array_backend.asnumpy(arr)

    def to_cpu_array(self, arr: NDArray) -> NDArray:
        return self.to_numpy_array(arr)

    def from_sharedarr(self, arr: CupyArray) -> CupyArray:
        return arr

    @staticmethod
    def to_sharedarr(arr: CupyArray, shared_memory_handler: type = None) -> shm_type:
        return arr

    def zeros(self, *args, **kwargs):
        return self._array_backend.zeros(*args, **kwargs)

    def unravel_index(self, indices, shape):
        return self._array_backend.unravel_index(indices=indices, dims=shape)

    def free_cache(self):
        self._array_backend.fft.config.get_plan_cache().clear()

    def rfftn(self, arr: CupyArray, out: CupyArray = None, **kwargs) -> CupyArray:
        return self._cufft.rfftn(arr, **kwargs)

    def irfftn(self, arr: CupyArray, out: CupyArray = None, **kwargs) -> CupyArray:
        return self._cufft.irfftn(arr, **kwargs)

    def compute_convolution_shapes(
        self, arr1_shape: Tuple[int], arr2_shape: Tuple[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        from cupyx.scipy.fft import next_fast_len

        convolution_shape = [int(x + y - 1) for x, y in zip(arr1_shape, arr2_shape)]
        fast_shape = [next_fast_len(x, real=True) for x in convolution_shape]
        fast_ft_shape = list(fast_shape[:-1]) + [fast_shape[-1] // 2 + 1]

        return convolution_shape, fast_shape, fast_ft_shape

    def _get_texture(self, arr: CupyArray, order: int = 3, prefilter: bool = False):
        key = id(arr)
        if key in TEXTURE_CACHE:
            return TEXTURE_CACHE[key]

        from voltools import StaticVolume

        # Only keep template, mask and noise template in cache
        if len(TEXTURE_CACHE) >= 3:
            TEXTURE_CACHE.clear()

        interpolation = "filt_bspline"
        if order == 1:
            interpolation = "linear"
        elif order == 3 and not prefilter:
            interpolation = "bspline"

        current_device = self._array_backend.cuda.device.get_device_id()
        TEXTURE_CACHE[key] = StaticVolume(
            arr, interpolation=interpolation, device=f"gpu:{current_device}"
        )

        return TEXTURE_CACHE[key]

    def _transform(
        self,
        data: CupyArray,
        matrix: CupyArray,
        output: CupyArray,
        prefilter: bool,
        order: int,
        cache: bool = False,
    ) -> CupyArray:
        if data.ndim == 3 and cache and self.texture_available:
            texture = self._get_texture(data, order=order, prefilter=prefilter)
            texture.affine(transform_m=matrix, profile=False, output=output)
            return output

        return self.affine_transform(
            input=data,
            matrix=matrix,
            mode="constant",
            output=output,
            order=order,
            prefilter=prefilter,
        )

    def get_available_memory(self) -> int:
        with self._array_backend.cuda.Device():
            free_memory, _ = self._array_backend.cuda.runtime.memGetInfo()
        return free_memory

    @contextmanager
    def set_device(self, device_index: int):
        with self._array_backend.cuda.Device(device_index):
            yield

    def device_count(self) -> int:
        return self._array_backend.cuda.runtime.getDeviceCount()

    def max_score_over_rotations(
        self,
        scores: CupyArray,
        max_scores: CupyArray,
        rotations: CupyArray,
        rotation_index: int,
    ) -> Tuple[CupyArray, CupyArray]:
        return self._max_score_over_rotations(
            max_scores,
            scores,
            rotation_index,
            max_scores,
            rotations,
        )
