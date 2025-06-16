"""
Backend using numpy and pyFFTW for template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
from psutil import virtual_memory
from contextlib import contextmanager
from typing import Tuple, Dict, List, Type

import scipy
import numpy as np
from scipy.ndimage import maximum_filter, affine_transform
from pyfftw.builders import rfftn as rfftn_builder, irfftn as irfftn_builder
from pyfftw import zeros_aligned, simd_alignment, FFTW, next_fast_len, interfaces

from ..types import NDArray, BackendArray, shm_type
from .matching_backend import MatchingBackend, _create_metafunction

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYFFTW_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def create_ufuncs(obj):
    ufuncs = [
        "add",
        "subtract",
        "multiply",
        "divide",
        "mod",
        "sum",
        "where",
        "einsum",
        "mean",
        "einsum",
        "std",
        "max",
        "min",
        "maximum",
        "minimum",
        "sqrt",
        "square",
        "abs",
        "power",
        "full",
        "clip",
        "arange",
        "stack",
        "concatenate",
        "repeat",
        "indices",
        "unique",
        "argsort",
        "tril_indices",
        "reshape",
        "identity",
        "dot",
        "copy",
    ]
    for ufunc in ufuncs:
        setattr(obj, ufunc, _create_metafunction(ufunc))
    return obj


@create_ufuncs
class _NumpyWrapper:
    """
    MatchingBackend prohibits using create_ufuncs on NumpyFFTWBackend directly.
    """

    pass


class NumpyFFTWBackend(_NumpyWrapper, MatchingBackend):
    """
    A numpy and pyfftw-based matching backend.
    """

    def __init__(
        self,
        array_backend=np,
        float_dtype=np.float32,
        complex_dtype=np.complex64,
        int_dtype=np.int32,
        overflow_safe_dtype=np.float32,
        **kwargs,
    ):
        super().__init__(
            array_backend=array_backend,
            float_dtype=float_dtype,
            complex_dtype=complex_dtype,
            int_dtype=int_dtype,
            overflow_safe_dtype=overflow_safe_dtype,
        )
        self.affine_transform = affine_transform

        self.cholesky = self._linalg_cholesky
        self.solve_triangular = self._solve_triangular
        self.linalg.solve_triangular = scipy.linalg.solve_triangular

    def _linalg_cholesky(self, arr, lower=False, *args, **kwargs):
        # Upper argument is not supported until numpy 2.0
        ret = self._array_backend.linalg.cholesky(arr, *args, **kwargs)
        if not lower:
            axes = list(range(ret.ndim))
            axes[-2:] = (ret.ndim - 1, ret.ndim - 2)
            ret = self._array_backend.transpose(ret, axes)
        return ret

    def _solve_triangular(self, a, b, lower=True, *args, **kwargs):
        mask = self._array_backend.tril if lower else self._array_backend.triu
        return self._array_backend.linalg.solve(mask(a), b, *args, **kwargs)

    def to_backend_array(self, arr: NDArray) -> NDArray:
        if isinstance(arr, self._array_backend.ndarray):
            return arr
        return self._array_backend.asarray(arr)

    def to_numpy_array(self, arr: NDArray) -> NDArray:
        return np.array(arr)

    def to_cpu_array(self, arr: NDArray) -> NDArray:
        return arr

    def get_fundamental_dtype(self, arr: NDArray) -> Type:
        dt = arr.dtype
        if self._array_backend.issubdtype(dt, self._array_backend.integer):
            return int
        elif self._array_backend.issubdtype(dt, self._array_backend.floating):
            return float
        elif self._array_backend.issubdtype(dt, self._array_backend.complexfloating):
            return complex
        return float

    def free_cache(self):
        pass

    def transpose(self, arr: NDArray, *args, **kwargs) -> NDArray:
        return self._array_backend.transpose(arr, *args, **kwargs)

    def tobytes(self, arr: NDArray) -> str:
        return arr.tobytes()

    def size(self, arr: NDArray) -> int:
        return arr.size

    def fill(self, arr: NDArray, value: float) -> NDArray:
        arr.fill(value)
        return arr

    def eps(self, dtype: type) -> NDArray:
        return self._array_backend.finfo(dtype).eps

    def datatype_bytes(self, dtype: type) -> NDArray:
        temp = self._array_backend.zeros(1, dtype=dtype)
        return temp.nbytes

    def astype(self, arr, dtype: Type) -> NDArray:
        if self._array_backend.iscomplexobj(arr):
            arr = arr.real
        return arr.astype(dtype)

    @staticmethod
    def at(arr, idx, value) -> NDArray:
        arr[idx] = value
        return arr

    def addat(self, arr, indices, *args, **kwargs) -> NDArray:
        self._array_backend.add.at(arr, indices, *args, **kwargs)
        return arr

    def topk_indices(self, arr: NDArray, k: int):
        temp = arr.reshape(-1)
        indices = self._array_backend.argpartition(temp, -k)[-k:][:k]
        sorted_indices = indices[self._array_backend.argsort(temp[indices])][::-1]
        sorted_indices = self.unravel_index(indices=sorted_indices, shape=arr.shape)
        return sorted_indices

    def indices(self, *args, **kwargs) -> NDArray:
        return self._array_backend.indices(*args, **kwargs)

    def roll(
        self, a: NDArray, shift: Tuple[int], axis: Tuple[int], **kwargs
    ) -> NDArray:
        return self._array_backend.roll(
            a,
            shift=shift,
            axis=axis,
            **kwargs,
        )

    def unravel_index(self, indices: NDArray, shape: Tuple[int]) -> NDArray:
        return self._array_backend.unravel_index(indices=indices, shape=shape)

    def max_filter_coordinates(self, score_space: NDArray, min_distance: Tuple[int]):
        score_box = tuple(min_distance for _ in range(score_space.ndim))
        max_filter = maximum_filter(score_space, size=score_box, mode="constant")
        max_filter = max_filter == score_space

        peaks = np.array(np.nonzero(max_filter)).T
        return peaks

    @staticmethod
    def zeros(shape: Tuple[int], dtype: type = None) -> NDArray:
        arr = zeros_aligned(shape, dtype=dtype, n=simd_alignment)
        return arr

    def from_sharedarr(self, args) -> NDArray:
        if len(args) == 1:
            return args[0]
        shm, shape, dtype = args
        return self.ndarray(shape, dtype, shm.buf)

    def to_sharedarr(
        self, arr: NDArray, shared_memory_handler: type = None
    ) -> shm_type:
        if shared_memory_handler is None:
            return (arr,)

        shm = shared_memory_handler.SharedMemory(size=arr.nbytes)
        np_array = self.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        np_array[:] = arr[:].copy()
        return shm, arr.shape, arr.dtype

    def topleft_pad(self, arr: NDArray, shape: Tuple[int], padval: int = 0) -> NDArray:
        b = self.zeros(shape, arr.dtype)
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
        fwd_shape: Tuple[int],
        inv_shape: Tuple[int],
        real_dtype: type,
        cmpl_dtype: type,
        fftargs: Dict = {},
        inv_output_shape: Tuple[int] = None,
        temp_fwd: NDArray = None,
        temp_inv: NDArray = None,
        fwd_axes: Tuple[int] = None,
        inv_axes: Tuple[int] = None,
    ) -> Tuple[FFTW, FFTW]:
        if temp_fwd is None:
            temp_fwd = (
                self.zeros(fwd_shape, real_dtype) if temp_fwd is None else temp_fwd
            )
        if temp_inv is None:
            temp_inv = (
                self.zeros(inv_shape, cmpl_dtype) if temp_inv is None else temp_inv
            )

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

        rfft_shape = self._format_fft_shape(temp_fwd.shape, fwd_axes)
        _rfftn = rfftn_builder(temp_fwd, s=rfft_shape, axes=fwd_axes, **fftargs)
        overwrite_input = fftargs.pop("overwrite_input", None)

        irfft_shape = fwd_shape if inv_output_shape is None else inv_output_shape
        irfft_shape = self._format_fft_shape(irfft_shape, inv_axes)
        _irfftn = irfftn_builder(temp_inv, s=irfft_shape, axes=inv_axes, **fftargs)

        def _rfftn_wrapper(arr, out, *args, **kwargs):
            return _rfftn(arr, out)

        def _irfftn_wrapper(arr, out, *args, **kwargs):
            return _irfftn(arr, out)

        fftargs["overwrite_input"] = overwrite_input
        return _rfftn_wrapper, _irfftn_wrapper

    @staticmethod
    def _format_fft_shape(shape: Tuple[int], axes: Tuple[int] = None):
        if axes is None:
            return shape
        axes = tuple(sorted(range(len(shape))[i] for i in axes))
        return tuple(shape[i] for i in axes)

    def rfftn(self, arr: NDArray, *args, **kwargs) -> NDArray:
        return interfaces.numpy_fft.rfftn(arr, **kwargs)

    def irfftn(self, arr: NDArray, *args, **kwargs) -> NDArray:
        return interfaces.numpy_fft.irfftn(arr, **kwargs)

    def extract_center(self, arr: NDArray, newshape: Tuple[int]) -> NDArray:
        new_shape = self.to_backend_array(newshape)
        current_shape = self.to_backend_array(arr.shape)
        starts = self.subtract(current_shape, new_shape)
        starts = self.astype(self.divide(starts, 2), self._int_dtype)
        stops = self.astype(self.add(starts, new_shape), self._int_dtype)
        box = tuple(slice(start, stop) for start, stop in zip(starts, stops))
        return arr[box]

    def compute_convolution_shapes(
        self, arr1_shape: Tuple[int], arr2_shape: Tuple[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        convolution_shape = [int(x + y - 1) for x, y in zip(arr1_shape, arr2_shape)]
        fast_shape = [next_fast_len(x) for x in convolution_shape]
        fast_ft_shape = list(fast_shape[:-1]) + [fast_shape[-1] // 2 + 1]

        return convolution_shape, fast_shape, fast_ft_shape

    def _rigid_transform_matrix(
        self,
        rotation_matrix: NDArray,
        translation: NDArray = None,
        center: NDArray = None,
    ) -> NDArray:
        ndim = rotation_matrix.shape[0]
        matrix = self.identity(ndim + 1, dtype=self._float_dtype)

        if translation is not None:
            translation_matrix = self.identity(ndim + 1, dtype=self._float_dtype)
            translation_matrix[:ndim, ndim] = -translation
            self.dot(matrix, translation_matrix, out=matrix)

        if center is not None:
            center_matrix = self.identity(ndim + 1, dtype=self._float_dtype)
            center_matrix[:ndim, ndim] = center
            self.dot(matrix, center_matrix, out=matrix)

        if rotation_matrix is not None:
            rmat = self.identity(ndim + 1, dtype=self._float_dtype)
            rmat[:ndim, :ndim] = self._array_backend.linalg.inv(rotation_matrix)
            self.dot(matrix, rmat, out=matrix)

        if center is not None:
            center_matrix[:ndim, ndim] = -center_matrix[:ndim, ndim]
            self.dot(matrix, center_matrix, out=matrix)

        matrix /= matrix[ndim, ndim]
        return matrix

    def _rigid_transform(
        self,
        data: NDArray,
        matrix: NDArray,
        output: NDArray,
        prefilter: bool,
        order: int,
        cache: bool = False,
        batched=False,
    ) -> None:
        if batched:
            for i in range(data.shape[0]):
                self._rigid_transform(
                    data=data[i],
                    matrix=matrix,
                    output=output[i],
                    prefilter=prefilter,
                    order=order,
                    cache=cache,
                    batched=False,
                )
            return None

        out_slice = tuple(slice(0, stop) for stop in data.shape)
        self.affine_transform(
            input=data,
            matrix=matrix,
            mode="constant",
            output=output[out_slice],
            order=order,
            prefilter=prefilter,
        )

    def rigid_transform(
        self,
        arr: NDArray,
        rotation_matrix: NDArray,
        arr_mask: NDArray = None,
        translation: NDArray = None,
        use_geometric_center: bool = False,
        out: NDArray = None,
        out_mask: NDArray = None,
        order: int = 3,
        cache: bool = False,
    ) -> Tuple[NDArray, NDArray]:
        out = self.zeros_like(arr) if out is None else out
        batched = arr.ndim != rotation_matrix.shape[0]

        center = self.divide(self.to_backend_array(arr.shape) - 1, 2)
        if not use_geometric_center:
            center = self.center_of_mass(arr, cutoff=0)

        offset = int(arr.ndim - rotation_matrix.shape[0])
        center = center[offset:]
        translation = self.zeros(center.size) if translation is None else translation
        matrix = self._rigid_transform_matrix(
            rotation_matrix=rotation_matrix,
            translation=translation,
            center=center,
        )

        subset = tuple(slice(None) for _ in range(arr.ndim))
        if offset > 1:
            subset = tuple(
                0 if i < (offset - 1) else slice(None) for i in range(arr.ndim)
            )

        self._rigid_transform(
            data=arr[subset],
            matrix=matrix,
            output=out[subset],
            order=order,
            prefilter=True,
            cache=cache,
            batched=batched,
        )

        # Applying the prefilter leads to artifacts in the mask.
        if arr_mask is not None:
            out_mask = self.zeros_like(arr_mask) if out_mask is None else out_mask
            self._rigid_transform(
                data=arr_mask[subset],
                matrix=matrix,
                output=out_mask[subset],
                order=order,
                prefilter=False,
                cache=cache,
                batched=batched,
            )

        return out, out_mask

    def center_of_mass(self, arr: BackendArray, cutoff: float = None) -> BackendArray:
        """
        Computes the center of mass of a numpy ndarray instance using all available
        elements. For template matching it typically makes sense to only input
        positive densities.

        Parameters
        ----------
        arr : BackendArray
            Array to compute the center of mass of.
        cutoff : float, optional
            Densities less than or equal to cutoff are nullified for center
            of mass computation. By default considers all values.

        Returns
        -------
        BackendArray
            Center of mass with shape (arr.ndim).
        """
        cutoff = self.min(arr) - 1 if cutoff is None else cutoff

        arr = self.where(arr > cutoff, arr, 0)
        denominator = self.sum(arr)

        grids = []
        for i, x in enumerate(arr.shape):
            baseline_dims = tuple(1 if i != t else x for t in range(len(arr.shape)))
            grids.append(
                self.reshape(self.arange(x, dtype=self._float_dtype), baseline_dims)
            )

        center_of_mass = [self.sum((arr * grid) / denominator) for grid in grids]

        return self.to_backend_array(center_of_mass)

    def get_available_memory(self) -> int:
        return virtual_memory().available

    @contextmanager
    def set_device(self, device_index: int):
        yield None

    def device_count(self) -> int:
        return 1

    @staticmethod
    def reverse(arr: NDArray, axis: Tuple[int] = None) -> NDArray:
        if axis is None:
            axis = tuple(range(arr.ndim))
        keep, rev = slice(None, None), slice(None, None, -1)
        return arr[tuple(rev if i in axis else keep for i in range(arr.ndim))]

    def max_score_over_rotations(
        self,
        scores: BackendArray,
        max_scores: BackendArray,
        rotations: BackendArray,
        rotation_index: int,
    ) -> None:
        """
        Update elements in ``max_scores`` and ``rotations`` where scores is larger than
        max_scores with score and rotation_index, respectivelty.

        .. warning:: ``max_scores`` and ``rotations`` are modified in-place.

        Parameters
        ----------
        scores : BackendArray
            The score space to compare against max_scores.
        max_scores : BackendArray
            Maximum score observed for each element in an array.
        rotations : BackendArray
            Rotation used to achieve a given max_score.
        rotation_index : int
            The index representing the current rotation.

        Returns
        -------
        Tuple[BackendArray, BackendArray]
            Updated ``max_scores`` and ``rotations``.
        """
        indices = scores > max_scores
        max_scores[indices] = scores[indices]
        rotations[indices] = rotation_index
        return max_scores, rotations

    def norm_scores(
        self,
        arr: BackendArray,
        exp_sq: BackendArray,
        sq_exp: BackendArray,
        n_obs: int,
        eps: float,
        out: BackendArray,
    ) -> BackendArray:
        """
        Normalizes ``arr`` by the standard deviation ensuring numerical stability.

        Parameters
        ----------
        arr : BackendArray
            The input array to be normalized.
        exp_sq : BackendArray
            Non-normalized expectation square.
        sq_exp : BackendArray
            Non-normalized expectation.
        n_obs : int
            Number of observations for normalization.
        eps : float
            Numbers below this threshold will be ignored in division.
        out : BackendArray
            Output array to write the result to.

        Returns
        -------
        BackendArray
            The normalized array with the same shape as `arr`.

        See Also
        --------
        :py:meth:`tme.matching_exhaustive.flc_scoring`
        """
        # Squared expected value (E(X)^2)
        sq_exp = self.divide(sq_exp, n_obs, out=sq_exp)
        sq_exp = self.square(sq_exp, out=sq_exp)
        # Expected squared value (E(X^2))
        exp_sq = self.divide(exp_sq, n_obs, out=exp_sq)
        # Variance
        sq_exp = self.subtract(exp_sq, sq_exp, out=sq_exp)
        sq_exp = self.maximum(sq_exp, 0.0, out=sq_exp)
        sq_exp = self.sqrt(sq_exp, out=sq_exp)

        # Assume that low stdev regions also have low scores
        # See :py:meth:`tme.matching_exhaustive.flcSphericalMask_setup` for correct norm
        sq_exp[sq_exp < eps] = 1
        sq_exp = self.multiply(sq_exp, n_obs, out=sq_exp)
        return self.divide(arr, sq_exp, out=out)
