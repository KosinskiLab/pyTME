"""
Backend using jax for template matching.

Copyright (c) 2023-2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from functools import wraps
from typing import Tuple, List, Dict, Any

import numpy as np

from ..types import JaxArray
from .npfftw_backend import NumpyFFTWBackend, shm_type


def emulate_out(func):
    """
    Adds an out argument to write output of ``func`` to.
    """

    @wraps(func)
    def inner(*args, out=None, **kwargs):
        ret = func(*args, **kwargs)
        if out is not None:
            out = out.at[:].set(ret)
            return out
        return ret

    return inner


class JaxBackend(NumpyFFTWBackend):
    """
    A jax-based matching backend.
    """

    def __init__(self, float_dtype=None, complex_dtype=None, int_dtype=None, **kwargs):
        import jax.scipy as jsp
        import jax.numpy as jnp

        float_dtype = jnp.float32 if float_dtype is None else float_dtype
        complex_dtype = jnp.complex64 if complex_dtype is None else complex_dtype
        int_dtype = jnp.int32 if int_dtype is None else int_dtype

        super().__init__(
            array_backend=jnp,
            float_dtype=float_dtype,
            complex_dtype=complex_dtype,
            int_dtype=int_dtype,
            overflow_safe_dtype=float_dtype,
        )
        self.scipy = jsp
        self._create_ufuncs()

    def from_sharedarr(self, arr: JaxArray) -> JaxArray:
        return arr

    @staticmethod
    def to_sharedarr(arr: JaxArray, shared_memory_handler: type = None) -> shm_type:
        return arr

    @staticmethod
    def at(arr, idx, value) -> JaxArray:
        return arr.at[idx].set(value)

    def addat(self, arr, indices, values):
        return arr.at[indices].add(values)

    def topleft_pad(
        self, arr: JaxArray, shape: Tuple[int], padval: int = 0
    ) -> JaxArray:
        b = self.full(shape=shape, dtype=arr.dtype, fill_value=padval)
        aind = [slice(None, None)] * arr.ndim
        bind = [slice(None, None)] * arr.ndim
        for i in range(arr.ndim):
            if arr.shape[i] > shape[i]:
                aind[i] = slice(0, shape[i])
            elif arr.shape[i] < shape[i]:
                bind[i] = slice(0, arr.shape[i])
        b = b.at[tuple(bind)].set(arr[tuple(aind)])
        return b

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
            "mod",
            "dot",
        ]
        for ufunc in ufuncs:
            backend_method = emulate_out(getattr(self._array_backend, ufunc))
            setattr(self, ufunc, staticmethod(backend_method))

        ufuncs = ["zeros", "full"]
        for ufunc in ufuncs:
            backend_method = getattr(self._array_backend, ufunc)
            setattr(self, ufunc, staticmethod(backend_method))

    def fill(self, arr: JaxArray, value: float) -> JaxArray:
        return self._array_backend.full(
            shape=arr.shape, dtype=arr.dtype, fill_value=value
        )

    def rfftn(self, arr: JaxArray, *args, **kwargs) -> JaxArray:
        return self._array_backend.fft.rfftn(arr, **kwargs)

    def irfftn(self, arr: JaxArray, *args, **kwargs) -> JaxArray:
        return self._array_backend.fft.irfftn(arr, **kwargs)

    def _interpolate(self, arr, indices, order: int = 1):
        ret = self.scipy.ndimage.map_coordinates(arr, indices, order=order)
        return ret.reshape(arr.shape)

    def _index_grid(self, shape: Tuple[int]) -> JaxArray:
        """
        Create homogeneous coordinate grid.

        Parameters
        ----------
        shape : tuple of int
            Shape to create the grid for

        Returns
        -------
        JaxArray
            Coordinate grid of shape (ndim + int(homogeneous), n_points)
        """
        indices = self._array_backend.indices(shape, dtype=self._float_dtype)
        indices = indices.reshape((len(shape), -1))
        ones = self._array_backend.ones((1, indices.shape[1]), dtype=indices.dtype)
        return self._array_backend.concatenate([indices, ones], axis=0)

    def _transform_indices(self, indices: JaxArray, matrix: JaxArray) -> JaxArray:
        return self._array_backend.matmul(matrix[:-1], indices)

    def _rigid_transform(
        self,
        arr: JaxArray,
        matrix: JaxArray,
        out: JaxArray = None,
        out_mask: JaxArray = None,
        arr_mask: JaxArray = None,
        order: int = 1,
        **kwargs,
    ) -> Tuple[JaxArray, JaxArray]:
        indices = self._index_grid(arr.shape)
        indices = self._transform_indices(indices, matrix)

        arr = self._interpolate(arr, indices, order)
        if arr_mask is not None:
            arr_mask = self._interpolate(out_mask, indices, order)
        return arr, arr_mask

    def max_score_over_rotations(
        self,
        scores: JaxArray,
        max_scores: JaxArray,
        rotations: JaxArray,
        rotation_index: int,
    ) -> Tuple[JaxArray, JaxArray]:
        update = self.greater(max_scores, scores)
        max_scores = max_scores.at[:].set(self.where(update, max_scores, scores))
        rotations = rotations.at[:].set(self.where(update, rotations, rotation_index))
        return max_scores, rotations

    def compute_convolution_shapes(
        self, arr1_shape: Tuple[int], arr2_shape: Tuple[int]
    ) -> Tuple[List[int], List[int], List[int]]:
        from scipy.fft import next_fast_len

        convolution_shape = [int(x + y - 1) for x, y in zip(arr1_shape, arr2_shape)]
        fast_shape = [next_fast_len(x, real=True) for x in convolution_shape]
        fast_ft_shape = list(fast_shape[:-1]) + [fast_shape[-1] // 2 + 1]

        return convolution_shape, fast_shape, fast_ft_shape

    def _to_hashable(self, obj: Any) -> Tuple[str, Tuple]:
        if isinstance(obj, np.ndarray):
            return ("array", (tuple(obj.flatten().tolist()), obj.shape))
        return ("other", obj)

    def _from_hashable(self, type_info: str, data: Any) -> Any:
        if type_info == "array":
            data, shape = data
            return self.array(data).reshape(shape)
        return data

    def _dict_to_tuple(self, data: Dict) -> Tuple:
        return tuple((k, self._to_hashable(v)) for k, v in data.items())

    def _tuple_to_dict(self, data: Tuple) -> Dict:
        return {x[0]: self._from_hashable(*x[1]) for x in data}

    def _unbatch(self, data, target_ndim, index):
        if not isinstance(data, type(self.zeros(1))):
            return data
        elif data.ndim <= target_ndim:
            return data
        return data[index]

    def scan(
        self,
        matching_data: type,
        splits: Tuple[Tuple[slice, slice]],
        n_jobs: int,
        callback_class: object,
        callback_class_args: Dict,
        rotate_mask: bool = False,
        background_correction: str = None,
        match_projection: bool = False,
        **kwargs,
    ) -> List:
        """
        Emulates output of :py:meth:`tme.matching_exhaustive._match_exhaustive`.
        """
        from ._jax_utils import setup_scan
        from ..matching_utils import setup_filter
        from ..analyzer import MaxScoreOverRotations

        pad_target = True if len(splits) > 1 else False
        target_pad = matching_data.target_padding(pad_target=pad_target)
        template_shape = matching_data._batch_shape(
            matching_data.template.shape, matching_data._target_batch
        )

        score_mask = 1
        target_shape = tuple(
            (x.stop - x.start + p) for x, p in zip(splits[0][0], target_pad)
        )
        conv_shape, fast_shape, fast_ft_shape, shift = matching_data.fourier_padding(
            target_shape=target_shape
        )

        analyzer_args = {
            "shape": fast_shape,
            "fourier_shift": shift,
            "fast_shape": fast_shape,
            "templateshape": template_shape,
            "convolution_shape": conv_shape,
            "convolution_mode": "valid" if pad_target else "same",
            "thread_safe": False,
            "aggregate_axis": matching_data._batch_axis(matching_data._batch_mask),
            "n_rotations": matching_data.rotations.shape[0],
            "jax_mode": True,
        }
        analyzer_args.update(callback_class_args)

        create_target_filter = matching_data.target_filter is not None
        create_template_filter = matching_data.template_filter is not None
        create_filter = create_target_filter or create_template_filter

        bg_tmpl = 1
        if background_correction == "phase-scrambling":
            bg_tmpl = matching_data.transform_template(
                "phase_randomization", reverse=True
            )
            bg_tmpl = self.astype(bg_tmpl, self._float_dtype)

        rotations = self.astype(matching_data.rotations, self._float_dtype)
        ret, template_filter, target_filter = [], 1, 1
        rotation_mapping = {
            self.tobytes(rotations[i]): i for i in range(rotations.shape[0])
        }
        for split_start in range(0, len(splits), n_jobs):

            analyzer_kwargs = []
            split_subset = splits[split_start : (split_start + n_jobs)]
            if not len(split_subset):
                continue

            targets = []
            for target_split, template_split in split_subset:
                base, translation_offset = matching_data.subset_by_slice(
                    target_slice=target_split,
                    target_pad=target_pad,
                    template_slice=template_split,
                )
                cur_args = analyzer_args.copy()
                cur_args["offset"] = translation_offset
                cur_args["targetshape"] = base._output_shape
                analyzer_kwargs.append(cur_args)

                if pad_target:
                    score_mask = base._score_mask(fast_shape, shift)

                # We prepad outside of jit to guarantee the stack operation works
                targets.append(self.topleft_pad(base._target, fast_shape))

            if create_filter:
                # This is technically inaccurate for whitening filters
                template_filter, target_filter = setup_filter(
                    matching_data=base,
                    fast_shape=fast_shape,
                    fast_ft_shape=fast_ft_shape,
                    pad_template_filter=False,
                    apply_target_filter=False,
                )

                # For projection matching we allow broadcasting the first dimension
                # This becomes problematic when applying per-tilt filters to the target
                # as the number of tilts does not necessarily coincide with the ideal
                # fourier shape. Hence we pad the target_filter with zeros here
                if target_filter.shape != (1,):
                    target_filter = self.topleft_pad(target_filter, fast_ft_shape)

            base, targets = None, self._array_backend.stack(targets)
            scan_inner = setup_scan(
                analyzer_kwargs=analyzer_kwargs,
                analyzer=callback_class,
                fast_shape=fast_shape,
                rotate_mask=rotate_mask,
                match_projection=match_projection,
            )

            states = scan_inner(
                self.astype(targets, self._float_dtype),
                self.astype(matching_data.template, self._float_dtype),
                self.astype(matching_data.template_mask, self._float_dtype),
                rotations,
                template_filter,
                target_filter,
                score_mask,
                bg_tmpl,
            )

            ndim = targets.ndim - 1
            for index in range(targets.shape[0]):
                kwargs = analyzer_kwargs[index]
                analyzer = callback_class(**kwargs)
                state = [self._unbatch(x, ndim, index) for x in states]

                if isinstance(analyzer, MaxScoreOverRotations):
                    state[2] = rotation_mapping

                ret.append(analyzer.result(state, **kwargs))
        return ret

    def get_available_memory(self) -> int:
        import jax

        _memory = {"cpu": 0, "gpu": 0}
        for device in jax.devices():
            if device.platform == "cpu":
                _memory["cpu"] = super().get_available_memory()
            else:
                mem_stats = device.memory_stats()
                _memory["gpu"] += mem_stats.get("bytes_limit", 0)

        if _memory["gpu"] > 0:
            return _memory["gpu"]
        return _memory["cpu"]
