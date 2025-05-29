""" Backend using jax for template matching.

    Copyright (c) 2023-2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from functools import wraps
from typing import Tuple, List, Callable

from ..types import BackendArray
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
        try:
            from ._jax_utils import scan as _

            self.scan = self._scan
        except Exception:
            pass

    def from_sharedarr(self, arr: BackendArray) -> BackendArray:
        return arr

    @staticmethod
    def to_sharedarr(arr: BackendArray, shared_memory_handler: type = None) -> shm_type:
        return arr

    @staticmethod
    def at(arr, idx, value) -> BackendArray:
        arr = arr.at[idx].set(value)
        return arr

    def topleft_pad(
        self, arr: BackendArray, shape: Tuple[int], padval: int = 0
    ) -> BackendArray:
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
        ]
        for ufunc in ufuncs:
            backend_method = emulate_out(getattr(self._array_backend, ufunc))
            setattr(self, ufunc, staticmethod(backend_method))

        ufuncs = ["zeros", "full"]
        for ufunc in ufuncs:
            backend_method = getattr(self._array_backend, ufunc)
            setattr(self, ufunc, staticmethod(backend_method))

    def fill(self, arr: BackendArray, value: float) -> BackendArray:
        return self._array_backend.full(
            shape=arr.shape, dtype=arr.dtype, fill_value=value
        )

    def build_fft(
        self,
        fwd_shape: Tuple[int],
        inv_shape: Tuple[int] = None,
        inv_output_shape: Tuple[int] = None,
        fwd_axes: Tuple[int] = None,
        inv_axes: Tuple[int] = None,
        **kwargs,
    ) -> Tuple[Callable, Callable]:
        rfft_shape = self._format_fft_shape(fwd_shape, fwd_axes)
        irfft_shape = fwd_shape if inv_output_shape is None else inv_output_shape
        irfft_shape = self._format_fft_shape(irfft_shape, inv_axes)

        def rfftn(arr, out=None, s=rfft_shape, axes=fwd_axes):
            return self._array_backend.fft.rfftn(arr, s=s, axes=axes)

        def irfftn(arr, out=None, s=irfft_shape, axes=inv_axes):
            return self._array_backend.fft.irfftn(arr, s=s, axes=axes)

        return rfftn, irfftn

    def rfftn(self, arr: BackendArray, *args, **kwargs) -> BackendArray:
        return self._array_backend.fft.rfftn(arr, **kwargs)

    def irfftn(self, arr: BackendArray, *args, **kwargs) -> BackendArray:
        return self._array_backend.fft.irfftn(arr, **kwargs)

    def rigid_transform(
        self,
        arr: BackendArray,
        rotation_matrix: BackendArray,
        out: BackendArray = None,
        out_mask: BackendArray = None,
        translation: BackendArray = None,
        arr_mask: BackendArray = None,
        order: int = 1,
        **kwargs,
    ) -> Tuple[BackendArray, BackendArray]:
        rotate_mask = arr_mask is not None

        # This approach is only valid for order <= 1
        if arr.ndim != rotation_matrix.shape[0]:
            matrix = self._array_backend.zeros((arr.ndim, arr.ndim))
            matrix = matrix.at[0, 0].set(1)
            matrix = matrix.at[1:, 1:].add(rotation_matrix)
            rotation_matrix = matrix

        center = self.divide(self.to_backend_array(arr.shape) - 1, 2)[:, None]
        indices = self._array_backend.indices(arr.shape, dtype=self._float_dtype)
        indices = indices.reshape((arr.ndim, -1))
        indices = indices.at[:].add(-center)
        indices = self._array_backend.matmul(rotation_matrix.T, indices)
        indices = indices.at[:].add(center)
        if translation is not None:
            indices = indices.at[:].add(translation)

        out = self.scipy.ndimage.map_coordinates(arr, indices, order=order).reshape(
            arr.shape
        )

        out_mask = arr_mask
        if rotate_mask:
            out_mask = self.scipy.ndimage.map_coordinates(
                arr_mask, indices, order=order
            ).reshape(arr_mask.shape)

        return out, out_mask

    def max_score_over_rotations(
        self,
        scores: BackendArray,
        max_scores: BackendArray,
        rotations: BackendArray,
        rotation_index: int,
    ) -> Tuple[BackendArray, BackendArray]:
        update = self.greater(max_scores, scores)
        max_scores = max_scores.at[:].set(self.where(update, max_scores, scores))
        rotations = rotations.at[:].set(self.where(update, rotations, rotation_index))
        return max_scores, rotations

    def _scan(
        self,
        matching_data: type,
        splits: Tuple[Tuple[slice, slice]],
        n_jobs: int,
        callback_class,
        rotate_mask: bool = False,
        **kwargs,
    ) -> List:
        """
        Emulates output of :py:meth:`tme.matching_exhaustive.scan` using
        :py:class:`tme.analyzer.MaxScoreOverRotations`.
        """
        from ._jax_utils import scan as scan_inner

        pad_target = True if len(splits) > 1 else False
        convolution_mode = "valid" if pad_target else "same"
        target_pad = matching_data.target_padding(pad_target=pad_target)

        target_shape = tuple(
            (x.stop - x.start + p) for x, p in zip(splits[0][0], target_pad)
        )
        conv_shape, fast_shape, fast_ft_shape, shift = matching_data._fourier_padding(
            target_shape=self.to_numpy_array(target_shape),
            template_shape=self.to_numpy_array(matching_data._template.shape),
            pad_fourier=False,
        )

        analyzer_args = {
            "convolution_mode": convolution_mode,
            "fourier_shift": shift,
            "targetshape": target_shape,
            "templateshape": matching_data.template.shape,
            "convolution_shape": conv_shape,
        }

        create_target_filter = matching_data.target_filter is not None
        create_template_filter = matching_data.template_filter is not None
        create_filter = create_target_filter or create_template_filter

        # Applying the filter leads to more FFTs
        fastt_shape = matching_data._template.shape
        if create_template_filter:
            fastt_shape = matching_data._template.shape

        ret, template_filter, target_filter = [], 1, 1
        rotation_mapping = {
            self.tobytes(matching_data.rotations[i]): i
            for i in range(matching_data.rotations.shape[0])
        }
        for split_start in range(0, len(splits), n_jobs):
            split_subset = splits[split_start : (split_start + n_jobs)]
            if not len(split_subset):
                continue

            targets, translation_offsets = [], []
            for target_split, template_split in split_subset:
                base = matching_data.subset_by_slice(
                    target_slice=target_split,
                    target_pad=target_pad,
                    template_slice=template_split,
                )
                translation_offsets.append(base._translation_offset)
                targets.append(self.topleft_pad(base._target, fast_shape))

            if create_filter:
                filter_args = {
                    "data_rfft": self.fft.rfftn(targets[0]),
                    "return_real_fourier": True,
                    "shape_is_real_fourier": False,
                }

            if create_template_filter:
                template_filter = matching_data.template_filter(
                    shape=fastt_shape, **filter_args
                )["data"]
                template_filter = template_filter.at[(0,) * template_filter.ndim].set(0)

            if create_target_filter:
                target_filter = matching_data.target_filter(
                    shape=fast_shape, **filter_args
                )["data"]
                target_filter = target_filter.at[(0,) * target_filter.ndim].set(0)

            create_filter, create_template_filter, create_target_filter = (False,) * 3
            base, targets = None, self._array_backend.stack(targets)
            scores, rotations = scan_inner(
                self.astype(targets, self._float_dtype),
                matching_data.template,
                matching_data.template_mask,
                matching_data.rotations,
                template_filter,
                target_filter,
                fast_shape,
                rotate_mask,
            )

            for index in range(scores.shape[0]):
                temp = callback_class(
                    shape=scores.shape,
                    scores=scores[index],
                    rotations=rotations[index],
                    thread_safe=False,
                    offset=translation_offsets[index],
                )
                temp.rotation_mapping = rotation_mapping
                ret.append(tuple(temp._postprocess(**analyzer_args)))

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
