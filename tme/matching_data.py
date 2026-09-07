"""
Class representation of template matching data.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Tuple, List, Optional

import numpy as np

from . import Density
from .filters import Compose
from .backends import backend as be
from .memory import compute_schedule
from .types import BackendArray, NDArray
from .matching_utils import copy_docstring


__all__ = ["MatchingData"]


class MatchingData:
    """
    Contains data required for template matching.

    Parameters
    ----------
    target : np.ndarray or :py:class:`tme.density.Density`
        Target data.
    template : np.ndarray or :py:class:`tme.density.Density`
        Template data.
    target_mask : np.ndarray or :py:class:`tme.density.Density`, optional
        Target mask data.
    template_mask : np.ndarray or :py:class:`tme.density.Density`, optional
        Template mask data.
    invert_target : bool, optional
        Whether to invert the target before template matching.
    rotations: np.ndarray, optional
        Template rotations to sample. Can be a single (d, d) or a stack (n, d, d)
        of rotation matrices where d is the dimension of the template.

    Examples
    --------
    The following achieves the minimal definition of a :py:class:`MatchingData` instance.

    >>> import numpy as np
    >>> from tme.matching_data import MatchingData
    >>> target = np.random.rand(50,40,60)
    >>> template = target[15:25, 10:20, 30:40]
    >>> matching_data = MatchingData(target=target, template=template)

    """

    def __init__(
        self,
        target: NDArray,
        template: NDArray,
        template_mask: NDArray = None,
        target_mask: NDArray = None,
        invert_target: bool = False,
        rotations: NDArray = None,
    ):
        self.target = target
        self.target_mask = target_mask

        self.template = template
        if template_mask is not None:
            self.template_mask = template_mask
        self._invert_target = invert_target

        self.set_matching_dimension()
        self.rotations = rotations

    @staticmethod
    def _shape_to_slice(shape: Tuple[int]) -> Tuple[slice]:
        return tuple(slice(0, dim) for dim in shape)

    @classmethod
    def _slice_to_mesh(cls, slice_variable: Tuple[slice], shape: Tuple[int]) -> NDArray:
        if slice_variable is None:
            slice_variable = cls._shape_to_slice(shape)
        ranges = [range(slc.start, slc.stop) for slc in slice_variable]
        indices = np.meshgrid(*ranges, sparse=True, indexing="ij")
        return indices

    def subset_array(
        self,
        arr: NDArray,
        arr_slice: Tuple[slice],
        padding: NDArray,
        invert: bool = False,
    ) -> NDArray:
        """
        Extract a subset of the input array according to the given slice and
        apply padding. If the padding exceeds the array dimensions, the
        padded regions are filled by reflection of the boundaries. Otherwise,
        the values in ``arr`` are used.

        Parameters
        ----------
        arr : NDArray
            The input array from which a subset is extracted.
        arr_slice : tuple of slice
            Defines the region of the input array to be extracted.
        padding : NDArray
            Padding values for each dimension.
        invert : bool, optional
            Whether the returned array should be inverted.

        Returns
        -------
        NDArray
            Subset of the input array with padding applied.
        """
        padding = be.to_numpy_array(padding)
        padding = np.maximum(padding, 0).astype(int)

        slice_start = np.array([x.start for x in arr_slice], dtype=int)
        slice_stop = np.array([x.stop for x in arr_slice], dtype=int)

        left_pad = np.divide(padding, 2).astype(int)
        right_pad = np.add(left_pad, np.mod(padding, 2))

        data_voxels_left = np.minimum(slice_start, left_pad)
        data_voxels_right = np.minimum(
            np.subtract(arr.shape, slice_stop), right_pad
        ).astype(int)

        arr_start = np.subtract(slice_start, data_voxels_left)
        arr_stop = np.add(slice_stop, data_voxels_right)
        arr_slice = tuple(slice(*pos) for pos in zip(arr_start, arr_stop))
        arr_mesh = self._slice_to_mesh(arr_slice, arr.shape)

        # Inputs are either tme.density.Density objects or regular numpy arrays
        if isinstance(arr, Density):
            # Memmaps created by Density contain only the array of interest, while
            # joblib memmap files contain multiple objects. Hence we need to
            # distinguish between them in the following
            if isinstance(arr.data, np.memmap):
                try:
                    arr = Density.from_file(arr.data.filename, subset=arr_slice).data
                except Exception:
                    arr = np.asarray(arr.data[*arr_mesh])
            else:
                arr = np.asarray(arr.data[*arr_mesh])
        else:
            arr = np.asarray(arr[*arr_mesh])

        padding = tuple(
            (left, right)
            for left, right in zip(
                np.subtract(left_pad, data_voxels_left),
                np.subtract(right_pad, data_voxels_right),
            )
        )
        # The reflections are later cropped from the scores
        arr = np.pad(arr, padding, mode="symmetric")

        if invert:
            arr = -arr
        return arr

    def subset_by_slice(
        self,
        target_slice: Tuple[slice] = None,
        template_slice: Tuple[slice] = None,
        target_pad: NDArray = None,
        template_pad: NDArray = None,
        invert_target: bool = False,
        return_global_position: bool = False,
    ) -> Tuple["MatchingData", Tuple]:
        """
        Subset class instance based on slices.

        Parameters
        ----------
        target_slice : tuple of slice, optional
            Target subset to use, all by default.
        template_slice : tuple of slice, optional
            Template subset to use, all by default.
        target_pad : BackendArray, optional
            Target padding, zero by default.
        template_pad : BackendArray, optional
            Template padding, zero by default.

        Returns
        -------
        :py:class:`MatchingData`
            Newly allocated subset of class instance.
        Tuple
            Translation offset to merge analyzers.

        Examples
        --------
        >>> import numpy as np
        >>> from tme.matching_data import MatchingData
        >>> target = np.random.rand(50,40,60)
        >>> template = target[15:25, 10:20, 30:40]
        >>> matching_data = MatchingData(target=target, template=template)
        >>> subset = matching_data.subset_by_slice(
        >>>     target_slice=(slice(0, 10), slice(10,20), slice(15,35))
        >>> )
        """
        if target_slice is None:
            target_slice = self._shape_to_slice(self._target.shape)
        if template_slice is None:
            template_slice = self._shape_to_slice(self._template.shape)

        if target_pad is None:
            target_pad = np.zeros(len(self._target.shape), dtype=int)
        if template_pad is None:
            template_pad = np.zeros(len(self._template.shape), dtype=int)

        target_mask, template_mask = None, None
        target_subset = self.subset_array(
            self._target, target_slice, target_pad, invert=self._invert_target
        )
        template_subset = self.subset_array(
            arr=self._template, arr_slice=template_slice, padding=template_pad
        )
        if self._target_mask is not None:
            mask_slice = zip(target_slice, self._target_mask.shape)
            mask_slice = tuple(x if t != 1 else slice(0, 1) for x, t in mask_slice)
            target_mask = self.subset_array(
                arr=self._target_mask, arr_slice=mask_slice, padding=target_pad
            )
        if self._template_mask is not None:
            mask_slice = zip(template_slice, self._template_mask.shape)
            mask_slice = tuple(x if t != 1 else slice(0, 1) for x, t in mask_slice)
            template_mask = self.subset_array(
                arr=self._template_mask, arr_slice=mask_slice, padding=template_pad
            )

        ret = self.__class__(
            target=target_subset,
            template=template_subset,
            template_mask=template_mask,
            target_mask=target_mask,
            rotations=self.rotations,
            invert_target=self._invert_target,
        )
        ret.set_matching_dimension(
            target_batched=self._target_batched,
            template_batched=self._template_batched,
        )
        ret.target_filter = self.target_filter
        ret.template_filter = self.template_filter

        starts = [s.start for s in target_slice]

        initial_shape = self._output_target_shape
        return_shape = ret._output_target_shape
        global_pos = tuple(
            int((y + z // 2) - x // 2)
            for x, y, z in zip(initial_shape, starts, return_shape)
        )
        if self._has_batch:
            starts.insert(1, 0)

        if return_global_position:
            return ret, tuple(int(x) for x in starts), global_pos
        return ret, tuple(int(x) for x in starts)

    def to_backend(self):
        """
        Transfer and convert types of internal data arrays to the current backend.

        Examples
        --------
        >>> matching_data.to_backend()
        """
        backend_arr = type(be.zeros((1), dtype=be._float))
        for attr_name, attr_value in vars(self).items():
            converted_array = None
            if isinstance(attr_value, np.ndarray):
                converted_array = be.to_backend_array(attr_value.copy())
            elif isinstance(attr_value, backend_arr):
                converted_array = be.to_backend_array(attr_value)
            else:
                continue

            current_dtype = be.get_fundamental_dtype(converted_array)
            target_dtype = be._fundamental_dtypes[current_dtype]

            # Optional, but scores are float so we avoid casting and potential issues
            if attr_name in ("_template", "_template_mask", "_target", "_target_mask"):
                target_dtype = be._float

            if target_dtype != current_dtype:
                converted_array = be.astype(converted_array, target_dtype)

            setattr(self, attr_name, converted_array)

    def set_matching_dimension(
        self, target_batched: bool = False, template_batched: bool = False
    ):
        """
        Configure batch dimensions for target and template

        Parameters
        ----------
        target_batched : bool, optional
            Whether the target has a leading batch dimension.
        template_batched : bool, optional
            Whether the template has a leading batch dimension.

        Examples
        --------
        >>> matching_data.set_matching_dimension(target_batched=True)

        Notes
        -----
        The batch dimension, if present, is always at position 0. When either
        side is batched, the other gets a singleton leading dimension so that
        both output shapes share the same number of dimensions.
        """
        self._target_batched = target_batched
        self._template_batched = template_batched
        self._has_batch = target_batched or template_batched

    @property
    def _output_target_shape(self):
        if self._has_batch and not self._target_batched:
            return (1,) + self._target.shape
        return self._target.shape

    @property
    def _output_template_shape(self):
        if self._has_batch and not self._template_batched:
            return (1,) + self._template.shape
        return self._template.shape

    def _batch_shape(
        self, shape: Tuple[int], target: bool = True
    ) -> Tuple[Tuple[int], Tuple[int]]:
        pad_shape = tuple(shape)
        reduced = axes = tuple(range(len(shape)))
        if self._has_batch:
            axes = axes[2:]
            reduced = tuple(range(1, len(shape) - 1))
            pad_shape = (shape[0], 1) if target else (1, shape[1])
            pad_shape = pad_shape + shape[2:]
        return pad_shape, axes, reduced

    def _to_full_batch(self, arr, target=True):
        if self._has_batch:
            return arr[:, None, ...] if target else arr[None, ...]
        return arr

    def _matching_shapes(self):
        targetshape = self._output_target_shape
        templateshape = self._output_template_shape
        if self._has_batch:
            targetshape = (targetshape[0], 1) + targetshape[1:]
            templateshape = (1,) + templateshape
        return targetshape, templateshape

    def target_padding(self, pad_target: bool = False) -> Tuple[int]:
        """
        Return padding to full convolution shape given the template.

        Parameters
        ----------
        pad_target : bool, optional
            Whether output shape is full convolution or same shape as target.

        Returns
        -------
        tuple of int
            Padding along each dimension.
        """
        padding = (0,) * len(self._output_target_shape)
        if pad_target:
            padding = np.subtract(self._output_template_shape, 1)
            if self._has_batch:
                padding[0] = 0
        return tuple(int(x) for x in padding)

    def fourier_padding(self, target_shape=None, template_shape=None) -> Tuple:
        """
        Computes efficient shape for Fourier transforms and potential associated shifts.

        Returns
        -------
        Tuple[tuple of int, tuple of int, tuple of int, tuple of int]
            Tuple with convolution, forward FT, inverse FT shape and corresponding shift.
            When batched, shapes are prefixed with (target_batch, template_batch).
        """
        batch_prefix = ()

        if target_shape is None:
            target_shape = self._output_target_shape
        if template_shape is None:
            template_shape = self._output_template_shape

        if self._has_batch:
            batch_prefix = (int(target_shape[0]), int(template_shape[0]))
            target_shape = target_shape[1:]
            template_shape = template_shape[1:]

        pad_shape = np.maximum(target_shape, template_shape)
        conv, fwd, inv = be.compute_convolution_shapes(
            pad_shape, np.ones_like(pad_shape)
        )

        fourier_shift = (
            1 - np.divide(template_shape, 2).astype(int) - np.mod(template_shape, 2)
        )

        shape_diff = np.subtract(target_shape, template_shape)
        if np.sum(shape_diff < 0):
            warnings.warn(
                "Template is larger than target and padding is turned off. Consider "
                "swapping them or activate padding. Correcting the shift for now."
            )
            shape_shift = np.divide(shape_diff, 2)
            offset = np.mod(shape_diff, 2)
            shape_shift = np.multiply(np.add(shape_shift, offset), shape_diff < 0)
            fourier_shift = np.subtract(fourier_shift, shape_shift).astype(int)

        fourier_shift = tuple(int(x) for x in fourier_shift)

        conv = batch_prefix + tuple(conv)
        fwd = batch_prefix + tuple(fwd)
        inv = batch_prefix + tuple(inv)
        fourier_shift = (0,) * len(batch_prefix) + fourier_shift

        return conv, fwd, inv, fourier_shift

    def _score_mask(self, fast_shape: Tuple[int], shift: Tuple[int]) -> BackendArray:
        """
        Create a boolean mask to exclude scores derived from padding in template matching.
        """
        padding = self.target_padding(True)
        offset = tuple(x // 2 for x in padding)
        shape = tuple(y - x for x, y in zip(padding, self.target.shape))

        # Spatial-only slicing; batch dims handled by prefixing slice(None)
        skip = 1 if self._has_batch else 0
        subset = [slice(None)] * (2 * skip)
        for i in range(skip, len(offset)):
            subset.append(slice(offset[i], offset[i] + shape[i]))

        score_mask = np.zeros(fast_shape, dtype=bool)
        score_mask[tuple(subset)] = 1
        score_mask = np.roll(
            score_mask,
            shift=tuple(-x for x in shift),
            axis=tuple(i for i in range(len(shift))),
        )
        return be.to_backend_array(score_mask)

    def _transform_data(
        self, method: str, data: BackendArray, batched: bool = False, **kwargs
    ) -> BackendArray:
        """
        Transform data using the specified method.

        Parameters
        ----------
        data : BackendArray
            Data to transform.
        method : str, optional
            Transformation method, default "phase_randomization".
            - "phase_randomization": Scrambles phase while preserving amplitude spectrum
            - "standardize": Standardize to zero mean and unit variance
            - "laplace": Applies Laplacian edge detection filter
        batched : bool
            Whether data has a leading batch dimension.
        **kwargs : dict
            Method-specific arguments (e.g., mode="wrap" for laplace).

        Returns
        -------
        BackendArray
            Transformed data.
        """
        from scipy.ndimage import laplace
        from .matching_utils import scramble_phases, standardize

        _methods = {
            "phase_randomization": lambda a, **kw: scramble_phases(
                be.to_numpy_array(a), **kw
            ),
            "laplace": lambda a, **kw: laplace(be.to_numpy_array(a), **kw),
            "standardize": lambda a, **kw: standardize(a, 1, be.size(a)),
        }
        func = _methods.get(method)
        if func is None:
            _supported = ",".join([str(x) for x in _methods])
            raise ValueError(f"Only methods {_supported} are supported.")

        if not batched:
            return be.to_backend_array(func(data, **kwargs))

        ret = be.zeros(data.shape, data.dtype)
        for i in range(data.shape[0]):
            slc = slice(i, i + 1)
            ret = be.at(ret, slc, be.to_backend_array(func(data[slc], **kwargs)))
        return ret

    @copy_docstring(_transform_data)
    def transform_target(self, method: str = "phase_randomization", **kwargs):
        ret = self._transform_data(method, self.target, self._target_batched, **kwargs)
        if self._has_batch and not self._target_batched:
            return ret[0]
        return ret

    @copy_docstring(_transform_data)
    def transform_template(self, method: str = "phase_randomization", **kwargs):
        template = self._get_data(
            self._template,
            self._output_template_shape,
            False,
            (0,) if self._has_batch else (),
        )
        ret = self._transform_data(method, template, self._template_batched, **kwargs)
        if self._has_batch and not self._template_batched:
            return ret[0]
        return ret

    def computation_schedule(
        self,
        matching_method: str = "FLCSphericalMask",
        max_workers: int = 1,
        pad_fourier: bool = False,
        pad_target_edges: bool = False,
        analyzer_method: str = None,
        max_memory: int = None,
        **mode_kwargs,
    ) -> Tuple[Tuple[Tuple[slice, ...]], Tuple[int, int]]:
        """
        Computes a parallelization schedule for a given template matching operation.

        Parameters
        ----------
        matching_method : str
            Matching method to use, default "FLCSphericalMask".
        max_workers : int, optional
            Maximum number of concurrent workers.
        pad_fourier : bool, optional
            Apply Fourier padding, default False.
        pad_target_edges : bool, optional
            Apply padding to target edges, default False.
        analyzer_method : str, optional
            Method used for score analysis, default None.
        max_memory : int, optional
            Maximum amount of memory that can be used in bytes.
        **mode_kwargs:
            Keyword arguments passed to :py:mesh:`tme.memory.compute_schedule`.

        Returns
        -------
        tuple of tuple of slice
            Tuple of slices defining a region in shape1 coordinates.
        tuple of int int
            Parallelization strategy as n_outer_jobs, n_inner_workers.
        """
        if max_memory is None:
            max_memory = be.get_available_memory() * be.device_count()

        shape = target = self._output_target_shape
        template = self._output_template_shape
        if self._has_batch:
            target = (target[0], 1) + target[1:]
            template = (1, template[0]) + template[1:]
            shape = np.broadcast_shapes(target, template)

        padding = tuple(0 for _ in target)
        if pad_target_edges:
            padding = template if not self._has_batch else (0, 0) + template[2:]

        if "split_axes" not in mode_kwargs and self._target_batched:
            mode_kwargs["split_axes"] = (0,)

        mode = mode_kwargs.get("mode")
        if self._has_batch and mode != "uniform":
            warnings.warn(
                f"'{mode}' is not supported for batches. Falling back to 'uniform'"
            )
            mode_kwargs["mode"] = "uniform"

        return compute_schedule(
            shape=shape,
            padding=padding,
            max_workers=max_workers,
            max_memory=max_memory,
            matching_method=matching_method,
            analyzer_method=analyzer_method,
            backend=be._backend_name,
            float_nbytes=be.datatype_bytes(be._float),
            complex_nbytes=be.datatype_bytes(be._complex),
            integer_nbytes=be.datatype_bytes(be._int),
            equal_shape=be._backend_name == "jax" and max_workers > 1,
            **mode_kwargs,
        )

    @property
    def rotations(self):
        """Return stored rotation matrices."""
        return self._rotations

    @rotations.setter
    def rotations(self, rotations: BackendArray):
        """
        Set :py:attr:`MatchingData.rotations`.

        Parameters
        ----------
        rotations : BackendArray
            Rotations matrices with shape (d, d) or (n, d, d).
        """
        if rotations is None:
            rotations = np.eye(len(self._target.shape) - int(self._target_batched))

        if rotations.ndim not in (2, 3):
            raise ValueError("Rotations have to be a rank 2 or 3 array.")
        elif rotations.ndim == 2:
            rotations = rotations.reshape(1, *rotations.shape)
        self._rotations = rotations.astype(np.float32)

    @staticmethod
    def _get_data(
        attribute,
        output_shape: Tuple[int],
        reverse: bool = False,
        axis: Tuple[int] = None,
    ):
        if isinstance(attribute, Density):
            attribute = attribute.data

        if attribute is not None:
            if reverse:
                rev_axis = tuple(i for i in range(attribute.ndim) if i not in axis)
                attribute = be.reverse(attribute, axis=rev_axis)
            attribute = attribute.reshape(tuple(int(x) for x in output_shape))

        return attribute

    @property
    def target(self) -> BackendArray:
        """Return the target."""
        return self._get_data(self._target, self._output_target_shape, False)

    @property
    def target_mask(self) -> BackendArray:
        """Return the target mask."""
        return self._get_data(self._target_mask, self._output_target_shape, False)

    @property
    def template(self) -> BackendArray:
        """Return the reversed template."""
        return self._get_data(
            self._template,
            self._output_template_shape,
            True,
            (0,) if self._template_batched else (),
        )

    @property
    def template_mask(self) -> BackendArray:
        """Return the reversed template mask."""
        return self._get_data(
            self._template_mask,
            self._output_template_shape,
            True,
            (0,) if self._template_batched else (),
        )

    @target.setter
    def target(self, arr: NDArray):
        """
        Set :py:attr:`MatchingData.target`.

        Parameters
        ----------
        arr : NDArray
            Array to set as the target.
        """
        self._target = arr

    @template.setter
    def template(self, arr: NDArray):
        """
        Set :py:attr:`MatchingData.template` and initializes
        :py:attr:`MatchingData.template_mask` to an to an uninformative
        mask filled with ones if not already defined.

        Parameters
        ----------
        arr : NDArray
            Array to set as the template.
        """
        self._template = arr
        if getattr(self, "_template_mask", None) is None:
            self._template_mask = np.full(
                shape=arr.shape, dtype=np.float32, fill_value=1
            )

    @staticmethod
    def _set_mask(mask, shape: Tuple[int]):
        if mask is not None:
            if np.broadcast_shapes(mask.shape, shape) != shape:
                raise ValueError("Mask and data shape need to be broadcastable.")
        return mask

    @target_mask.setter
    def target_mask(self, arr: NDArray):
        """
        Set :py:attr:`MatchingData.target_mask`.

        Parameters
        ----------
        arr : NDArray
            Array to set as the target_mask.
        """
        self._target_mask = self._set_mask(mask=arr, shape=self._target.shape)

    @template_mask.setter
    def template_mask(self, arr: NDArray):
        """
        Set :py:attr:`MatchingData.template_mask`.

        Parameters
        ----------
        arr : NDArray
            Array to set as the template_mask.
        """
        self._template_mask = self._set_mask(mask=arr, shape=self._template.shape)

    @staticmethod
    def _set_filter(composable_filter) -> Optional[Compose]:
        if composable_filter is None:
            return None

        if not isinstance(composable_filter, Compose):
            warnings.warn(
                "Custom filters are not sanitized and need to be correctly shaped."
            )

        return composable_filter

    @property
    def template_filter(self) -> Optional[Compose]:
        """
        Returns the template filter.

        Returns
        -------
        :py:class:`tme.preprocessing.compose.Compose` | BackendArray | None
            Composable filter, a backend array or None.
        """
        return getattr(self, "_template_filter", None)

    @property
    def target_filter(self) -> Optional[Compose]:
        """
        Returns the target filter.

        Returns
        -------
        :py:class:`tme.preprocessing.compose.Compose` | BackendArray | None
            Composable filter, a backend array or None.
        """
        return getattr(self, "_target_filter", None)

    @template_filter.setter
    def template_filter(self, template_filter):
        self._template_filter = self._set_filter(template_filter)

    @target_filter.setter
    def target_filter(self, target_filter):
        self._target_filter = self._set_filter(target_filter)

    def _split_rotations_on_jobs(self, n_jobs: int) -> List[NDArray]:
        """
        Split the rotation matrices into parts based on the number of jobs.

        Parameters
        ----------
        n_jobs : int
            Number of jobs for splitting.

        Returns
        -------
        list of NDArray
            List of split rotation matrices.
        """
        nrot_per_job = int(self.rotations.shape[0] // n_jobs)
        rot_list = []
        for n in range(n_jobs):
            init_rot = n * nrot_per_job
            end_rot = init_rot + nrot_per_job
            if n == n_jobs - 1:
                end_rot = None
            rot_list.append(self.rotations[init_rot:end_rot])
        return rot_list

    def free(self):
        """
        Dereference data arrays owned by the class instance.
        """
        attrs = ("_target", "_template", "_template_mask", "_target_mask")
        for attr in attrs:
            setattr(self, attr, None)
