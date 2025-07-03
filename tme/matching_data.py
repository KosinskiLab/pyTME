"""
Class representation of template matching data.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Tuple, List, Optional, Generator, Dict

import numpy as np

from . import Density
from .filters import Compose
from .backends import backend as be
from .types import BackendArray, NDArray
from .matching_utils import compute_parallelization_schedule

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

        self.rotations = rotations
        self._invert_target = invert_target
        self._translation_offset = tuple(0 for _ in range(len(target.shape)))

        self._set_matching_dimension()

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

    @staticmethod
    def _load_array(arr: BackendArray) -> BackendArray:
        """Load ``arr``, if ``arr`` type is a :obj:`numpy.memmap`, reload from disk."""
        if isinstance(arr, np.memmap):
            return np.memmap(arr.filename, mode="r", shape=arr.shape, dtype=arr.dtype)
        return arr

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

        # We are deviating from our typical right_pad + mod here
        # because cropping from full convolution mode to target shape
        # is defined from the perspective of the origin
        right_pad = np.divide(padding, 2).astype(int)
        left_pad = np.add(right_pad, np.mod(padding, 2))

        data_voxels_left = np.minimum(slice_start, left_pad)
        data_voxels_right = np.minimum(
            np.subtract(arr.shape, slice_stop), right_pad
        ).astype(int)

        arr_start = np.subtract(slice_start, data_voxels_left)
        arr_stop = np.add(slice_stop, data_voxels_right)
        arr_slice = tuple(slice(*pos) for pos in zip(arr_start, arr_stop))
        arr_mesh = self._slice_to_mesh(arr_slice, arr.shape)

        # Note different from joblib memmaps, the memmaps created by
        # Density are guaranteed to only contain the array of interest
        if isinstance(arr, Density):
            if isinstance(arr.data, np.memmap):
                arr = Density.from_file(arr.data.filename, subset=arr_slice).data
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
        arr = np.pad(arr, padding, mode="reflect")

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

        # Deal with splitting offsets
        mask = np.subtract(1, self._template_batch).astype(bool)
        target_offset = np.zeros(len(self._output_target_shape), dtype=int)
        target_offset[mask] = [x.start for x in target_slice]
        mask = np.subtract(1, self._target_batch).astype(bool)
        template_offset = np.zeros(len(self._output_template_shape), dtype=int)
        template_offset[mask] = [x.start for x, b in zip(template_slice, mask) if b]

        translation_offset = tuple(x for x in target_offset)

        ret.target_filter = self.target_filter
        ret.template_filter = self.template_filter

        ret.set_matching_dimension(
            target_dim=getattr(self, "_target_dim", None),
            template_dim=getattr(self, "_template_dim", None),
        )

        return ret, translation_offset

    def to_backend(self):
        """
        Transfer and convert types of internal data arrays to the current backend.

        Examples
        --------
        >>> matching_data.to_backend()
        """
        backend_arr = type(be.zeros((1), dtype=be._float_dtype))
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
                target_dtype = be._float_dtype

            if target_dtype != current_dtype:
                converted_array = be.astype(converted_array, target_dtype)

            setattr(self, attr_name, converted_array)

    def set_matching_dimension(self, target_dim: int = None, template_dim: int = None):
        """
        Sets matching dimensions for target and template.

        Parameters
        ----------
        target_dim : int, optional
            Target batch dimension, None by default.
        template_dim : int, optional
            Template batch dimension, None by default.

        Examples
        --------
        >>> matching_data.set_matching_dimension(target_dim=0, template_dim=None)

        Notes
        -----
        If target and template share a batch dimension, the target will take
        precendence and the template dimension will be shifted to the right. If target
        and template have the same dimension, but target specifies batch dimensions,
        the leftmost template dimensions are assumed to be collapse dimensions.
        """
        target_ndim = len(self._target.shape)
        _, target_dims = self._compute_batch_dims(target_dim, ndim=target_ndim)
        template_ndim = len(self._template.shape)
        _, template_dims = self._compute_batch_dims(template_dim, ndim=template_ndim)

        target_ndim -= len(target_dims)
        template_ndim -= len(template_dims)
        self._set_matching_dimension(
            target_dims=target_dims, template_dims=template_dims
        )

    def _set_matching_dimension(
        self, target_dims: Tuple[int] = (), template_dims: Tuple[int] = ()
    ):
        self._target_dim, self._template_dim = target_dims, template_dims

        target_ndim, template_ndim = len(self._target.shape), len(self._template.shape)
        batch_dims = len(target_dims) + len(template_dims)
        target_measurement_dims = target_ndim - len(target_dims)
        collapse_dims = max(
            template_ndim - len(template_dims) - target_measurement_dims, 0
        )
        matching_dims = target_measurement_dims + batch_dims

        target_shape = np.full(shape=matching_dims, fill_value=1, dtype=int)
        template_shape = np.full(shape=matching_dims, fill_value=1, dtype=int)
        template_batch = np.full(shape=matching_dims, fill_value=1, dtype=int)
        target_batch = np.full(shape=matching_dims, fill_value=1, dtype=int)

        target_index, template_index = 0, 0
        for k in range(matching_dims):
            target_dim = k - target_index
            template_dim = k - template_index

            if target_dim in target_dims:
                target_shape[k] = self._target.shape[target_dim]
                template_batch[k] = 0
                if target_index == len(template_dims) and collapse_dims > 0:
                    template_shape[k] = self._template.shape[template_dim]
                    collapse_dims -= 1
                template_index += 1
                continue

            if template_dim in template_dims:
                template_shape[k] = self._template.shape[template_dim]
                target_batch[k] = 0
                target_index += 1
                continue

            target_batch[k] = template_batch[k] = 0
            if target_dim < target_ndim:
                target_shape[k] = self._target.shape[target_dim]
            if template_dim < template_ndim:
                template_shape[k] = self._template.shape[template_dim]

        batch_mask = np.logical_or(target_batch, template_batch)
        self._output_target_shape = tuple(int(x) for x in target_shape)
        self._output_template_shape = tuple(int(x) for x in template_shape)
        self._batch_mask = tuple(int(x) for x in batch_mask)
        self._template_batch = tuple(int(x) for x in template_batch)
        self._target_batch = tuple(int(x) for x in target_batch)

        output_shape = np.add(
            self._output_target_shape,
            np.multiply(self._template_batch, self._output_template_shape),
        )
        output_shape = np.subtract(output_shape, self._template_batch)
        self._output_shape = tuple(int(x) for x in output_shape)

    @staticmethod
    def _compute_batch_dims(batch_dims: Tuple[int], ndim: int) -> Tuple:
        """
        Computes a mask for the batch dimensions and the validated batch dimensions.

        Parameters
        ----------
        batch_dims : tuple of int
            A tuple of integers representing the batch dimensions.
        ndim : int
            The number of dimensions of the array.

        Returns
        -------
        Tuple[ArrayLike, tuple of int]
            Mask and the corresponding batch dimensions.

        Raises
        ------
        ValueError
            If any dimension in batch_dims is not less than ndim.
        """
        mask = np.zeros(ndim, dtype=int)
        if batch_dims is None:
            return mask, ()

        if isinstance(batch_dims, int):
            batch_dims = (batch_dims,)

        for dim in batch_dims:
            if dim < ndim:
                mask[dim] = 1
                continue
            raise ValueError(f"Batch indices needs to be < {ndim}, got {dim}.")

        return mask, batch_dims

    @staticmethod
    def _batch_shape(shape: Tuple[int], mask: Tuple[int], keepdims=True) -> Tuple[int]:
        if keepdims:
            return tuple(x if y == 0 else 1 for x, y in zip(shape, mask))
        return tuple(x for x, y in zip(shape, mask) if y == 0)

    @staticmethod
    def _batch_iter(shape: Tuple[int], mask: Tuple[int]) -> Generator:
        def _recursive_gen(current_shape, current_mask, current_slices):
            if not current_shape:
                yield current_slices
                return

            if current_mask[0] == 1:
                for i in range(current_shape[0]):
                    new_slices = current_slices + (slice(i, i + 1),)
                    yield from _recursive_gen(
                        current_shape[1:], current_mask[1:], new_slices
                    )
            else:
                new_slices = current_slices + (slice(None),)
                yield from _recursive_gen(
                    current_shape[1:], current_mask[1:], new_slices
                )

        return _recursive_gen(shape, mask, ())

    @staticmethod
    def _batch_axis(mask: Tuple[int]) -> Tuple[int]:
        return tuple(i for i in range(len(mask)) if mask[i] == 0)

    def target_padding(self, pad_target: bool = False) -> Tuple[int]:
        """
        Computes the padding of the target to the full convolution
        shape given the registered template.

        Parameters
        ----------
        pad_target : bool, optional
            Whether to pad the target, defaults to False.

        Returns
        -------
        tuple of int
            Padding along each dimension.

        Examples
        --------
        >>> matching_data.target_padding(pad_target=True)
        """
        padding = np.zeros(len(self._output_target_shape), dtype=int)
        if pad_target:
            padding = np.subtract(self._output_template_shape, 1)
            if hasattr(self, "_target_batch"):
                padding = np.multiply(padding, np.subtract(1, self._target_batch))

        if hasattr(self, "_template_batch"):
            padding = tuple(x for x, i in zip(padding, self._template_batch) if i == 0)

        return tuple(int(x) for x in padding)

    @staticmethod
    def _fourier_padding(
        target_shape: Tuple[int],
        template_shape: Tuple[int],
        pad_target: bool = False,
        batch_mask: Tuple[int] = None,
    ) -> Tuple[Tuple, Tuple, Tuple, Tuple]:
        if batch_mask is None:
            batch_mask = np.zeros_like(template_shape)
        batch_mask = np.asarray(batch_mask)

        fourier_pad = np.ones(len(template_shape), dtype=int)
        fourier_pad = np.multiply(fourier_pad, 1 - batch_mask)
        fourier_pad = np.add(fourier_pad, batch_mask)

        # Avoid padding batch dimensions
        pad_shape = np.maximum(target_shape, template_shape)
        pad_shape = np.maximum(target_shape, np.multiply(1 - batch_mask, pad_shape))
        ret = be.compute_convolution_shapes(pad_shape, fourier_pad)
        conv_shape, fast_shape, fast_ft_shape = ret

        template_mod = np.mod(template_shape, 2)
        fourier_shift = 1 - np.divide(template_shape, 2).astype(int)
        fourier_shift = np.subtract(fourier_shift, template_mod)

        shape_diff = np.multiply(
            np.subtract(target_shape, template_shape), 1 - batch_mask
        )
        shape_mask = shape_diff < 0
        if np.sum(shape_mask):
            shape_shift = np.divide(shape_diff, 2)
            offset = np.mod(shape_diff, 2)
            warnings.warn(
                "Template is larger than target and padding is turned off. Consider "
                "swapping them or activate padding. Correcting the shift for now."
            )
            shape_shift = np.multiply(np.add(shape_shift, offset), shape_mask)
            fourier_shift = np.subtract(fourier_shift, shape_shift).astype(int)

        if pad_target:
            fourier_shift = np.subtract(fourier_shift, np.subtract(1, template_mod))

        fourier_shift = tuple(np.multiply(fourier_shift, 1 - batch_mask).astype(int))
        return tuple(conv_shape), tuple(fast_shape), tuple(fast_ft_shape), fourier_shift

    def fourier_padding(self, pad_target: bool = False) -> Tuple:
        """
        Computes efficient shape four Fourier transforms and potential associated shifts.

        Parameters
        ----------
        pad_target : bool, optional
            Whether the target has been padded to the full convolution shape.

        Returns
        -------
        Tuple[tuple of int, tuple of int, tuple of int, tuple of int]
            Tuple with convolution, forward FT, inverse FT shape and corresponding shift.

        Examples
        --------
        >>> conv, fwd, inv, shift = matching_data.fourier_padding(pad_fourier=True)
        """
        return self._fourier_padding(
            target_shape=be.to_numpy_array(self._output_target_shape),
            template_shape=be.to_numpy_array(self._output_template_shape),
            batch_mask=be.to_numpy_array(self._batch_mask),
            pad_target=pad_target,
        )

    def computation_schedule(
        self,
        matching_method: str = "FLCSphericalMask",
        max_cores: int = 1,
        use_gpu: bool = False,
        pad_fourier: bool = False,
        pad_target_edges: bool = False,
        analyzer_method: str = None,
        available_memory: int = None,
        max_splits: int = 256,
    ) -> Tuple[Dict, Tuple]:
        """
        Computes a parallelization schedule for a given template matching operation.

        Parameters
        ----------
        matching_method : str
            Matching method to use, default "FLCSphericalMask".
        max_cores : int, optional
            Maximum number of CPU cores to use, default 1.
        use_gpu : bool, optional
            Whether to utilize GPU acceleration, default False.
        pad_fourier : bool, optional
            Apply Fourier padding, default False.
        pad_target_edges : bool, optional
            Apply padding to target edges, default False.
        analyzer_method : str, optional
            Method used for score analysis, default None.
        available_memory : int, optional
            Available memory in bytes. If None, uses all available system memory.
        max_splits : int, optional
            Maximum number of splits to consider, default 256.

        Returns
        -------
        target_splits : dict
            Optimal splits for each axis of the target tensor
        schedule : tuple
            (n_outer_jobs, n_inner_jobs_per_outer) defining the parallelization schedule
        """

        if available_memory is None:
            available_memory = be.get_available_memory() * be.device_count()

        _template = self._output_template_shape
        shape1 = np.broadcast_shapes(
            self._output_target_shape,
            self._batch_shape(_template, np.subtract(1, self._template_batch)),
        )

        shape2 = tuple(0 for _ in _template)
        if pad_fourier:
            shape2 = np.multiply(_template, np.subtract(1, self._batch_mask))

        padding = tuple(0 for _ in self._output_target_shape)
        if pad_target_edges:
            padding = tuple(
                x if y == 0 else 1 for x, y in zip(_template, self._template_batch)
            )

        return compute_parallelization_schedule(
            shape1=shape1,
            shape2=shape2,
            shape1_padding=padding,
            max_cores=max_cores,
            max_ram=available_memory,
            matching_method=matching_method,
            analyzer_method=analyzer_method,
            backend=be._backend_name,
            float_nbytes=be.datatype_bytes(be._float_dtype),
            complex_nbytes=be.datatype_bytes(be._complex_dtype),
            integer_nbytes=be.datatype_bytes(be._int_dtype),
            split_only_outer=use_gpu,
            split_axes=self._target_dim if len(self._target_dim) else None,
            max_splits=max_splits,
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
            print("No rotations provided, assuming identity for now.")
            rotations = np.eye(len(self._target.shape))

        if rotations.ndim not in (2, 3):
            raise ValueError("Rotations have to be a rank 2 or 3 array.")
        elif rotations.ndim == 2:
            print("Reshaping rotations array to rank 3.")
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
        target_mask = getattr(self, "_target_mask", None)
        if target_mask is None:
            return None

        _output_shape = self._output_target_shape
        if be.size(target_mask) != np.prod(_output_shape):
            _output_shape = self._batch_shape(_output_shape, self._target_batch, True)

        return self._get_data(target_mask, _output_shape, False)

    @property
    def template(self) -> BackendArray:
        """Return the reversed template."""
        _output_shape = self._output_template_shape
        return self._get_data(self._template, _output_shape, True, self._template_dim)

    @property
    def template_mask(self) -> BackendArray:
        """Return the reversed template mask."""
        template_mask = getattr(self, "_template_mask", None)
        if template_mask is None:
            return None

        _output_shape = self._output_template_shape
        if np.prod([int(i) for i in template_mask.shape]) != np.prod(_output_shape):
            _output_shape = self._batch_shape(_output_shape, self._template_batch, True)

        return self._get_data(template_mask, _output_shape, True, self._template_dim)

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
        nrot_per_job = self.rotations.shape[0] // n_jobs
        rot_list = []
        for n in range(n_jobs):
            init_rot = n * nrot_per_job
            end_rot = init_rot + nrot_per_job
            if n == n_jobs - 1:
                end_rot = None
            rot_list.append(self.rotations[init_rot:end_rot])
        return rot_list

    def _free_data(self):
        """
        Dereference data arrays owned by the class instance.
        """
        attrs = ("_target", "_template", "_template_mask", "_target_mask")
        for attr in attrs:
            setattr(self, attr, None)
