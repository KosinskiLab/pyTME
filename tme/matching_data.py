""" Class representation of template matching data.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import warnings
from typing import Tuple, List, Optional

import numpy as np
from numpy.typing import NDArray

from . import Density
from .types import ArrayLike
from .preprocessing import Compose
from .backends import backend as be
from .matching_utils import compute_full_convolution_index


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
        Whether to invert the target before template matching..
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
        self._translation_offset = np.zeros(len(target.shape), dtype=int)
        self._invert_target = invert_target

        self._set_matching_dimension()

    @staticmethod
    def _shape_to_slice(shape: Tuple[int]):
        return tuple(slice(0, dim) for dim in shape)

    @classmethod
    def _slice_to_mesh(cls, slice_variable: (slice,), shape: (int,)):
        if slice_variable is None:
            slice_variable = cls._shape_to_slice(shape)
        ranges = [range(slc.start, slc.stop) for slc in slice_variable]
        indices = np.meshgrid(*ranges, sparse=True, indexing="ij")
        return indices

    @staticmethod
    def _load_array(arr: NDArray):
        """
        Load ``arr``,  If ``arr`` type is memmap, reload from disk.

        Parameters
        ----------
        arr : NDArray
            Array to load.

        Returns
        -------
        NDArray
            Loaded array.
        """
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
        apply padding.

        Parameters
        ----------
        arr : NDArray
            The input array from which a subset is extracted.
        arr_slice : tuple of slice
            Defines the region of the input array to be extracted.
        padding : NDArray
            Padding values for each dimension. If the padding exceeds the array
            dimensions, the extra regions are filled with the mean of the array
            values, otherwise, the values in ``arr`` are used.
        invert : bool, optional
            Whether the returned array should be inverted and normalized to the interval
            [0, 1]. If available, uses the metadata information of the Density object,
            otherwise computes min and max on the extracted subset.

        Returns
        -------
        NDArray
            Subset of the input array with padding applied.
        """
        padding = be.to_numpy_array(padding)
        padding = np.maximum(padding, 0).astype(int)

        slice_start = np.array([x.start for x in arr_slice], dtype=int)
        slice_stop = np.array([x.stop for x in arr_slice], dtype=int)

        padding = np.add(padding, np.mod(padding, 2))
        left_pad = right_pad = np.divide(padding, 2).astype(int)

        data_voxels_left = np.minimum(slice_start, left_pad)
        data_voxels_right = np.minimum(
            np.subtract(arr.shape, slice_stop), right_pad
        ).astype(int)

        arr_start = np.subtract(slice_start, data_voxels_left)
        arr_stop = np.add(slice_stop, data_voxels_right)
        arr_slice = tuple(slice(*pos) for pos in zip(arr_start, arr_stop))
        arr_mesh = self._slice_to_mesh(arr_slice, arr.shape)

        if isinstance(arr, Density):
            if isinstance(arr.data, np.memmap):
                arr = Density.from_file(arr.data.filename, subset=arr_slice).data
            else:
                arr = np.asarray(arr.data[*arr_mesh])
        else:
            if isinstance(arr, np.memmap):
                arr = np.memmap(
                    arr.filename, mode="r", shape=arr.shape, dtype=arr.dtype
                )
            arr = np.asarray(arr[*arr_mesh])

        padding = tuple(
            (left, right)
            for left, right in zip(
                np.subtract(left_pad, data_voxels_left),
                np.subtract(right_pad, data_voxels_right),
            )
        )
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
    ) -> "MatchingData":
        """
        Subset class instance based on slices.

        Parameters
        ----------
        target_slice : tuple of slice, optional
            Target subset to use, all by default.
        template_slice : tuple of slice, optional
            Template subset to use, all by default.
        target_pad : NDArray, optional
            Target padding, zero by default.
        template_pad : NDArray, optional
            Template padding, zero by default.

        Returns
        -------
        :py:class:`MatchingData`
            Newly allocated subset of class instance.

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
            target_mask = self.subset_array(
                arr=self._target_mask, arr_slice=target_slice, padding=target_pad
            )
        if self._template_mask is not None:
            template_mask = self.subset_array(
                arr=self._template_mask, arr_slice=template_slice, padding=template_pad
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
        target_offset = np.zeros(len(self._output_target_shape), dtype=int)
        offset = target_offset.size - len(target_slice)
        target_offset[offset:] = [x.start for x in target_slice]
        template_offset = np.zeros(len(self._output_target_shape), dtype=int)
        offset = template_offset.size - len(template_slice)
        template_offset[offset:] = [x.start for x in template_slice]
        ret._translation_offset = target_offset
        if len(self._target.shape) == len(self._template.shape):
            ret.indices = compute_full_convolution_index(
                outer_shape=self._target.shape,
                inner_shape=self._template.shape,
                outer_split=target_slice,
                inner_split=template_slice,
            )

        ret._is_padded = be.sum(be.to_backend_array(target_pad)) > 0
        ret.target_filter = self.target_filter
        ret.template_filter = self.template_filter

        ret._set_matching_dimension(
            target_dims=getattr(self, "_target_dims", None),
            template_dims=getattr(self, "_template_dims", None),
        )

        return ret

    def to_backend(self) -> None:
        """
        Transfer and convert types of class instance's data arrays to the current backend
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

    def _set_matching_dimension(
        self, target_dims: Tuple[int] = None, template_dims: Tuple[int] = None
    ) -> None:
        """
        Sets matching dimensions for target and template.
        Parameters
        ----------
        target_dims : tuple of ints, optional
            Target batch dimensions, None by default.
        template_dims : tuple of ints, optional
            Template batch dimensions, None by default.

        Notes
        -----
        If the target and template share a batch dimension, the target will
        take precendence and the template dimension will be shifted to the right.
        If target and template have the same dimension, but target specifies batch
        dimensions, the leftmost template dimensions are assumed to be a collapse
        dimension that operates on a measurement dimension.
        """
        self._target_dims = target_dims
        self._template_dims = template_dims

        target_ndim = len(self._target.shape)
        self._is_target_batch, target_dims = self._compute_batch_dimension(
            batch_dims=target_dims, ndim=target_ndim
        )
        template_ndim = len(self._template.shape)
        self._is_template_batch, template_dims = self._compute_batch_dimension(
            batch_dims=template_dims, ndim=template_ndim
        )

        batch_dims = len(target_dims) + len(template_dims)
        target_measurement_dims = target_ndim - len(target_dims)

        collapse_dims = max(
            template_ndim - len(template_dims) - target_measurement_dims, 0
        )

        matching_dims = target_measurement_dims + batch_dims

        target_shape = np.full(shape=matching_dims, fill_value=1, dtype=int)
        template_shape = np.full(shape=matching_dims, fill_value=1, dtype=int)
        batch_mask = np.full(shape=matching_dims, fill_value=1, dtype=int)

        target_index, template_index = 0, 0
        for k in range(matching_dims):
            target_dim = k - target_index
            template_dim = k - template_index

            if target_dim in target_dims:
                target_shape[k] = self._target.shape[target_dim]
                if target_index == len(template_dims) and collapse_dims > 0:
                    template_shape[k] = self._template.shape[template_dim]
                    collapse_dims -= 1
                template_index += 1
                continue

            if template_dim in template_dims:
                template_shape[k] = self._template.shape[template_dim]
                target_index += 1
                continue

            batch_mask[k] = 0
            if target_dim < target_ndim:
                target_shape[k] = self._target.shape[target_dim]
            if template_dim < template_ndim:
                template_shape[k] = self._template.shape[template_dim]

        self._output_target_shape = tuple(int(x) for x in target_shape)
        self._output_template_shape = tuple(int(x) for x in template_shape)
        self._batch_mask = tuple(int(x) for x in batch_mask)

    @staticmethod
    def _compute_batch_dimension(
        batch_dims: Tuple[int], ndim: int
    ) -> Tuple[ArrayLike, Tuple]:
        """
        Computes a mask for the batch dimensions and the validated batch dimensions.

        Parameters
        ----------
        batch_dims : tuple of ints
            A tuple of integers representing the batch dimensions.
        ndim : int
            The number of dimensions of the array.

        Returns
        -------
        Tuple[ArrayLike, tuple of ints]
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

    def target_padding(self, pad_target: bool = False) -> Tuple[int]:
        """
        Computes padding for the target based on the template's shape.

        Parameters
        ----------
        pad_target : bool, default False
            Whether to pad the target, default returns an array of zeros.

        Returns
        -------
        tuple of ints
            Padding along each dimension of the target.
        """
        target_padding = np.zeros(len(self._output_target_shape), dtype=int)
        if pad_target:
            target_padding = np.subtract(
                self._output_template_shape,
                np.mod(self._output_template_shape, 2),
            )
            if hasattr(self, "_is_target_batch"):
                target_padding = np.multiply(
                    target_padding,
                    np.subtract(1, self._is_target_batch),
                )

        return tuple(int(x) for x in target_padding)

    @staticmethod
    def _fourier_padding(
        target_shape: NDArray,
        template_shape: NDArray,
        batch_mask: NDArray = None,
        pad_fourier: bool = False,
    ) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Determines an efficient shape for Fourier transforms considering zero-padding.
        """
        fourier_pad = template_shape
        fourier_shift = np.zeros_like(template_shape)

        if batch_mask is None:
            batch_mask = np.zeros_like(template_shape)
        batch_mask = np.asarray(batch_mask)

        if not pad_fourier:
            fourier_pad = np.ones(len(fourier_pad), dtype=int)
        fourier_pad = np.multiply(fourier_pad, 1 - batch_mask)
        fourier_pad = np.add(fourier_pad, batch_mask)

        pad_shape = np.maximum(target_shape, template_shape)
        ret = be.compute_convolution_shapes(pad_shape, fourier_pad)
        convolution_shape, fast_shape, fast_ft_shape = ret
        if not pad_fourier:
            fourier_shift = 1 - np.divide(template_shape, 2).astype(int)
            fourier_shift -= np.mod(template_shape, 2)
            shape_diff = np.subtract(fast_shape, convolution_shape)
            shape_diff = np.divide(shape_diff, 2).astype(int)
            shape_diff = np.multiply(shape_diff, 1 - batch_mask)
            np.add(fourier_shift, shape_diff, out=fourier_shift)

        fourier_shift = fourier_shift.astype(int)

        shape_diff = np.subtract(target_shape, template_shape)
        shape_diff = np.multiply(shape_diff, 1 - batch_mask)
        if np.sum(shape_diff < 0) and not pad_fourier:
            warnings.warn(
                "Template is larger than target and Fourier padding is turned off. "
                "This may lead to inaccurate results. Prefer swapping template and target, "
                "enable padding or turn off template centering."
            )
            fourier_shift = np.subtract(fourier_shift, np.divide(shape_diff, 2))
            fourier_shift = fourier_shift.astype(int)

        return tuple(fast_shape), tuple(fast_ft_shape), tuple(fourier_shift)

    def fourier_padding(self, pad_fourier: bool = False) -> Tuple[Tuple, Tuple, Tuple]:
        """
        Computes efficient shape four Fourier transforms and potential associated shifts.

        Parameters
        ----------
        pad_fourier : bool, default False
            If true, returns the shape of the full-convolution defined as sum of target
            shape and template shape minus one, False by default.

        Returns
        -------
        Tuple[tuple of int, tuple of int, tuple of int]
            Tuple with real and complex Fourier transform shape, and corresponding shift.
        """
        return self._fourier_padding(
            target_shape=be.to_numpy_array(self._output_target_shape),
            template_shape=be.to_numpy_array(self._output_template_shape),
            batch_mask=be.to_numpy_array(self._batch_mask),
            pad_fourier=pad_fourier,
        )

    @property
    def rotations(self):
        """Return stored rotation matrices."""
        return self._rotations

    @rotations.setter
    def rotations(self, rotations: NDArray):
        """
        Set :py:attr:`MatchingData.rotations`.

        Parameters
        ----------
        rotations : NDArray
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
    def _get_data(attribute, output_shape: Tuple[int], reverse: bool = False):
        if isinstance(attribute, Density):
            attribute = attribute.data

        if attribute is not None:
            if reverse:
                attribute = be.reverse(attribute)
            attribute = attribute.reshape(tuple(int(x) for x in output_shape))

        return attribute

    @property
    def target(self):
        """
        Return the target.

        Returns
        -------
        NDArray
            Output data.
        """
        return self._get_data(self._target, self._output_target_shape, False)

    @property
    def target_mask(self):
        """
        Return the target mask.

        Returns
        -------
        NDArray
            Output data.
        """
        target_mask = getattr(self, "_target_mask", None)
        return self._get_data(target_mask, self._output_target_shape, False)

    @property
    def template(self):
        """
        Return the reversed template.

        Returns
        -------
        NDArray
            Output data.
        """
        return self._get_data(self._template, self._output_template_shape, True)

    @property
    def template_mask(self):
        """
        Return the reversed template mask.

        Returns
        -------
        NDArray
            Output data.
        """
        template_mask = getattr(self, "_template_mask", None)
        return self._get_data(template_mask, self._output_template_shape, True)

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
            self._template_mask = be.full(
                shape=arr.shape, dtype=be._float_dtype, fill_value=1
            )

    @staticmethod
    def _set_mask(mask, shape: Tuple[int]):
        if mask is not None:
            if mask.shape != shape:
                raise ValueError(
                    "Mask and respective data have to have the same shape."
                )
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
        if isinstance(composable_filter, Compose):
            return composable_filter
        return None

    @property
    def template_filter(self) -> Optional[Compose]:
        """
        Returns the composable template filter.

        Returns
        -------
        :py:class:`tme.preprocessing.Compose` | None
            Composable template filter or None.
        """
        return getattr(self, "_template_filter", None)

    @property
    def target_filter(self) -> Optional[Compose]:
        """
        Returns the composable target filter.

        Returns
        -------
        :py:class:`tme.preprocessing.Compose` | None
            Composable filter or None.
        """
        return getattr(self, "_target_filter", None)

    @template_filter.setter
    def template_filter(self, composable_filter: Compose):
        self._template_filter = self._set_filter(composable_filter)

    @target_filter.setter
    def target_filter(self, composable_filter: Compose):
        self._target_filter = self._set_filter(composable_filter)

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
        Free (dereference) data arrays owned by the class instance.
        """
        attrs = ("_target", "_template", "_template_mask", "_target_mask")
        for attr in attrs:
            setattr(self, attr, None)
