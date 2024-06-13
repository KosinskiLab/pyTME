""" Data class for holding template matching data.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import warnings
from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray

from . import Density
from .types import ArrayLike
from .backends import backend
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
        Whether to invert and rescale the target before template matching..
    rotations: np.ndarray, optional
        Template rotations to sample. Can be a single (d x d) or a stack (n x d x d)
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
        self._target = target
        self._target_mask = target_mask
        self._template_mask = template_mask
        self._translation_offset = np.zeros(len(target.shape), dtype=int)

        self.template = template

        self._target_pad = np.zeros(len(target.shape), dtype=int)
        self._template_pad = np.zeros(len(template.shape), dtype=int)

        self.template_filter = {}
        self.target_filter = {}

        self._invert_target = invert_target

        self._rotations = rotations
        if rotations is not None:
            self.rotations = rotations

        self._set_batch_dimension()

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

        if type(arr) == np.memmap:
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
        padding = backend.to_numpy_array(padding)
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

        arr_min, arr_max = None, None
        if type(arr) == Density:
            if type(arr.data) == np.memmap:
                dens = Density.from_file(arr.data.filename, subset=arr_slice)
                arr = dens.data
                arr_min = dens.metadata.get("min", None)
                arr_max = dens.metadata.get("max", None)
            else:
                arr = np.asarray(arr.data[*arr_mesh])
        else:
            if type(arr) == np.memmap:
                arr = np.memmap(
                    arr.filename, mode="r", shape=arr.shape, dtype=arr.dtype
                )
            arr = np.asarray(arr[*arr_mesh])

        def _warn_on_mismatch(
            expectation: float, computation: float, name: str
        ) -> float:
            if expectation is None:
                expectation = computation
            expectation, computation = float(expectation), float(computation)

            if abs(computation) > abs(expectation):
                warnings.warn(
                    f"Computed {name} value is more extreme than value in file header"
                    f" (|{computation}| > |{expectation}|). This may lead to issues"
                    " with padding and contrast inversion."
                )

            return expectation

        padding = tuple(
            (left, right)
            for left, right in zip(
                np.subtract(left_pad, data_voxels_left),
                np.subtract(right_pad, data_voxels_right),
            )
        )
        ret = np.pad(arr, padding, mode="reflect")

        if invert:
            arr_min = _warn_on_mismatch(arr_min, arr.min(), "min")
            arr_max = _warn_on_mismatch(arr_max, arr.max(), "max")

            # Avoid in-place operation in case ret is not floating point
            ret = (
                -np.divide(np.subtract(ret, arr_min), np.subtract(arr_max, arr_min)) + 1
            )
        return ret

    def subset_by_slice(
        self,
        target_slice: Tuple[slice] = None,
        template_slice: Tuple[slice] = None,
        target_pad: NDArray = None,
        template_pad: NDArray = None,
        invert_target: bool = False,
    ) -> "MatchingData":
        """
        Slice the instance arrays based on the provided slices.

        Parameters
        ----------
        target_slice : tuple of slice, optional
            Slices for the target. If not provided, the full shape is used.
        template_slice : tuple of slice, optional
            Slices for the template. If not provided, the full shape is used.
        target_pad : NDArray, optional
            Padding for target. Defaults to zeros. If padding exceeds target,
            pad with mean.
        template_pad : NDArray, optional
            Padding for template. Defaults to zeros. If padding exceeds template,
            pad with mean.

        Returns
        -------
        MatchingData
            Newly allocated sliced class instance.
        """
        target_shape = self._target.shape
        template_shape = self._template.shape

        if target_slice is None:
            target_slice = self._shape_to_slice(target_shape)
        if template_slice is None:
            template_slice = self._shape_to_slice(template_shape)

        if target_pad is None:
            target_pad = np.zeros(len(self._target.shape), dtype=int)
        if template_pad is None:
            template_pad = np.zeros(len(self._template.shape), dtype=int)

        indices = None
        if len(self._target.shape) == len(self._template.shape):
            indices = compute_full_convolution_index(
                outer_shape=self._target.shape,
                inner_shape=self._template.shape,
                outer_split=target_slice,
                inner_split=template_slice,
            )

        target_subset = self.subset_array(
            arr=self._target,
            arr_slice=target_slice,
            padding=target_pad,
            invert=self._invert_target,
        )

        template_subset = self.subset_array(
            arr=self._template,
            arr_slice=template_slice,
            padding=template_pad,
        )
        ret = self.__class__(target=target_subset, template=template_subset)

        target_offset = np.zeros(len(self._output_target_shape), dtype=int)
        target_offset[(target_offset.size - len(target_slice)) :] = [
            x.start for x in target_slice
        ]
        template_offset = np.zeros(len(self._output_target_shape), dtype=int)
        template_offset[(template_offset.size - len(template_slice)) :] = [
            x.start for x in template_slice
        ]
        ret._translation_offset = target_offset

        ret.template_filter = self.template_filter
        ret.target_filter = self.target_filter
        ret._rotations, ret.indices = self.rotations, indices
        ret._target_pad, ret._template_pad = target_pad, template_pad
        ret._invert_target = self._invert_target

        if self._target_mask is not None:
            ret._target_mask = self.subset_array(
                arr=self._target_mask, arr_slice=target_slice, padding=target_pad
            )
        if self._template_mask is not None:
            ret.template_mask = self.subset_array(
                arr=self._template_mask,
                arr_slice=template_slice,
                padding=template_pad,
            )

        target_dims, template_dims = None, None
        if hasattr(self, "_target_dims"):
            target_dims = self._target_dims

        if hasattr(self, "_template_dims"):
            template_dims = self._template_dims

        ret._set_batch_dimension(target_dims=target_dims, template_dims=template_dims)

        return ret

    def to_backend(self) -> None:
        """
        Transfer and convert types of class instance's data arrays to the current backend
        """
        backend_arr = type(backend.zeros((1), dtype=backend._float_dtype))
        for attr_name, attr_value in vars(self).items():
            converted_array = None
            if isinstance(attr_value, np.ndarray):
                converted_array = backend.to_backend_array(attr_value.copy())
            elif isinstance(attr_value, backend_arr):
                converted_array = backend.to_backend_array(attr_value)
            else:
                continue

            current_dtype = backend.get_fundamental_dtype(converted_array)
            target_dtype = backend._fundamental_dtypes[current_dtype]

            # Optional, but scores are float so we avoid casting and potential issues
            if attr_name in ("_template", "_template_mask", "_target", "_target_mask"):
                target_dtype = backend._float_dtype

            if target_dtype != current_dtype:
                converted_array = backend.astype(converted_array, target_dtype)

            setattr(self, attr_name, converted_array)

    def _set_batch_dimension(
        self, target_dims: Tuple[int] = None, template_dims: Tuple[int] = None
    ) -> None:
        """
        Sets the shapes of target and template for template matching considering
        their corresponding batch dimensions.

        Parameters
        ----------
        target_dims : Tuple[int], optional
            A tuple of integers specifying the batch dimensions of the target. If None,
            the target is assumed not to have batch dimensions.
        template_dims : Tuple[int], optional
            A tuple of integers specifying the batch dimensions of the template. If None,
            the template is assumed not to have batch dimensions.

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

        target_shape = backend.full(
            shape=(matching_dims,), fill_value=1, dtype=backend._int_dtype
        )
        template_shape = backend.full(
            shape=(matching_dims,), fill_value=1, dtype=backend._int_dtype
        )
        batch_mask = backend.full(
            shape=(matching_dims,), fill_value=1, dtype=backend._int_dtype
        )

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

        self._output_target_shape = target_shape
        self._output_template_shape = template_shape
        self._batch_mask = batch_mask

    @staticmethod
    def _compute_batch_dimension(
        batch_dims: Tuple[int], ndim: int
    ) -> Tuple[ArrayLike, Tuple]:
        """
        Computes a mask for the batch dimensions and the validated batch dimensions.

        Parameters
        ----------
        batch_dims : Tuple[int]
            A tuple of integers representing the batch dimensions.
        ndim : int
            The number of dimensions of the array.

        Returns
        -------
        Tuple[ArrayLike, Tuple]
            A tuple containing the mask (as an ArrayLike) and the validated batch dimensions.

        Raises
        ------
        ValueError
            If any dimension in batch_dims is not less than ndim.
        """
        mask = backend.zeros(ndim, dtype=bool)
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

    def target_padding(self, pad_target: bool = False) -> ArrayLike:
        """
        Computes padding for the target based on the template's shape.

        Parameters
        ----------
        pad_target : bool, default False
            If True, computes the padding required for the target. If False,
            an array of zeros is returned.

        Returns
        -------
        ArrayLike
            An array indicating the padding for each dimension of the target.
        """
        target_padding = backend.zeros(
            len(self._output_target_shape), dtype=backend._int_dtype
        )

        if pad_target:
            backend.subtract(
                self._output_template_shape,
                backend.mod(self._output_template_shape, 2),
                out=target_padding,
            )
            if hasattr(self, "_is_target_batch"):
                target_padding[self._is_target_batch] = 0

        return target_padding

    def fourier_padding(
        self, pad_fourier: bool = False
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """
        Computes an efficient shape for the forward Fourier transform, the
        corresponding shape of the real-valued FFT, and the associated
        translation shift.

        Parameters
        ----------
        pad_fourier : bool, default False
            If true, returns the shape of the full-convolution defined as sum of target
            shape and template shape minus one. By default, returns unpadded transform.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike, ArrayLike]
            A tuple containing the calculated fast shape, fast Fourier transform shape,
            and the Fourier shift values, respectively.
        """
        template_shape = self._template.shape
        if hasattr(self, "_output_template_shape"):
            template_shape = self._output_template_shape
        template_shape = backend.to_backend_array(template_shape)

        target_shape = self._target.shape
        if hasattr(self, "_output_target_shape"):
            target_shape = self._output_target_shape
        target_shape = backend.to_backend_array(target_shape)

        fourier_pad = backend.to_backend_array(template_shape)
        fourier_shift = backend.zeros(len(fourier_pad))

        if not pad_fourier:
            fourier_pad = backend.full(
                shape=(len(fourier_pad),),
                fill_value=1,
                dtype=backend._int_dtype,
            )

        fourier_pad = backend.to_backend_array(fourier_pad)
        if hasattr(self, "_batch_mask"):
            batch_mask = backend.to_backend_array(self._batch_mask)
            backend.multiply(fourier_pad, 1 - batch_mask, out=fourier_pad)
            backend.add(fourier_pad, batch_mask, out=fourier_pad)

        pad_shape = backend.maximum(target_shape, template_shape)
        ret = backend.compute_convolution_shapes(pad_shape, fourier_pad)
        convolution_shape, fast_shape, fast_ft_shape = ret
        if not pad_fourier:
            fourier_shift = 1 - backend.astype(backend.divide(template_shape, 2), int)
            fourier_shift -= backend.mod(template_shape, 2)
            shape_diff = backend.subtract(fast_shape, convolution_shape)
            shape_diff = backend.astype(backend.divide(shape_diff, 2), int)

            if hasattr(self, "_batch_mask"):
                batch_mask = backend.to_backend_array(self._batch_mask)
                backend.multiply(shape_diff, 1 - batch_mask, out=shape_diff)

            backend.add(fourier_shift, shape_diff, out=fourier_shift)

        fourier_shift = backend.astype(fourier_shift, backend._int_dtype)

        return fast_shape, fast_ft_shape, fourier_shift

    @property
    def rotations(self):
        """Return stored rotation matrices.."""
        return self._rotations

    @rotations.setter
    def rotations(self, rotations: NDArray):
        """
        Set and reshape the rotation matrices for template matching.

        Parameters
        ----------
        rotations : NDArray
            Rotations in shape (k x k), or (n x k x k).
        """
        if rotations.__class__ != np.ndarray:
            raise ValueError("Rotation set has to be of type numpy ndarray.")
        if rotations.ndim == 2:
            print("Reshaping rotations array to rank 3.")
            rotations = rotations.reshape(1, *rotations.shape)
        elif rotations.ndim == 3:
            pass
        else:
            raise ValueError("Rotations have to be a rank 2 or 3 array.")
        self._rotations = rotations.astype(np.float32)

    @property
    def target(self):
        """Returns the target."""
        target = self._target
        if isinstance(self._target, Density):
            target = self._target.data

        out_shape = tuple(int(x) for x in self._output_target_shape)
        return target.reshape(out_shape)

    @property
    def template(self):
        """Returns the reversed template."""
        template = self._template
        if isinstance(self._template, Density):
            template = self._template.data
        template = backend.reverse(template)
        out_shape = tuple(int(x) for x in self._output_template_shape)
        return template.reshape(out_shape)

    @template.setter
    def template(self, template: NDArray):
        """
        Set the template array. If not already defined, also initializes
        :py:attr:`MatchingData.template_mask` to an uninformative mask filled with
        ones.

        Parameters
        ----------
        template : NDArray
            Array to set as the template.
        """
        self._templateshape = template.shape[::-1]
        if self._template_mask is None:
            self._template_mask = backend.full(
                shape=template.shape, dtype=float, fill_value=1
            )

        self._template = template

    @property
    def target_mask(self):
        """Returns the target mask NDArray."""
        target_mask = self._target_mask
        if isinstance(self._target_mask, Density):
            target_mask = self._target_mask.data

        if target_mask is not None:
            out_shape = tuple(int(x) for x in self._output_target_shape)
            target_mask = target_mask.reshape(out_shape)

        return target_mask

    @target_mask.setter
    def target_mask(self, mask: NDArray):
        """Sets the target mask."""
        if not np.all(self.target.shape == mask.shape):
            raise ValueError("Target and its mask have to have the same shape.")

        self._target_mask = mask

    @property
    def template_mask(self):
        """
        Set the template mask array after reversing it.

        Parameters
        ----------
        template : NDArray
            Array to set as the template.
        """
        mask = self._template_mask
        if isinstance(self._template_mask, Density):
            mask = self._template_mask.data

        if mask is not None:
            mask = backend.reverse(mask)
            out_shape = tuple(int(x) for x in self._output_template_shape)
            mask = mask.reshape(out_shape)
        return mask

    @template_mask.setter
    def template_mask(self, mask: NDArray):
        """Returns the reversed template mask NDArray."""
        if not np.all(self._templateshape[::-1] == mask.shape):
            raise ValueError("Template and its mask have to have the same shape.")

        self._template_mask = mask

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
