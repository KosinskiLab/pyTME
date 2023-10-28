""" Data class for holding template matching data.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, List

import numpy as np
from numpy.typing import NDArray

from . import Density
from .backends import backend
from .matching_utils import compute_full_convolution_index


class MatchingData:
    """
    Contains data required for template matching.

    Parameters
    ----------
    target : np.ndarray or Density
        Target data array for template matching.
    template : np.ndarray or Density
        Template data array for template matching.

    """

    def __init__(self, target: NDArray, template: NDArray):
        self._default_dtype = np.float32
        self._complex_dtype = np.complex64

        self._target = target
        self._target_mask = None
        self._template_mask = None
        self._translation_offset = np.zeros(len(target.shape), dtype=int)

        self.template = template

        self._target_pad = np.zeros(len(target.shape), dtype=int)
        self._template_pad = np.zeros(len(template.shape), dtype=int)

        self.template_filter = {}
        self.target_filter = {}

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
        self, arr: NDArray, arr_slice: Tuple[slice], padding: NDArray
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
            values, otherwise, the
            values in ``arr`` are used.

        Returns
        -------
        NDArray
            Subset of the input array with padding applied.
        """
        padding = np.maximum(padding, 0)

        slice_start = np.array([x.start for x in arr_slice], dtype=int)
        slice_stop = np.array([x.stop for x in arr_slice], dtype=int)
        slice_shape = np.subtract(slice_stop, slice_start)

        padding = np.add(padding, np.mod(padding, 2))
        left_pad = right_pad = np.divide(padding, 2).astype(int)

        data_voxels_left = np.minimum(slice_start, left_pad)
        data_voxels_right = np.minimum(
            np.subtract(arr.shape, slice_stop), right_pad
        ).astype(int)

        ret_shape = np.add(slice_shape, padding)

        arr_start = np.subtract(slice_start, data_voxels_left)
        arr_stop = np.add(slice_stop, data_voxels_right)
        arr_slice = tuple(slice(*pos) for pos in zip(arr_start, arr_stop))
        arr_mesh = self._slice_to_mesh(arr_slice, arr.shape)

        subset_start = np.subtract(left_pad, data_voxels_left)
        subset_stop = np.add(subset_start, np.subtract(arr_stop, arr_start))
        subset_slice = tuple(slice(*prod) for prod in zip(subset_start, subset_stop))
        subset_mesh = self._slice_to_mesh(subset_slice, ret_shape)

        if type(arr) == Density:
            if type(arr.data) == np.memmap:
                arr = Density.from_file(arr.data.filename, subset=arr_slice).data
            else:
                arr = np.asarray(arr.data[*arr_mesh])
        else:
            if type(arr) == np.memmap:
                arr = np.memmap(
                    arr.filename, mode="r", shape=arr.shape, dtype=arr.dtype
                )
            arr = np.asarray(arr[*arr_mesh])

        ret = np.full(
            shape=np.add(slice_shape, padding), fill_value=arr.mean(), dtype=arr.dtype
        )
        ret[*subset_mesh] = arr

        return ret

    def subset_by_slice(
        self,
        target_slice: Tuple[slice] = None,
        template_slice: Tuple[slice] = None,
        target_pad: NDArray = None,
        template_pad: NDArray = None,
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
        target_shape = self.target.shape
        template_shape = self._template.shape

        if target_slice is None:
            target_slice = self._shape_to_slice(target_shape)
        if template_slice is None:
            template_slice = self._shape_to_slice(template_shape)

        if target_pad is None:
            target_pad = np.zeros(len(self.target.shape), dtype=int)
        if template_pad is None:
            template_pad = np.zeros(len(self.target.shape), dtype=int)

        indices = compute_full_convolution_index(
            outer_shape=self.target.shape,
            inner_shape=self.template.shape,
            outer_split=target_slice,
            inner_split=template_slice,
        )

        target_subset = self.subset_array(
            arr=self._target, arr_slice=target_slice, padding=target_pad
        )
        template_subset = self.subset_array(
            arr=self._template,
            arr_slice=template_slice,
            padding=template_pad,
        )
        ret = self.__class__(target=target_subset, template=template_subset)

        ret._translation_offset = np.add(
            [x.start for x in target_slice],
            [x.start for x in template_slice],
        )
        ret.template_filter = self.template_filter

        ret.rotations, ret.indices = self.rotations, indices
        ret._target_pad, ret._template_pad = target_pad, template_pad

        if self._target_mask is not None:
            ret.target_mask = self.subset_array(
                arr=self._target_mask, arr_slice=target_slice, padding=target_pad
            )
        if self._template_mask is not None:
            ret.template_mask = self.subset_array(
                arr=self._template_mask,
                arr_slice=template_slice,
                padding=template_pad,
            )

        return ret

    def to_backend(self) -> None:
        """
        Transfer the class instance's numpy arrays to the current backend.
        """
        for attr_name, attr_value in vars(self).items():
            if isinstance(attr_value, np.ndarray):
                converted_array = backend.to_backend_array(attr_value.copy())
                setattr(self, attr_name, converted_array)

        self._default_dtype = backend._default_dtype
        self._complex_dtype = backend._complex_dtype

    @property
    def rotations(self):
        """Return rotation matrices used for fitting."""
        return self._rotations

    @rotations.setter
    def rotations(self, rotations: NDArray):
        """
        Set and reshape the rotation matrices for fitting.

        Parameters
        ----------
        rotations : NDArray
            Rotations in shape (3 x 3), (1 x 3 x 3), or (n x k x k).
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
        self._rotations = rotations.astype(self._default_dtype)

    @property
    def target(self):
        """Returns the target NDArray."""
        if type(self._target) == Density:
            return self._target.data
        return self._target

    @property
    def template(self):
        """Returns the reversed template NDArray."""
        if type(self._template) == Density:
            return backend.reverse(self._template.data)
        return backend.reverse(self._template)

    @template.setter
    def template(self, template: NDArray):
        """
        Set the template array.

        Parameters
        ----------
        template : NDArray
            Array to set as the template.
        """
        if type(template) == Density:
            template.data = template.data.astype(self._default_dtype, copy=False)
            self._template = template
            self._templateshape = self._template.shape[::-1]
            return None
        self._template = template.astype(self._default_dtype, copy=False)
        self._templateshape = self._template.shape[::-1]

    @property
    def target_mask(self):
        """Returns the target mask NDArray."""
        if type(self._target_mask) == Density:
            return self._target_mask.data
        return self._target_mask

    @target_mask.setter
    def target_mask(self, mask: NDArray):
        """Sets the target mask."""
        if not np.all(self.target.shape == mask.shape):
            raise ValueError("Target and its mask have to have the same shape.")

        if type(mask) == Density:
            mask.data = mask.data.astype(self._default_dtype, copy=False)
            self._target_mask = mask
            self._targetmaskshape = self._target_mask.shape[::-1]
            return None
        self._target_mask = mask.astype(self._default_dtype, copy=False)
        self._targetmaskshape = self._target_mask.shape

    @property
    def template_mask(self):
        """
        Set the template mask array after reversing it.

        Parameters
        ----------
        template : NDArray
            Array to set as the template.
        """
        if type(self._template_mask) == Density:
            return backend.reverse(self._template_mask.data)
        return backend.reverse(self._template_mask)

    @template_mask.setter
    def template_mask(self, mask: NDArray):
        """Returns the reversed template mask NDArray."""
        if not np.all(self._template.shape == mask.shape):
            raise ValueError("Target and its mask have to have the same shape.")

        if type(mask) == Density:
            mask.data = mask.data.astype(self._default_dtype, copy=False)
            self._template_mask = mask
            self._templatemaskshape = self._template_mask.shape[::-1]
            return None

        self._template_mask = mask.astype(self._default_dtype, copy=False)
        self._templatemaskshape = self._template_mask.shape[::-1]

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
