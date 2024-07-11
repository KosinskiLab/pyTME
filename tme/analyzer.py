""" Implements classes to analyze outputs from exhaustive template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from contextlib import nullcontext
from abc import ABC, abstractmethod
from multiprocessing import Manager, Lock
from typing import Tuple, List, Dict, Generator

import numpy as np
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
from skimage.registration._phase_cross_correlation import _upsampled_dft

from .backends import backend as be
from .types import BackendArray, NDArray
from .extensions import max_index_by_label, find_candidate_indices
from .matching_utils import (
    split_shape,
    array_to_memmap,
    generate_tempfile_name,
    euler_to_rotationmatrix,
    apply_convolution_mode,
)

PeakType = Tuple[BackendArray, BackendArray]


def _filter_bucket(coordinates: BackendArray, min_distance: Tuple[int]) -> BackendArray:
    coordinates = be.subtract(coordinates, be.min(coordinates, axis=0))
    bucket_indices = be.astype(be.divide(coordinates, min_distance), int)
    multiplier = be.power(
        be.max(bucket_indices, axis=0) + 1, be.arange(bucket_indices.shape[1])
    )
    be.multiply(bucket_indices, multiplier, out=bucket_indices)
    flattened_indices = be.sum(bucket_indices, axis=1)
    _, unique_indices = be.unique(flattened_indices, return_index=True)
    unique_indices = unique_indices[be.argsort(unique_indices)]
    return unique_indices


def filter_points_indices(
    coordinates: BackendArray,
    min_distance: float,
    bucket_cutoff: int = 1e4,
    batch_dims: Tuple[int] = None,
) -> BackendArray:
    if min_distance <= 0:
        return be.arange(coordinates.shape[0])
    if coordinates.shape[0] == 0:
        return ()

    if batch_dims is not None:
        coordinates_new = be.zeros(coordinates.shape, coordinates.dtype)
        coordinates_new[:] = coordinates
        coordinates_new[..., batch_dims] = be.astype(
            coordinates[..., batch_dims] * (2 * min_distance), coordinates_new.dtype
        )
        coordinates = coordinates_new

    if isinstance(coordinates, np.ndarray):
        return find_candidate_indices(coordinates, min_distance)
    elif coordinates.shape[0] > bucket_cutoff or not isinstance(
        coordinates, np.ndarray
    ):
        return _filter_bucket(coordinates, min_distance)
    distances = np.linalg.norm(coordinates[:, None] - coordinates, axis=-1)
    distances = np.tril(distances)
    keep = np.sum(distances > min_distance, axis=1)
    indices = np.arange(coordinates.shape[0])
    return indices[keep == indices]


def filter_points(
    coordinates: NDArray, min_distance: Tuple[int], batch_dims: Tuple[int] = None
) -> BackendArray:
    unique_indices = filter_points_indices(coordinates, min_distance, batch_dims)
    coordinates = coordinates[unique_indices]
    return coordinates


class PeakCaller(ABC):
    """
    Base class for peak calling algorithms.

    Parameters
    ----------
    number_of_peaks : int, optional
        Number of candidate peaks to consider.
    min_distance : int, optional
        Minimum distance between peaks.
    min_boundary_distance : int, optional
        Minimum distance to array boundaries.
    batch_dims : int, optional
        Peak calling batch dimensions.
    minimum_score : float
        Minimum score from which to consider peaks. If provided, superseeds limits
        presented by :py:attr:`PeakCaller.number_of_peaks`.
    maximum_score : float
        Maximum score upon which to consider peaks,
    **kwargs
        Optional keyword arguments.

    Raises
    ------
    ValueError
        If number_of_peaks is less than or equal to zero.
        If min_distances is less than zero.
    """

    def __init__(
        self,
        number_of_peaks: int = 1000,
        min_distance: int = 1,
        min_boundary_distance: int = 0,
        batch_dims: Tuple[int] = None,
        minimum_score: float = None,
        maximum_score: float = None,
        **kwargs,
    ):
        if number_of_peaks <= 0:
            raise ValueError(
                f"number_of_peaks has to be larger than 0, got {number_of_peaks}"
            )
        if min_distance < 0:
            raise ValueError(f"min_distance has to be non-negative, got {min_distance}")
        if min_boundary_distance < 0:
            raise ValueError(
                f"min_boundary_distance has to be non-negative, got {min_boundary_distance}"
            )

        self.peak_list = []
        self.min_distance = int(min_distance)
        self.number_of_peaks = int(number_of_peaks)
        self.min_boundary_distance = int(min_boundary_distance)

        self.batch_dims = batch_dims
        if batch_dims is not None:
            self.batch_dims = tuple(int(x) for x in self.batch_dims)

        self.minimum_score, self.maximum_score = minimum_score, maximum_score

        # Postprocesing arguments
        self.fourier_shift = kwargs.get("fourier_shift", None)
        self.convolution_mode = kwargs.get("convolution_mode", None)
        self.targetshape = kwargs.get("targetshape", None)
        self.templateshape = kwargs.get("templateshape", None)

    def __iter__(self) -> Generator:
        """
        Returns a generator to list objects containing translation,
        rotation, score and details of a given candidate.
        """
        self.peak_list = [be.to_cpu_array(arr) for arr in self.peak_list]
        yield from self.peak_list

    @staticmethod
    def _batchify(shape: Tuple[int], batch_dims: Tuple[int] = None) -> List:
        if batch_dims is None:
            yield (tuple(slice(None) for _ in shape), tuple(0 for _ in shape))
            return None

        batch_ranges = [range(shape[dim]) for dim in batch_dims]

        def _generate_slices_recursive(current_dim, current_indices):
            if current_dim == len(batch_dims):
                slice_list, offset_list, batch_index = [], [], 0
                for i in range(len(shape)):
                    if i in batch_dims:
                        index = current_indices[batch_index]
                        slice_list.append(slice(index, index + 1))
                        offset_list.append(index)
                        batch_index += 1
                    else:
                        slice_list.append(slice(None))
                        offset_list.append(0)
                yield (tuple(slice_list), tuple(offset_list))
            else:
                for index in batch_ranges[current_dim]:
                    yield from _generate_slices_recursive(
                        current_dim + 1, current_indices + (index,)
                    )

        yield from _generate_slices_recursive(0, ())

    def __call__(self, scores: BackendArray, rotation_matrix: BackendArray, **kwargs):
        """
        Update the internal parameter store based on input array.

        Parameters
        ----------
        scores : BackendArray
            Score space data.
        rotation_matrix : BackendArray
            Rotation matrix used to obtain the score array.
        **kwargs
            Optional keyword aguments passed to :py:meth:`PeakCaller.call_peaks`.
        """
        minimum_score, maximum_score = self.minimum_score, self.maximum_score
        for subset, batch_offset in self._batchify(scores.shape, self.batch_dims):
            batch_offset = be.to_backend_array(batch_offset)
            peak_positions, peak_details = self.call_peaks(
                scores=scores[subset],
                rotation_matrix=rotation_matrix,
                minimum_score=minimum_score,
                maximum_score=maximum_score,
                **kwargs,
            )

            if peak_positions is None:
                continue
            if peak_positions.shape[0] == 0:
                continue

            if peak_details is None:
                peak_details = be.full((peak_positions.shape[0],), fill_value=-1)

            peak_positions = be.to_backend_array(peak_positions)
            peak_positions = be.add(peak_positions, batch_offset, out=peak_positions)
            peak_positions = be.astype(peak_positions, int)
            if self.min_boundary_distance > 0:
                upper_limit = be.subtract(
                    be.to_backend_array(scores.shape), self.min_boundary_distance
                )
                valid_peaks = be.multiply(
                    peak_positions < upper_limit,
                    peak_positions >= self.min_boundary_distance,
                )
                if self.batch_dims is not None:
                    valid_peaks[..., self.batch_dims] = True

                valid_peaks = be.sum(valid_peaks, axis=1) == peak_positions.shape[1]

                if be.sum(valid_peaks) == 0:
                    continue
                peak_positions = peak_positions[valid_peaks]
                peak_details = peak_details[valid_peaks]

            peak_scores = scores[tuple(peak_positions.T)]
            if minimum_score is not None:
                valid_peaks = peak_scores >= minimum_score
                peak_positions, peak_details, peak_scores = (
                    peak_positions[valid_peaks],
                    peak_details[valid_peaks],
                    peak_scores[valid_peaks],
                )
            if maximum_score is not None:
                valid_peaks = peak_scores <= maximum_score
                peak_positions, peak_details, peak_scores = (
                    peak_positions[valid_peaks],
                    peak_details[valid_peaks],
                    peak_scores[valid_peaks],
                )

            if peak_positions.shape[0] == 0:
                continue

            rotations = be.repeat(
                rotation_matrix.reshape(1, *rotation_matrix.shape),
                peak_positions.shape[0],
                axis=0,
            )

            self._update(
                peak_positions=peak_positions,
                peak_details=peak_details,
                peak_scores=peak_scores,
                rotations=rotations,
            )

        return None

    @abstractmethod
    def call_peaks(self, scores: BackendArray, **kwargs) -> PeakType:
        """
        Call peaks in the score space.

        Parameters
        ----------
        scores : BackendArray
            Score array.
        **kwargs : dict
            Optional keyword arguments passed to underlying implementations.

        Returns
        -------
        Tuple[BackendArray, BackendArray]
            Array of peak coordinates and peak details.
        """

    @classmethod
    def merge(cls, candidates=List[List], **kwargs) -> Tuple:
        """
        Merge multiple instances of :py:class:`PeakCaller`.

        Parameters
        ----------
        candidates : list of lists
            Obtained by invoking list on the generator returned by __iter__.
        **kwargs
            Optional keyword arguments.

        Returns
        -------
        Tuple
            Tuple of translation, rotation, score and details of candidates.
        """
        base = cls(**kwargs)
        for candidate in candidates:
            if len(candidate) == 0:
                continue
            peak_positions, rotations, peak_scores, peak_details = candidate
            base._update(
                peak_positions=be.to_backend_array(peak_positions),
                peak_details=be.to_backend_array(peak_details),
                peak_scores=be.to_backend_array(peak_scores),
                rotations=be.to_backend_array(rotations),
                offset=kwargs.get("offset", None),
            )
        return tuple(base)

    @staticmethod
    def oversample_peaks(
        scores: BackendArray, peak_positions: BackendArray, oversampling_factor: int = 8
    ):
        """
        Refines peaks positions in the corresponding score space.

        Parameters
        ----------
        scores : BackendArray
            The d-dimensional array representing the score space.
        peak_positions : BackendArray
            An array of shape (n, d) containing the peak coordinates
            to be refined, where n is the number of peaks and d is the
            dimensionality of the score space.
        oversampling_factor : int, optional
            The oversampling factor for Fourier transforms. Defaults to 8.

        Returns
        -------
        BackendArray
            An array of shape (n, d) containing the refined subpixel
            coordinates of the peaks.

        Notes
        -----
        Floating point peak positions are determined by oversampling the
        scores around peak_positions. The accuracy
        of refinement scales with 1 / oversampling_factor.

        References
        ----------
        .. [1]  https://scikit-image.org/docs/stable/api/skimage.registration.html
        .. [2]  Manuel Guizar-Sicairos, Samuel T. Thurman, and
                James R. Fienup, “Efficient subpixel image registration
                algorithms,” Optics Letters 33, 156-158 (2008).
                DOI:10.1364/OL.33.000156

        """
        scores = be.to_numpy_array(scores)
        peak_positions = be.to_numpy_array(peak_positions)

        peak_positions = np.round(
            np.divide(
                np.multiply(peak_positions, oversampling_factor), oversampling_factor
            )
        )
        upsampled_region_size = np.ceil(np.multiply(oversampling_factor, 1.5))
        dftshift = np.round(np.divide(upsampled_region_size, 2.0))
        sample_region_offset = np.subtract(
            dftshift, np.multiply(peak_positions, oversampling_factor)
        )

        scores_ft = np.fft.fftn(scores).conj()
        for index in range(sample_region_offset.shape[0]):
            cross_correlation_upsampled = _upsampled_dft(
                data=scores_ft,
                upsampled_region_size=upsampled_region_size,
                upsample_factor=oversampling_factor,
                axis_offsets=sample_region_offset[index],
            ).conj()

            maxima = np.unravel_index(
                np.argmax(np.abs(cross_correlation_upsampled)),
                cross_correlation_upsampled.shape,
            )
            maxima = np.divide(np.subtract(maxima, dftshift), oversampling_factor)
            peak_positions[index] = np.add(peak_positions[index], maxima)

        peak_positions = be.to_backend_array(peak_positions)

        return peak_positions

    def _update(
        self,
        peak_positions: BackendArray,
        peak_details: BackendArray,
        peak_scores: BackendArray,
        rotations: BackendArray,
        offset: BackendArray = None,
    ):
        """
        Update internal parameter store.

        Parameters
        ----------
        peak_positions : BackendArray
            Position of peaks (n, d).
        peak_details : BackendArray
            Details of each peak (n, ).
        rotations: BackendArray
            Rotation at each peak (n, d, d).
        rotations: BackendArray
            Rotation at each peak (n, d, d).
        offset : BackendArray, optional
            Translation offset, e.g. from splitting, (n, ).
        """
        if offset is not None:
            offset = be.astype(offset, peak_positions.dtype)
            peak_positions = be.add(peak_positions, offset, out=peak_positions)

        if not len(self.peak_list):
            self.peak_list = [peak_positions, rotations, peak_scores, peak_details]
        else:
            peak_positions = be.concatenate((self.peak_list[0], peak_positions))
            rotations = be.concatenate((self.peak_list[1], rotations))
            peak_scores = be.concatenate((self.peak_list[2], peak_scores))
            peak_details = be.concatenate((self.peak_list[3], peak_details))

        if self.batch_dims is None:
            top_n = min(be.size(peak_scores), self.number_of_peaks)
            top_scores, *_ = be.topk_indices(peak_scores, top_n)
        else:
            # Not very performant but fairly robust
            batch_indices = peak_positions[..., self.batch_dims]
            batch_indices = be.subtract(batch_indices, be.min(batch_indices, axis=0))
            multiplier = be.power(
                be.max(batch_indices, axis=0) + 1,
                be.arange(batch_indices.shape[1]),
            )
            batch_indices = be.multiply(batch_indices, multiplier, out=batch_indices)
            batch_indices = be.sum(batch_indices, axis=1)
            unique_indices, batch_counts = be.unique(batch_indices, return_counts=True)
            total_indices = be.arange(peak_scores.shape[0])
            batch_indices = [total_indices[batch_indices == x] for x in unique_indices]
            top_scores = be.concatenate(
                [
                    total_indices[indices][
                        be.topk_indices(
                            peak_scores[indices], min(y, self.number_of_peaks)
                        )
                    ]
                    for indices, y in zip(batch_indices, batch_counts)
                ]
            )

        final_order = top_scores[
            filter_points_indices(
                coordinates=peak_positions[top_scores],
                min_distance=self.min_distance,
                batch_dims=self.batch_dims,
            )
        ]

        self.peak_list[0] = peak_positions[final_order,]
        self.peak_list[1] = rotations[final_order,]
        self.peak_list[2] = peak_scores[final_order]
        self.peak_list[3] = peak_details[final_order]

    def _postprocess(
        self,
        fast_shape: Tuple[int],
        targetshape: Tuple[int],
        templateshape: Tuple[int],
        fourier_shift: Tuple[int] = None,
        convolution_mode: str = None,
        shared_memory_handler=None,
        **kwargs,
    ):
        if not len(self.peak_list):
            return self

        peak_positions = self.peak_list[0]
        if not len(peak_positions):
            return self

        # Wrap peaks around score space
        fast_shape = be.to_backend_array(fast_shape)
        if fourier_shift is not None:
            fourier_shift = be.to_backend_array(fourier_shift)
            peak_positions = be.add(peak_positions, fourier_shift)
            peak_positions = be.subtract(
                peak_positions,
                be.multiply(
                    be.astype(be.divide(peak_positions, fast_shape), int),
                    fast_shape,
                ),
            )

        # Remove padding to fast Fourier (and potential full convolution) shape
        targetshape = be.to_backend_array(targetshape)
        templateshape = be.to_backend_array(templateshape)
        fast_shape = be.minimum(be.add(targetshape, templateshape) - 1, fast_shape)
        output_shape = fast_shape
        if convolution_mode == "same":
            output_shape = targetshape
        elif convolution_mode == "valid":
            output_shape = be.add(
                be.subtract(targetshape, templateshape),
                be.mod(templateshape, 2),
            )

        output_shape = be.to_backend_array(output_shape)
        starts = be.astype(
            be.divide(be.subtract(fast_shape, output_shape), 2),
            be._int_dtype,
        )
        stops = be.add(starts, output_shape)

        valid_peaks = be.multiply(peak_positions > starts, peak_positions <= stops)
        valid_peaks = be.sum(valid_peaks, axis=1) == peak_positions.shape[1]

        self.peak_list[0] = be.subtract(peak_positions, starts)
        self.peak_list = [x[valid_peaks] for x in self.peak_list]
        return self


class PeakCallerSort(PeakCaller):
    """
    A :py:class:`PeakCaller` subclass that first selects ``number_of_peaks``
    highest scores.
    """

    def call_peaks(self, scores: BackendArray, **kwargs) -> PeakType:
        flat_scores = scores.reshape(-1)
        k = min(self.number_of_peaks, be.size(flat_scores))

        top_k_indices, *_ = be.topk_indices(flat_scores, k)

        coordinates = be.unravel_index(top_k_indices, scores.shape)
        coordinates = be.transpose(be.stack(coordinates))

        return coordinates, None


class PeakCallerMaximumFilter(PeakCaller):
    """
    Find local maxima by applying a maximum filter and enforcing a distance
    constraint subsequently. This is similar to the strategy implemented in
    :obj:`skimage.feature.peak_local_max`.
    """

    def call_peaks(self, scores: BackendArray, **kwargs) -> PeakType:
        return be.max_filter_coordinates(scores, self.min_distance), None


class PeakCallerFast(PeakCaller):
    """
    Subdivides the score space into squares with edge length ``min_distance``
    and determiens maximum value for each. In a second pass, all local maxima
    that are not the local maxima in a ``min_distance`` square centered around them
    are removed.

    """

    def call_peaks(self, scores: BackendArray, **kwargs) -> PeakType:
        splits = {
            axis: scores.shape[axis] // self.min_distance for axis in range(scores.ndim)
        }
        slices = split_shape(scores.shape, splits)

        coordinates = be.to_backend_array(
            [
                be.unravel_index(be.argmax(scores[subvol]), scores[subvol].shape)
                for subvol in slices
            ]
        )
        offset = be.to_backend_array(
            [tuple(x.start for x in subvol) for subvol in slices]
        )
        be.add(coordinates, offset, out=coordinates)
        coordinates = coordinates[
            be.flip(be.argsort(scores[tuple(coordinates.T)]), (0,))
        ]

        if coordinates.shape[0] == 0:
            return None

        starts = be.maximum(coordinates - self.min_distance, 0)
        stops = be.minimum(coordinates + self.min_distance, scores.shape)
        slices_list = [
            tuple(slice(*coord) for coord in zip(start_row, stop_row))
            for start_row, stop_row in zip(starts, stops)
        ]

        keep = [
            score_subvol >= be.max(scores[subvol])
            for subvol, score_subvol in zip(slices_list, scores[tuple(coordinates.T)])
        ]
        coordinates = coordinates[keep,]

        if len(coordinates) == 0:
            return coordinates, None

        return coordinates, None


class PeakCallerRecursiveMasking(PeakCaller):
    """
    Identifies peaks iteratively by selecting the top score and masking
    a region around it.
    """

    def call_peaks(
        self,
        scores: BackendArray,
        rotation_matrix: BackendArray,
        mask: BackendArray = None,
        minimum_score: float = None,
        rotation_space: BackendArray = None,
        rotation_mapping: Dict = None,
        **kwargs,
    ) -> PeakType:
        """
        Call peaks in the score space.

        Parameters
        ----------
        scores : BackendArray
            Data array of scores.
        rotation_matrix : BackendArray
            Rotation matrix.
        mask : BackendArray, optional
            Mask array, by default None.
        rotation_space : BackendArray, optional
            Rotation space array, by default None.
        rotation_mapping : Dict optional
            Dictionary mapping values in rotation_space to Euler angles.
            By default None
        minimum_score : float
            Minimum score value to consider. If provided, superseeds limit given
            by :py:attr:`PeakCaller.number_of_peaks`.

        Returns
        -------
        Tuple[BackendArray, BackendArray]
            Array of peak coordinates and peak details.

        Notes
        -----
        By default, scores are masked using a box with edge length self.min_distance.
        If mask is provided, elements around each peak will be multiplied by the mask
        values. If rotation_space and rotation_mapping is provided, the respective
        rotation will be applied to the mask, otherwise rotation_matrix is used.
        """
        coordinates, masking_function = [], self._mask_scores_rotate

        if mask is None:
            masking_function = self._mask_scores_box
            shape = tuple(self.min_distance for _ in range(scores.ndim))
            mask = be.zeros(shape, dtype=be._float_dtype)

        rotated_template = be.zeros(mask.shape, dtype=mask.dtype)

        peak_limit = self.number_of_peaks
        if minimum_score is not None:
            peak_limit = be.size(scores)
        else:
            minimum_score = be.min(scores) - 1

        scores_copy = be.zeros(scores.shape, dtype=scores.dtype)
        scores_copy[:] = scores

        while True:
            be.argmax(scores_copy)
            peak = be.unravel_index(
                indices=be.argmax(scores_copy), shape=scores_copy.shape
            )
            if scores_copy[tuple(peak)] < minimum_score:
                break

            coordinates.append(peak)

            current_rotation_matrix = self._get_rotation_matrix(
                peak=peak,
                rotation_space=rotation_space,
                rotation_mapping=rotation_mapping,
                rotation_matrix=rotation_matrix,
            )

            masking_function(
                scores=scores_copy,
                rotation_matrix=current_rotation_matrix,
                peak=peak,
                mask=mask,
                rotated_template=rotated_template,
            )

            if len(coordinates) >= peak_limit:
                break

        peaks = be.to_backend_array(coordinates)
        return peaks, None

    @staticmethod
    def _get_rotation_matrix(
        peak: BackendArray,
        rotation_space: BackendArray,
        rotation_mapping: BackendArray,
        rotation_matrix: BackendArray,
    ) -> BackendArray:
        """
        Get rotation matrix based on peak and rotation data.

        Parameters
        ----------
        peak : BackendArray
            Peak coordinates.
        rotation_space : BackendArray
            Rotation space array.
        rotation_mapping : Dict
            Dictionary mapping values in rotation_space to Euler angles.
        rotation_matrix : BackendArray
            Current rotation matrix.

        Returns
        -------
        BackendArray
            Rotation matrix.
        """
        if rotation_space is None or rotation_mapping is None:
            return rotation_matrix

        rotation = rotation_mapping[rotation_space[tuple(peak)]]

        # TODO: Newer versions of rotation mapping contain rotation matrices not angles
        if len(rotation) == 3:
            rotation = be.to_backend_array(
                euler_to_rotationmatrix(be.to_numpy_array(rotation))
            )
        return rotation

    @staticmethod
    def _mask_scores_box(
        scores: BackendArray, peak: BackendArray, mask: BackendArray, **kwargs: Dict
    ) -> None:
        """
        Mask scores in a box around a peak.

        Parameters
        ----------
        scores : BackendArray
            Data array of scores.
        peak : BackendArray
            Peak coordinates.
        mask : BackendArray
            Mask array.
        """
        start = be.maximum(be.subtract(peak, mask.shape), 0)
        stop = be.minimum(be.add(peak, mask.shape), scores.shape)
        start, stop = be.astype(start, int), be.astype(stop, int)
        coords = tuple(slice(*pos) for pos in zip(start, stop))
        scores[coords] = 0
        return None

    @staticmethod
    def _mask_scores_rotate(
        scores: BackendArray,
        peak: BackendArray,
        mask: BackendArray,
        rotated_template: BackendArray,
        rotation_matrix: BackendArray,
        **kwargs: Dict,
    ) -> None:
        """
        Mask scores using mask rotation around a peak.

        Parameters
        ----------
        scores : BackendArray
            Data array of scores.
        peak : BackendArray
            Peak coordinates.
        mask : BackendArray
            Mask array.
        rotated_template : BackendArray
            Empty array to write mask rotations to.
        rotation_matrix : BackendArray
            Rotation matrix.
        """
        left_pad = be.divide(mask.shape, 2).astype(int)
        right_pad = be.add(left_pad, be.mod(mask.shape, 2).astype(int))

        score_start = be.subtract(peak, left_pad)
        score_stop = be.add(peak, right_pad)

        template_start = be.subtract(be.maximum(score_start, 0), score_start)
        template_stop = be.subtract(score_stop, be.minimum(score_stop, scores.shape))
        template_stop = be.subtract(mask.shape, template_stop)

        score_start = be.maximum(score_start, 0)
        score_stop = be.minimum(score_stop, scores.shape)
        score_start = be.astype(score_start, int)
        score_stop = be.astype(score_stop, int)

        template_start = be.astype(template_start, int)
        template_stop = be.astype(template_stop, int)
        coords_score = tuple(slice(*pos) for pos in zip(score_start, score_stop))
        coords_template = tuple(
            slice(*pos) for pos in zip(template_start, template_stop)
        )

        rotated_template.fill(0)
        be.rigid_transform(
            arr=mask, rotation_matrix=rotation_matrix, order=1, out=rotated_template
        )

        scores[coords_score] = be.multiply(
            scores[coords_score], (rotated_template[coords_template] <= 0.1)
        )
        return None


class PeakCallerScipy(PeakCaller):
    """
    Peak calling using :obj:`skimage.feature.peak_local_max` to compute local maxima.
    """

    def call_peaks(
        self, scores: BackendArray, minimum_score: float = None, **kwargs
    ) -> PeakType:
        scores = be.to_numpy_array(scores)
        num_peaks = self.number_of_peaks
        if minimum_score is not None:
            num_peaks = np.inf

        non_squeezable_dims = tuple(i for i, x in enumerate(scores.shape) if x != 1)
        peaks = peak_local_max(
            np.squeeze(scores),
            num_peaks=num_peaks,
            min_distance=self.min_distance,
            threshold_abs=minimum_score,
        )
        peaks_full = np.zeros((peaks.shape[0], scores.ndim), peaks.dtype)
        peaks_full[..., non_squeezable_dims] = peaks[:]
        peaks = be.to_backend_array(peaks_full)
        return peaks, None


class PeakClustering(PeakCallerSort):
    """
    Use DBScan clustering to identify more reliable peaks.
    """

    def __init__(
        self,
        number_of_peaks: int = 1000,
        **kwargs,
    ):
        kwargs["min_distance"] = 0
        super().__init__(number_of_peaks=number_of_peaks, **kwargs)

    @classmethod
    def merge(cls, **kwargs) -> NDArray:
        """
        Merge multiple instances of Analyzer.

        Parameters
        ----------
        **kwargs
            Optional keyword arguments passed to :py:meth:`PeakCaller.merge`.

        Returns
        -------
        NDArray
            NDArray of candidates.
        """
        peaks, rotations, scores, details = super().merge(**kwargs)

        scores = np.array([candidate[2] for candidate in peaks])
        clusters = DBSCAN(eps=np.finfo(float).eps, min_samples=8).fit(peaks)
        labels = clusters.labels_.astype(int)

        label_max = max_index_by_label(labels=labels, scores=scores)
        if -1 in label_max:
            _ = label_max.pop(-1)
        representatives = set(label_max.values())

        keep = np.array(
            [
                True if index in representatives else False
                for index in range(peaks.shape[0])
            ]
        )
        peaks = peaks[keep,]
        rotations = rotations[keep,]
        scores = scores[keep]
        details = details[keep]

        return peaks, rotations, scores, details


class MaxScoreOverRotations:
    """
    Determine the rotation maximizing the score of all given translations.

    Attributes
    ----------
    shape : tuple of ints.
        Shape of ``scores`` and rotations.
    scores : BackendArray
        Array mapping translations to scores.
    rotations : BackendArray
        Array mapping translations to rotation indices.
    rotation_mapping : Dict
        Mapping of rotation matrix bytestrings to rotation indices.
    offset : BackendArray, optional
        Coordinate origin considered during merging, zeryo by default
    use_memmap : bool, optional
        Memmap scores and rotations arrays, False by default.
    thread_safe: bool, optional
        Allow class to be modified by multiple processes, True by default.
    only_unique_rotations : bool, optional
        Whether each rotation will be shown only once, False by default.

    Raises
    ------
    ValueError
        If the data shape cannot be determined from the parameters.

    Examples
    --------
    The following achieves the minimal definition of a :py:class:`MaxScoreOverRotations`
    instance

    >>> from tme.analyzer import MaxScoreOverRotations
    >>> analyzer = MaxScoreOverRotations(shape = (50, 50))

    The following simulates a template matching run by creating random data for a range
    of rotations and sending it to ``analyzer`` via its __call__ method

    >>> for rotation_number in range(10):
    >>>     scores = np.random.rand(50,50)
    >>>     rotation = np.random.rand(scores.ndim, scores.ndim)
    >>>     analyzer(scores = scores, rotation_matrix = rotation)

    The aggregated scores can be exctracted by invoking the __iter__ method of
    ``analyzer``

    >>> results = tuple(analyzer)

    The ``results`` tuple contains (1) the maximum scores for each translation,
    (2) an offset which is relevant when merging results from split template matching
    using :py:meth:`MaxScoreOverRotations.merge`, (3) the rotation used to obtain a
    score for a given translation, (4) a dictionary mapping rotation matrices to the
    indices used in (2).

    We can extract the ``optimal_score`, ``optimal_translation`` and ``optimal_rotation``
    as follows

    >>> optimal_score = results[0].max()
    >>> optimal_translation = np.where(results[0] == results[0].max())
    >>> optimal_rotation_index = results[2][optimal_translation]
    >>> for key, value in results[3].items():
    >>>     if value != optimal_rotation_index:
    >>>         continue
    >>>     optimal_rotation = np.frombuffer(key, rotation.dtype)
    >>>     optimal_rotation = optimal_rotation.reshape(scores.ndim, scores.ndim)

    The outlined procedure is a trivial method to identify high scoring peaks.
    Alternatively, :py:class:`PeakCaller` offers a range of more elaborate approaches
    that can be used.
    """

    def __init__(
        self,
        shape: Tuple[int] = None,
        scores: BackendArray = None,
        rotations: BackendArray = None,
        offset: BackendArray = None,
        score_threshold: float = 0,
        shared_memory_handler: object = None,
        use_memmap: bool = False,
        thread_safe: bool = True,
        only_unique_rotations: bool = False,
        **kwargs,
    ):
        if shape is None and scores is None:
            raise ValueError("Either scores_shape or scores need to be specified.")

        if scores is None:
            shape = tuple(int(x) for x in shape)
            scores = be.full(
                shape=shape,
                dtype=be._float_dtype,
                fill_value=score_threshold,
            )
        self.scores, self.shape = scores, scores.shape

        if rotations is None:
            rotations = be.full(shape, dtype=be._int_dtype, fill_value=-1)
        self.rotations = rotations

        self.scores_dtype = self.scores.dtype
        self.rotations_dtype = self.rotations.dtype
        self.scores = be.to_sharedarr(self.scores, shared_memory_handler)
        self.rotations = be.to_sharedarr(self.rotations, shared_memory_handler)

        if offset is None:
            offset = be.zeros(len(self.shape), be._int_dtype)
        self.offset = be.astype(offset, int)

        self.use_memmap = use_memmap
        self.lock = Manager().Lock() if thread_safe else nullcontext()
        self.lock_is_nullcontext = isinstance(self.scores, type(be.zeros((1))))
        self.rotation_mapping = Manager().dict() if thread_safe else {}
        self._inversion_mapping = self.lock_is_nullcontext and only_unique_rotations

    def _postprocess(
        self,
        targetshape: Tuple[int],
        templateshape: Tuple[int],
        fourier_shift: Tuple[int] = None,
        convolution_mode: str = None,
        shared_memory_handler=None,
        fast_shape: Tuple[int] = None,
        **kwargs,
    ) -> "MaxScoreOverRotations":
        """
        Correct padding to Fourier (and if requested convolution) shape.
        """
        scores = be.from_sharedarr(self.scores)
        rotations = be.from_sharedarr(self.rotations)
        if fourier_shift is not None:
            axis = tuple(i for i in range(len(fourier_shift)))
            scores = be.roll(scores, shift=fourier_shift, axis=axis)
            rotations = be.roll(rotations, shift=fourier_shift, axis=axis)

        convargs = {
            "s1": targetshape,
            "s2": templateshape,
            "convolution_mode": convolution_mode,
        }
        if convolution_mode is not None:
            scores = apply_convolution_mode(scores, **convargs)
            rotations = apply_convolution_mode(rotations, **convargs)

        self.shape = scores.shape
        self.scores = be.to_sharedarr(scores, shared_memory_handler)
        self.rotations = be.to_sharedarr(rotations, shared_memory_handler)
        return self

    def __iter__(self) -> Generator:
        scores = be.from_sharedarr(self.scores)
        rotations = be.from_sharedarr(self.rotations)

        scores = be.to_numpy_array(scores)
        rotations = be.to_numpy_array(rotations)
        if self.use_memmap:
            scores = np.memmap(
                array_to_memmap(scores),
                mode="r",
                dtype=scores.dtype,
                shape=scores.shape,
            )
            rotations = np.memmap(
                array_to_memmap(rotations),
                mode="r",
                dtype=rotations.dtype,
                shape=rotations.shape,
            )
        else:
            # Copy to avoid invalidation by shared memory handler
            scores, rotations = scores.copy(), rotations.copy()

        if self._inversion_mapping:
            self.rotation_mapping = {
                be.tobytes(v): k for k, v in self.rotation_mapping.items()
            }

        param_store = (
            scores,
            be.to_numpy_array(self.offset),
            rotations,
            dict(self.rotation_mapping),
        )
        yield from param_store

    def __call__(self, scores: BackendArray, rotation_matrix: BackendArray):
        """
        Update internal parameter store based on `scores`.

        Parameters
        ----------
        scores : BackendArray
            Array containing the score space.
        rotation_matrix : BackendArray
            Square matrix describing the current rotation.
        """
        # be.tobytes behaviour caused overhead for certain GPU/CUDA combinations
        # If the analyzer is not shared and each rotation is unique, we can
        # use index to rotation mapping and invert prior to merging.
        if self.lock_is_nullcontext:
            rotation_index = len(self.rotation_mapping)
            if self._inversion_mapping:
                self.rotation_mapping[rotation_index] = rotation_matrix
            else:
                rotation = be.tobytes(rotation_matrix)
                rotation_index = self.rotation_mapping.setdefault(
                    rotation, rotation_index
                )
            self.scores, self.rotations = be.max_score_over_rotations(
                scores=scores,
                max_scores=self.scores,
                rotations=self.rotations,
                rotation_index=rotation_index,
            )
            return None

        rotation = be.tobytes(rotation_matrix)
        with self.lock:
            rotation_index = self.rotation_mapping.setdefault(
                rotation, len(self.rotation_mapping)
            )
            internal_scores = be.from_sharedarr(self.scores)
            internal_rotations = be.from_sharedarr(self.rotations)
            internal_sores, internal_rotations = be.max_score_over_rotations(
                scores=scores,
                max_scores=internal_scores,
                rotations=internal_rotations,
                rotation_index=rotation_index,
            )
            return None

    @classmethod
    def merge(cls, param_stores=List[Tuple], **kwargs) -> Tuple[NDArray]:
        """
        Merges multiple instances of :py:class:`MaxScoreOverRotations`.

        Parameters
        ----------
        param_stores : list of tuples, optional
            Internal parameter store. Obtained by running `tuple(instance)`.
        **kwargs
            Optional keyword arguments.

        Returns
        -------
        tuple
            Max aggregated translation scores, corresponding rotations,
            translation offset that is zero by default and mapping between
            rotation index and rotation matrices.
        """
        if len(param_stores) == 1:
            return param_stores[0]

        # Determine output array shape and create consistent rotation map
        new_rotation_mapping, out_shape = {}, None
        for i in range(len(param_stores)):
            if param_stores[i] is None:
                continue

            scores, offset, rotations, rotation_mapping = param_stores[i]
            if out_shape is None:
                out_shape = np.zeros(scores.ndim, int)
                scores_dtype, rotations_dtype = scores.dtype, rotations.dtype
            out_shape = np.maximum(out_shape, np.add(offset, scores.shape))

            for key, value in rotation_mapping.items():
                if key not in new_rotation_mapping:
                    new_rotation_mapping[key] = len(new_rotation_mapping)

        if out_shape is None:
            return None

        out_shape = tuple(int(x) for x in out_shape)
        use_memmap = kwargs.get("use_memmap", False)
        if use_memmap:
            scores_out_filename = generate_tempfile_name()
            rotations_out_filename = generate_tempfile_name()

            scores_out = np.memmap(
                scores_out_filename, mode="w+", shape=out_shape, dtype=scores_dtype
            )
            scores_out.fill(kwargs.get("score_threshold", 0))
            scores_out.flush()
            rotations_out = np.memmap(
                rotations_out_filename,
                mode="w+",
                shape=out_shape,
                dtype=rotations_dtype,
            )
            rotations_out.fill(-1)
            rotations_out.flush()
        else:
            scores_out = np.full(
                out_shape,
                fill_value=kwargs.get("score_threshold", 0),
                dtype=scores_dtype,
            )
            rotations_out = np.full(out_shape, fill_value=-1, dtype=rotations_dtype)

        for i in range(len(param_stores)):
            if param_stores[i] is None:
                continue

            if use_memmap:
                scores_out = np.memmap(
                    scores_out_filename,
                    mode="r+",
                    shape=out_shape,
                    dtype=scores_dtype,
                )
                rotations_out = np.memmap(
                    rotations_out_filename,
                    mode="r+",
                    shape=out_shape,
                    dtype=rotations_dtype,
                )
            scores, offset, rotations, rotation_mapping = param_stores[i]
            stops = np.add(offset, scores.shape).astype(int)
            indices = tuple(slice(*pos) for pos in zip(offset, stops))

            indices_update = scores > scores_out[indices]
            scores_out[indices][indices_update] = scores[indices_update]

            lookup_table = np.arange(
                len(rotation_mapping) + 1, dtype=rotations_out.dtype
            )
            for key, value in rotation_mapping.items():
                lookup_table[value] = new_rotation_mapping[key]

            updated_rotations = rotations[indices_update]
            if len(updated_rotations):
                rotations_out[indices][indices_update] = lookup_table[updated_rotations]

            if use_memmap:
                scores._mmap.close()
                rotations._mmap.close()
                scores_out.flush()
                rotations_out.flush()
                scores_out, rotations_out = None, None

            param_stores[i] = None
            scores, rotations = None, None

        if use_memmap:
            scores_out = np.memmap(
                scores_out_filename, mode="r", shape=out_shape, dtype=scores_dtype
            )
            rotations_out = np.memmap(
                rotations_out_filename,
                mode="r",
                shape=out_shape,
                dtype=rotations_dtype,
            )
        return (
            scores_out,
            np.zeros(scores_out.ndim, dtype=int),
            rotations_out,
            new_rotation_mapping,
        )

    @property
    def shared(self):
        return True


class _MaxScoreOverTranslations(MaxScoreOverRotations):
    """
    Obtain the maximum translation score over various rotations.

    Attributes
    ----------
    scores : BackendArray
        The score space for the observed rotations.
    rotations : BackendArray
        The rotation identifiers for each score.
    translation_offset : BackendArray, optional
        The offset applied during translation.
    observed_rotations : int
        Count of observed rotations.
    use_memmap : bool, optional
        Whether to offload internal data arrays to disk
    thread_safe: bool, optional
        Whether access to internal data arrays should be thread safe
    """

    def __call__(
        self, scores: BackendArray, rotation_matrix: BackendArray, **kwargs
    ) -> None:
        """
        Update internal parameter store based on `scores`.

        Parameters
        ----------
        scores : BackendArray
            Numpy array containing the score space.
        rotation_matrix : BackendArray
            Square matrix describing the current rotation.
        **kwargs
            Optional keyword arguments.
        """
        from tme.matching_utils import centered_mask

        with self.lock:
            rotation = be.tobytes(rotation_matrix)
            if rotation not in self.observed_rotations:
                self.observed_rotations[rotation] = len(self.observed_rotations)
            scores = centered_mask(scores, kwargs["template_shape"])
            rotation_index = self.observed_rotations[rotation]
            internal_scores = be.from_sharedarr(
                shape=self.shape,
                dtype=self.scores_dtype,
                shm=self.scores,
            )
            max_score = scores.max(axis=(1, 2, 3))
            mean_score = scores.mean(axis=(1, 2, 3))
            std_score = scores.std(axis=(1, 2, 3))
            z_score = (max_score - mean_score) / std_score
            internal_scores[rotation_index] = z_score


class MemmapHandler:
    """
    Create numpy memmap objects to write score spaces to.

    This is useful in cases where not the entire score space is sampled at once.

    Parameters
    ----------
    path_translation : dict
        Translation between rotation matrix and memmap file path.
    shape : tuple of int
        Shape of the memmap array.
    dtype : type
        Numpy dtype of the memmap array.
    mode : str, optional
        Mode to open the memmap array with.
    indices : tuple of slice, optional
        Slices specifying which parts of the memmap array will be updated by `__call__`.
    **kwargs
        Optional keyword arguments.
    """

    def __init__(
        self,
        path_translation: Dict,
        shape: Tuple[int],
        dtype: type,
        mode: str = "r+",
        indices: Tuple[slice] = None,
        **kwargs,
    ):
        filepaths = list(path_translation.values())
        _ = [
            np.memmap(filepath, mode=mode, shape=shape, dtype=dtype)
            for filepath in filepaths
        ]
        self._path_translation = path_translation
        self.lock = Lock()
        self.shape = shape
        self.dtype = dtype
        self._indices = indices

    def __call__(self, scores: NDArray, rotation_matrix: NDArray) -> None:
        """
        Write `scores` to memmap object on disk.

        Parameters
        ----------
        scores : ndarray
            Numpy array containing the score space.
        rotation_matrix : ndarray
            Square matrix describing the current rotation.
        **kwargs
            Arbitrary keyword arguments.
        """
        current_object = self._rotation_matrix_to_filepath(rotation_matrix)

        array = np.memmap(current_object, mode="r+", shape=self.shape, dtype=self.dtype)
        # Does not really need a lock because processes operate on different rotations
        with self.lock:
            array[self._indices] += scores
            array.flush()

    def __iter__(self):
        yield None

    @classmethod
    def merge(cls, *args, **kwargs) -> None:
        """
        Placeholder merge method. Does nothing.
        """
        return None

    def update_indices(self, indices: Tuple[slice]) -> None:
        """
        Change which parts of the memmap array will be updated.

        Parameters
        ----------
        indices : tuple of slice
            Slices specifying which parts of the memmap array will be
            updated by `__call__`.
        """
        self._indices = indices

    def _rotation_matrix_to_filepath(self, rotation_matrix: NDArray) -> str:
        """
        Create string representation of `rotation_matrix`.

        Parameters
        ----------
        rotation_matrix : ndarray
            Rotation matrix to convert to string.

        Returns
        -------
        str
            String representation of the rotation matrix.
        """
        rotation_string = "_".join(rotation_matrix.ravel().astype(str))
        return self._path_translation[rotation_string]
