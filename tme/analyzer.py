""" Implements classes to analyze outputs from exhaustive template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
from time import sleep
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod
from contextlib import nullcontext
from multiprocessing import RawValue, Manager, Lock

import numpy as np
from numpy.typing import NDArray
from scipy.stats import entropy
from sklearn.cluster import DBSCAN
from skimage.feature import peak_local_max
from skimage.registration._phase_cross_correlation import _upsampled_dft
from .extensions import max_index_by_label, online_statistics, find_candidate_indices
from .matching_utils import (
    split_numpy_array_slices,
    array_to_memmap,
    generate_tempfile_name,
    euler_to_rotationmatrix,
    apply_convolution_mode,
)
from .backends import backend


def filter_points_indices_bucket(
    coordinates: NDArray, min_distance: Tuple[int]
) -> NDArray:
    coordinates = backend.subtract(coordinates, backend.min(coordinates, axis=0))
    bucket_indices = backend.astype(backend.divide(coordinates, min_distance), int)
    multiplier = backend.power(
        backend.max(bucket_indices, axis=0) + 1, backend.arange(bucket_indices.shape[1])
    )
    backend.multiply(bucket_indices, multiplier, out=bucket_indices)
    flattened_indices = backend.sum(bucket_indices, axis=1)
    _, unique_indices = backend.unique(flattened_indices, return_index=True)
    unique_indices = unique_indices[backend.argsort(unique_indices)]
    return unique_indices


def filter_points_indices(
    coordinates: NDArray,
    min_distance: float,
    bucket_cutoff: int = 1e4,
    batch_dims: Tuple[int] = None,
) -> NDArray:
    if min_distance <= 0:
        return backend.arange(coordinates.shape[0])
    if coordinates.shape[0] == 0:
        return ()

    if batch_dims is not None:
        coordinates_new = backend.zeros(coordinates.shape, coordinates.dtype)
        coordinates_new[:] = coordinates
        coordinates_new[..., batch_dims] = backend.astype(
            coordinates[..., batch_dims] * (2 * min_distance), coordinates_new.dtype
        )
        coordinates = coordinates_new

    if isinstance(coordinates, np.ndarray):
        return find_candidate_indices(coordinates, min_distance)
    elif coordinates.shape[0] > bucket_cutoff or not isinstance(
        coordinates, np.ndarray
    ):
        return filter_points_indices_bucket(coordinates, min_distance)
    distances = np.linalg.norm(coordinates[:, None] - coordinates, axis=-1)
    distances = np.tril(distances)
    keep = np.sum(distances > min_distance, axis=1)
    indices = np.arange(coordinates.shape[0])
    return indices[keep == indices]


def filter_points(
    coordinates: NDArray, min_distance: Tuple[int], batch_dims: Tuple[int] = None
) -> NDArray:
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
    **kwargs
        Additional keyword arguments.

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
        **kwargs,
    ):
        number_of_peaks = int(number_of_peaks)
        min_distance, min_boundary_distance = int(min_distance), int(
            min_boundary_distance
        )
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
        self.min_distance = min_distance
        self.min_boundary_distance = min_boundary_distance
        self.number_of_peaks = number_of_peaks

        self.batch_dims = batch_dims
        if batch_dims is not None:
            self.batch_dims = tuple(int(x) for x in self.batch_dims)

        # Postprocesing arguments
        self.fourier_shift = kwargs.get("fourier_shift", None)
        self.convolution_mode = kwargs.get("convolution_mode", None)
        self.targetshape = kwargs.get("targetshape", None)
        self.templateshape = kwargs.get("templateshape", None)

    def __iter__(self):
        """
        Returns a generator to list objects containing translation,
        rotation, score and details of a given candidate.
        """
        self.peak_list = [backend.to_cpu_array(arr) for arr in self.peak_list]
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

    def __call__(
        self,
        score_space: NDArray,
        rotation_matrix: NDArray,
        minimum_score: float = None,
        maximum_score: float = None,
        **kwargs,
    ) -> None:
        """
        Update the internal parameter store based on input array.

        Parameters
        ----------
        score_space : NDArray
            Array containing the score space.
        rotation_matrix : NDArray
            Rotation matrix used to obtain the score array.
        minimum_score : float
            Minimum score from which to consider peaks. If provided, superseeds limits
            presented by :py:attr:`PeakCaller.number_of_peaks`.
        maximum_score : float
            Maximum score upon which to consider peaks,
        **kwargs
            Optional keyword arguments passed to :py:meth:`PeakCaller.call_peak`.
        """
        for subset, offset in self._batchify(score_space.shape, self.batch_dims):
            peak_positions, peak_details = self.call_peaks(
                score_space=score_space[subset],
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
                peak_details = backend.full((peak_positions.shape[0],), fill_value=-1)

            backend.add(peak_positions, offset, out=peak_positions)
            peak_positions = backend.astype(peak_positions, int)
            if self.min_boundary_distance > 0:
                upper_limit = backend.subtract(
                    score_space.shape, self.min_boundary_distance
                )
                valid_peaks = backend.multiply(
                    peak_positions < upper_limit,
                    peak_positions >= self.min_boundary_distance,
                )
                if self.batch_dims is not None:
                    valid_peaks[..., self.batch_dims] = True

                valid_peaks = (
                    backend.sum(valid_peaks, axis=1) == peak_positions.shape[1]
                )

                if backend.sum(valid_peaks) == 0:
                    continue

                peak_positions, peak_details = (
                    peak_positions[valid_peaks],
                    peak_details[valid_peaks],
                )

            peak_scores = score_space[tuple(peak_positions.T)]
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

            rotations = backend.repeat(
                rotation_matrix.reshape(1, *rotation_matrix.shape),
                peak_positions.shape[0],
                axis=0,
            )

            self._update(
                peak_positions=peak_positions,
                peak_details=peak_details,
                peak_scores=peak_scores,
                rotations=rotations,
                batch_offset=offset,
                **kwargs,
            )

        return None

    @abstractmethod
    def call_peaks(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        This function is not intended to be called directly, but should rather be
        defined by classes inheriting from :py:class:`PeakCaller` to execute a given
        peak calling algorithm.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        **kwargs : Dict, optional
            Keyword arguments passed to __call__.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Array of peak coordinates and peak details.
        """

    @classmethod
    def merge(cls, candidates=List[List], **kwargs) -> NDArray:
        """
        Merge multiple instances of :py:class:`PeakCaller`.

        Parameters
        ----------
        candidate_fits : list of lists
            Obtained by invoking list on the generator returned by __iter__.
        param_stores : list of tuples, optional
            List of parameter stores. Each tuple contains candidate data and number
            of candidates.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        NDArray
            NDArray of candidates.
        """
        base = cls(**kwargs)
        for candidate in candidates:
            if len(candidate) == 0:
                continue
            peak_positions, rotations, peak_scores, peak_details = candidate
            kwargs["translation_offset"] = backend.zeros(peak_positions.shape[1])
            base._update(
                peak_positions=backend.to_backend_array(peak_positions),
                peak_details=backend.to_backend_array(peak_details),
                peak_scores=backend.to_backend_array(peak_scores),
                rotations=backend.to_backend_array(rotations),
                **kwargs,
            )
        return tuple(base)

    @staticmethod
    def oversample_peaks(
        score_space: NDArray, peak_positions: NDArray, oversampling_factor: int = 8
    ):
        """
        Refines peaks positions in the corresponding score space.

        Parameters
        ----------
        score_space : NDArray
            The d-dimensional array representing the score space.
        peak_positions : NDArray
            An array of shape (n, d) containing the peak coordinates
            to be refined, where n is the number of peaks and d is the
            dimensionality of the score space.
        oversampling_factor : int, optional
            The oversampling factor for Fourier transforms. Defaults to 8.

        Returns
        -------
        NDArray
            An array of shape (n, d) containing the refined subpixel
            coordinates of the peaks.

        Notes
        -----
        Floating point peak positions are determined by oversampling the
        score_space around peak_positions. The accuracy
        of refinement scales with 1 / oversampling_factor.

        References
        ----------
        .. [1]  https://scikit-image.org/docs/stable/api/skimage.registration.html
        .. [2]  Manuel Guizar-Sicairos, Samuel T. Thurman, and
                James R. Fienup, “Efficient subpixel image registration
                algorithms,” Optics Letters 33, 156-158 (2008).
                DOI:10.1364/OL.33.000156

        """
        score_space = backend.to_numpy_array(score_space)
        peak_positions = backend.to_numpy_array(peak_positions)

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

        score_space_ft = np.fft.fftn(score_space).conj()
        for index in range(sample_region_offset.shape[0]):
            cross_correlation_upsampled = _upsampled_dft(
                data=score_space_ft,
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

        peak_positions = backend.to_backend_array(peak_positions)

        return peak_positions

    def _update(
        self,
        peak_positions: NDArray,
        peak_details: NDArray,
        peak_scores: NDArray,
        rotations: NDArray,
        **kwargs,
    ) -> None:
        """
        Update internal parameter store.

        Parameters
        ----------
        peak_positions : NDArray
            Position of peaks with shape n x d where n is the number of
            peaks and d the dimension.
        peak_scores : NDArray
            Corresponding score obtained at each peak.
        translation_offset : NDArray, optional
            Offset of the score_space, occurs e.g. when template matching
            to parts of a tomogram.
        rotations: NDArray
            Rotations used to obtain the score space from which
            the candidate stem.
        """
        translation_offset = kwargs.get(
            "translation_offset", backend.zeros(peak_positions.shape[1])
        )
        translation_offset = backend.astype(translation_offset, peak_positions.dtype)

        backend.add(peak_positions, translation_offset, out=peak_positions)
        if not len(self.peak_list):
            self.peak_list = [peak_positions, rotations, peak_scores, peak_details]

        peak_positions = backend.concatenate((self.peak_list[0], peak_positions))
        rotations = backend.concatenate((self.peak_list[1], rotations))
        peak_scores = backend.concatenate((self.peak_list[2], peak_scores))
        peak_details = backend.concatenate((self.peak_list[3], peak_details))

        if self.batch_dims is None:
            top_n = min(backend.size(peak_scores), self.number_of_peaks)
            top_scores, *_ = backend.topk_indices(peak_scores, top_n)
        else:
            # Not very performant but fairly robust
            batch_indices = peak_positions[..., self.batch_dims]
            backend.subtract(
                batch_indices, backend.min(batch_indices, axis=0), out=batch_indices
            )
            multiplier = backend.power(
                backend.max(batch_indices, axis=0) + 1,
                backend.arange(batch_indices.shape[1]),
            )
            backend.multiply(batch_indices, multiplier, out=batch_indices)
            batch_indices = backend.sum(batch_indices, axis=1)
            unique_indices, batch_counts = backend.unique(
                batch_indices, return_counts=True
            )
            total_indices = backend.arange(peak_scores.shape[0])
            batch_indices = [total_indices[batch_indices == x] for x in unique_indices]
            top_scores = backend.concatenate(
                [
                    total_indices[indices][
                        backend.topk_indices(
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
        self, fourier_shift, convolution_mode, targetshape, templateshape, **kwargs
    ):
        if not len(self.peak_list):
            return self

        peak_positions = self.peak_list[0]
        if not len(peak_positions):
            return self

        if targetshape is None or templateshape is None:
            return self

        # Remove padding to next fast fourier length
        score_space_shape = backend.add(targetshape, templateshape) - 1

        if fourier_shift is not None:
            peak_positions = backend.add(peak_positions, fourier_shift)

            backend.subtract(
                peak_positions,
                backend.multiply(
                    backend.astype(
                        backend.divide(peak_positions, score_space_shape), int
                    ),
                    score_space_shape,
                ),
                out=peak_positions,
            )

        if convolution_mode is None:
            return None

        if convolution_mode == "full":
            output_shape = score_space_shape
        elif convolution_mode == "same":
            output_shape = targetshape
        elif convolution_mode == "valid":
            output_shape = backend.add(
                backend.subtract(targetshape, templateshape),
                backend.mod(templateshape, 2),
            )

        output_shape = backend.to_backend_array(output_shape)
        starts = backend.divide(backend.subtract(score_space_shape, output_shape), 2)
        starts = backend.astype(starts, int)
        stops = backend.add(starts, output_shape)

        valid_peaks = (
            backend.sum(
                backend.multiply(peak_positions > starts, peak_positions <= stops),
                axis=1,
            )
            == peak_positions.shape[1]
        )
        self.peak_list[0] = backend.subtract(peak_positions, starts)
        self.peak_list = [x[valid_peaks] for x in self.peak_list]
        return self


class PeakCallerSort(PeakCaller):
    """
    A :py:class:`PeakCaller` subclass that first selects ``number_of_peaks``
    highest scores.
    """

    def call_peaks(self, score_space: NDArray, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Array of peak coordinates and peak details.
        """
        flat_score_space = score_space.reshape(-1)
        k = min(self.number_of_peaks, backend.size(flat_score_space))

        top_k_indices, *_ = backend.topk_indices(flat_score_space, k)

        coordinates = backend.unravel_index(top_k_indices, score_space.shape)
        coordinates = backend.transpose(backend.stack(coordinates))

        return coordinates, None


class PeakCallerMaximumFilter(PeakCaller):
    """
    Find local maxima by applying a maximum filter and enforcing a distance
    constraint subsequently. This is similar to the strategy implemented in
    skimage.feature.peak_local_max.
    """

    def call_peaks(self, score_space: NDArray, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        kwargs: Dict, optional
            Optional keyword arguments.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Array of peak coordinates and peak details.
        """
        peaks = backend.max_filter_coordinates(score_space, self.min_distance)

        return peaks, None


class PeakCallerFast(PeakCaller):
    """
    Subdivides the score space into squares with edge length ``min_distance``
    and determiens maximum value for each. In a second pass, all local maxima
    that are not the local maxima in a ``min_distance`` square centered around them
    are removed.

    """

    def call_peaks(self, score_space: NDArray, **kwargs) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Array of peak coordinates and peak details.
        """
        splits = {
            axis: score_space.shape[axis] // self.min_distance
            for axis in range(score_space.ndim)
        }
        slices = split_numpy_array_slices(score_space.shape, splits)

        coordinates = backend.to_backend_array(
            [
                backend.unravel_index(
                    backend.argmax(score_space[subvol]), score_space[subvol].shape
                )
                for subvol in slices
            ]
        )
        offset = backend.to_backend_array(
            [tuple(x.start for x in subvol) for subvol in slices]
        )
        backend.add(coordinates, offset, out=coordinates)
        coordinates = coordinates[
            backend.flip(backend.argsort(score_space[tuple(coordinates.T)]), (0,))
        ]

        if coordinates.shape[0] == 0:
            return None

        starts = backend.maximum(coordinates - self.min_distance, 0)
        stops = backend.minimum(coordinates + self.min_distance, score_space.shape)
        slices_list = [
            tuple(slice(*coord) for coord in zip(start_row, stop_row))
            for start_row, stop_row in zip(starts, stops)
        ]

        scores = score_space[tuple(coordinates.T)]
        keep = [
            score >= backend.max(score_space[subvol])
            for subvol, score in zip(slices_list, scores)
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
        score_space: NDArray,
        rotation_matrix: NDArray,
        mask: NDArray = None,
        minimum_score: float = None,
        rotation_space: NDArray = None,
        rotation_mapping: Dict = None,
        **kwargs,
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        rotation_matrix : NDArray
            Rotation matrix.
        mask : NDArray, optional
            Mask array, by default None.
        rotation_space : NDArray, optional
            Rotation space array, by default None.
        rotation_mapping : Dict optional
            Dictionary mapping values in rotation_space to Euler angles.
            By default None
        minimum_score : float
            Minimum score value to consider. If provided, superseeds limit given
            by :py:attr:`PeakCaller.number_of_peaks`.

        Returns
        -------
        Tuple[NDArray, NDArray]
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
            shape = tuple(self.min_distance for _ in range(score_space.ndim))
            mask = backend.zeros(shape, dtype=backend._float_dtype)

        rotated_template = backend.zeros(mask.shape, dtype=mask.dtype)

        peak_limit = self.number_of_peaks
        if minimum_score is not None:
            peak_limit = backend.size(score_space)
        else:
            minimum_score = backend.min(score_space) - 1

        scores = backend.zeros(score_space.shape, dtype=score_space.dtype)
        scores[:] = score_space

        while True:
            backend.argmax(scores)
            peak = backend.unravel_index(
                indices=backend.argmax(scores), shape=scores.shape
            )
            if scores[tuple(peak)] < minimum_score:
                break

            coordinates.append(peak)

            current_rotation_matrix = self._get_rotation_matrix(
                peak=peak,
                rotation_space=rotation_space,
                rotation_mapping=rotation_mapping,
                rotation_matrix=rotation_matrix,
            )

            masking_function(
                score_space=scores,
                rotation_matrix=current_rotation_matrix,
                peak=peak,
                mask=mask,
                rotated_template=rotated_template,
            )

            if len(coordinates) >= peak_limit:
                break

        peaks = backend.to_backend_array(coordinates)
        return peaks, None

    @staticmethod
    def _get_rotation_matrix(
        peak: NDArray,
        rotation_space: NDArray,
        rotation_mapping: NDArray,
        rotation_matrix: NDArray,
    ) -> NDArray:
        """
        Get rotation matrix based on peak and rotation data.

        Parameters
        ----------
        peak : NDArray
            Peak coordinates.
        rotation_space : NDArray
            Rotation space array.
        rotation_mapping : Dict
            Dictionary mapping values in rotation_space to Euler angles.
        rotation_matrix : NDArray
            Current rotation matrix.

        Returns
        -------
        NDArray
            Rotation matrix.
        """
        if rotation_space is None or rotation_mapping is None:
            return rotation_matrix

        rotation = rotation_mapping[rotation_space[tuple(peak)]]

        rotation_matrix = backend.to_backend_array(
            euler_to_rotationmatrix(backend.to_numpy_array(rotation))
        )
        return rotation_matrix

    @staticmethod
    def _mask_scores_box(
        score_space: NDArray, peak: NDArray, mask: NDArray, **kwargs: Dict
    ) -> None:
        """
        Mask scores in a box around a peak.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        peak : NDArray
            Peak coordinates.
        mask : NDArray
            Mask array.
        """
        start = backend.maximum(backend.subtract(peak, mask.shape), 0)
        stop = backend.minimum(backend.add(peak, mask.shape), score_space.shape)
        start, stop = backend.astype(start, int), backend.astype(stop, int)
        coords = tuple(slice(*pos) for pos in zip(start, stop))
        score_space[coords] = 0
        return None

    @staticmethod
    def _mask_scores_rotate(
        score_space: NDArray,
        peak: NDArray,
        mask: NDArray,
        rotated_template: NDArray,
        rotation_matrix: NDArray,
        **kwargs: Dict,
    ) -> None:
        """
        Mask score_space using mask rotation around a peak.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        peak : NDArray
            Peak coordinates.
        mask : NDArray
            Mask array.
        rotated_template : NDArray
            Empty array to write mask rotations to.
        rotation_matrix : NDArray
            Rotation matrix.
        """
        left_pad = backend.divide(mask.shape, 2).astype(int)
        right_pad = backend.add(left_pad, backend.mod(mask.shape, 2).astype(int))

        score_start = backend.subtract(peak, left_pad)
        score_stop = backend.add(peak, right_pad)

        template_start = backend.subtract(backend.maximum(score_start, 0), score_start)
        template_stop = backend.subtract(
            score_stop, backend.minimum(score_stop, score_space.shape)
        )
        template_stop = backend.subtract(mask.shape, template_stop)

        score_start = backend.maximum(score_start, 0)
        score_stop = backend.minimum(score_stop, score_space.shape)
        score_start = backend.astype(score_start, int)
        score_stop = backend.astype(score_stop, int)

        template_start = backend.astype(template_start, int)
        template_stop = backend.astype(template_stop, int)
        coords_score = tuple(slice(*pos) for pos in zip(score_start, score_stop))
        coords_template = tuple(
            slice(*pos) for pos in zip(template_start, template_stop)
        )

        rotated_template.fill(0)
        backend.rotate_array(
            arr=mask, rotation_matrix=rotation_matrix, order=1, out=rotated_template
        )

        score_space[coords_score] = backend.multiply(
            score_space[coords_score], (rotated_template[coords_template] <= 0.1)
        )
        return None


class PeakCallerScipy(PeakCaller):
    """
    Peak calling using skimage.feature.peak_local_max to compute local maxima.
    """

    def call_peaks(
        self, score_space: NDArray, minimum_score: float = None, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        minimum_score : float
            Minimum score value to consider. If provided, superseeds limit given
            by :py:attr:`PeakCaller.number_of_peaks`.

        Returns
        -------
        Tuple[NDArray, NDArray]
            Array of peak coordinates and peak details.
        """
        score_space = backend.to_numpy_array(score_space)
        num_peaks = self.number_of_peaks
        if minimum_score is not None:
            num_peaks = np.inf

        non_squeezable_dims = tuple(
            i for i, x in enumerate(score_space.shape) if x != 1
        )
        peaks = peak_local_max(
            np.squeeze(score_space),
            num_peaks=num_peaks,
            min_distance=self.min_distance,
            threshold_abs=minimum_score,
        )
        peaks_full = np.zeros((peaks.shape[0], score_space.ndim), peaks.dtype)
        peaks_full[..., non_squeezable_dims] = peaks[:]
        peaks = backend.to_backend_array(peaks_full)
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
            Additional keyword arguments passed to :py:meth:`PeakCaller.merge`.

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


class ScoreStatistics(PeakCallerFast):
    """
    Compute basic statistics on score spaces with respect to a reference
    score or value.

    This class is used to evaluate a blurring or scoring method when the correct fit
    is known. It is thread-safe and is designed to be shared among multiple processes
    with write permissions to the internal parameters.

    After instantiation, the class's functionality can be accessed through the
    `__call__` method.

    Parameters
    ----------
    reference_position : int, optional
        Index of the correct fit in the array passed to call. Defaults to None.
    min_distance : float, optional
        Minimum distance for local maxima. Defaults to None.
    reference_fit : float, optional
        Score of the correct fit. If set, `reference_position` will be ignored.
        Defaults to None.
    number_of_peaks : int, optional
        Number of candidate fits to consider. Defaults to 1.
    """

    def __init__(
        self,
        reference_position: Tuple[int] = None,
        min_distance: float = 10,
        reference_fit: float = None,
        number_of_peaks: int = 1,
    ):
        super().__init__(number_of_peaks=number_of_peaks, min_distance=min_distance)
        self.lock = Lock()

        self.n = RawValue("Q", 0)
        self.rmean = RawValue("d", 0)
        self.ssqd = RawValue("d", 0)
        self.nbetter_or_equal = RawValue("Q", 0)
        self.maximum_value = RawValue("f", 0)
        self.minimum_value = RawValue("f", 2**32)
        self.shannon_entropy = Manager().list()
        self.candidate_fits = Manager().list()
        self.rotation_names = Manager().list()
        self.reference_fit = RawValue("f", 0)
        self.has_reference = RawValue("i", 0)

        self.reference_position = reference_position
        if reference_fit is not None:
            self.reference_fit.value = reference_fit
            self.has_reference.value = 1

    def __call__(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> None:
        """
        Processes the input array and rotation matrix.

        Parameters
        ----------
        arr : NDArray
            Input data array.
        rotation_matrix : NDArray
            Rotation matrix for processing.
        """
        self.set_reference(score_space, rotation_matrix)

        while not self.has_reference.value:
            print("Stalling processes until reference_fit has been set.")
            sleep(0.5)

        name = "_".join([str(value) for value in rotation_matrix.ravel()])
        n, rmean, ssqd, nbetter_or_equal, max_value, min_value = online_statistics(
            score_space, 0, 0.0, 0.0, self.reference_fit.value
        )

        freq, _ = np.histogram(score_space, bins=100)
        shannon_entropy = entropy(freq / score_space.size)

        peaks, _ = super().call_peaks(
            score_space=score_space, rotation_matrix=rotation_matrix, **kwargs
        )
        scores = score_space[tuple(peaks.T)]
        rotations = np.repeat(
            rotation_matrix.reshape(1, *rotation_matrix.shape),
            peaks.shape[0],
            axis=0,
        )
        distances = np.linalg.norm(peaks - self.reference_position[None, :], axis=1)

        self._update(
            peak_positions=peaks,
            rotations=rotations,
            peak_scores=scores,
            peak_details=distances,
            n=n,
            rmean=rmean,
            ssqd=ssqd,
            nbetter_or_equal=nbetter_or_equal,
            max_value=max_value,
            min_value=min_value,
            entropy=shannon_entropy,
            name=name,
        )

    def __iter__(self):
        param_store = (
            self.peak_list[0],
            self.peak_list[1],
            self.peak_list[2],
            self.peak_list[3],
            self.n.value,
            self.rmean.value,
            self.ssqd.value,
            self.nbetter_or_equal.value,
            self.maximum_value.value,
            self.minimum_value.value,
            list(self.shannon_entropy),
            list(self.rotation_names),
            self.reference_fit.value,
        )
        yield from param_store

    def _update(
        self,
        n: int,
        rmean: float,
        ssqd: float,
        nbetter_or_equal: int,
        max_value: float,
        min_value: float,
        entropy: float,
        name: str,
        **kwargs,
    ) -> None:
        """
        Updates the internal statistics of the analyzer.

        Parameters
        ----------
        n : int
            Sample size.
        rmean : float
            Running mean.
        ssqd : float
            Sum of squared differences.
        nbetter_or_equal : int
            Number of values better or equal to reference.
        max_value : float
            Maximum value.
        min_value : float
            Minimum value.
        entropy : float
            Shannon entropy.
        candidates : list
            List of candidate fits.
        name : str
            Name or label for the data.
        kwargs : dict
            Keyword arguments passed to PeakCaller._update.
        """
        with self.lock:
            super()._update(**kwargs)

            n_total = self.n.value + n
            delta = rmean - self.rmean.value
            delta2 = delta * delta
            self.rmean.value += delta * n / n_total
            self.ssqd.value += ssqd + delta2 * (n * self.n.value) / n_total
            self.n.value = n_total
            self.nbetter_or_equal.value += nbetter_or_equal
            self.minimum_value.value = min(self.minimum_value.value, min_value)
            self.maximum_value.value = max(self.maximum_value.value, max_value)
            self.shannon_entropy.append(entropy)
            self.rotation_names.append(name)

    @classmethod
    def merge(cls, param_stores: List[Tuple]) -> Tuple:
        """
        Merges multiple instances of :py:class`ScoreStatistics`.

        Parameters
        ----------
        param_stores : list of tuple
            Internal parameter store. Obtained by running `tuple(instance)`.
            Defaults to a list with two empty tuples.

        Returns
        -------
        tuple
            Contains the reference fit, the z-transform of the reference fit,
            number of scores, and various other statistics.
        """
        base = cls(reference_position=np.zeros(3, int))
        for param_store in param_stores:
            base._update(
                peak_positions=param_store[0],
                rotations=param_store[1],
                peak_scores=param_store[2],
                peak_details=param_store[3],
                n=param_store[4],
                rmean=param_store[5],
                ssqd=param_store[6],
                nbetter_or_equal=param_store[7],
                max_value=param_store[8],
                min_value=param_store[9],
                entropy=param_store[10],
                name=param_store[11],
            )
        base.reference_fit.value = param_store[12]
        return tuple(base)

    def set_reference(self, score_space: NDArray, rotation_matrix: NDArray) -> None:
        """
        Sets the reference for the analyzer based on the input array
        and rotation matrix.

        Parameters
        ----------
        score_space : NDArray
            Input data array.
        rotation_matrix : NDArray
            Rotation matrix for setting reference.
        """
        is_ref = np.allclose(
            rotation_matrix,
            np.eye(rotation_matrix.shape[0], dtype=rotation_matrix.dtype),
        )
        if not is_ref:
            return None

        reference_position = self.reference_position
        if reference_position is None:
            reference_position = np.divide(score_space.shape, 2).astype(int)
        self.reference_position = reference_position
        self.reference_fit.value = score_space[tuple(reference_position)]
        self.has_reference.value = 1


class MaxScoreOverRotations:
    """
    Obtain the maximum translation score over various rotations.

    Attributes
    ----------
    score_space : NDArray
        The score space for the observed rotations.
    rotations : NDArray
        The rotation identifiers for each score.
    translation_offset : NDArray, optional
        The offset applied during translation.
    observed_rotations : int
        Count of observed rotations.
    use_memmap : bool, optional
        Whether to offload internal data arrays to disk
    thread_safe: bool, optional
        Whether access to internal data arrays should be thread safe

    Examples
    --------
    The following achieves the minimal definition of a :py:class:`MaxScoreOverRotations`
    instance

    >>> from tme.analyzer import MaxScoreOverRotations
    >>> analyzer = MaxScoreOverRotations(
    >>>    score_space_shape = (50, 50),
    >>>    score_space_dtype = np.float32,
    >>>    rotation_space_dtype = np.int32,
    >>> )

    The following simulates a template matching run by creating random data for a range
    of rotations and sending it to ``analyzer`` via its __call__ method

    >>> for rotation_number in range(10):
    >>>     scores = np.random.rand(50,50)
    >>>     rotation = np.random.rand(scores.ndim, scores.ndim)
    >>>     analyzer(score_space = scores, rotation_matrix = rotation)

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
        score_space_shape: Tuple[int],
        score_space_dtype: type,
        translation_offset: NDArray = None,
        score_threshold: float = 0,
        shared_memory_handler: object = None,
        rotation_space_dtype: type = int,
        use_memmap: bool = False,
        thread_safe: bool = True,
        **kwargs,
    ):
        score_space_shape = tuple(int(x) for x in score_space_shape)
        self.score_space = backend.arr_to_sharedarr(
            backend.full(
                shape=score_space_shape,
                dtype=score_space_dtype,
                fill_value=score_threshold,
            ),
            shared_memory_handler=shared_memory_handler,
        )
        self.rotations = backend.arr_to_sharedarr(
            backend.full(score_space_shape, dtype=rotation_space_dtype, fill_value=-1),
            shared_memory_handler,
        )
        if translation_offset is None:
            translation_offset = backend.zeros(len(score_space_shape))

        self.translation_offset = backend.astype(translation_offset, int)
        self.score_space_shape = score_space_shape
        self.rotation_space_dtype = rotation_space_dtype
        self.score_space_dtype = score_space_dtype

        self.use_memmap = use_memmap
        self.lock = Manager().Lock() if thread_safe else nullcontext()
        self.lock_is_nullcontext = isinstance(
            self.score_space, type(backend.zeros((1)))
        )
        self.observed_rotations = Manager().dict() if thread_safe else {}

    def _postprocess(
        self,
        fourier_shift,
        convolution_mode,
        targetshape,
        templateshape,
        shared_memory_handler=None,
        **kwargs,
    ):
        internal_scores = backend.sharedarr_to_arr(
            shape=self.score_space_shape,
            dtype=self.score_space_dtype,
            shm=self.score_space,
        )
        internal_rotations = backend.sharedarr_to_arr(
            shape=self.score_space_shape,
            dtype=self.rotation_space_dtype,
            shm=self.rotations,
        )

        if fourier_shift is not None:
            axis = tuple(i for i in range(len(fourier_shift)))
            internal_scores = backend.roll(
                internal_scores, shift=fourier_shift, axis=axis
            )
            internal_rotations = backend.roll(
                internal_rotations, shift=fourier_shift, axis=axis
            )

        if convolution_mode is not None:
            internal_scores = apply_convolution_mode(
                internal_scores,
                convolution_mode=convolution_mode,
                s1=targetshape,
                s2=templateshape,
            )
            internal_rotations = apply_convolution_mode(
                internal_rotations,
                convolution_mode=convolution_mode,
                s1=targetshape,
                s2=templateshape,
            )

        self.score_space_shape = internal_scores.shape
        self.score_space = backend.arr_to_sharedarr(
            internal_scores, shared_memory_handler
        )
        self.rotations = backend.arr_to_sharedarr(
            internal_rotations, shared_memory_handler
        )
        return self

    def __iter__(self):
        internal_scores = backend.sharedarr_to_arr(
            shape=self.score_space_shape,
            dtype=self.score_space_dtype,
            shm=self.score_space,
        )
        internal_rotations = backend.sharedarr_to_arr(
            shape=self.score_space_shape,
            dtype=self.rotation_space_dtype,
            shm=self.rotations,
        )

        internal_scores = backend.to_numpy_array(internal_scores)
        internal_rotations = backend.to_numpy_array(internal_rotations)
        if self.use_memmap:
            internal_scores_filename = array_to_memmap(internal_scores)
            internal_rotations_filename = array_to_memmap(internal_rotations)
            internal_scores = np.memmap(
                internal_scores_filename,
                mode="r",
                dtype=internal_scores.dtype,
                shape=internal_scores.shape,
            )
            internal_rotations = np.memmap(
                internal_rotations_filename,
                mode="r",
                dtype=internal_rotations.dtype,
                shape=internal_rotations.shape,
            )
        else:
            # Avoid invalidation by shared memory handler with copy
            internal_scores = internal_scores.copy()
            internal_rotations = internal_rotations.copy()

        param_store = (
            internal_scores,
            backend.to_numpy_array(self.translation_offset),
            internal_rotations,
            dict(self.observed_rotations),
        )
        yield from param_store

    def __call__(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> None:
        """
        Update internal parameter store based on `score_space`.

        Parameters
        ----------
        score_space : ndarray
            Numpy array containing the score space.
        rotation_matrix : ndarray
            Square matrix describing the current rotation.
        **kwargs
            Arbitrary keyword arguments.
        """
        rotation = backend.tobytes(rotation_matrix)

        if self.lock_is_nullcontext:
            rotation_index = self.observed_rotations.setdefault(
                rotation, len(self.observed_rotations)
            )
            backend.max_score_over_rotations(
                score_space=score_space,
                internal_scores=self.score_space,
                internal_rotations=self.rotations,
                rotation_index=rotation_index,
            )
            return None

        with self.lock:
            rotation_index = self.observed_rotations.setdefault(
                rotation, len(self.observed_rotations)
            )
            internal_scores = backend.sharedarr_to_arr(
                shape=self.score_space_shape,
                dtype=self.score_space_dtype,
                shm=self.score_space,
            )
            internal_rotations = backend.sharedarr_to_arr(
                shape=self.score_space_shape,
                dtype=self.rotation_space_dtype,
                shm=self.rotations,
            )

            backend.max_score_over_rotations(
                score_space=score_space,
                internal_scores=internal_scores,
                internal_rotations=internal_rotations,
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
            Arbitrary keyword arguments.

        Returns
        -------
        tuple
            Max aggregated translation scores, corresponding rotations,
            translation offset that is zero by default and mapping between
            rotation index and rotation matrices.
        """
        if len(param_stores) == 1:
            return param_stores[0]

        new_rotation_mapping, base_max = {}, None
        scores_out_dtype, rotations_out_dtype = None, None
        for i in range(len(param_stores)):
            if param_stores[i] is None:
                continue
            score_space, offset, rotations, rotation_mapping = param_stores[i]
            if base_max is None:
                base_max = np.zeros(score_space.ndim, int)
                scores_out_dtype = score_space.dtype
                rotations_out_dtype = rotations.dtype
            np.maximum(base_max, np.add(offset, score_space.shape), out=base_max)

            for key, value in rotation_mapping.items():
                if key not in new_rotation_mapping:
                    new_rotation_mapping[key] = len(new_rotation_mapping)

        if base_max is None:
            return None

        base_max = tuple(int(x) for x in base_max)
        use_memmap = kwargs.get("use_memmap", False)
        if use_memmap:
            scores_out_filename = generate_tempfile_name()
            rotations_out_filename = generate_tempfile_name()

            scores_out = np.memmap(
                scores_out_filename, mode="w+", shape=base_max, dtype=scores_out_dtype
            )
            scores_out.fill(kwargs.get("score_threshold", 0))
            scores_out.flush()
            rotations_out = np.memmap(
                rotations_out_filename,
                mode="w+",
                shape=base_max,
                dtype=rotations_out_dtype,
            )
            rotations_out.fill(-1)
            rotations_out.flush()
        else:
            scores_out = np.full(
                base_max,
                fill_value=kwargs.get("score_threshold", 0),
                dtype=scores_out_dtype,
            )
            rotations_out = np.full(base_max, fill_value=-1, dtype=rotations_out_dtype)

        for i in range(len(param_stores)):
            if param_stores[i] is None:
                continue

            if use_memmap:
                scores_out = np.memmap(
                    scores_out_filename,
                    mode="r+",
                    shape=base_max,
                    dtype=scores_out_dtype,
                )
                rotations_out = np.memmap(
                    rotations_out_filename,
                    mode="r+",
                    shape=base_max,
                    dtype=rotations_out_dtype,
                )
            score_space, offset, rotations, rotation_mapping = param_stores[i]
            stops = np.add(offset, score_space.shape).astype(int)
            indices = tuple(slice(*pos) for pos in zip(offset, stops))

            indices_update = score_space > scores_out[indices]
            scores_out[indices][indices_update] = score_space[indices_update]

            lookup_table = np.arange(
                len(rotation_mapping) + 1, dtype=rotations_out.dtype
            )
            for key, value in rotation_mapping.items():
                lookup_table[value] = new_rotation_mapping[key]

            updated_rotations = rotations[indices_update]
            if len(updated_rotations):
                rotations_out[indices][indices_update] = lookup_table[updated_rotations]

            if use_memmap:
                score_space._mmap.close()
                rotations._mmap.close()
                scores_out.flush()
                rotations_out.flush()
                scores_out, rotations_out = None, None

            param_stores[i] = None
            score_space, rotations = None, None

        if use_memmap:
            scores_out = np.memmap(
                scores_out_filename, mode="r", shape=base_max, dtype=scores_out_dtype
            )
            rotations_out = np.memmap(
                rotations_out_filename,
                mode="r",
                shape=base_max,
                dtype=rotations_out_dtype,
            )
        return (
            scores_out,
            np.zeros(scores_out.ndim, dtype=int),
            rotations_out,
            new_rotation_mapping,
        )


class _MaxScoreOverTranslations(MaxScoreOverRotations):
    """
    Obtain the maximum translation score over various rotations.

    Attributes
    ----------
    score_space : NDArray
        The score space for the observed rotations.
    rotations : NDArray
        The rotation identifiers for each score.
    translation_offset : NDArray, optional
        The offset applied during translation.
    observed_rotations : int
        Count of observed rotations.
    use_memmap : bool, optional
        Whether to offload internal data arrays to disk
    thread_safe: bool, optional
        Whether access to internal data arrays should be thread safe
    """

    def __call__(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> None:
        """
        Update internal parameter store based on `score_space`.

        Parameters
        ----------
        score_space : ndarray
            Numpy array containing the score space.
        rotation_matrix : ndarray
            Square matrix describing the current rotation.
        **kwargs
            Arbitrary keyword arguments.
        """
        from tme.matching_utils import centered_mask

        with self.lock:
            rotation = backend.tobytes(rotation_matrix)
            if rotation not in self.observed_rotations:
                self.observed_rotations[rotation] = len(self.observed_rotations)
            score_space = centered_mask(score_space, kwargs["template_shape"])
            rotation_index = self.observed_rotations[rotation]
            internal_scores = backend.sharedarr_to_arr(
                shape=self.score_space_shape,
                dtype=self.score_space_dtype,
                shm=self.score_space,
            )
            max_score = score_space.max(axis=(1, 2, 3))
            mean_score = score_space.mean(axis=(1, 2, 3))
            std_score = score_space.std(axis=(1, 2, 3))
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
        Arbitrary keyword arguments.
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

    def __call__(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> None:
        """
        Write `score_space` to memmap object on disk.

        Parameters
        ----------
        score_space : ndarray
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
            array[self._indices] += score_space
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
