""" Implements classes to analyze outputs from exhaustive template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from functools import wraps
from abc import ABC, abstractmethod
from typing import Tuple, List, Dict, Generator

import numpy as np
from skimage.feature import peak_local_max
from skimage.registration._phase_cross_correlation import _upsampled_dft

from ._utils import score_to_cart
from ..backends import backend as be
from ..matching_utils import split_shape
from ..types import BackendArray, NDArray
from ..rotations import euler_to_rotationmatrix

__all__ = [
    "PeakCaller",
    "PeakCallerSort",
    "PeakCallerMaximumFilter",
    "PeakCallerFast",
    "PeakCallerRecursiveMasking",
    "PeakCallerScipy",
    "PeakClustering",
    "filter_points",
    "filter_points_indices",
]

PeakType = Tuple[BackendArray, BackendArray]


def _filter_bucket(
    coordinates: BackendArray, min_distance: Tuple[float], scores: BackendArray = None
) -> BackendArray:
    coordinates = be.subtract(coordinates, be.min(coordinates, axis=0))
    bucket_indices = be.astype(be.divide(coordinates, min_distance), int)
    multiplier = be.power(
        be.max(bucket_indices, axis=0) + 1, be.arange(bucket_indices.shape[1])
    )
    bucket_indices = be.multiply(bucket_indices, multiplier, out=bucket_indices)
    flattened_indices = be.sum(bucket_indices, axis=1)

    if scores is not None:
        _, inverse_indices = be.unique(flattened_indices, return_inverse=True)

        # Avoid bucket index overlap
        scores = be.subtract(scores, be.min(scores))
        scores = be.divide(scores, be.max(scores) + 0.1, out=scores)
        scores = be.subtract(inverse_indices, scores)

        indices = be.argsort(scores)
        sorted_buckets = inverse_indices[indices]
        mask = sorted_buckets[1:] != sorted_buckets[:-1]
        mask = be.concatenate((be.full((1,), fill_value=1, dtype=mask.dtype), mask))
        return indices[mask]

    _, unique_indices = be.unique(flattened_indices, return_index=True)
    return unique_indices[be.argsort(unique_indices)]


def filter_points_indices(
    coordinates: BackendArray,
    min_distance: float,
    bucket_cutoff: int = 1e5,
    batch_dims: Tuple[int] = None,
    scores: BackendArray = None,
) -> BackendArray:
    from ..extensions import find_candidate_indices

    if min_distance <= 0:
        return be.arange(coordinates.shape[0])

    n_coords = coordinates.shape[0]
    if n_coords == 0:
        return ()

    if batch_dims is not None:
        coordinates_new = be.zeros(coordinates.shape, coordinates.dtype)
        coordinates_new[:] = coordinates
        coordinates_new[..., batch_dims] = be.astype(
            coordinates[..., batch_dims] * (2 * min_distance), coordinates_new.dtype
        )
        coordinates = coordinates_new

    if isinstance(coordinates, np.ndarray) and n_coords < bucket_cutoff:
        if scores is not None:
            sorted_indices = np.argsort(-scores)
            coordinates = coordinates[sorted_indices]
        indices = find_candidate_indices(coordinates, min_distance)
        if scores is not None:
            return sorted_indices[indices]
    elif n_coords > bucket_cutoff or not isinstance(coordinates, np.ndarray):
        return _filter_bucket(coordinates, min_distance, scores)

    distances = be.linalg.norm(coordinates[:, None] - coordinates, axis=-1)
    distances = be.tril(distances)
    keep = be.sum(distances > min_distance, axis=1)
    indices = be.arange(coordinates.shape[0])
    return indices[keep == indices]


def filter_points(
    coordinates: NDArray, min_distance: Tuple[int], batch_dims: Tuple[int] = None
) -> BackendArray:
    unique_indices = filter_points_indices(coordinates, min_distance, batch_dims)
    coordinates = coordinates[unique_indices]
    return coordinates


def batchify(shape: Tuple[int], batch_dims: Tuple[int] = None) -> List:
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


class PeakCaller(ABC):
    """
    Base class for peak calling algorithms.

    Parameters
    ----------
    shape : tuple of int
        Score space shape. Used to determine dimension of peak calling problem.
    num_peaks : int, optional
        Number of candidate peaks to consider.
    min_distance : int, optional
        Minimum distance between peaks, 1 by default
    min_boundary_distance : int, optional
        Minimum distance to array boundaries, 0 by default.
    min_score : float, optional
        Minimum score from which to consider peaks.
    max_score : float, optional
        Maximum score upon which to consider peaks.
    batch_dims : int, optional
        Peak calling batch dimensions.
    **kwargs
        Optional keyword arguments.

    Raises
    ------
    ValueError
        If num_peaks is less than or equal to zero.
        If min_distances is less than zero.
    """

    def __init__(
        self,
        shape: int,
        num_peaks: int = 1000,
        min_distance: int = 1,
        min_boundary_distance: int = 0,
        min_score: float = None,
        max_score: float = None,
        batch_dims: Tuple[int] = None,
        shm_handler: object = None,
        **kwargs,
    ):
        if num_peaks <= 0:
            raise ValueError("num_peaks has to be larger than 0.")
        if min_distance < 0:
            raise ValueError("min_distance has to be non-negative.")
        if min_boundary_distance < 0:
            raise ValueError("min_boundary_distance has to be non-negative.")

        ndim = len(shape)
        self.translations = be.full(
            (num_peaks, ndim), fill_value=-1, dtype=be._int_dtype
        )
        self.rotations = be.full(
            (num_peaks, ndim, ndim), fill_value=0, dtype=be._float_dtype
        )
        for i in range(ndim):
            self.rotations[:, i, i] = 1.0

        self.scores = be.full((num_peaks,), fill_value=0, dtype=be._float_dtype)
        self.details = be.full((num_peaks,), fill_value=0, dtype=be._float_dtype)

        self.num_peaks = int(num_peaks)
        self.min_distance = int(min_distance)
        self.min_boundary_distance = int(min_boundary_distance)

        self.batch_dims = batch_dims
        if batch_dims is not None:
            self.batch_dims = tuple(int(x) for x in self.batch_dims)

        self.min_score, self.max_score = min_score, max_score

        # Postprocessing arguments
        self.fourier_shift = kwargs.get("fourier_shift", None)
        self.convolution_mode = kwargs.get("convolution_mode", None)
        self.targetshape = kwargs.get("targetshape", None)
        self.templateshape = kwargs.get("templateshape", None)

    def __iter__(self) -> Generator:
        """
        Returns a generator to list objects containing translation,
        rotation, score and details of a given candidate.
        """
        self.peak_list = [
            be.to_cpu_array(self.translations),
            be.to_cpu_array(self.rotations),
            be.to_cpu_array(self.scores),
            be.to_cpu_array(self.details),
        ]
        yield from self.peak_list

    def _get_peak_mask(self, peaks: BackendArray, scores: BackendArray) -> BackendArray:
        if not len(peaks):
            return None

        valid_peaks = be.full((peaks.shape[0],), fill_value=1) == 1
        if self.min_boundary_distance > 0:
            upper_limit = be.subtract(
                be.to_backend_array(scores.shape), self.min_boundary_distance
            )
            valid_peaks = be.multiply(
                peaks < upper_limit,
                peaks >= self.min_boundary_distance,
            )
            if self.batch_dims is not None:
                valid_peaks[..., self.batch_dims] = True

            valid_peaks = be.sum(valid_peaks, axis=1) == peaks.shape[1]

        # Score thresholds and nan removal
        peak_scores = scores[tuple(peaks.T)]
        valid_peaks = be.multiply(peak_scores == peak_scores, valid_peaks)
        if self.min_score is not None:
            valid_peaks = be.multiply(peak_scores >= self.min_score, valid_peaks)

        if self.max_score is not None:
            valid_peaks = be.multiply(peak_scores <= self.max_score, valid_peaks)

        if be.sum(valid_peaks) == 0:
            return None

        # Ensure consistent upper limit of input peaks for _update step
        if (
            be.sum(valid_peaks) > self.num_peaks
            or peak_scores.shape[0] > self.num_peaks
        ):
            peak_indices = self._top_peaks(
                peaks, scores=peak_scores * valid_peaks, num_peaks=2 * self.num_peaks
            )
            valid_peaks = be.full(peak_scores.shape, 0, bool)
            valid_peaks[peak_indices] = True

            if self.min_score is not None:
                valid_peaks = be.multiply(peak_scores >= self.min_score, valid_peaks)

            if self.max_score is not None:
                valid_peaks = be.multiply(peak_scores <= self.max_score, valid_peaks)

        if valid_peaks.shape[0] != peaks.shape[0]:
            return None
        return valid_peaks

    def _apply_over_batch(func):
        @wraps(func)
        def wrapper(self, scores, rotation_matrix, **kwargs):
            for subset, batch_offset in batchify(scores.shape, self.batch_dims):
                yield func(
                    self,
                    scores=scores[subset],
                    rotation_matrix=rotation_matrix,
                    batch_offset=batch_offset,
                    **kwargs,
                )

        return wrapper

    @_apply_over_batch
    def _call_peaks(self, scores, rotation_matrix, batch_offset=None, **kwargs):
        peak_positions, peak_details = self.call_peaks(
            scores=scores,
            rotation_matrix=rotation_matrix,
            min_score=self.min_score,
            max_score=self.max_score,
            batch_offset=batch_offset,
            **kwargs,
        )
        if peak_positions is None:
            return None, None

        peak_positions = be.to_backend_array(peak_positions)
        if batch_offset is not None:
            batch_offset = be.to_backend_array(batch_offset)
            peak_positions = be.add(peak_positions, batch_offset, out=peak_positions)

        peak_positions = be.astype(peak_positions, int)
        return peak_positions, peak_details

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
        for ret in self._call_peaks(scores=scores, rotation_matrix=rotation_matrix):
            peak_positions, peak_details = ret
            if peak_positions is None:
                continue

            valid_peaks = self._get_peak_mask(peaks=peak_positions, scores=scores)
            if valid_peaks is None:
                continue

            peak_positions = peak_positions[valid_peaks]
            peak_scores = scores[tuple(peak_positions.T)]
            if peak_details is not None:
                peak_details = peak_details[valid_peaks]
                # peak_details, peak_scores = peak_scores, -peak_details
            else:
                peak_details = be.full(peak_scores.shape, fill_value=-1)

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
        if "shape" not in kwargs:
            kwargs["shape"] = tuple(1 for _ in range(candidates[0][0].shape[1]))

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

    def _top_peaks(self, positions, scores, num_peaks: int = None):
        num_peaks = be.size(scores) if not num_peaks else num_peaks

        if self.batch_dims is None:
            top_n = min(be.size(scores), num_peaks)
            top_scores, *_ = be.topk_indices(scores, top_n)
            return top_scores

        # Not very performant but fairly robust
        batch_indices = positions[..., self.batch_dims]
        batch_indices = be.subtract(batch_indices, be.min(batch_indices, axis=0))
        multiplier = be.power(
            be.max(batch_indices, axis=0) + 1,
            be.arange(batch_indices.shape[1]),
        )
        batch_indices = be.multiply(batch_indices, multiplier, out=batch_indices)
        batch_indices = be.sum(batch_indices, axis=1)
        unique_indices, batch_counts = be.unique(batch_indices, return_counts=True)
        total_indices = be.arange(scores.shape[0])
        batch_indices = [total_indices[batch_indices == x] for x in unique_indices]
        top_scores = be.concatenate(
            [
                total_indices[indices][
                    be.topk_indices(scores[indices], min(y, num_peaks))
                ]
                for indices, y in zip(batch_indices, batch_counts)
            ]
        )
        return top_scores

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
        peak_scores: BackendArray
            Score at each peak (n,).
        rotations: BackendArray
            Rotation at each peak (n, d, d).
        offset : BackendArray, optional
            Translation offset, e.g. from splitting, (d, ).
        """
        if offset is not None:
            offset = be.astype(be.to_backend_array(offset), peak_positions.dtype)
            peak_positions = be.add(peak_positions, offset, out=peak_positions)

        positions = be.concatenate((self.translations, peak_positions))
        rotations = be.concatenate((self.rotations, rotations))
        scores = be.concatenate((self.scores, peak_scores))
        details = be.concatenate((self.details, peak_details))

        # topk filtering after distances yields more distributed peak calls
        distance_order = filter_points_indices(
            coordinates=positions,
            min_distance=self.min_distance,
            batch_dims=self.batch_dims,
            scores=scores,
        )

        top_scores = self._top_peaks(
            positions[distance_order, :], scores[distance_order], self.num_peaks
        )
        final_order = distance_order[top_scores]

        self.translations = positions[final_order, :]
        self.rotations = rotations[final_order, :]
        self.scores = scores[final_order]
        self.details = details[final_order]

    def _postprocess(self, **kwargs):
        if not len(self.translations):
            return self

        positions, valid_peaks = score_to_cart(self.translations, **kwargs)

        self.translations = positions[valid_peaks]
        self.rotations = self.rotations[valid_peaks]
        self.scores = self.scores[valid_peaks]
        self.details = self.details[valid_peaks]
        return self


class PeakCallerSort(PeakCaller):
    """
    A :py:class:`PeakCaller` subclass that first selects ``num_peaks``
    highest scores.
    """

    def call_peaks(self, scores: BackendArray, **kwargs) -> PeakType:
        flat_scores = scores.reshape(-1)
        k = min(self.num_peaks, be.size(flat_scores))

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
        splits = {i: x // self.min_distance for i, x in enumerate(scores.shape)}
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
        coordinates = coordinates[be.argsort(-scores[tuple(coordinates.T)])]

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
        min_score: float = None,
        rotations: BackendArray = None,
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
        rotations : BackendArray, optional
            Rotation space array, by default None.
        rotation_mapping : Dict optional
            Dictionary mapping values in rotations to Euler angles.
            By default None
        min_score : float
            Minimum score value to consider. If provided, superseeds limit given
            by :py:attr:`PeakCaller.num_peaks`.

        Returns
        -------
        Tuple[BackendArray, BackendArray]
            Array of peak coordinates and peak details.

        Notes
        -----
        By default, scores are masked using a box with edge length self.min_distance.
        If mask is provided, elements around each peak will be multiplied by the mask
        values. If rotations and rotation_mapping is provided, the respective
        rotation will be applied to the mask, otherwise rotation_matrix is used.
        """
        coordinates, masking_function = [], self._mask_scores_rotate

        if mask is None:
            masking_function = self._mask_scores_box
            shape = tuple(self.min_distance for _ in range(scores.ndim))
            mask = be.zeros(shape, dtype=be._float_dtype)

        rotated_template = be.zeros(mask.shape, dtype=mask.dtype)

        peak_limit = self.num_peaks
        if min_score is not None:
            peak_limit = be.size(scores)
        else:
            min_score = be.min(scores) - 1

        scores_copy = be.zeros(scores.shape, dtype=scores.dtype)
        scores_copy[:] = scores

        while True:
            be.argmax(scores_copy)
            peak = be.unravel_index(
                indices=be.argmax(scores_copy), shape=scores_copy.shape
            )
            if scores_copy[tuple(peak)] < min_score:
                break

            coordinates.append(peak)

            current_rotation_matrix = self._get_rotation_matrix(
                peak=peak,
                rotation_space=rotations,
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
        if rotation.ndim != 2:
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
        self, scores: BackendArray, min_score: float = None, **kwargs
    ) -> PeakType:
        scores = be.to_numpy_array(scores)
        num_peaks = self.num_peaks
        if min_score is not None:
            num_peaks = np.inf

        non_squeezable_dims = tuple(i for i, x in enumerate(scores.shape) if x != 1)
        peaks = peak_local_max(
            np.squeeze(scores),
            num_peaks=num_peaks,
            min_distance=self.min_distance,
            threshold_abs=min_score,
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
        num_peaks: int = 1000,
        **kwargs,
    ):
        kwargs["min_distance"] = 0
        super().__init__(num_peaks=num_peaks, **kwargs)

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
        from sklearn.cluster import DBSCAN
        from ..extensions import max_index_by_label

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
