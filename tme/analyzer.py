""" Implements classes to analyze score spaces from systematic fitting.

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

from .extensions import max_index_by_label, online_statistics
from .matching_utils import (
    split_numpy_array_slices,
    array_to_memmap,
    generate_tempfile_name,
)

from .backends import backend


def filter_points_indices(coordinates: NDArray, min_distance: Tuple[int]):
    if min_distance <= 0:
        return backend.arange(coordinates.shape[0])
    bucket_indices = backend.astype(backend.divide(coordinates, min_distance), int)
    multiplier = backend.power(
        backend.max(bucket_indices, axis=0) + 1, backend.arange(bucket_indices.shape[1])
    )
    backend.multiply(bucket_indices, multiplier, out=bucket_indices)
    flattened_indices = backend.sum(bucket_indices, axis=1)
    _, unique_indices = backend.unique(flattened_indices, return_index=True)
    unique_indices = unique_indices[backend.argsort(unique_indices)]
    return unique_indices


def filter_points(coordinates: NDArray, min_distance: Tuple[int]):
    unique_indices = filter_points_indices(coordinates, min_distance)
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
        **kwargs,
    ):
        number_of_peaks, min_distance = int(number_of_peaks), int(min_distance)
        if number_of_peaks <= 0:
            raise ValueError(
                f"number_of_peaks has to be larger than 0, got {number_of_peaks}"
            )
        if min_distance < 0:
            raise ValueError(f"min_distance has to be non-negative, got {min_distance}")

        self.peak_list = []
        self.min_distance = min_distance
        self.number_of_peaks = number_of_peaks

    def __iter__(self):
        """
        Returns a generator to list objects containing translation,
        rotation, score and details of a given candidate.
        """
        self.peak_list = [backend.to_cpu_array(arr) for arr in self.peak_list]
        yield from self.peak_list

    def __call__(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> None:
        """
        Update the internal parameter store based on input array.

        Parameters
        ----------
        score_space : NDArray
            Array containing the score space.
        rotation_matrix : NDArray
            Rotation matrix used to obtain the score array.
        **kwargs
            Additional keyword arguments.
        """
        peak_positions, peak_details = self.call_peaks(
            score_space=score_space, rotation_matrix=rotation_matrix, **kwargs
        )

        if peak_positions is None:
            return None

        peak_positions = backend.astype(peak_positions, int)
        if peak_positions.shape[0] == 0:
            return None

        fourier_shift = kwargs.get(
            "fourier_shift", backend.zeros(peak_positions.shape[1], dtype=int)
        )
        if np.any(fourier_shift) != 0:
            peak_positions = backend.mod(
                backend.add(peak_positions, fourier_shift), score_space.shape
            )
            peak_positions = backend.astype(peak_positions, int)

        if peak_details is None:
            peak_details = backend.to_backend_array([-1] * peak_positions.shape[0])

        rotations = backend.repeat(
            rotation_matrix.reshape(1, *rotation_matrix.shape),
            peak_positions.shape[0],
            axis=0,
        )

        self._update(
            peak_positions=peak_positions,
            peak_details=peak_details,
            peak_scores=score_space[tuple(peak_positions.T)],
            rotations=rotations,
            **kwargs,
        )

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
        minimum_score : float
            Minimum score value to consider.
        min_distance : float
            Minimum distance between maxima.

        Returns
        -------
        NDArray
            Array of peak coordiantes.
        NDArray
            Array of peak details.
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
            base._update(
                peak_positions=backend.to_backend_array(peak_positions),
                peak_details=backend.to_backend_array(peak_details),
                peak_scores=backend.to_backend_array(peak_scores),
                rotations=backend.to_backend_array(rotations),
                **kwargs,
            )
        return tuple(base)

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
            return None

        peaks = backend.concatenate((self.peak_list[0], peak_positions))
        rotations = backend.concatenate((self.peak_list[1], rotations))
        peak_scores = backend.concatenate((self.peak_list[2], peak_scores))
        peak_details = backend.concatenate((self.peak_list[3], peak_details))

        top_n = min(backend.size(peak_scores), self.number_of_peaks)
        top_scores, *_ = backend.topk_indices(peak_scores, top_n)

        final_order = top_scores[
            filter_points_indices(peaks[top_scores], self.min_distance)
        ]

        self.peak_list[0] = peaks[final_order,]
        self.peak_list[1] = rotations[final_order,]
        self.peak_list[2] = peak_scores[final_order]
        self.peak_list[3] = peak_details[final_order]


class PeakCallerSort(PeakCaller):
    """
    A :py:class:`PeakCaller` subclass that first selects ``number_of_peaks``
    highest scores and subsequently filters local maxima to suffice a distance
    from one another of ``min_distance``.

    """

    def call_peaks(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        minimum_score : float
            Minimum score value to consider.
        min_distance : float
            Minimum distance between maxima.

        Returns
        -------
        NDArray
            Array of peak coordiantes.
        NDArray
            Array of peak details.
        """
        flat_score_space = score_space.reshape(-1)
        k = min(self.number_of_peaks, backend.size(flat_score_space))

        top_k_indices, *_ = backend.topk_indices(flat_score_space, k)

        coordinates = backend.unravel_index(top_k_indices, score_space.shape)
        coordinates = backend.transpose(backend.stack(coordinates))

        peaks = filter_points(coordinates, self.min_distance)
        return peaks, None


class PeakCallerMaximumFilter(PeakCaller):
    """
    Find local maxima by applying a maximum filter and enforcing a distance
    constraint subseqquently. This is similar to the strategy implemented in
    skimage.feature.peak_local_max.
    """

    def call_peaks(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        minimum_score : float
            Minimum score value to consider.
        min_distance : float
            Minimum distance between maxima.

        Returns
        -------
        NDArray
            Array of peak coordiantes.
        NDArray
            Array of peak details.
        """
        peaks = backend.max_filter_coordinates(score_space, self.min_distance)

        input_candidates = min(
            self.number_of_peaks, peaks.shape[0] - 1, backend.size(score_space) - 1
        )
        top_indices = backend.topk_indices(
            score_space[tuple(peaks.T)], input_candidates
        )
        peaks = peaks[top_indices]

        return peaks, None


class PeakCallerFast(PeakCaller):
    """
    Subdivides the score space into squares with edge length ``min_distance``
    and determiens maximum value for each. In a second pass, all local maxima
    that are not the local maxima in a ``min_distance`` square centered around them
    are removed.

    """

    def call_peaks(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        minimum_score : float
            Minimum score value to consider.
        min_distance : float
            Minimum distance between maxima.

        Returns
        -------
        NDArray
            Array of peak coordiantes.
        NDArray
            Array of peak details.
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

        peaks = filter_points(coordinates, self.min_distance)

        starts = backend.maximum(peaks - self.min_distance, 0)
        stops = backend.minimum(peaks + self.min_distance, score_space.shape)

        slices_list = [
            tuple(slice(*coord) for coord in zip(start_row, stop_row))
            for start_row, stop_row in zip(starts, stops)
        ]

        scores = score_space[tuple(peaks.T)]
        keep = [
            score >= backend.max(score_space[subvol])
            for subvol, score in zip(slices_list, scores)
        ]
        peaks = peaks[keep,]

        if len(peaks) == 0:
            return peaks, None

        return peaks, None


class PeakCallerRecursiveMasking(PeakCaller):
    """
    Identifies peaks iteratively by selecting the top score and masking
    a region around it.
    """

    def call_peaks(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        minimum_score : float
            Minimum score value to consider.
        min_distance : float
            Minimum distance between maxima.

        Returns
        -------
        NDArray
            Array of peak coordiantes.
        NDArray
            Array of peak details.
        """
        score_box = tuple(self.min_distance for _ in range(score_space.ndim))
        coordinates = []
        while True:
            backend.argmax(score_space)
            max_coord = backend.unravel_index(
                indices=backend.argmax(score_space), shape=score_space.shape
            )
            coordinates.append(max_coord)
            start = backend.maximum(backend.subtract(max_coord, score_box), 0)
            stop = backend.minimum(backend.add(max_coord, score_box), score_space.shape)
            start, stop = backend.astype(start, int), backend.astype(stop, int)
            coords = tuple(slice(*pos) for pos in zip(start, stop))
            score_space[coords] = 0
            if len(coordinates) >= self.number_of_peaks:
                break
        peaks = backend.to_backend_array(coordinates)
        return peaks, None


class PeakCallerScipy(PeakCaller):
    """
    Peak calling using skimage.feature.peak_local_max to compute local maxima.
    """

    def call_peaks(
        self, score_space: NDArray, rotation_matrix: NDArray, **kwargs
    ) -> Tuple[NDArray, NDArray]:
        """
        Call peaks in the score space.

        Parameters
        ----------
        score_space : NDArray
            Data array of scores.
        minimum_score : float
            Minimum score value to consider.
        min_distance : float
            Minimum distance between maxima.

        Returns
        -------
        NDArray
            Array of peak coordiantes.
        NDArray
            Array of peak details.
        """
        peaks = peak_local_max(
            score_space,
            num_peaks=self.number_of_peaks,
            min_distance=self.min_distance,
        )
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
    """

    def __init__(
        self,
        score_space_shape: Tuple[int],
        score_space_dtype: type,
        translation_offset: NDArray = None,
        shared_memory_handler: object = None,
        rotation_space_dtype: type = int,
        use_memmap: bool = False,
        thread_safe: bool = True,
        **kwargs,
    ):
        score_space_shape = tuple(int(x) for x in score_space_shape)
        self.score_space = backend.arr_to_sharedarr(
            backend.zeros(shape=score_space_shape, dtype=score_space_dtype),
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
        self.observed_rotations = Manager().dict() if thread_safe else {}

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
        with self.lock:
            rotation = backend.tobytes(rotation_matrix)
            if rotation not in self.observed_rotations:
                self.observed_rotations[rotation] = len(self.observed_rotations)
            rotation_index = self.observed_rotations[rotation]
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
            indices = score_space > internal_scores
            internal_scores[indices] = score_space[indices]
            internal_rotations[indices] = rotation_index

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
        fourier_shift = kwargs.get(
            "fourier_shift", backend.zeros(len(base_max), dtype=int)
        )
        fourier_shift = backend.to_numpy_array(fourier_shift)

        use_memmap = kwargs.get("use_memmap", False)
        if use_memmap:
            scores_out_filename = generate_tempfile_name()
            rotations_out_filename = generate_tempfile_name()

            scores_out = np.memmap(
                scores_out_filename, mode="w+", shape=base_max, dtype=scores_out_dtype
            )
            rotations_out = np.memmap(
                rotations_out_filename,
                mode="w+",
                shape=base_max,
                dtype=rotations_out_dtype,
            )
        else:
            scores_out = np.zeros(base_max, dtype=scores_out_dtype)
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
            scores_temp, offset, rotations_temp, rotation_mapping = param_stores[i]
            stops = np.add(offset, scores_temp.shape).astype(int)
            indices = tuple(slice(*pos) for pos in zip(offset, stops))

            if np.any(fourier_shift) != 0:
                axis = tuple(range(scores_temp.ndim))
                score_space = np.roll(scores_temp, shift=fourier_shift, axis=axis)
                rotations = np.roll(rotations_temp, shift=fourier_shift, axis=axis)
            else:
                score_space, rotations = scores_temp, rotations_temp

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
                scores_temp._mmap.close()
                rotations_temp._mmap.close()
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
