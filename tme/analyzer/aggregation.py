""" Implements classes to analyze outputs from exhaustive template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from contextlib import nullcontext
from multiprocessing import Manager
from typing import Tuple, List, Dict, Generator

import numpy as np

from ..types import BackendArray
from ._utils import cart_to_score
from ..backends import backend as be
from ..matching_utils import (
    create_mask,
    array_to_memmap,
    generate_tempfile_name,
    apply_convolution_mode,
)


__all__ = [
    "MaxScoreOverRotations",
    "MaxScoreOverTranslations",
]


class MaxScoreOverRotations:
    """
    Determine the rotation maximizing the score over all possible translations.

    Parameters
    ----------
    shape : tuple of int
        Shape of array passed to :py:meth:`MaxScoreOverRotations.__call__`.
    scores : BackendArray, optional
        Array mapping translations to scores.
    rotations : BackendArray, optional
        Array mapping translations to rotation indices.
    offset : BackendArray, optional
        Coordinate origin considered during merging, zero by default.
    score_threshold : float, optional
        Minimum score to be considered, zero by default.
    shm_handler : :class:`multiprocessing.managers.SharedMemoryManager`, optional
        Shared memory manager, defaults to memory not being shared.
    use_memmap : bool, optional
        Memmap internal arrays, False by default.
    thread_safe: bool, optional
        Allow class to be modified by multiple processes, True by default.
    only_unique_rotations : bool, optional
        Whether each rotation will be shown only once, False by default.

    Attributes
    ----------
    scores : BackendArray
        Mapping of translations to scores.
    rotations : BackendArray
        Mmapping of translations to rotation indices.
    rotation_mapping : Dict
        Mapping of rotations to rotation indices.
    offset : BackendArray, optional
        Coordinate origin considered during merging, zero by default

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

    The aggregated scores can be extracted by invoking the __iter__ method of
    ``analyzer``

    >>> results = tuple(analyzer)

    The ``results`` tuple contains (1) the maximum scores for each translation,
    (2) an offset which is relevant when merging results from split template matching
    using :py:meth:`MaxScoreOverRotations.merge`, (3) the rotation used to obtain a
    score for a given translation, (4) a dictionary mapping rotation matrices to the
    indices used in (2).

    We can extract the ``optimal_score``, ``optimal_translation`` and ``optimal_rotation``
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
        shape: Tuple[int],
        scores: BackendArray = None,
        rotations: BackendArray = None,
        offset: BackendArray = None,
        score_threshold: float = 0,
        shm_handler: object = None,
        use_memmap: bool = False,
        thread_safe: bool = True,
        only_unique_rotations: bool = False,
        **kwargs,
    ):
        self._shape = tuple(int(x) for x in shape)

        self.scores = scores
        if self.scores is None:
            self.scores = be.full(
                shape=self._shape, dtype=be._float_dtype, fill_value=score_threshold
            )
        self.rotations = rotations
        if self.rotations is None:
            self.rotations = be.full(self._shape, dtype=be._int_dtype, fill_value=-1)

        self.scores = be.to_sharedarr(self.scores, shm_handler)
        self.rotations = be.to_sharedarr(self.rotations, shm_handler)

        if offset is None:
            offset = be.zeros(len(self._shape), be._int_dtype)
        self.offset = be.astype(be.to_backend_array(offset), int)

        self._use_memmap = use_memmap
        self._lock = Manager().Lock() if thread_safe else nullcontext()
        self._lock_is_nullcontext = isinstance(self.scores, type(be.zeros((1))))
        self._inversion_mapping = self._lock_is_nullcontext and only_unique_rotations
        self.rotation_mapping = Manager().dict() if thread_safe else {}

    def _postprocess(
        self,
        targetshape: Tuple[int],
        templateshape: Tuple[int],
        convolution_shape: Tuple[int],
        fourier_shift: Tuple[int] = None,
        convolution_mode: str = None,
        shm_handler=None,
        **kwargs,
    ) -> "MaxScoreOverRotations":
        """Correct padding to Fourier shape and convolution mode."""
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
            "convolution_shape": convolution_shape,
        }
        if convolution_mode is not None:
            scores = apply_convolution_mode(scores, **convargs)
            rotations = apply_convolution_mode(rotations, **convargs)

        self._shape, self.scores, self.rotations = scores.shape, scores, rotations
        if shm_handler is not None:
            self.scores = be.to_sharedarr(scores, shm_handler)
            self.rotations = be.to_sharedarr(rotations, shm_handler)
        return self

    def __iter__(self) -> Generator:
        scores = be.from_sharedarr(self.scores)
        rotations = be.from_sharedarr(self.rotations)

        scores = be.to_numpy_array(scores)
        rotations = be.to_numpy_array(rotations)
        if self._use_memmap:
            scores = array_to_memmap(scores)
            rotations = array_to_memmap(rotations)
        else:
            if type(self.scores) is not type(scores):
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
        Update the parameter store.

        Parameters
        ----------
        scores : BackendArray
            Array of scores.
        rotation_matrix : BackendArray
            Square matrix describing the current rotation.
        """
        # be.tobytes behaviour caused overhead for certain GPU/CUDA combinations
        # If the analyzer is not shared and each rotation is unique, we can
        # use index to rotation mapping and invert prior to merging.
        if self._lock_is_nullcontext:
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
        with self._lock:
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
    def merge(cls, param_stores: List[Tuple], **kwargs) -> Tuple:
        """
        Merge multiple instances of the current class.

        Parameters
        ----------
        param_stores : list of tuple
            List of instance's internal state created by applying `tuple(instance)`.
        **kwargs : dict, optional
            Optional keyword arguments.

        Returns
        -------
        NDArray
            Maximum score of each translation over all observed rotations.
        NDArray
            Translation offset, zero by default.
        NDArray
            Mapping between translations and rotation indices.
        Dict
            Mapping between rotations and rotation indices.
        """
        use_memmap = kwargs.get("use_memmap", False)
        if len(param_stores) == 1:
            ret = param_stores[0]
            if use_memmap:
                scores, offset, rotations, rotation_mapping = ret
                scores = array_to_memmap(scores)
                rotations = array_to_memmap(rotations)
                ret = (scores, offset, rotations, rotation_mapping)

            return ret

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
    def is_shareable(self) -> bool:
        """Boolean indicating whether class instance can be shared across processes."""
        return True


class MaxScoreOverTranslations(MaxScoreOverRotations):
    """
    Determine the translation maximizing the score over all possible rotations.

    Parameters
    ----------
    shape : tuple of int
        Shape of array passed to :py:meth:`MaxScoreOverTranslations.__call__`.
    n_rotations : int
        Number of rotations to aggregate over.
    aggregate_axis : tuple of int, optional
        Array axis to aggregate over, None by default.
    shm_handler : :class:`multiprocessing.managers.SharedMemoryManager`, optional
        Shared memory manager, defaults to memory not being shared.
    **kwargs: dict, optional
        Keyword arguments passed to the constructor of the parent class.
    """

    def __init__(
        self,
        shape: Tuple[int],
        n_rotations: int,
        aggregate_axis: Tuple[int] = None,
        shm_handler: object = None,
        offset: Tuple[int] = None,
        **kwargs: Dict,
    ):
        shape_reduced = [x for i, x in enumerate(shape) if i not in aggregate_axis]
        shape_reduced.insert(0, n_rotations)

        if offset is None:
            offset = be.zeros(len(shape), be._int_dtype)
        offset = [x for i, x in enumerate(offset) if i not in aggregate_axis]
        offset.insert(0, 0)

        super().__init__(
            shape=shape_reduced, shm_handler=shm_handler, offset=offset, **kwargs
        )

        self.rotations = be.full(1, dtype=be._int_dtype, fill_value=-1)
        self.rotations = be.to_sharedarr(self.rotations, shm_handler)
        self._aggregate_axis = aggregate_axis

    def __call__(self, scores: BackendArray, rotation_matrix: BackendArray):
        if self._lock_is_nullcontext:
            rotation_index = len(self.rotation_mapping)
            if self._inversion_mapping:
                self.rotation_mapping[rotation_index] = rotation_matrix
            else:
                rotation = be.tobytes(rotation_matrix)
                rotation_index = self.rotation_mapping.setdefault(
                    rotation, rotation_index
                )
            max_score = be.max(scores, axis=self._aggregate_axis)
            self.scores[rotation_index] = max_score
            return None

        rotation = be.tobytes(rotation_matrix)
        with self._lock:
            rotation_index = self.rotation_mapping.setdefault(
                rotation, len(self.rotation_mapping)
            )
            internal_scores = be.from_sharedarr(self.scores)
            max_score = be.max(scores, axis=self._aggregate_axis)
            internal_scores[rotation_index] = max_score
            return None

    @classmethod
    def merge(cls, param_stores: List[Tuple], **kwargs) -> Tuple:
        """
        Merge multiple instances of the current class.

        Parameters
        ----------
        param_stores : list of tuple
            List of instance's internal state created by applying `tuple(instance)`.
        **kwargs : dict, optional
            Optional keyword arguments.

        Returns
        -------
        NDArray
            Maximum score of each rotation over all observed translations.
        NDArray
            Translation offset, zero by default.
        NDArray
            Mapping between translations and rotation indices.
        Dict
            Mapping between rotations and rotation indices.
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
                scores_dtype, rotations_out = scores.dtype, rotations
            out_shape = np.maximum(out_shape, np.add(offset, scores.shape))

            for key, value in rotation_mapping.items():
                if key not in new_rotation_mapping:
                    new_rotation_mapping[key] = len(new_rotation_mapping)

        if out_shape is None:
            return None

        out_shape[0] = len(new_rotation_mapping)
        out_shape = tuple(int(x) for x in out_shape)

        use_memmap = kwargs.get("use_memmap", False)
        if use_memmap:
            scores_out_filename = generate_tempfile_name()
            scores_out = np.memmap(
                scores_out_filename, mode="w+", shape=out_shape, dtype=scores_dtype
            )
            scores_out.fill(kwargs.get("score_threshold", 0))
            scores_out.flush()
        else:
            scores_out = np.full(
                out_shape,
                fill_value=kwargs.get("score_threshold", 0),
                dtype=scores_dtype,
            )

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
            scores, offset, rotations, rotation_mapping = param_stores[i]

            outer_table = np.arange(len(rotation_mapping), dtype=int)
            lookup_table = np.array(
                [new_rotation_mapping[key] for key in rotation_mapping.keys()],
                dtype=int,
            )

            stops = np.add(offset, scores.shape).astype(int)
            indices = [slice(*pos) for pos in zip(offset[1:], stops[1:])]
            indices.insert(0, lookup_table)
            indices = tuple(indices)

            scores_out[indices] = np.maximum(scores_out[indices], scores[outer_table])

            if use_memmap:
                scores._mmap.close()
                scores_out.flush()
                scores_out = None

            param_stores[i], scores = None, None

        if use_memmap:
            scores_out = np.memmap(
                scores_out_filename, mode="r", shape=out_shape, dtype=scores_dtype
            )

        return (
            scores_out,
            np.zeros(scores_out.ndim, dtype=int),
            rotations_out,
            new_rotation_mapping,
        )

    def _postprocess(self, **kwargs):
        return self
