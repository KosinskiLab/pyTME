"""
Implements classes to analyze outputs from exhaustive template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, List, Dict

import numpy as np

from .base import AbstractAnalyzer
from ..types import BackendArray
from ._utils import cart_to_score
from ..backends import backend as be
from ..matching_utils import (
    create_mask,
    array_to_memmap,
    apply_convolution_mode,
    generate_tempfile_name,
)

__all__ = [
    "MaxScoreOverRotations",
    "MaxScoreOverRotationsConstrained",
    "MaxScoreOverTranslations",
]


class MaxScoreOverRotations(AbstractAnalyzer):
    """
    Determine the rotation maximizing the score over all possible translations.

    Parameters
    ----------
    shape : tuple of int
        Shape of array passed to :py:meth:`MaxScoreOverRotations.__call__`.
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
    inversion_mapping : bool, optional
        Do not use rotation matrix bytestrings for intermediate data handling.
        This is useful for GPU backend where analyzers are not shared across
        devices and every rotation is only observed once. It is generally
        safe to deactivate inversion mapping, but at a cost of performance.

    Examples
    --------
    The following achieves the minimal definition of a :py:class:`MaxScoreOverRotations`
    instance

    >>> import numpy as np
    >>> from tme.analyzer import MaxScoreOverRotations
    >>> analyzer = MaxScoreOverRotations(shape=(50, 50))

    The following simulates a template matching run by creating random data for a range
    of rotations and sending it to ``analyzer`` via its __call__ method

    >>> state = analyzer.init_state()
    >>> for rotation_number in range(10):
    >>>     scores = np.random.rand(50,50)
    >>>     rotation = np.random.rand(scores.ndim, scores.ndim)
    >>>     state = analyzer(state, scores=scores, rotation_matrix=rotation)

    The aggregated scores can be extracted by invoking the result method of
    ``analyzer``

    >>> results = analyzer.result(state)

    The ``results`` tuple contains (1) the maximum scores for each translation,
    (2) an offset which is relevant when merging results from split template matching
    using :py:meth:`MaxScoreOverRotations.merge`, (3) the rotation used to obtain a
    score for a given translation, (4) a dictionary mapping indices used in (2) to
    rotation matrices (2).

    We can extract the ``optimal_score``, ``optimal_translation`` and ``optimal_rotation``
    as follows

    >>> optimal_score = results[0].max()
    >>> optimal_translation = np.where(results[0] == results[0].max())
    >>> optimal_rotation = results[2][optimal_translation]

    The outlined procedure is a trivial method to identify high scoring peaks.
    Alternatively, :py:class:`PeakCaller` offers a range of more elaborate approaches
    that can be used.
    """

    def __init__(
        self,
        shape: Tuple[int],
        offset: BackendArray = None,
        score_threshold: float = 0,
        shm_handler: object = None,
        use_memmap: bool = False,
        inversion_mapping: bool = False,
        jax_mode: bool = False,
        **kwargs,
    ):
        self._use_memmap = use_memmap
        self._score_threshold = score_threshold
        self._shape = tuple(int(x) for x in shape)
        self._inversion_mapping = inversion_mapping

        self._jax_mode = jax_mode
        if self._jax_mode:
            self._inversion_mapping = False

        if offset is None:
            offset = be.zeros(len(self._shape), be._int_dtype)
        self._offset = be.astype(be.to_backend_array(offset), int)

    @property
    def shareable(self):
        return True

    def init_state(self):
        """
        Initialize the analysis state.

        Returns
        -------
        tuple
            Initial state tuple containing (scores, rotations, rotation_mapping) where:
            - scores : BackendArray of shape `self._shape` filled with `score_threshold`.
            - rotations : BackendArray of shape `self._shape` filled with -1.
            - rotation_mapping : dict, empty mapping from rotation bytes to indices.
        """
        scores = be.full(
            shape=self._shape, dtype=be._float_dtype, fill_value=self._score_threshold
        )
        rotations = be.full(self._shape, dtype=be._int_dtype, fill_value=-1)
        return scores, rotations, {}

    def __call__(
        self,
        state: Tuple,
        scores: BackendArray,
        rotation_matrix: BackendArray,
        **kwargs,
    ) -> Tuple:
        """
        Update the parameter store.

        Parameters
        ----------
        state : tuple
            Current state tuple (scores, rotations, rotation_mapping) where:
            - scores : BackendArray, current maximum scores.
            - rotations : BackendArray, current rotation indices.
            - rotation_mapping : dict, mapping from rotation bytes to indices.
        scores : BackendArray
            Array of new scores to update analyzer with.
        rotation_matrix : BackendArray
            Square matrix used to obtain the current rotation.
        Returns
        -------
        tuple
            Updated state tuple (scores, rotations, rotation_mapping).
        """
        # be.tobytes behaviour caused overhead for certain GPU/CUDA combinations
        # If the analyzer is not shared and each rotation is unique, we can
        # use index to rotation mapping and invert prior to merging.
        prev_scores, rotations, rotation_mapping = state

        rotation_index = len(rotation_mapping)
        rotation_matrix = be.astype(rotation_matrix, be._float_dtype)
        if self._inversion_mapping:
            rotation_mapping[rotation_index] = rotation_matrix
        elif self._jax_mode:
            rotation_index = kwargs.get("rotation_index", 0)
        else:
            rotation = be.tobytes(rotation_matrix)
            rotation_index = rotation_mapping.setdefault(rotation, rotation_index)

        scores, rotations = be.max_score_over_rotations(
            scores=scores,
            max_scores=prev_scores,
            rotations=rotations,
            rotation_index=rotation_index,
        )
        return scores, rotations, rotation_mapping

    @staticmethod
    def _invert_rmap(rotation_mapping: dict) -> dict:
        """
        Invert dictionary from rotation matrix bytestrings mapping to rotation
        indices ro rotation indices mapping to rotation matrices.
        """
        new_map, ndim = {}, None
        for k, v in rotation_mapping.items():
            nbytes = be.datatype_bytes(be._float_dtype)
            dtype = np.float32 if nbytes == 4 else np.float16
            rmat = np.frombuffer(k, dtype=dtype)
            if ndim is None:
                ndim = int(np.sqrt(rmat.size))
            new_map[v] = rmat.reshape(ndim, ndim)
        return new_map

    def result(
        self,
        state,
        targetshape: Tuple[int] = None,
        templateshape: Tuple[int] = None,
        convolution_shape: Tuple[int] = None,
        fourier_shift: Tuple[int] = None,
        convolution_mode: str = None,
        **kwargs,
    ) -> Tuple:
        """
        Finalize the analysis result with optional postprocessing.

        Parameters
        ----------
        state : tuple
            Current state tuple (scores, rotations, rotation_mapping) where:
            - scores : BackendArray, current maximum scores.
            - rotations : BackendArray, current rotation indices.
            - rotation_mapping : dict, mapping from rotation indices to matrices.
        targetshape : Tuple[int], optional
            Shape of the target for convolution mode correction.
        templateshape : Tuple[int], optional
            Shape of the template for convolution mode correction.
        convolution_shape : Tuple[int], optional
            Shape used for convolution.
        fourier_shift : Tuple[int], optional.
            Shift to apply for Fourier correction.
        convolution_mode : str, optional
            Convolution mode for padding correction.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        tuple
            Final result tuple (scores, offset, rotations, rotation_mapping).
        """
        scores, rotations, rotation_mapping = state

        # Apply postprocessing if parameters are provided
        if fourier_shift is not None:
            axis = tuple(i for i in range(len(fourier_shift)))
            scores = be.roll(scores, shift=fourier_shift, axis=axis)
            rotations = be.roll(rotations, shift=fourier_shift, axis=axis)

        if convolution_mode is not None:
            convargs = {
                "s1": targetshape,
                "s2": templateshape,
                "convolution_mode": convolution_mode,
                "convolution_shape": convolution_shape,
            }
            scores = apply_convolution_mode(scores, **convargs)
            rotations = apply_convolution_mode(rotations, **convargs)

        scores = be.to_numpy_array(scores)
        rotations = be.to_numpy_array(rotations)
        if self._use_memmap:
            scores = array_to_memmap(scores)
            rotations = array_to_memmap(rotations)

        if self._inversion_mapping:
            rotation_mapping = {be.tobytes(v): k for k, v in rotation_mapping.items()}

        return (
            scores,
            be.to_numpy_array(self._offset),
            rotations,
            self._invert_rmap(rotation_mapping),
        )

    def _harmonize_states(states: List[Tuple]):
        """
        Create consistent reference frame for merging different analyzer
        instances, w.r.t. to rotations and output shape from different
        splits of the target.
        """
        new_rotation_mapping, out_shape = {}, None
        for i in range(len(states)):
            if states[i] is None:
                continue

            scores, offset, rotations, rotation_mapping = states[i]
            if out_shape is None:
                out_shape = np.zeros(scores.ndim, int)
            out_shape = np.maximum(out_shape, np.add(offset, scores.shape))

            new_param = {}
            for key, value in rotation_mapping.items():
                rotation_bytes = be.tobytes(value)
                new_param[rotation_bytes] = key
                if rotation_bytes not in new_rotation_mapping:
                    new_rotation_mapping[rotation_bytes] = len(new_rotation_mapping)
            states[i] = (scores, offset, rotations, new_param)
        out_shape = tuple(int(x) for x in out_shape)
        return new_rotation_mapping, out_shape, states

    @classmethod
    def merge(cls, results: List[Tuple], **kwargs) -> Tuple:
        """
        Merge multiple instances of the current class.

        Parameters
        ----------
        results : list of tuple
            List of instance's internal state created by applying `result`.
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
        if len(results) == 1:
            ret = results[0]
            if use_memmap:
                scores, offset, rotations, rotation_mapping = ret
                scores = array_to_memmap(scores)
                rotations = array_to_memmap(rotations)
                ret = (scores, offset, rotations, rotation_mapping)

            return ret

        # Determine output array shape and create consistent rotation map
        master_rotation_mapping, out_shape, results = cls._harmonize_states(results)
        if out_shape is None:
            return None

        scores_dtype = results[0][0].dtype
        rotations_dtype = results[0][2].dtype
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

        for i in range(len(results)):
            if results[i] is None:
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
            scores, offset, rotations, rotation_mapping = results[i]
            stops = np.add(offset, scores.shape).astype(int)
            indices = tuple(slice(*pos) for pos in zip(offset, stops))

            indices_update = scores > scores_out[indices]
            scores_out[indices][indices_update] = scores[indices_update]

            lookup_table = np.arange(
                len(rotation_mapping) + 1, dtype=rotations_out.dtype
            )
            for key, value in rotation_mapping.items():
                lookup_table[value] = master_rotation_mapping[key]

            updated_rotations = rotations[indices_update]
            if len(updated_rotations):
                rotations_out[indices][indices_update] = lookup_table[updated_rotations]

            if use_memmap:
                scores._mmap.close()
                rotations._mmap.close()
                scores_out.flush()
                rotations_out.flush()
                scores_out, rotations_out = None, None

            results[i] = None
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
            cls._invert_rmap(master_rotation_mapping),
        )


class MaxScoreOverRotationsConstrained(MaxScoreOverRotations):
    """
    Implements constrained template matching using rejection sampling.

    Parameters
    ----------
    cone_angle : float
        Maximum accepted rotational deviation in degrees.
    positions : BackendArray
        Array of shape (n, d) with n seed point translations.
    positions : BackendArray
        Array of shape (n, d, d) with n seed point rotation matrices.
    reference : BackendArray
        Reference orientation of the template, wlog defaults to (0,0,1).
    acceptance_radius : int or tuple of ints
        Translational acceptance radius around seed point in voxels.
    **kwargs : dict, optional
        Keyword aguments passed to the constructor of :py:class:`MaxScoreOverRotations`.
    """

    def __init__(
        self,
        cone_angle: float,
        positions: BackendArray,
        rotations: BackendArray,
        reference: BackendArray = (0, 0, 1),
        acceptance_radius: int = 10,
        **kwargs,
    ):
        MaxScoreOverRotations.__init__(self, **kwargs)

        if not isinstance(acceptance_radius, (int, Tuple)):
            raise ValueError("acceptance_radius needs to be of type int or tuple.")

        if isinstance(acceptance_radius, int):
            acceptance_radius = (
                acceptance_radius,
                acceptance_radius,
                acceptance_radius,
            )
        acceptance_radius = tuple(int(x) for x in acceptance_radius)

        self._cone_angle = float(np.radians(cone_angle))
        self._cone_cutoff = float(np.tan(self._cone_angle))
        self._reference = be.astype(
            be.reshape(be.to_backend_array(reference), (-1,)), be._float_dtype
        )
        positions = be.astype(be.to_backend_array(positions), be._int_dtype)

        ndim = positions.shape[1]
        rotate_mask = len(set(acceptance_radius)) != 1
        extend = max(acceptance_radius)
        mask = create_mask(
            mask_type="ellipse",
            radius=acceptance_radius,
            shape=tuple(2 * extend + 1 for _ in range(ndim)),
            center=tuple(extend for _ in range(ndim)),
        )
        self._score_mask = be.astype(be.to_backend_array(mask), be._float_dtype)

        # Map position from real space to shifted score space
        lower_limit = be.to_backend_array(self._offset)
        positions = be.subtract(positions, lower_limit)
        positions, valid_positions = cart_to_score(
            positions=positions,
            fast_shape=kwargs.get("fast_shape", None),
            targetshape=kwargs.get("targetshape", None),
            templateshape=kwargs.get("templateshape", None),
            fourier_shift=kwargs.get("fourier_shift", None),
            convolution_mode=kwargs.get("convolution_mode", None),
            convolution_shape=kwargs.get("convolution_shape", None),
        )

        self._positions = positions[valid_positions]
        rotations = be.to_backend_array(rotations)[valid_positions]
        ex = be.astype(be.to_backend_array((1, 0, 0)), be._float_dtype)
        ey = be.astype(be.to_backend_array((0, 1, 0)), be._float_dtype)
        ez = be.astype(be.to_backend_array((0, 0, 1)), be._float_dtype)

        self._normals_x = (rotations @ ex[..., None])[..., 0]
        self._normals_y = (rotations @ ey[..., None])[..., 0]
        self._normals_z = (rotations @ ez[..., None])[..., 0]

        # Periodic wrapping could be avoided by padding the target
        shape = be.to_backend_array(self._shape)
        starts = be.subtract(self._positions, extend)
        ret, (n, d), mshape = [], self._positions.shape, self._score_mask.shape
        if starts.shape[0] > 0:
            for i in range(d):
                indices = starts[:, slice(i, i + 1)] + be.arange(mshape[i])[None]
                indices = be.mod(indices, shape[i], out=indices)
                indices_shape = (n, *tuple(1 if k != i else -1 for k in range(d)))
                ret.append(be.reshape(indices, indices_shape))

        self._index_grid = tuple(ret)
        self._mask_shape = tuple(1 if i != 0 else -1 for i in range(1 + ndim))

        if rotate_mask:
            self._score_mask = be.zeros(
                (rotations.shape[0], *self._score_mask.shape), dtype=be._float_dtype
            )
            for i in range(rotations.shape[0]):
                mask = create_mask(
                    mask_type="ellipse",
                    radius=acceptance_radius,
                    shape=tuple(2 * extend + 1 for _ in range(ndim)),
                    center=tuple(extend for _ in range(ndim)),
                    orientation=be.to_numpy_array(rotations[i]),
                )
                self._score_mask[i] = be.astype(
                    be.to_backend_array(mask), be._float_dtype
                )

    def __call__(
        self, state: Tuple, scores: BackendArray, rotation_matrix: BackendArray
    ) -> Tuple:
        mask = self._get_constraint(rotation_matrix)
        mask = self._get_score_mask(mask=mask, scores=scores)

        scores = be.multiply(scores, mask, out=scores)
        return super().__call__(state, scores=scores, rotation_matrix=rotation_matrix)

    def _get_constraint(self, rotation_matrix: BackendArray) -> BackendArray:
        """
        Determine whether the angle between projection of reference w.r.t to
        a given rotation matrix and a set of rotations fall within the set
        cone_angle cutoff.

        Parameters
        ----------
        rotation_matrix : BackendArray
            Rotation matrix with shape (d,d).

        Returns
        -------
        BackerndArray
            Boolean mask of shape (n, )
        """
        template_rot = rotation_matrix @ self._reference

        x = be.sum(be.multiply(self._normals_x, template_rot), axis=1)
        y = be.sum(be.multiply(self._normals_y, template_rot), axis=1)
        z = be.sum(be.multiply(self._normals_z, template_rot), axis=1)

        return be.sqrt(x**2 + y**2) <= (z * self._cone_cutoff)

    def _get_score_mask(self, mask: BackendArray, scores: BackendArray, **kwargs):
        score_mask = be.zeros(scores.shape, scores.dtype)

        if be.sum(mask) == 0:
            return score_mask
        mask = be.reshape(mask, self._mask_shape)

        score_mask = be.addat(score_mask, self._index_grid, self._score_mask * mask)
        return score_mask > 0


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
        self._aggregate_axis = aggregate_axis

    def init_state(self):
        scores = be.full(
            shape=self._shape, dtype=be._float_dtype, fill_value=self._score_threshold
        )
        rotations = be.full(1, dtype=be._int_dtype, fill_value=-1)
        return scores, rotations, {}

    def __call__(
        self, state, scores: BackendArray, rotation_matrix: BackendArray
    ) -> Tuple:
        prev_scores, rotations, rotation_mapping = state

        rotation_index = len(rotation_mapping)
        if self._inversion_mapping:
            rotation_mapping[rotation_index] = rotation_matrix
        else:
            rotation = be.tobytes(rotation_matrix)
            rotation_index = rotation_mapping.setdefault(rotation, rotation_index)
        max_score = be.max(scores, axis=self._aggregate_axis)

        update = prev_scores[rotation_index]
        update = be.maximum(max_score, update, out=update)
        return prev_scores, rotations, rotation_mapping

    @classmethod
    def merge(cls, states: List[Tuple], **kwargs) -> Tuple:
        """
        Merge multiple instances of the current class.

        Parameters
        ----------
        states : list of tuple
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
        if len(states) == 1:
            return states[0]

        # Determine output array shape and create consistent rotation map
        states, master_rotation_mapping, out_shape = cls._harmonize_states(states)
        if out_shape is None:
            return None

        out_shape[0] = len(master_rotation_mapping)
        out_shape = tuple(int(x) for x in out_shape)

        scores_dtype = states[0][0].dtype
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

        for i in range(len(states)):
            if states[i] is None:
                continue

            if use_memmap:
                scores_out = np.memmap(
                    scores_out_filename,
                    mode="r+",
                    shape=out_shape,
                    dtype=scores_dtype,
                )
            scores, offset, rotations, rotation_mapping = states[i]

            outer_table = np.arange(len(rotation_mapping), dtype=int)
            lookup_table = np.array(
                [master_rotation_mapping[key] for key in rotation_mapping.keys()],
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

            states[i], scores = None, None

        if use_memmap:
            scores_out = np.memmap(
                scores_out_filename, mode="r", shape=out_shape, dtype=scores_dtype
            )

        return (
            scores_out,
            np.zeros(scores_out.ndim, dtype=int),
            states[2],
            cls._invert_rmap(master_rotation_mapping),
        )

    def result(self, state: Tuple, **kwargs) -> Tuple:
        return state
