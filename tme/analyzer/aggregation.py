"""
Implements classes to analyze outputs from exhaustive template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Tuple, List, Dict, Optional

import numpy as np

from .base import AbstractAnalyzer
from ..types import BackendArray, NDArray
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
    offset : tuple of int, optional
        Coordinate origin considered during merging, zero by default.
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
        shape: Tuple[int, ...],
        offset: Optional[Tuple[int, ...]] = None,
        shm_handler: Optional[object] = None,
        use_memmap: bool = False,
        inversion_mapping: bool = False,
        jax_mode: bool = False,
        **kwargs,
    ):
        self._use_memmap = use_memmap
        self._shape = tuple(int(x) for x in shape)
        self._inversion_mapping = inversion_mapping

        self._jax_mode = jax_mode
        if self._jax_mode:
            self._inversion_mapping = False

        if offset is None:
            offset = (0,) * len(self._shape)
        self._offset = tuple(int(x) for x in offset)

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
            - ssum : BackendArray, accumulator for sum of squared scores.
        """
        scores = be.full(self._shape, dtype=be._float, fill_value=-float("inf"))
        rotations = be.full(self._shape, dtype=be._int, fill_value=-1)
        ssum = be.full((1), dtype=be._float, fill_value=0)
        return scores, rotations, {}, ssum

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
            - ssum : BackendArray, accumulator for sum of squared scores.
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
        prev_scores, rotations, rotation_mapping, ssum = state

        rotation_index = len(rotation_mapping)
        rotation_matrix = be.astype(rotation_matrix, be._float)
        if self._inversion_mapping:
            rotation_mapping[rotation_index] = rotation_matrix
        elif self._jax_mode:
            rotation_index = kwargs.get("rotation_index", 0)
        else:
            rotation = be.tobytes(rotation_matrix)
            rotation_index = rotation_mapping.setdefault(rotation, rotation_index)

        if not kwargs.get("_skip_ssum", False):
            ssum = be.add(ssum, be.ssum(scores), out=ssum)
        scores, rotations = be.max_score_over_rotations(
            scores=scores,
            max_scores=prev_scores,
            rotations=rotations,
            rotation_index=rotation_index,
        )
        return scores, rotations, rotation_mapping, ssum

    def correct_background(self, state, mean=0, inv_std=1, **kwargs):
        scores, rotations, rotation_mapping, ssum = state

        scores = be.subtract(scores, mean, out=scores)
        scores = be.multiply(scores, inv_std, out=scores)

        return scores, rotations, rotation_mapping, ssum

    @staticmethod
    def _invert_rmap(rotation_mapping: dict) -> dict:
        """
        Invert dictionary from rotation matrix bytestrings mapping to rotation
        indices ro rotation indices mapping to rotation matrices.
        """
        new_map, ndim = {}, None

        nbytes = be.datatype_bytes(be._float)
        for k, v in rotation_mapping.items():
            dtype = np.float16
            if nbytes == 8:
                dtype = np.float64
            elif nbytes == 4:
                dtype = np.float32
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
            - ssum : BackendArray, accumulator for sum of squared scores.
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
            Final result tuple (scores, offset, rotations, rotation_mapping, ssum).
        """
        scores, rotations, rotation_mapping, ssum = state

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

        n_rotations = max(len(rotation_mapping), 1)
        return (
            scores,
            be.to_numpy_array(self._offset),
            rotations,
            self._invert_rmap(rotation_mapping),
            be.to_numpy_array(ssum) / (scores.size * n_rotations),
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

            scores, offset, rotations, rotation_mapping, ssum = states[i]
            if out_shape is None:
                out_shape = np.zeros(scores.ndim, int)
            out_shape = np.maximum(out_shape, np.add(offset, scores.shape))

            new_param = {}
            for key, value in rotation_mapping.items():
                rotation_bytes = np.asarray(value).tobytes()
                new_param[rotation_bytes] = key
                if rotation_bytes not in new_rotation_mapping:
                    new_rotation_mapping[rotation_bytes] = len(new_rotation_mapping)
            states[i] = (scores, offset, rotations, new_param, ssum)
        out_shape = tuple(int(x) for x in out_shape)
        return new_rotation_mapping, out_shape, states

    @classmethod
    def merge(
        cls,
        results: List[Tuple],
        use_memmap: bool = False,
        output_shape: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> Tuple:
        """
        Merge multiple instances of the current class.

        Parameters
        ----------
        results : list of tuple
            List of instance's internal state created by applying `result`.
        use_memmap : bool
            Whether to memmap results, defaults to False.
        output_shape : bool
            Override internal output shape (for subset matching).
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
        # In this case we do not need to acount for offsets and merging
        if len(results) == 1 and output_shape is None:
            scores, offset, rotations, rotation_mapping, ssum = results[0]
            if use_memmap:
                scores = array_to_memmap(scores)
                rotations = array_to_memmap(rotations)
            return scores, offset, rotations, rotation_mapping, ssum

        # Determine output array shape and create consistent rotation map
        master_rotation_mapping, out_shape, results = cls._harmonize_states(results)
        out_shape = out_shape if output_shape is None else output_shape
        if out_shape is None:
            return None

        f_dtype, i_dtype = results[0][0].dtype, results[0][2].dtype
        if use_memmap:
            scores_fname = generate_tempfile_name()
            rotations_fname = generate_tempfile_name()

            scores_out = np.memmap(
                scores_fname, mode="w+", shape=out_shape, dtype=f_dtype
            )
            scores_out.fill(-float("inf"))
            scores_out.flush()
            rotations_out = np.memmap(
                rotations_fname,
                mode="w+",
                shape=out_shape,
                dtype=i_dtype,
            )
            rotations_out.fill(-1)
            rotations_out.flush()
        else:
            scores_out = np.full(out_shape, fill_value=-float("inf"), dtype=f_dtype)
            rotations_out = np.full(out_shape, fill_value=-1, dtype=i_dtype)

        total_ssum = 0
        for i in range(len(results)):
            if results[i] is None:
                continue

            if use_memmap:
                scores_out = np.memmap(
                    scores_fname,
                    mode="r+",
                    shape=out_shape,
                    dtype=f_dtype,
                )
                rotations_out = np.memmap(
                    rotations_fname,
                    mode="r+",
                    shape=out_shape,
                    dtype=i_dtype,
                )
            scores, offset, rotations, rotation_mapping, ssum = results[i]

            total_ssum = np.add(total_ssum, ssum)
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
                scores_fname, mode="r", shape=out_shape, dtype=f_dtype
            )
            rotations_out = np.memmap(
                rotations_fname,
                mode="r",
                shape=out_shape,
                dtype=i_dtype,
            )

        return (
            scores_out,
            np.zeros(scores_out.ndim, dtype=int),
            rotations_out,
            cls._invert_rmap(master_rotation_mapping),
            total_ssum / len(results),
        )


class MaxScoreOverRotationsConstrained(MaxScoreOverRotations):
    """
    Implements constrained template matching using rejection sampling.

    Parameters
    ----------
    positions : NDArray
        Array of shape (n, d) with n seed point translations.
    rotations : NDArray
        Array of shape (n, d, d) with n seed point rotation matrices.
    cone_angle : float, optional
        Maximum accepted rotational deviation in degrees. Default is unconstrained.
    cone_offset : float, optional
        Accept matches between cone_offset and cone_angle instead of 0 and cone angle.
    reference : tuple of ints
        Reference orientation of the template, defaults to (0,0,1).
    acceptance_radius : tuple of ints, optional
        Translational acceptance radius around seed point in voxels.
    unique_positions : bool, optional
        If True, assumes positions are unique and uses optimized indexing.
        Only valid when acceptance_radius is None. Default is False.
    **kwargs : dict, optional
        Keyword aguments passed to the constructor of :py:class:`MaxScoreOverRotations`.
    """

    def __init__(
        self,
        positions: NDArray,
        rotations: NDArray,
        cone_angle: Optional[float] = None,
        cone_offset: Optional[float] = None,
        reference: Tuple[int, int, int] = (0, 0, 1),
        acceptance_radius: Optional[Tuple[int, int, int]] = None,
        unique_positions: bool = False,
        **kwargs,
    ):
        MaxScoreOverRotations.__init__(self, **kwargs)

        if acceptance_radius is not None:
            acceptance_radius = tuple(int(x) for x in acceptance_radius)

        self._index_grid = None
        norm = np.linalg.norm(reference)
        if norm < 1e-3:
            raise ValueError("reference needs to be non zero (got norm < 1e-3.")

        reference = np.divide(reference, norm)
        self._reference = be.reshape(be.to_backend_array(reference, be._float), (-1,))

        # Map position from real space to shifted score space
        positions = np.subtract(positions, self._offset)
        score_positions, valid_positions = cart_to_score(
            positions=positions,
            fast_shape=self._shape,
            targetshape=kwargs.get("targetshape", None),
            templateshape=kwargs.get("templateshape", None),
            fourier_shift=kwargs.get("fourier_shift", None),
            convolution_mode=kwargs.get("convolution_mode", None),
            convolution_shape=kwargs.get("convolution_shape", None),
        )

        positions = score_positions[valid_positions]
        rotations = rotations[valid_positions]

        # All scores will be rejected in this case. We should think about a
        # unified interface for checking analyzer validity to skip such runs
        if positions.shape[0] == 0:

            def _get_score_mask(*args, **kwargs):
                return 0

            self._get_score_mask = _get_score_mask
            self._get_constraint = _get_score_mask
            return None

        # Omits orientational constraints
        self._n_rotations = rotations.shape[0]
        self._get_constraint = self._get_constraint_null
        if cone_angle is not None and cone_angle > 0:
            # cone_angle = max(min(float(cone_angle), 90), 0)
            # self._cone_cutoff = float(np.tan(np.radians(cone_angle)))
            self._cone_cutoff = float(np.cos(np.radians(cone_angle)))
            self._get_constraint = self._get_constraint_cone

            if cone_offset is not None and cone_offset > 0:
                self._cone_cutoff_lower = float(
                    np.cos(np.radians(cone_angle + cone_offset))
                )
                self._cone_cutoff_upper = float(
                    np.cos(np.radians(max(cone_offset - cone_angle, 0.0)))
                )
                self._get_constraint = self._get_constraint_cone_offset

            # Setup local coordinate systems, this is equivalent to the previous R.T @ e_i
            # self._rotations = be.astype(be.to_backend_array(rotations), be._float16)
            self._rotations = rotations.transpose(0, 2, 1)
            self._rotations = be.to_backend_array(self._rotations) @ self._reference
            self._rotations = be.astype(self._rotations, be._float16)

        # Add translational uncertainty, i.e., seed points are not dense
        positions = be.to_backend_array(positions, be._uint16)
        if acceptance_radius is not None:
            ndim = positions.shape[-1]
            extend = max(acceptance_radius)
            mask_center = tuple(extend for _ in range(ndim))
            mask_shape = tuple(2 * extend + 1 for _ in range(ndim))

            mask = create_mask(
                mask_type="ellipse",
                radius=acceptance_radius,
                shape=mask_shape,
                center=mask_center,
            )
            self._score_mask = be.to_backend_array(mask > 0, bool)

            shape = be.to_backend_array(self._shape)
            starts = be.subtract(positions, extend)
            ret, (n, d), mshape = [], positions.shape, mask_shape
            if starts.shape[0] > 0:
                for i in range(d):
                    indices = starts[:, slice(i, i + 1)] + be.arange(mshape[i])[None]
                    indices = be.mod(indices, shape[i], out=indices)
                    indices = be.astype(indices, be._int)
                    indices_shape = (n, *tuple(1 if k != i else -1 for k in range(d)))
                    ret.append(be.reshape(indices, indices_shape))

            self._index_grid = tuple(ret)
            self._mask_shape = tuple(1 if i != 0 else -1 for i in range(1 + ndim))
            if len(set(acceptance_radius)) != 1:
                n_rotations = rotations.shape[0]

                self._score_mask = be.zeros((n_rotations, *mask_shape), dtype=bool)
                for i in range(n_rotations):
                    mask = create_mask(
                        mask_type="ellipse",
                        radius=acceptance_radius,
                        shape=mask_shape,
                        center=mask_center,
                        orientation=rotations[i].T,
                    )
                    self._score_mask = be.at(
                        self._score_mask,
                        i,
                        be.to_backend_array(mask > 0, bool),
                    )
        else:
            self._score_mask = 1
            self._mask_shape = (-1,)
            self._index_grid = tuple(positions[:, i] for i in range(positions.shape[1]))

            if unique_positions:
                self._get_score_mask = self._get_score_mask_unique

    def __call__(
        self,
        state: Tuple,
        scores: BackendArray,
        rotation_matrix: BackendArray,
        **kwargs,
    ) -> Tuple:
        # Accumulate ssum before masking so the variance estimate
        # reflects the global background, consistent with unconstrained matching.
        prev_scores, rotations, rotation_mapping, ssum = state
        ssum = be.add(ssum, be.ssum(scores), out=ssum)
        state = (prev_scores, rotations, rotation_mapping, ssum)

        mask = self._get_constraint(rotation_matrix)
        mask = self._get_score_mask(mask=mask, scores=scores)

        scores = be.multiply(scores, mask, out=scores)
        return super().__call__(
            state,
            scores=scores,
            rotation_matrix=rotation_matrix,
            _skip_ssum=True,
            **kwargs,
        )

    def _get_constraint_null(self, rotation_matrix: BackendArray) -> BackendArray:
        return be.full((self._n_rotations,), fill_value=1, dtype=bool)

    def _get_constraint_cone(self, rotation_matrix: BackendArray) -> BackendArray:
        template_rot = rotation_matrix.T @ self._reference
        template_rot = be.astype(template_rot, be._float16)
        ret = self._rotations @ template_rot
        return ret >= self._cone_cutoff

    def _get_constraint_cone_offset(
        self, rotation_matrix: BackendArray
    ) -> BackendArray:
        template_rot = rotation_matrix.T @ self._reference
        template_rot = be.astype(template_rot, be._float16)
        ret = self._rotations @ template_rot

        # x, y, z = ret.T
        # return be.sqrt(x**2 + y**2) <= (z * self._cone_cutoff)

        return (ret >= self._cone_cutoff_lower) & (ret <= self._cone_cutoff_upper)

    def _get_score_mask(self, mask: BackendArray, scores: BackendArray, **kwargs):
        score_mask = be.zeros(scores.shape, be._float)

        mask = be.reshape(mask, self._mask_shape)

        # Ideally score mask would be bool but thats not supported by addat
        score_mask = be.addat(score_mask, self._index_grid, self._score_mask * mask)
        return score_mask > 0

    def _get_score_mask_unique(
        self, mask: BackendArray, scores: BackendArray, **kwargs
    ):
        """Fast path for unique positions without acceptance radius."""
        score_mask = be.zeros(scores.shape, dtype=bool)
        return be.at(score_mask, self._index_grid, mask)

    def correct_background(self, state, mean=0, inv_std=1, **kwargs):
        scores, *_ = state

        # Only apply the spatial constrain to the background scores
        score_mask = be.zeros(scores.shape, be._float)
        score_mask = be.addat(score_mask, self._index_grid, self._score_mask) > 0
        return super().correct_background(
            state,
            mean=be.multiply(mean, score_mask),
            inv_std=be.multiply(inv_std, score_mask),
            **kwargs,
        )

    def result(self, state, *args, **kwargs) -> Tuple:
        scores, rotations, rotation_mapping, ssum = state

        if self._index_grid is None:
            state = (scores * 0, rotations, rotation_mapping, 0)
            return super().result(state, *args, **kwargs)

        mask = be.full((self._index_grid[0].shape[0],), fill_value=1) > 0
        mask = self._get_score_mask(mask=mask, scores=scores)

        # ssum is now accumulated before masking in __call__, so it reflects
        # global background variance. Spatial rescaling commented out for now.
        # mask_sum = mask.sum()
        # ssum = be.where(mask_sum > 0, ssum * be.size(scores) / mask_sum, ssum)
        # state = (scores, rotations, rotation_mapping, ssum)
        return super().result(state, *args, **kwargs)


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
        if aggregate_axis is None:
            aggregate_axis = tuple(range(len(shape)))

        shape_reduced = [x for i, x in enumerate(shape) if i not in aggregate_axis]
        shape_reduced.insert(0, n_rotations)

        if offset is None:
            offset = be.zeros(len(shape), be._int)
        offset = [x for i, x in enumerate(offset) if i not in aggregate_axis]
        offset.insert(0, 0)

        super().__init__(
            shape=shape_reduced, shm_handler=shm_handler, offset=offset, **kwargs
        )
        self._aggregate_axis = aggregate_axis

    def init_state(self):
        scores, rotations, rotation_mapping, ssum = super().init_state()
        rotations = be.full(self._shape, dtype=be._int, fill_value=0)
        return scores, rotations, rotation_mapping, ssum

    def __call__(
        self,
        state,
        scores: BackendArray,
        rotation_matrix: BackendArray,
        **kwargs,
    ) -> Tuple:

        prev_scores, rotations, rotation_mapping, ssum = state

        rotation_index = len(rotation_mapping)
        rotation_matrix = be.astype(rotation_matrix, be._float)
        if self._inversion_mapping:
            rotation_mapping[rotation_index] = rotation_matrix
        elif self._jax_mode:
            rotation_index = kwargs.get("rotation_index", 0)
        else:
            rotation = be.tobytes(rotation_matrix)
            rotation_index = rotation_mapping.setdefault(rotation, rotation_index)

        ssum = be.add(ssum, be.ssum(scores), out=ssum)
        scores = be.max(scores, axis=self._aggregate_axis)
        scores = be.maximum(scores, prev_scores[rotation_index])
        prev_scores = be.at(prev_scores, rotation_index, scores)

        return prev_scores, rotations, rotation_mapping, ssum

    def correct_background(self, state, mean=0, inv_std=1, **kwargs):
        warnings.warn(
            "MaxScoreOverTranslations does not support background correction."
        )
        return state
