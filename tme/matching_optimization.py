""" Implements various methods for non-exhaustive template matching
    based on numerical optimization.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Dict
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import (
    differential_evolution,
    LinearConstraint,
    basinhopping,
)
from scipy.ndimage import laplace
from scipy.spatial import KDTree

from .matching_utils import rigid_transform, euler_to_rotationmatrix


class MatchCoordinatesToDensity(ABC):
    """
    A class to template match coordinate sets.

    Parameters
    ----------
    target_coordinates : NDArray
        The coordinates of the target.
    template_coordinates : NDArray
        The coordinates of the template.
    target_weights : NDArray
        The weights of the target.
    template_weights : NDArray
        The weights of the template.
    sampling_rate : NDArray
        The size of the voxel.
    template_mask_coordinates : NDArray, optional
        The coordinates of the template mask. Default is None.
    target_mask_coordinates : NDArray, optional
        The coordinates of the target mask. Default is None.
    **kwargs : dict, optional
        Other keyword arguments.
    """

    def __init__(
        self,
        target_coordinates: NDArray,
        template_coordinates: NDArray,
        target_weights: NDArray,
        template_weights: NDArray,
        sampling_rate: NDArray,
        template_mask_coordinates: NDArray = None,
        target_mask_coordinates: NDArray = None,
        **kwargs,
    ):
        target, _, origin = FitRefinement.array_from_coordinates(
            target_coordinates, target_weights, sampling_rate
        )
        self.target_density = target
        self.target_origin = origin
        self.sampling_rate = sampling_rate

        self.template_weights = template_weights
        self.template_coordinates = template_coordinates
        self.template_coordinates_rotated = np.empty(
            self.template_coordinates.shape, dtype=np.float32
        )

        self.target_mask_density = None
        if target_mask_coordinates is not None:
            target_mask, *_ = FitRefinement.array_from_coordinates(
                coordinates=target_mask_coordinates.astype(np.float32),
                weights=np.ones(target_mask_coordinates.shape[1]),
                shape=self.target_density.shape,
                origin=self.target_origin,
                sampling_rate=self.sampling_rate,
            )
            self.target_mask_density = target_mask

        self.template_mask_coordinates = None
        self.template_mask_coordinates_rotated = None
        if template_mask_coordinates is not None:
            self.template_mask_coordinates = template_mask_coordinates
            self.template_mask_coordinates_rotated = np.empty(
                self.template_mask_coordinates.shape, dtype=np.float32
            )

    def __call__(self, x: NDArray):
        """
        Return the score for a given transformation.

        Parameters
        ----------
        x : NDArray
            The input transformation parameters.

        Returns
        -------
        float
            The negative score from the scoring function.
        """
        translation, rotation = x[:3], x[3:]
        rotation_matrix = euler_to_rotationmatrix(rotation)

        rigid_transform(
            coordinates=self.template_coordinates,
            coordinates_mask=self.template_mask_coordinates,
            rotation_matrix=rotation_matrix,
            translation=translation,
            out=self.template_coordinates_rotated,
            out_mask=self.template_mask_coordinates_rotated,
            use_geometric_center=False,
        )

        mapping = FitRefinement.map_coordinates_to_array(
            coordinates=self.template_coordinates_rotated,
            coordinates_mask=self.template_mask_coordinates_rotated,
            array_origin=self.target_origin,
            array_shape=self.target_density.shape,
            sampling_rate=self.sampling_rate,
        )

        return -self.scoring_function(
            transformed_coordinates=mapping[0],
            transformed_coordinates_mask=mapping[1],
            in_volume=mapping[2],
            in_volume_mask=mapping[3],
        )

    @abstractmethod
    def scoring_function(*args, **kwargs):
        """
        Computes a scoring metric for a given set of coordinates.

        This function is not intended to be called directly, but should rather be
        defined by classes inheriting from :py:class:`MatchCoordinatesToDensity`
        to parse a given file format.
        """


class MatchCoordinatesToCoordinates(ABC):
    """
    A class to template match coordinate sets.

    Parameters
    ----------
    target_coordinates : NDArray
        The coordinates of the target.
    template_coordinates : NDArray
        The coordinates of the template.
    target_weights : NDArray
        The weights of the target.
    template_weights : NDArray
        The weights of the template.
    sampling_rate : NDArray
        The size of the voxel.
    template_mask_coordinates : NDArray, optional
        The coordinates of the template mask. Default is None.
    target_mask_coordinates : NDArray, optional
        The coordinates of the target mask. Default is None.
    **kwargs : dict, optional
        Other keyword arguments.
    """

    def __init__(
        self,
        target_coordinates: NDArray,
        template_coordinates: NDArray,
        target_weights: NDArray,
        template_weights: NDArray,
        template_mask_coordinates: NDArray = None,
        target_mask_coordinates: NDArray = None,
        **kwargs,
    ):
        self.target_weights = target_weights
        self.target_coordinates = target_coordinates

        self.template_weights = template_weights
        self.template_coordinates = template_coordinates
        self.template_coordinates_rotated = np.empty(
            self.template_coordinates.shape, dtype=np.float32
        )
        self.target_mask_coordinates = target_mask_coordinates

        self.template_mask_coordinates = None
        self.template_mask_coordinates_rotated = None
        if template_mask_coordinates is not None:
            self.template_mask_coordinates = template_mask_coordinates
            self.template_mask_coordinates_rotated = np.empty(
                self.template_mask_coordinates.shape, dtype=np.float32
            )

    def __call__(self, x: NDArray):
        """
        Return the score for a given transformation.

        Parameters
        ----------
        x : NDArray
            The input transformation parameters.

        Returns
        -------
        float
            The negative score from the scoring function.
        """
        translation, rotation = x[:3], x[3:]
        rotation_matrix = euler_to_rotationmatrix(rotation)

        rigid_transform(
            coordinates=self.template_coordinates,
            coordinates_mask=self.template_mask_coordinates,
            rotation_matrix=rotation_matrix,
            translation=translation,
            out=self.template_coordinates_rotated,
            out_mask=self.template_mask_coordinates_rotated,
            use_geometric_center=False,
        )

        return -self.scoring_function(
            transformed_coordinates=self.template_coordinates_rotated,
            transformed_coordinates_mask=self.template_mask_coordinates_rotated,
        )

    @abstractmethod
    def scoring_function(*args, **kwargs):
        """
        Computes a scoring metric for a given set of coordinates.

        This function is not intended to be called directly, but should rather be
        defined by classes inheriting from :py:class:`MatchCoordinatesToDensity`
        to parse a given file format.
        """


class CrossCorrelation(MatchCoordinatesToDensity):
    """
    Class representing the Cross-Correlation matching score.

    Cross-Correlation score formula:

    .. math::

        \\text{score} = \\text{target_weights} \\cdot \\text{template_weights}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.denominator = 1

    def scoring_function(
        self,
        transformed_coordinates: NDArray,
        transformed_coordinates_mask: NDArray,
        in_volume: NDArray,
        in_volume_mask: NDArray,
    ) -> float:
        """
        Compute the Cross-Correlation score.

        Parameters
        ----------
        transformed_coordinates : NDArray
            Transformed coordinates.
        transformed_coordinates_mask : NDArray
            Mask for the transformed coordinates.
        in_volume : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target volume.
        in_volume_mask : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target mask volume.
        Returns
        -------
        float
            The Cross-Correlation score.
        """
        score = np.dot(
            self.target_density[tuple(transformed_coordinates[:, in_volume])],
            self.template_weights[in_volume],
        )
        score /= self.denominator
        return score


class LaplaceCrossCorrelation(CrossCorrelation):
    """
    Class representing the Laplace Cross-Correlation matching score.

    The score is computed like CrossCorrelation, but with Laplace filtered
    weights, indicated by the Laplace operator :math:`\\nabla^{2}`.

    .. math::

        \\text{score} = \\nabla^{2} \\text{target_weights} \\cdot
                        \\nabla^{2} \\text{template_weights}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_density = laplace(self.target_density)

        arr, positions, _ = FitRefinement.array_from_coordinates(
            self.template_coordinates, self.template_weights, self.sampling_rate
        )
        self.template_weights = laplace(arr)[tuple(positions)]


class NormalizedCrossCorrelation(CrossCorrelation):
    """
    Class representing the Normalized Cross-Correlation matching score.

    The score is computed by normalizing the dot product of `target_weights` and
    `template_weights` with the product of their norms. This normalization ensures
    the score lies between -1 and 1, providing a measure of similarity that's invariant
    to scale.

    .. math::

        \\text{score} = \\frac{\\text{target_weights} \\cdot \\text{template_weights}}
                        {\\text{max(target_norm} \\times \\text{template_norm, eps)}}

    Where:

    .. math::

        \\text{target_norm} = ||\\text{target_weights}||

    .. math::

        \\text{template_norm} = ||\\text{template_weights}||

    Here, :math:`||.||` denotes the L2 (Euclidean) norm.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        target_norm = np.linalg.norm(self.target_density[self.target_density != 0])
        template_norm = np.linalg.norm(self.template_weights)
        self.denominator = np.fmax(target_norm * template_norm, np.finfo(float).eps)


class NormalizedCrossCorrelationMean(NormalizedCrossCorrelation):
    """
    Class representing the Mean Normalized Cross-Correlation matching score.

    This class extends the Normalized Cross-Correlation by computing the score
    after subtracting the mean from both `target_weights` and `template_weights`.
    This modification enhances the matching score's sensitivity to patterns
    over flat regions in the data.

    Mathematically, the Mean Normalized Cross-Correlation score is computed as:

    .. math::

        \\text{score} = \\frac{(\\text{target_weights} - \\text{mean(target_weights)})
                        \\cdot (\\text{template_weights} -
                        \\text{mean(template_weights)})}
                        {\\text{max(target_norm} \\times \\text{template_norm, eps)}}

    Where:

    .. math::

        \\text{target_norm} = ||\\text{target_weights} - \\text{mean(target_weights)}||

    .. math::

        \\text{template_norm} = ||\\text{template_weights} -
        \\text{mean(template_weights)}||

    Here, :math:`||.||` denotes the L2 (Euclidean) norm, and :math:`\\text{mean(.)}`
    computes the mean of the respective weights.
    """

    def __init__(self, **kwargs):
        print(kwargs["target_weights"].mean())
        kwargs["target_weights"] -= kwargs["target_weights"].mean()
        kwargs["template_weights"] -= kwargs["template_weights"].mean()
        super().__init__(**kwargs)


class MaskedCrossCorrelation(MatchCoordinatesToDensity):
    """
    Class representing the Masked Cross-Correlation matching score.

    The Masked Cross-Correlation computes the similarity between `target_weights`
    and `template_weights` under respective masks. The score is normalized and lies
    between -1 and 1, providing a measure of similarity even in the presence of
    missing or masked data.

    The formula for the Masked Cross-Correlation is:

    .. math::
        \\text{numerator} = \\text{dot}(\\text{target_weights},
                            \\text{template_weights}) -
                            \\frac{\\text{sum}(\\text{mask_target}) \\times
                            \\text{sum}(\\text{mask_template})}
                                  {\\text{mask_overlap}}

    .. math::
        \\text{denominator1} = \\text{sum}(\\text{mask_target}^2) -
                               \\frac{\\text{sum}(\\text{mask_target})^2}
                                     {\\text{mask_overlap}}

    .. math::
        \\text{denominator2} = \\text{sum}(\\text{mask_template}^2) -
                               \\frac{\\text{sum}(\\text{mask_template})^2}
                                     {\\text{mask_overlap}}

    .. math::
        \\text{denominator} = \\sqrt{\\text{denominator1} \\times \\text{denominator2}}

    .. math::
        \\text{score} = \\frac{\\text{numerator}}{\\text{denominator}}
                        \\text{ if denominator } \\neq 0
                        \\text{ else } 0

    Where:

    -   mask_target and mask_template are binary masks for the target_weights
        and template_weights respectively.

    -   mask_overlap represents the number of overlapping non-zero elements in
        the masks.

    References
    ----------
    .. [1]  Masked FFT registration, Dirk Padfield, CVPR 2010 conference
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def scoring_function(
        self,
        transformed_coordinates: NDArray,
        transformed_coordinates_mask: NDArray,
        in_volume: NDArray,
        in_volume_mask: NDArray,
    ) -> float:
        """
        Compute the Masked Cross-Correlation score.

        Parameters
        ----------
        transformed_coordinates : NDArray
            Transformed coordinates.
        transformed_coordinates_mask : NDArray
            Mask for the transformed coordinates.
        in_volume : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target volume.
        in_volume_mask : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target mask volume.

        Returns
        -------
        float
            The Masked Cross-Correlation score.
        """
        mask_overlap = np.sum(
            self.target_mask_density[
                tuple(transformed_coordinates_mask[:, in_volume_mask])
            ],
        )
        mask_overlap = np.fmax(mask_overlap, np.finfo(float).eps)

        mask_target = self.target_density[
            tuple(transformed_coordinates_mask[:, in_volume_mask])
        ]
        denominator1 = np.subtract(
            np.sum(mask_target**2),
            np.divide(np.square(np.sum(mask_target)), mask_overlap),
        )
        mask_template = np.multiply(
            self.template_weights[in_volume],
            self.target_mask_density[tuple(transformed_coordinates[:, in_volume])],
        )
        denominator2 = np.subtract(
            np.sum(mask_template**2),
            np.divide(np.square(np.sum(mask_template)), mask_overlap),
        )

        denominator1 = np.fmax(denominator1, 0.0)
        denominator2 = np.fmax(denominator2, 0.0)
        denominator = np.sqrt(np.multiply(denominator1, denominator2))

        numerator = np.dot(
            self.target_density[tuple(transformed_coordinates[:, in_volume])],
            self.template_weights[in_volume],
        )

        numerator -= np.divide(
            np.multiply(np.sum(mask_target), np.sum(mask_template)), mask_overlap
        )

        if denominator == 0:
            return 0

        score = numerator / denominator
        return score


class PartialLeastSquareDifference(MatchCoordinatesToDensity):
    """
    Class representing the Partial Least Square Difference matching score.

    The Partial Least Square Difference (PLSQ) between the target :math:`f` and the
    template :math:`g` is calculated as:

    .. math::

        \\text{d(f,g)} = \\sum_{i=1}^{n} \\| f(\\mathbf{p}_i) - g(\\mathbf{q}_i) \\|^2

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def scoring_function(
        self,
        transformed_coordinates: NDArray,
        transformed_coordinates_mask: NDArray,
        in_volume: NDArray,
        in_volume_mask: NDArray,
    ) -> float:
        """
        Compute the Partial Least Square Difference score.

        Given the transformed coordinates and their associated mask, this function
        computes the difference between target and template densities.

        Parameters
        ----------
        transformed_coordinates : NDArray
            Transformed coordinates.
        transformed_coordinates_mask : NDArray
            Mask for the transformed coordinates.
        in_volume : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target volume.
        in_volume_mask : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target mask volume.

        Returns
        -------
        float
            The negative of the Partial Least Square Difference score.
        """
        score = np.sum(
            np.square(
                np.subtract(
                    self.target_density[tuple(transformed_coordinates[:, in_volume])],
                    self.template_weights[in_volume],
                )
            )
        )
        score += np.sum(np.square(self.template_weights[np.invert(in_volume)]))

        return -score


class Chamfer(MatchCoordinatesToCoordinates):
    """
    Class representing the Chamfer matching score.

    The Chamfer distance is computed as:

    .. math::

        \\text{d(f,g)} = \\frac{1}{|X|} \\sum_{\\mathbf{f}_i \\in X}
        \\inf_{\\mathbf{g} \\in Y} ||\\mathbf{f}_i - \\mathbf{g}||_2

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.target_tree = KDTree(self.target_coordinates.T)

    def scoring_function(
        self,
        transformed_coordinates: NDArray,
        transformed_coordinates_mask: NDArray,
        **kwargs,
    ) -> float:
        """
        Compute the Chamfer distance score.

        Given the transformed coordinates and their associated mask, this function
        calculates the average distance between the rotated template coordinates
        and the nearest target coordinates.

        Parameters
        ----------
        transformed_coordinates : NDArray
            Transformed coordinates.

        Returns
        -------
        float
            The negative of the Chamfer distance score.
        """
        dist, _ = self.target_tree.query(self.template_coordinates_rotated.T)
        score = np.mean(dist)
        return -score


class MutualInformation(MatchCoordinatesToDensity):
    """
    Class representing the Mutual Information matching score.

    The Mutual Information (MI) score is calculated as:

    .. math::

        \\text{d(f,g)} = \\sum_{f,g} p(f,g) \\log \\frac{p(f,g)}{p(f)p(g)}

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def scoring_function(
        self,
        transformed_coordinates: NDArray,
        transformed_coordinates_mask: NDArray,
        in_volume: NDArray,
        in_volume_mask: NDArray,
    ) -> float:
        """
        Compute the Mutual Information score.

        Given the transformed coordinates and their associated mask, this function
        computes the mutual information between the target and template densities.

        Parameters
        ----------
        transformed_coordinates : NDArray
            Transformed coordinates.
        transformed_coordinates_mask : NDArray
            Mask for the transformed coordinates.
        in_volume : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target volume.
        in_volume_mask : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target mask volume.

        Returns
        -------
        float
            The Mutual Information score.
        """
        p_xy, target, template = np.histogram2d(
            self.target_density[tuple(transformed_coordinates[:, in_volume])],
            self.template_weights[in_volume],
        )
        p_x, p_y = np.sum(p_xy, axis=1), np.sum(p_xy, axis=0)

        p_xy /= p_xy.sum()
        p_x /= p_x.sum()
        p_y /= p_y.sum()

        logprob = np.divide(p_xy, p_x[:, None] * p_y[None, :] + np.finfo(float).eps)
        score = np.nansum(p_xy * logprob)

        return score


class Envelope(MatchCoordinatesToDensity):
    """
    Class representing the Envelope matching score.

    The Envelope score (ENV) is calculated as:

    .. math::

        \\text{d(f,g)} =    \\sum_{\\mathbf{p} \\in P} f'(\\mathbf{p})
                            \\cdot g'(\\mathbf{p})

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012
    """

    def __init__(self, target_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self.target_density = np.where(self.target_density > target_threshold, -1, 1)
        self.target_density_present = np.sum(self.target_density == -1)
        self.target_density_absent = np.sum(self.target_density == 1)
        self.template_weights = np.ones_like(self.template_weights)

    def scoring_function(
        self,
        transformed_coordinates: NDArray,
        transformed_coordinates_mask: NDArray,
        in_volume: NDArray,
        in_volume_mask: NDArray,
    ) -> float:
        """
        Compute the Envelope score.

        Given the transformed coordinates and their associated mask, this function
        computes the envelope score based on target density thresholds.

        Parameters
        ----------
        transformed_coordinates : NDArray
            Transformed coordinates.
        transformed_coordinates_mask : NDArray
            Mask for the transformed coordinates.
        in_volume : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target volume.
        in_volume_mask : NDArray
            Binary mask indicating which ``transformed_coordinates`` are in the
            target mask volume.

        Returns
        -------
        float
            The Envelope score.
        """
        score = self.target_density[tuple(transformed_coordinates[:, in_volume])]
        unassigned_density = self.target_density_present - (score == -1).sum()

        score = score.sum() - unassigned_density - 2 * np.sum(np.invert(in_volume))
        min_score = -self.target_density_present - 2 * self.target_density_absent
        score = (score - 2 * min_score) / (2 * self.target_density_present - min_score)

        return score


class NormalVectorScore(MatchCoordinatesToCoordinates):
    """
    Class representing the Normal Vector matching score.

    The Normal Vector Score (NVS) is calculated as:

    .. math::

        \\text{d(f,g)} = \\frac{1}{N} \\sum_{i=1}^{N}
        \\frac{
            {\\vec{f}_i} \\cdot {\\vec{g}_i}
        }{
            ||\\vec{f}_i|| \\, ||\\vec{g}_i||
        }

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def scoring_function(
        self,
        transformed_coordinates: NDArray,
        transformed_coordinates_mask: NDArray,
        **kwargs,
    ) -> float:
        """
        Compute the Normal Vector Score.

        Given the template and target vectors, this function computes the average
        cosine similarity between the two sets of vectors.

        Parameters
        ----------
        template_vectors : NDArray
            Normal vectors derived from the template.
        target_vectors : NDArray
            Normal vectors derived from the target.

        Returns
        -------
        float
            The Normal Vector Score.
        """
        numerator = np.multiply(transformed_coordinates, self.target_coordinates)
        denominator = np.linalg.norm(transformed_coordinates)
        denominator *= np.linalg.norm(self.target_coordinates)
        score = np.mean(numerator / denominator)
        return score


MATCHING_OPTIMIZATION_REGISTER = {
    "CrossCorrelation": CrossCorrelation,
    "LaplaceCrossCorrelation": LaplaceCrossCorrelation,
    "NormalizedCrossCorrelationMean": NormalizedCrossCorrelationMean,
    "NormalizedCrossCorrelation": NormalizedCrossCorrelation,
    "MaskedCrossCorrelation": MaskedCrossCorrelation,
    "PartialLeastSquareDifference": PartialLeastSquareDifference,
    "Envelope": Envelope,
    "Chamfer": Chamfer,
    "MutualInformation": MutualInformation,
    "NormalVectorScore": NormalVectorScore,
}


def register_matching_optimization(match_name: str, match_class: type):
    """
    Registers a class to be used by :py:class:`FitRefinement`.

    Parameters
    ----------
    match_name : str
        Name of the matching instance.
    match_class : type
        Class pointer.

    Raises
    ------
    ValueError
        If any of the required methods is not defined.
    """
    methods_to_check = ["__init__", "__call__", "scoring_function"]

    for method in methods_to_check:
        if not hasattr(match_class, method):
            raise ValueError(
                f"Method '{method}' is not defined in the provided class or object."
            )
    MATCHING_OPTIMIZATION_REGISTER[match_name] = match_class


class FitRefinement:
    """
    A class to refine the fit between target and template coordinates.

    Notes
    -----
    By default scipy.optimize.differential_evolution or scipy.optimize.basinhopping
    are used which can be unreliable if the initial alignment is very poor. Other
    optimizers can be implemented by subclassing :py:class:`FitRefinement` and
    overwriting the :py:meth:`FitRefinement.refine` function.

    """

    @staticmethod
    def map_coordinates_to_array(
        coordinates: NDArray,
        array_shape: NDArray,
        array_origin: NDArray,
        sampling_rate: NDArray,
        coordinates_mask: NDArray = None,
    ) -> Tuple[NDArray, NDArray]:
        """
        Map coordinates to a volume based on given voxel size and origin.

        Parameters
        ----------
        coordinates : NDArray
            An array representing the coordinates to be mapped [d x N].
        array_shape : NDArray
            The shape of the array to which the coordinates are mapped.
        array_origin : NDArray
            The origin of the array to which the coordinates are mapped.
        sampling_rate : NDArray
            The size of a voxel in the array.
        coordinates_mask : NDArray, optional
            An array representing the mask for the coordinates [d x T].

        Returns
        -------
        tuple
            Returns transformed coordinates, transformed coordinates mask,
            mask for in_volume points, and mask for in_volume points in mask.
        """
        coordinates = coordinates.astype(sampling_rate.dtype)
        np.divide(
            coordinates - array_origin[:, None], sampling_rate[:, None], out=coordinates
        )
        transformed_coordinates = coordinates.astype(int)
        in_volume = np.logical_and(
            transformed_coordinates < np.array(array_shape)[:, None],
            transformed_coordinates >= 0,
        ).min(axis=0)

        transformed_coordinates_mask, in_volume_mask = None, None

        if coordinates_mask is not None:
            coordinates_mask = coordinates_mask.astype(sampling_rate.dtype)
            np.divide(
                coordinates_mask - array_origin[:, None],
                sampling_rate[:, None],
                out=coordinates_mask,
            )
            transformed_coordinates_mask = coordinates_mask.astype(int)
            in_volume_mask = np.logical_and(
                transformed_coordinates_mask < np.array(array_shape)[:, None],
                transformed_coordinates_mask >= 0,
            ).min(axis=0)

        return (
            transformed_coordinates,
            transformed_coordinates_mask,
            in_volume,
            in_volume_mask,
        )

    @staticmethod
    def array_from_coordinates(
        coordinates: NDArray,
        weights: NDArray,
        sampling_rate: NDArray,
        origin: NDArray = None,
        shape: NDArray = None,
    ) -> Tuple[NDArray, NDArray, NDArray]:
        """
        Create a volume from coordinates, using given weights and voxel size.

        Parameters
        ----------
        coordinates : NDArray
            An array representing the coordinates [d x N].
        weights : NDArray
            An array representing the weights for each coordinate [N].
        sampling_rate : NDArray
            The size of a voxel in the volume.
        origin : NDArray, optional
            The origin of the volume.
        shape : NDArray, optional
            The shape of the volume.

        Returns
        -------
        tuple
            Returns the generated volume, positions of coordinates, and origin.
        """
        if origin is None:
            origin = coordinates.min(axis=1)

        positions = np.divide(coordinates - origin[:, None], sampling_rate[:, None])
        positions = positions.astype(int)

        if shape is None:
            shape = positions.max(axis=1) + 1

        arr = np.zeros(shape, dtype=np.float32)
        np.add.at(arr, tuple(positions), weights)
        return arr, positions, origin

    def refine(
        self,
        target_coordinates: NDArray,
        target_weights: NDArray,
        template_coordinates: NDArray,
        template_weights: NDArray,
        sampling_rate: float = None,
        translational_uncertainty: Tuple[float] = None,
        rotational_uncertainty: Tuple[float] = None,
        scoring_class: str = "CrossCorrelation",
        scoring_class_parameters: Dict = dict(),
        local_optimization: bool = True,
        maxiter: int = 100,
    ) -> (NDArray, NDArray):
        """
        Refines the alignment of template coordinates to target coordinates.

        Parameters
        ----------
        target_coordinates : NDArray
            The coordinates of the target.

        target_weights : NDArray
            The weights of the target.

        template_coordinates : NDArray
            The coordinates of the template.

        template_weights : NDArray
            The weights of the template.

        sampling_rate : float, optional
            The size of the voxel. Default is None.

        translational_uncertainty : (float,), optional
            The translational uncertainty. Default is None.

        rotational_uncertainty : (float,), optional
            The rotational uncertainty. Default is None.

        scoring_class : str, optional
            The scoring class to be used. Default is "CC".

        scoring_class_parameters : dict, optional
            The parameters for the scoring class. Default is an empty dictionary.

        local_optimization : bool, optional
            Whether to use local optimization. Default is True.

        maxiter : int, optional
            The maximum number of iterations. Default is 100.

        Returns
        -------
        tuple
            A tuple containing the translation and rotation matrix of the refinement,
            as well as the score of the refinement.

        Raises
        ------
        NotNotImplementedError
            If scoring class is not a part of `MATCHING_OPTIMIZATION_REGISTER`.
            Individual scores can be added via
            :py:meth:`register_matching_optimization`.

        See Also
        --------
        :py:meth:`register_matching_optimization`
        """
        if scoring_class not in MATCHING_OPTIMIZATION_REGISTER:
            raise NotImplementedError(
                f"Parameter score has to be one of "
                f"{', '.join(MATCHING_OPTIMIZATION_REGISTER.keys())}."
            )
        scoring_class = MATCHING_OPTIMIZATION_REGISTER.get(scoring_class, None)

        if sampling_rate is None:
            sampling_rate = np.ones(1)
        sampling_rate = np.repeat(
            sampling_rate, target_coordinates.shape[0] // sampling_rate.size
        )

        score = scoring_class(
            target_coordinates=target_coordinates,
            template_coordinates=template_coordinates,
            target_weights=target_weights,
            template_weights=template_weights,
            sampling_rate=sampling_rate,
            **scoring_class_parameters,
        )

        initial_score = score(np.zeros(6))

        mass_center_target = np.dot(target_coordinates, target_weights)
        mass_center_target /= target_weights.sum()
        mass_center_template = np.dot(template_coordinates, template_weights)
        mass_center_template /= template_weights.sum()

        if translational_uncertainty is None:
            mass_center_difference = np.ceil(
                np.subtract(mass_center_target, mass_center_template)
            ).astype(int)
            target_range = np.ceil(
                np.divide(
                    np.subtract(
                        target_coordinates.max(axis=1), target_coordinates.min(axis=1)
                    ),
                    2,
                )
            ).astype(int)
            translational_uncertainty = tuple(
                (center - start, center + start)
                for center, start in zip(mass_center_difference, target_range)
            )
        if rotational_uncertainty is None:
            rotational_uncertainty = tuple(
                (-90, 90) for _ in range(target_coordinates.shape[0])
            )

        uncertainty = (*translational_uncertainty, *rotational_uncertainty)
        bounds = [bound if bound != (0, 0) else (-1e-9, 1e-9) for bound in uncertainty]
        linear_constraint = LinearConstraint(
            np.eye(len(bounds)), np.min(bounds, axis=1), np.max(bounds, axis=1)
        )

        if local_optimization:
            result = basinhopping(
                x0=np.zeros(6),
                func=score,
                niter=maxiter,
                minimizer_kwargs={"method": "COBYLA", "constraints": linear_constraint},
            )
        else:
            result = differential_evolution(
                func=score,
                bounds=bounds,
                constraints=linear_constraint,
                maxiter=maxiter,
            )

        print(f"Initial score: {-initial_score} - Refined score: {-result.fun}")
        if initial_score < result.fun:
            result.x = np.zeros_like(result.x)
        translation, rotation = result.x[:3], result.x[3:]
        rotation_matrix = euler_to_rotationmatrix(rotation)
        return translation, rotation_matrix, -result.fun


