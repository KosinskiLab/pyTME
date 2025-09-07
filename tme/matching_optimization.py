"""
Implements methods for non-exhaustive template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Tuple, List, Dict
from abc import ABC, abstractmethod

import numpy as np
from scipy.ndimage import laplace, map_coordinates, sobel
from scipy.optimize import (
    minimize,
    basinhopping,
    LinearConstraint,
    differential_evolution,
)

from .backends import backend as be
from .types import ArrayLike, NDArray
from .matching_data import MatchingData
from .rotations import euler_to_rotationmatrix
from .matching_utils import _rigid_transform, standardize


def _format_rigid_transform(x: Tuple[float]) -> Tuple[ArrayLike, ArrayLike]:
    """
    Returns a formated rigid transform definition.

    Parameters
    ----------
    x : tuple of float
        Even-length tuple where the first half represents translations and the
        second half Euler angles in zyz convention for each dimension.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Translation of length [d, ] and rotation matrix with dimension [d x d].
    """
    split = len(x) // 2
    translation, angles = x[:split], x[split:]

    translation = be.to_backend_array(translation)
    rotation_matrix = euler_to_rotationmatrix(be.to_numpy_array(angles), "ZYZ")
    return translation, be.to_backend_array(rotation_matrix)


class _MatchDensityToDensity(ABC):
    """
    Parameters
    ----------
    target : array_like
        The target density array.
    template : array_like
        The template density array.
    template_mask : array_like, optional
        Mask array for the template density.
    target_mask : array_like, optional
        Mask array for the target density.
    pad_target_edges : bool, optional
        Whether to pad the edges of the target density array. Default is False.
    pad_fourier : bool, optional
        Whether to pad the Fourier transform of the target and template densities.
    rotate_mask : bool, optional
        Whether to rotate the mask arrays along with the densities. Default is True.
    interpolation_order : int, optional
        The interpolation order for rigid transforms. Default is 1.
    negate_score : bool, optional
        Whether the final score should be multiplied by negative one. Default is True.
    **kwargs : Dict, optional
        Keyword arguments propagated to downstream functions.
    """

    def __init__(
        self,
        target: ArrayLike,
        template: ArrayLike,
        template_mask: ArrayLike = None,
        target_mask: ArrayLike = None,
        pad_target_edges: bool = False,
        pad_fourier: bool = False,
        rotate_mask: bool = True,
        interpolation_order: int = 1,
        negate_score: bool = True,
        **kwargs: Dict,
    ):
        self.eps = be.eps(target.dtype)
        self.rotate_mask = rotate_mask
        self.interpolation_order = interpolation_order

        matching_data = MatchingData(target=target, template=template)
        if template_mask is not None:
            matching_data.template_mask = template_mask
        if target_mask is not None:
            matching_data.target_mask = target_mask

        self.target, self.target_mask = matching_data.target, matching_data.target_mask

        self.template = matching_data._template
        self.template_rot = be.zeros(template.shape, be._float_dtype)

        self.template_mask, self.template_mask_rot = 1, 1
        rotate_mask = False if matching_data._template_mask is None else rotate_mask
        if matching_data.template_mask is not None:
            self.template_mask = matching_data._template_mask
            self.template_mask_rot = be.topleft_pad(
                matching_data._template_mask, self.template_mask.shape
            )

        self.template_slices = tuple(slice(None) for _ in self.template.shape)
        self.target_slices = tuple(slice(0, x) for x in self.template.shape)

        self.score_sign = -1 if negate_score else 1

        if hasattr(self, "_post_init"):
            self._post_init(**kwargs)

    @staticmethod
    def _interpolate(data, positions, order: int = 1, out=None):
        return map_coordinates(
            data, positions, order=order, mode="constant", output=out
        )

    def score_translation(self, x: Tuple[float]) -> float:
        """
        Computes the score after a given translation.

        Parameters
        ----------
        x : tuple of float
            Tuple representing the translation transformation in each dimension.

        Returns
        -------
        float
            The score obtained for the translation transformation.
        """
        return self.score((*x, *[0 for _ in range(len(x))]))

    def score_angles(self, x: Tuple[float]) -> float:
        """
        Computes the score after a given rotation.

        Parameters
        ----------
        x : tuple of float
            Tuple of Euler angles in zyz convention for each dimension.

        Returns
        -------
        float
            The score obtained for the rotation transformation.
        """
        return self.score((*[0 for _ in range(len(x))], *x))

    def score(self, x: Tuple[float]) -> float:
        """
        Compute the matching score for the given transformation parameters.

        Parameters
        ----------
        x : tuple of float
            Even-length tuple where the first half represents translations and the
            second half Euler angles in zyz convention for each dimension.

        Returns
        -------
        float
            The matching score obtained for the transformation.
        """
        translation, rotation_matrix = _format_rigid_transform(x)
        self.template_rot.fill(0)

        voxel_translation = be.astype(translation, be._int_dtype)
        subvoxel_translation = be.subtract(translation, voxel_translation)

        center = be.astype(be.divide(self.template.shape, 2), be._int_dtype)
        right_pad = be.subtract(self.template.shape, center)

        translated_center = be.add(voxel_translation, center)

        target_starts = be.subtract(translated_center, center)
        target_stops = be.add(translated_center, right_pad)

        template_starts = be.subtract(be.maximum(target_starts, 0), target_starts)
        template_stops = be.subtract(
            target_stops, be.minimum(target_stops, self.target.shape)
        )
        template_stops = be.subtract(self.template.shape, template_stops)

        target_starts = be.maximum(target_starts, 0)
        target_stops = be.minimum(target_stops, self.target.shape)

        cand_start, cand_stop = template_starts.astype(int), template_stops.astype(int)
        obs_start, obs_stop = target_starts.astype(int), target_stops.astype(int)

        self.template_slices = tuple(slice(s, e) for s, e in zip(cand_start, cand_stop))
        self.target_slices = tuple(slice(s, e) for s, e in zip(obs_start, obs_stop))

        kw_dict = {
            "arr": self.template,
            "rotation_matrix": rotation_matrix,
            "translation": subvoxel_translation,
            "out": self.template_rot,
            "order": self.interpolation_order,
            "use_geometric_center": True,
        }
        if self.rotate_mask:
            self.template_mask_rot.fill(0)
            kw_dict["arr_mask"] = self.template_mask
            kw_dict["out_mask"] = self.template_mask_rot
        be.rigid_transform(**kw_dict)

        return self()

    @abstractmethod
    def __call__(self) -> float:
        """Returns the score of the current configuration."""


class _MatchCoordinatesToDensity(_MatchDensityToDensity):
    """
    Parameters
    ----------
    target : NDArray
        A d-dimensional target to match the template coordinate set to.
    template_coordinates : NDArray
        Template coordinate array with shape (d,n).
    template_weights : NDArray
        Template weight array with shape (n,).
    template_mask_coordinates : NDArray, optional
        Template mask coordinates with shape (d,n).
    target_mask : NDArray, optional
        A d-dimensional mask to be applied to the target.
    negate_score : bool, optional
        Whether the final score should be multiplied by negative one. Default is True.
    return_gradient : bool, optional
        Invoking __call_ returns a tuple of score and parameter gradient. Default is False.
    **kwargs : Dict, optional
        Keyword arguments propagated to downstream functions.
    """

    def __init__(
        self,
        target: NDArray,
        template_coordinates: NDArray,
        template_weights: NDArray,
        template_mask_coordinates: NDArray = None,
        target_mask: NDArray = None,
        negate_score: bool = True,
        return_gradient: bool = False,
        interpolation_order: int = 1,
        **kwargs: Dict,
    ):
        self.target = target.astype(np.float32)
        self.target_mask = None
        if target_mask is not None:
            self.target_mask = target_mask.astype(np.float32)

        self.eps = be.eps(self.target.dtype)

        self.target_grad = np.stack(
            [sobel(self.target, axis=i) for i in range(self.target.ndim)]
        )

        self.n_points = template_coordinates.shape[1]
        self.template = template_coordinates.astype(np.float32)
        self.template_rotated = np.zeros_like(self.template)
        self.template_weights = template_weights.astype(np.float32)
        self.template_center = np.mean(self.template, axis=1)[:, None]

        self.template_mask, self.template_mask_rotated = None, None
        if template_mask_coordinates is not None:
            self.template_mask = template_mask_coordinates.astype(np.float32)
            self.template_mask_rotated = np.empty_like(self.template_mask)

        self.denominator = 1
        self.score_sign = -1 if negate_score else 1
        self.interpolation_order = interpolation_order

        self._target_values = self._interpolate(
            self.target, self.template, order=self.interpolation_order
        )

        if return_gradient and not hasattr(self, "grad"):
            raise NotImplementedError(f"{type(self)} does not have grad method.")
        self.return_gradient = return_gradient

    def score(self, x: Tuple[float]):
        """
        Compute the matching score for the given transformation parameters.

        Parameters
        ----------
        x : tuple of float
            Even-length tuple where the first half represents translations and the
            second half Euler angles in zyz convention for each dimension.

        Returns
        -------
        float
            The matching score obtained for the transformation.
        """
        translation, rotation_matrix = _format_rigid_transform(x)

        _rigid_transform(
            coordinates=self.template,
            coordinates_mask=self.template_mask,
            rotation_matrix=rotation_matrix,
            translation=translation,
            out=self.template_rotated,
            out_mask=self.template_mask_rotated,
            use_geometric_center=False,
        )

        self._target_values = self._interpolate(
            self.target, self.template_rotated, order=self.interpolation_order
        )

        score = self()
        if not self.return_gradient:
            return score

        return score, self.grad()

    def _interpolate_gradient(self, positions):
        ret = be.zeros(positions.shape, dtype=positions.dtype)

        for k in range(self.target_grad.shape[0]):
            ret[k, :] = self._interpolate(
                self.target_grad[k], positions, order=self.interpolation_order
            )

        return ret

    @staticmethod
    def _torques(positions, center, gradients):
        positions_center = (positions - center).T
        return be.cross(positions_center, gradients.T).T


class _MatchCoordinatesToCoordinates(_MatchDensityToDensity):
    """
    Parameters
    ----------
    target_coordinates : NDArray
        The coordinates of the target with shape [d x N].
    template_coordinates : NDArray
        The coordinates of the template with shape [d x T].
    target_weights : NDArray
        The weights of the target with shape [N].
    template_weights : NDArray
        The weights of the template with shape [T].
    template_mask_coordinates : NDArray, optional
        The coordinates of the template mask with shape [d x T]. Default is None.
    target_mask_coordinates : NDArray, optional
        The coordinates of the target mask with shape [d X N]. Default is None.
    negate_score : bool, optional
        Whether the final score should be multiplied by negative one. Default is True.
    **kwargs : Dict, optional
        Keyword arguments propagated to downstream functions.
    """

    def __init__(
        self,
        target_coordinates: NDArray,
        template_coordinates: NDArray,
        target_weights: NDArray,
        template_weights: NDArray,
        template_mask_coordinates: NDArray = None,
        target_mask_coordinates: NDArray = None,
        negate_score: bool = True,
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
        self.score_sign = -1 if negate_score else 1

        if hasattr(self, "_post_init"):
            self._post_init(**kwargs)

    def score(self, x: Tuple[float]) -> float:
        """
        Compute the matching score for the given transformation parameters.

        Parameters
        ----------
        x : tuple of float
            Even-length tuple where the first half represents translations and the
            second half Euler angles in zyz convention for each dimension.

        Returns
        -------
        float
            The matching score obtained for the transformation.
        """
        translation, rotation_matrix = _format_rigid_transform(x)

        _rigid_transform(
            coordinates=self.template_coordinates,
            coordinates_mask=self.template_mask_coordinates,
            rotation_matrix=rotation_matrix,
            translation=translation,
            out=self.template_coordinates_rotated,
            out_mask=self.template_mask_coordinates_rotated,
            use_geometric_center=False,
        )

        return self(
            transformed_coordinates=self.template_coordinates_rotated,
            transformed_coordinates_mask=self.template_mask_coordinates_rotated,
        )


class FLC(_MatchDensityToDensity):
    """
    Computes a normalized cross-correlation score of a target f a template g
    and a mask m:

    .. math::

        \\frac{CC(f, \\frac{g*m - \\overline{g*m}}{\\sigma_{g*m}})}
        {N_m * \\sqrt{
            \\frac{CC(f^2, m)}{N_m} - (\\frac{CC(f, m)}{N_m})^2}
        }

    Where:

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*)

    and Nm is the number of voxels within the template mask m.
    """

    __doc__ += _MatchDensityToDensity.__doc__

    def _post_init(self, **kwargs: Dict):
        if self.target_mask is not None:
            be.multiply(self.target, self.target_mask, out=self.target)

        self.target_square = be.square(self.target)

        standardize(
            template=self.template,
            mask=self.template_mask,
            n_observations=be.sum(self.template_mask),
        )

    def __call__(self) -> float:
        """Returns the score of the current configuration."""
        n_obs = be.sum(self.template_mask_rot)

        standardize(
            template=self.template_rot,
            mask=self.template_mask_rot,
            n_observations=n_obs,
        )
        overlap = be.sum(
            be.multiply(
                self.template_rot[self.template_slices], self.target[self.target_slices]
            )
        )

        mask_rot = self.template_mask_rot[self.template_slices]
        exp_sq = be.sum(self.target_square[self.target_slices] * mask_rot) / n_obs
        sq_exp = be.square(be.sum(self.target[self.target_slices] * mask_rot) / n_obs)

        denominator = be.maximum(be.subtract(exp_sq, sq_exp), 0.0)
        denominator = be.sqrt(denominator)
        if denominator < self.eps:
            return 0

        score = be.divide(overlap, denominator * n_obs) * self.score_sign
        return score


class CrossCorrelation(_MatchCoordinatesToDensity):
    """
    Computes the Cross-Correlation score as:

    .. math::

        \\text{score} = \\text{target_weights} \\cdot \\text{template_weights}
    """

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __call__(self) -> float:
        """Returns the score of the current configuration."""
        score = be.dot(self._target_values, self.template_weights)
        score /= self.denominator * self.score_sign
        return score

    def grad(self):
        """
        Calculate the gradient of the cost function w.r.t. translation and rotation.

        .. math::

            \\nabla f = -\\frac{1}{N} \\begin{bmatrix}
            \\sum_i w_i \\nabla v(x_i) \\\\
            \\sum_i w_i (r_i \\times \\nabla v(x_i))
            \\end{bmatrix}

        where :math:`N` is the number of points, :math:`w_i` are weights,
        :math:`x_i` are rotated template positions, and :math:`r_i` are
        positions relative to the template center.

        Returns
        -------
        np.ndarray
            Negative gradient of the cost function: [dx, dy, dz, dRx, dRy, dRz].

        """
        grad = self._interpolate_gradient(positions=self.template_rotated)
        torque = self._torques(
            positions=self.template_rotated, gradients=grad, center=self.template_center
        )

        translation_grad = be.sum(grad * self.template_weights, axis=1)
        torque_grad = be.sum(torque * self.template_weights, axis=1)

        # <u, dv/dx> / <u, r x dv/dx>
        total_grad = be.concatenate([translation_grad, torque_grad])
        total_grad = be.divide(total_grad, self.n_points, out=total_grad)
        return -total_grad


class LaplaceCrossCorrelation(CrossCorrelation):
    """
    Uses the same formalism as :py:class:`CrossCorrelation` but with Laplace
    filtered weights (:math:`\\nabla^{2}`):

    .. math::

        \\text{score} = \\nabla^{2} \\text{target_weights} \\cdot
                        \\nabla^{2} \\text{template_weights}
    """

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __init__(self, **kwargs):
        kwargs["target"] = laplace(kwargs["target"])

        coordinates = kwargs["template_coordinates"]
        origin = coordinates.min(axis=1)
        positions = (coordinates - origin[:, None]).astype(int)
        shape = positions.max(axis=1) + 1
        arr = np.zeros(shape, dtype=np.float32)
        np.add.at(arr, tuple(positions), kwargs["template_weights"])

        kwargs["template_weights"] = laplace(arr)[tuple(positions)]
        super().__init__(**kwargs)


class NormalizedCrossCorrelation(CrossCorrelation):
    """
    Computes a normalized version of the :py:class:`CrossCorrelation` score based
    on the dot product of `target_weights` and `template_weights`, in order to
    reduce bias to regions of high local energy.

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

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __call__(self) -> float:
        denominator = be.multiply(
            np.linalg.norm(self.template_weights), np.linalg.norm(self._target_values)
        )

        if denominator <= 0:
            return 0.0

        self.denominator = denominator
        return super().__call__()

    def grad(self):
        """
        Calculate the normalized gradient of the cost function w.r.t. translation and rotation.

        .. math::

            \\nabla f = -\\frac{1}{N|w||v|^3} \\begin{bmatrix}
            (\\sum_i w_i \\nabla v(x_i))|v|^2 - (\\sum_i v(x_i)
                \\nabla v(x_i))(w \\cdot v) \\\\
            (\\sum_i w_i (r_i \\times \\nabla v(x_i)))|v|^2 - (\\sum_i v(x_i)
            (r_i \\times \\nabla v(x_i)))(w \\cdot v)
            \\end{bmatrix}

        where :math:`N` is the number of points, :math:`w` are weights,
        :math:`v` are target values, :math:`x_i` are rotated template positions,
        and :math:`r_i` are positions relative to the template center.

        Returns
        -------
        np.ndarray
            Negative normalized gradient: [dx, dy, dz, dRx, dRy, dRz].

        """
        grad = self._interpolate_gradient(positions=self.template_rotated)
        torque = self._torques(
            positions=self.template_rotated, gradients=grad, center=self.template_center
        )

        norm = be.multiply(
            be.power(be.sqrt(be.sum(be.square(self._target_values))), 3),
            be.sqrt(be.sum(be.square(self.template_weights))),
        )

        # (<u,dv/dx> * |v|**2 - <u,v> * <v,dv/dx>)/(|w|*|v|**3)
        translation_grad = be.multiply(
            be.sum(be.multiply(grad, self.template_weights), axis=1),
            be.sum(be.square(self._target_values)),
        )
        translation_grad -= be.multiply(
            be.sum(be.multiply(grad, self._target_values), axis=1),
            be.sum(be.multiply(self._target_values, self.template_weights)),
        )

        # (<u,r x dv/dx> * |v|**2 - <u,v> * <v,r x dv/dx>)/(|w|*|v|**3)
        torque_grad = be.multiply(
            be.sum(be.multiply(torque, self.template_weights), axis=1),
            be.sum(be.square(self._target_values)),
        )
        torque_grad -= be.multiply(
            be.sum(be.multiply(torque, self._target_values), axis=1),
            be.sum(be.multiply(self._target_values, self.template_weights)),
        )

        total_grad = be.concatenate([translation_grad, torque_grad])
        if norm > 0:
            total_grad = be.divide(total_grad, norm, out=total_grad)

        total_grad = be.divide(total_grad, self.n_points, out=total_grad)
        return -total_grad


class NormalizedCrossCorrelationMean(NormalizedCrossCorrelation):
    """
    Computes a similar score than :py:class:`NormalizedCrossCorrelation`, but
    additionally factors in the mean of template and target.

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

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __init__(self, **kwargs):
        kwargs["target"] = np.subtract(kwargs["target"], kwargs["target"].mean())
        kwargs["template_weights"] = np.subtract(
            kwargs["template_weights"], kwargs["template_weights"].mean()
        )
        super().__init__(**kwargs)


class MaskedCrossCorrelation(_MatchCoordinatesToDensity):
    """
    The Masked Cross-Correlation computes the similarity between `target_weights`
    and `template_weights` under respective masks. The score provides a measure of
    similarity even in the presence of missing or masked data.

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

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __call__(self) -> float:
        """Returns the score of the current configuration."""

        in_volume = np.logical_and(
            self.template_rotated < np.array(self.target.shape)[:, None],
            self.template_rotated >= 0,
        ).min(axis=0)
        in_volume_mask = np.logical_and(
            self.template_mask_rotated < np.array(self.target.shape)[:, None],
            self.template_mask_rotated >= 0,
        ).min(axis=0)

        mask_overlap = np.sum(
            self.target_mask[
                tuple(self.template_mask_rotated[:, in_volume_mask].astype(int))
            ],
        )
        mask_overlap = np.fmax(mask_overlap, np.finfo(float).eps)

        mask_target = self.target[
            tuple(self.template_mask_rotated[:, in_volume_mask].astype(int))
        ]
        denominator1 = np.subtract(
            np.sum(mask_target**2),
            np.divide(np.square(np.sum(mask_target)), mask_overlap),
        )
        mask_template = np.multiply(
            self.template_weights[in_volume],
            self.target_mask[tuple(self.template_rotated[:, in_volume].astype(int))],
        )
        denominator2 = np.subtract(
            np.sum(mask_template**2),
            np.divide(np.square(np.sum(mask_template)), mask_overlap),
        )

        denominator1 = np.fmax(denominator1, 0.0)
        denominator2 = np.fmax(denominator2, 0.0)
        denominator = np.sqrt(np.multiply(denominator1, denominator2))

        numerator = np.dot(
            self.target[tuple(self.template_rotated[:, in_volume].astype(int))],
            self.template_weights[in_volume],
        )

        numerator -= np.divide(
            np.multiply(np.sum(mask_target), np.sum(mask_template)), mask_overlap
        )

        if denominator == 0:
            return 0.0

        score = numerator / denominator
        return float(score * self.score_sign)


class PartialLeastSquareDifference(_MatchCoordinatesToDensity):
    """
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

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __call__(self) -> float:
        """Returns the score of the current configuration."""
        score = be.sum(
            be.square(be.subtract(self._target_values, self.template_weights))
        )
        return score * self.score_sign


class MutualInformation(_MatchCoordinatesToDensity):
    """
    The Mutual Information (MI) score between the target :math:`f` and the
    template :math:`g` is calculated as:

    .. math::

        \\text{d(f,g)} = \\sum_{f,g} p(f,g) \\log \\frac{p(f,g)}{p(f)p(g)}

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012

    """

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __call__(self) -> float:
        """Returns the score of the current configuration."""
        p_xy, target, template = np.histogram2d(
            self._target_values, self.template_weights
        )
        p_x, p_y = np.sum(p_xy, axis=1), np.sum(p_xy, axis=0)

        p_xy /= p_xy.sum()
        p_x /= p_x.sum()
        p_y /= p_y.sum()

        logprob = np.divide(p_xy, p_x[:, None] * p_y[None, :] + np.finfo(float).eps)
        score = np.nansum(p_xy * logprob)

        return score * self.score_sign


class Envelope(_MatchCoordinatesToDensity):
    """
    The Envelope score (ENV) between the target :math:`f` and the
    template :math:`g` is calculated as:

    .. math::

        \\text{d(f,g)} =    \\sum_{\\mathbf{p} \\in P} f'(\\mathbf{p})
                            \\cdot g'(\\mathbf{p})

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 1174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012
    """

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __init__(self, target_threshold: float = None, **kwargs):
        super().__init__(**kwargs)
        if target_threshold is None:
            target_threshold = np.mean(self.target)
        self.target = np.where(self.target > target_threshold, -1, 1)
        self.target_present = np.sum(self.target == -1)
        self.target_absent = np.sum(self.target == 1)
        self.template_weights = np.ones_like(self.template_weights)

    def __call__(self) -> float:
        """Returns the score of the current configuration."""
        score = self._target_values
        unassigned_density = self.target_present - (score == -1).sum()

        # Out of volume values will be set to 0
        score = score.sum() - unassigned_density
        score -= 2 * np.sum(np.invert(np.abs(self._target_values) > 0))
        min_score = -self.target_present - 2 * self.target_absent
        score = (score - 2 * min_score) / (2 * self.target_present - min_score)

        return score * self.score_sign


class Chamfer(_MatchCoordinatesToCoordinates):
    """
    The Chamfer distance between the target :math:`f` and the template :math:`g`
    is calculated as:

    .. math::

        \\text{d(f,g)} = \\frac{1}{|X|} \\sum_{\\mathbf{f}_i \\in X}
        \\inf_{\\mathbf{g} \\in Y} ||\\mathbf{f}_i - \\mathbf{g}||_2

    References
    ----------
    .. [1]  Daven Vasishtan and Maya Topf, "Scoring functions for cryoEM density
            fitting", Journal of Structural Biology, vol. 174, no. 2,
            pp. 333--343, 2011. DOI: https://doi.org/10.1016/j.jsb.2011.01.012
    """

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def _post_init(self, **kwargs):
        from scipy.spatial import KDTree

        self.target_tree = KDTree(self.target_coordinates.T)

    def __call__(self) -> float:
        """Returns the score of the current configuration."""
        dist, _ = self.target_tree.query(self.template_coordinates_rotated.T)
        score = np.mean(dist)
        return score * self.score_sign


class NormalVectorScore(_MatchCoordinatesToCoordinates):
    """
    The Normal Vector Score (NVS) between the target's :math:`f` and the template
    :math:`g`'s normal vectors is calculated as:

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

    __doc__ += _MatchCoordinatesToDensity.__doc__

    def __call__(self) -> float:
        """Returns the score of the current configuration."""
        numerator = np.multiply(
            self.template_coordinates_rotated, self.target_coordinates
        )
        denominator = np.linalg.norm(self.template_coordinates_rotated)
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
    "FLC": FLC,
}


def register_matching_optimization(match_name: str, match_class: type):
    """
    Registers a new mtaching method.

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
    methods_to_check = ["__init__", "__call__"]

    for method in methods_to_check:
        if not hasattr(match_class, method):
            raise ValueError(
                f"Method '{method}' is not defined in the provided class or object."
            )
    MATCHING_OPTIMIZATION_REGISTER[match_name] = match_class


def create_score_object(score: str, **kwargs) -> object:
    """
    Initialize score object with name ``score`` using ``**kwargs``.

    Parameters
    ----------
    score: str
        Name of the score.
    **kwargs: Dict
        Keyword arguments passed to the __init__ method of the score object.

    Returns
    -------
    object
        Initialized score object.

    Raises
    ------
    ValueError
        If ``score`` is not a key in MATCHING_OPTIMIZATION_REGISTER.

    See Also
    --------
    :py:meth:`register_matching_optimization`

    Examples
    --------
    >>> from tme import Density
    >>> from tme.mask import create_mask
    >>> from tme.matching_utils import euler_to_rotationmatrix
    >>> from tme.matching_optimization import CrossCorrelation, optimize_match
    >>> translation, rotation = (5, -2, 7), (5, -10, 2)
    >>> target = create_mask(
    >>>     mask_type="ellipse",
    >>>     radius=(5,5,5),
    >>>     shape=(51,51,51),
    >>>     center=(25,25,25),
    >>> ).astype(float)
    >>> template = Density(data=target)
    >>> template = template.rigid_transform(
    >>>     translation=translation,
    >>>     rotation_matrix=euler_to_rotationmatrix(rotation),
    >>> )
    >>> template_coordinates = template.to_pointcloud(0)
    >>> template_weights = template.data[tuple(template_coordinates)]
    >>> score_object = CrossCorrelation(
    >>>     target=target,
    >>>     template_coordinates=template_coordinates,
    >>>     template_weights=template_weights,
    >>>     negate_score=True # Multiply returned score with -1 for minimization
    >>> )
    """

    score_object = MATCHING_OPTIMIZATION_REGISTER.get(score, None)

    if score_object is None:
        raise ValueError(
            f"{score} is not defined. Please pick from "
            f" {', '.join(list(MATCHING_OPTIMIZATION_REGISTER.keys()))}."
        )

    score_object = score_object(**kwargs)
    return score_object


def optimize_match(
    score_object: object,
    bounds_translation: Tuple[Tuple[float]] = None,
    bounds_rotation: Tuple[Tuple[float]] = None,
    optimization_method: str = "basinhopping",
    maxiter: int = 50,
    x0: Tuple[float] = None,
) -> Tuple[ArrayLike, ArrayLike, float]:
    """
    Find the translation and rotation optimizing the score returned by ``score_object``
    with respect to provided bounds.

    Parameters
    ----------
    score_object: object
        Class object that defines a score method, which returns a floating point
        value given a tuple of floating points where the first half describes a
        translation and the second a rotation. The score will be minimized, i.e.
        it has to be negated if similarity should be optimized.
    bounds_translation : tuple of tuple float, optional
        Bounds on the evaluated translations. Has to be specified per dimension
        as tuple of (min, max). Default is None.
    bounds_rotation : tuple of tuple float, optional
        Bounds on the evaluated zyz Euler angles. Has to be specified per dimension
        as tuple of (min, max). Default is None.
    optimization_method : str, optional
        Optimizer that will be used, basinhopping by default. For further
        information refer to :doc:`scipy:reference/optimize`.

        +------------------------+-------------------------------------------+
        | differential_evolution | Highest accuracy but long runtime.        |
        |                        | Requires bounds on translation.           |
        +------------------------+-------------------------------------------+
        | basinhopping           | Decent accuracy, medium runtime.          |
        +------------------------+-------------------------------------------+
        | minimize               | If initial values are closed to optimum   |
        |                        | acceptable accuracy and short runtime     |
        +------------------------+-------------------------------------------+

    maxiter : int, optional
        The maximum number of iterations, 50 by default.
    x0 : tuple of floats, optional
        Initial values for the optimizer, zero by default.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike, float]
        Optimal translation, rotation matrix and corresponding score.

    Raises
    ------
    ValueError
        If ``optimization_method`` is not supported.

    Notes
    -----
    This function currently only supports three-dimensional optimization and
    ``score_object`` will be modified during this operation.

    Examples
    --------
    Having defined ``score_object``, for instance via :py:meth:`create_score_object`,
    non-exhaustive template matching can be performed as follows

    >>> translation_fit, rotation_fit, score = optimize_match(score_object)

    `translation_fit` and `rotation_fit` correspond to the inverse of the applied
    translation and rotation, so the following statements should hold within tolerance

    >>> np.allclose(translation, -translation_fit, atol = 1) # True
    >>> np.allclose(rotation, np.linalg.inv(rotation_fit), rtol = .1) # True

    Bounds on translation and rotation can be defined as follows

    >>> translation_fit, rotation_fit, score = optimize_match(
    >>>     score_object=score_object,
    >>>     bounds_translation=((-5,5),(-2,2),(0,0)),
    >>>     bounds_rotation=((-10,10), (-5,5), (0,0)),
    >>> )

    The optimization scheme and the initial parameter estimates can also be adapted

    >>> translation_fit, rotation_fit, score = optimize_match(
    >>>     score_object=score_object,
    >>>     optimization_method="minimize",
    >>>     x0=(0,0,0,5,3,-5),
    >>> )

    """
    ndim = 3
    _optimization_method = {
        "differential_evolution": differential_evolution,
        "basinhopping": basinhopping,
        "minimize": minimize,
    }
    if optimization_method not in _optimization_method:
        raise ValueError(
            f"{optimization_method} is not supported. "
            f"Pick from {', '.join(list(_optimization_method.keys()))}"
        )

    finfo = np.finfo(np.float32)

    # DE always requires bounds
    if optimization_method == "differential_evolution" and bounds_translation is None:
        bounds_translation = tuple((finfo.min, finfo.max) for _ in range(ndim))

    if bounds_translation is None and bounds_rotation is not None:
        bounds_translation = tuple((finfo.min, finfo.max) for _ in range(ndim))

    if bounds_rotation is None and bounds_translation is not None:
        bounds_rotation = tuple((-180, 180) for _ in range(ndim))

    bounds, linear_constraint = None, ()
    if bounds_rotation is not None and bounds_translation is not None:
        uncertainty = (*bounds_translation, *bounds_rotation)
        bounds = [
            bound if bound != (0, 0) else (-finfo.resolution, finfo.resolution)
            for bound in uncertainty
        ]
        linear_constraint = LinearConstraint(
            np.eye(len(bounds)), np.min(bounds, axis=1), np.max(bounds, axis=1)
        )

    x0 = np.zeros(2 * ndim) if x0 is None else x0

    return_gradient = getattr(score_object, "return_gradient", False)
    if optimization_method != "minimize" and return_gradient:
        warnings.warn("Gradient only considered for optimization_method='minimize'.")
        score_object.return_gradient = False

    initial_score = score_object.score(x=x0)
    if isinstance(initial_score, (List, Tuple)):
        initial_score = initial_score[0]

    if optimization_method == "basinhopping":
        result = basinhopping(
            x0=x0,
            func=score_object.score,
            niter=maxiter,
            minimizer_kwargs={"method": "COBYLA", "constraints": linear_constraint},
        )
    elif optimization_method == "differential_evolution":
        result = differential_evolution(
            func=score_object.score,
            bounds=bounds,
            constraints=linear_constraint,
            maxiter=maxiter,
        )
    elif optimization_method == "minimize":
        if hasattr(score_object, "grad") and not return_gradient:
            warnings.warn(
                "Consider initializing score object with return_gradient=True."
            )
        result = minimize(
            x0=x0,
            fun=score_object.score,
            jac=return_gradient,
            bounds=bounds,
            constraints=linear_constraint,
            options={"maxiter": maxiter},
        )
    print(f"Niter: {result.nit}, success : {result.success} ({result.message}).")
    print(f"Initial score: {initial_score} - Refined score: {result.fun}")
    if initial_score < result.fun:
        print("Initial score better than refined score. Returning identity.")
        result.x = np.zeros_like(result.x)
    translation, rotation = result.x[:ndim], result.x[ndim:]
    rotation_matrix = euler_to_rotationmatrix(rotation, "ZYZ")
    return translation, rotation_matrix, float(result.fun)
