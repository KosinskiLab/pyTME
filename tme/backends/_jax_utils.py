"""
Utility functions for jax backend.

Copyright (c) 2023-2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
from functools import partial

import jax.numpy as jnp
from jax import pmap, lax

from ..types import BackendArray
from ..backends import backend as be
from ..matching_utils import normalize_template as _normalize_template


def _correlate(template: BackendArray, ft_target: BackendArray) -> BackendArray:
    """
    Computes :py:meth:`tme.matching_exhaustive.cc_setup`.
    """
    template_ft = jnp.fft.rfftn(template, s=template.shape)
    template_ft = template_ft.at[:].multiply(ft_target)
    correlation = jnp.fft.irfftn(template_ft, s=template.shape)
    return correlation


def _flc_scoring(
    template: BackendArray,
    template_mask: BackendArray,
    ft_target: BackendArray,
    ft_target2: BackendArray,
    n_observations: BackendArray,
    eps: float,
    **kwargs,
) -> BackendArray:
    """
    Computes :py:meth:`tme.matching_exhaustive.flc_scoring`.
    """
    correlation = _correlate(template=template, ft_target=ft_target)
    inv_denominator = _reciprocal_target_std(
        ft_target=ft_target,
        ft_target2=ft_target2,
        template_mask=template_mask,
        eps=eps,
        n_observations=n_observations,
    )
    correlation = correlation.at[:].multiply(inv_denominator)
    return correlation


def _flcSphere_scoring(
    template: BackendArray,
    ft_target: BackendArray,
    inv_denominator: BackendArray,
    **kwargs,
) -> BackendArray:
    """
    Computes :py:meth:`tme.matching_exhaustive.flc_scoring`.
    """
    correlation = _correlate(template=template, ft_target=ft_target)
    correlation = correlation.at[:].multiply(inv_denominator)
    return correlation


def _reciprocal_target_std(
    ft_target: BackendArray,
    ft_target2: BackendArray,
    template_mask: BackendArray,
    n_observations: float,
    eps: float,
) -> BackendArray:
    """
    Computes reciprocal standard deviation of a target given a mask.

    See Also
    --------
    :py:meth:`tme.matching_exhaustive.flc_scoring`.
    """
    ft_shape = template_mask.shape
    ft_template_mask = jnp.fft.rfftn(template_mask, s=ft_shape)

    # E(X^2)- E(X)^2
    exp_sq = jnp.fft.irfftn(ft_target2 * ft_template_mask, s=ft_shape)
    exp_sq = exp_sq.at[:].divide(n_observations)

    ft_template_mask = ft_template_mask.at[:].multiply(ft_target)
    sq_exp = jnp.fft.irfftn(ft_template_mask, s=ft_shape)
    sq_exp = sq_exp.at[:].divide(n_observations)
    sq_exp = sq_exp.at[:].power(2)

    exp_sq = exp_sq.at[:].add(-sq_exp)
    exp_sq = exp_sq.at[:].max(0)
    exp_sq = exp_sq.at[:].power(0.5)

    exp_sq = exp_sq.at[:].set(
        jnp.where(exp_sq <= eps, 0, jnp.reciprocal(exp_sq * n_observations))
    )
    return exp_sq


def _apply_fourier_filter(arr: BackendArray, arr_filter: BackendArray) -> BackendArray:
    arr_ft = jnp.fft.rfftn(arr, s=arr.shape)
    arr_ft = arr_ft.at[:].multiply(arr_filter)
    return arr.at[:].set(jnp.fft.irfftn(arr_ft, s=arr.shape))


def _identity(arr: BackendArray, arr_filter: BackendArray) -> BackendArray:
    return arr


@partial(
    pmap,
    in_axes=(0,) + (None,) * 6,
    static_broadcasted_argnums=[6, 7],
)
def scan(
    target: BackendArray,
    template: BackendArray,
    template_mask: BackendArray,
    rotations: BackendArray,
    template_filter: BackendArray,
    target_filter: BackendArray,
    fast_shape: Tuple[int],
    rotate_mask: bool,
) -> Tuple[BackendArray, BackendArray]:
    eps = jnp.finfo(template.dtype).resolution

    if hasattr(target_filter, "shape"):
        target = _apply_fourier_filter(target, target_filter)

    ft_target = jnp.fft.rfftn(target, s=fast_shape)
    ft_target2 = jnp.fft.rfftn(jnp.square(target), s=fast_shape)
    inv_denominator, target, scoring_func = None, None, _flc_scoring
    if not rotate_mask:
        n_observations = jnp.sum(template_mask)
        inv_denominator = _reciprocal_target_std(
            ft_target=ft_target,
            ft_target2=ft_target2,
            template_mask=be.topleft_pad(template_mask, fast_shape),
            eps=eps,
            n_observations=n_observations,
        )
        ft_target2, scoring_func = None, _flcSphere_scoring

    _template_filter_func = _identity
    if template_filter.shape != ():
        _template_filter_func = _apply_fourier_filter

    def _sample_transform(ret, rotation_matrix):
        max_scores, rotations, index = ret
        template_rot, template_mask_rot = be.rigid_transform(
            arr=template,
            arr_mask=template_mask,
            rotation_matrix=rotation_matrix,
            order=1,  # thats all we get for now
        )

        n_observations = jnp.sum(template_mask_rot)
        template_rot = _template_filter_func(template_rot, template_filter)
        template_rot = _normalize_template(
            template_rot, template_mask_rot, n_observations
        )
        template_rot = be.topleft_pad(template_rot, fast_shape)
        template_mask_rot = be.topleft_pad(template_mask_rot, fast_shape)

        scores = scoring_func(
            template=template_rot,
            template_mask=template_mask_rot,
            ft_target=ft_target,
            ft_target2=ft_target2,
            inv_denominator=inv_denominator,
            n_observations=n_observations,
            eps=eps,
        )
        max_scores, rotations = be.max_score_over_rotations(
            scores, max_scores, rotations, index
        )
        return (max_scores, rotations, index + 1), None

    score_space = jnp.zeros(fast_shape)
    rotation_space = jnp.full(shape=fast_shape, dtype=jnp.int32, fill_value=-1)
    (score_space, rotation_space, _), _ = lax.scan(
        _sample_transform, (score_space, rotation_space, 0), rotations
    )

    return score_space, rotation_space
