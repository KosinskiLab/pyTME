"""
Utility functions for jax backend.

Copyright (c) 2023-2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
from functools import partial

import jax.numpy as jnp
from jax import pmap, lax, jit

from ..types import BackendArray
from ..backends import backend as be
from ..matching_utils import standardize, to_padded


__all__ = ["scan", "setup_scan"]


def _correlate(template: BackendArray, ft_target: BackendArray) -> BackendArray:
    """
    Computes :py:meth:`tme.matching_scores.cc_setup`.
    """
    template_ft = jnp.fft.rfftn(template, s=template.shape)
    template_ft = template_ft.at[:].multiply(ft_target)
    correlation = jnp.fft.irfftn(template_ft, s=template.shape)
    return correlation


def _flc_scoring(
    ft_target: BackendArray,
    ft_target2: BackendArray,
    template: BackendArray,
    template_mask: BackendArray,
    n_observations: BackendArray,
    eps: float,
    **kwargs,
) -> BackendArray:
    """
    Computes :py:meth:`tme.matching_scores.flc_scoring`.
    """
    inv_denominator = _reciprocal_target_std(
        ft_target=ft_target,
        ft_target2=ft_target2,
        template_mask=template_mask,
        eps=eps,
        n_observations=n_observations,
    )
    return _flcSphere_scoring(ft_target, template, inv_denominator)


def _flcSphere_scoring(
    ft_target: BackendArray,
    template: BackendArray,
    inv_denominator: BackendArray,
    **kwargs,
) -> BackendArray:
    """
    Computes :py:meth:`tme.matching_scores.corr_scoring`.
    """
    correlation = _correlate(template=template, ft_target=ft_target)
    return correlation.at[:].multiply(inv_denominator)


def _reciprocal_target_std(
    ft_target: BackendArray,
    ft_target2: BackendArray,
    template_mask: BackendArray,
    n_obs: float,
    eps: float,
) -> BackendArray:
    """
    Computes reciprocal standard deviation of a target given a mask.

    See Also
    --------
    :py:meth:`tme.matching_scores.flc_scoring`.
    """
    shape = template_mask.shape
    ft_template_mask = jnp.fft.rfftn(template_mask, s=shape)

    # E(X^2)- E(X)^2
    exp_sq = jnp.fft.irfftn(ft_target2 * ft_template_mask, s=shape)
    exp_sq = exp_sq.at[:].divide(n_obs)

    ft_template_mask = ft_template_mask.at[:].multiply(ft_target)
    sq_exp = jnp.fft.irfftn(ft_template_mask, s=shape)
    sq_exp = sq_exp.at[:].divide(n_obs)
    sq_exp = sq_exp.at[:].power(2)

    exp_sq = exp_sq.at[:].add(-sq_exp)
    exp_sq = exp_sq.at[:].max(0)
    exp_sq = exp_sq.at[:].power(0.5)

    exp_sq = exp_sq.at[:].set(
        jnp.where(exp_sq <= eps, 0, jnp.reciprocal(exp_sq * n_obs))
    )
    return exp_sq


def _apply_fourier_filter(arr: BackendArray, arr_filter: BackendArray) -> BackendArray:
    arr_ft = jnp.fft.rfftn(arr, s=arr.shape)
    arr_ft = arr_ft.at[:].multiply(arr_filter)
    return arr.at[:].set(jnp.fft.irfftn(arr_ft, s=arr.shape))


def setup_scan(analyzer_kwargs, analyzer, fast_shape, rotate_mask, match_projection):
    """Create separate scan function with initialized analyzer for each device"""
    device_scans = [
        partial(
            scan,
            fast_shape=fast_shape,
            rotate_mask=rotate_mask,
            analyzer=analyzer(**device_config),
        )
        for device_config in analyzer_kwargs
    ]

    @partial(
        pmap,
        in_axes=(0,) + (None,) * 7,
        axis_name="batch",
    )
    def scan_combined(
        target,
        template,
        template_mask,
        rotations,
        template_filter,
        target_filter,
        score_mask,
        background_template,
    ):
        return lax.switch(
            lax.axis_index("batch"),
            device_scans,
            target,
            template,
            template_mask,
            rotations,
            template_filter,
            target_filter,
            score_mask,
            background_template,
        )

    return scan_combined


@partial(jit, static_argnums=(8, 9, 10))
def scan(
    target: BackendArray,
    template: BackendArray,
    template_mask: BackendArray,
    rotations: BackendArray,
    template_filter: BackendArray,
    target_filter: BackendArray,
    score_mask: BackendArray,
    background_template: BackendArray,
    fast_shape: Tuple[int],
    rotate_mask: bool,
    analyzer: object,
) -> Tuple:
    eps = jnp.finfo(template.dtype).resolution

    if target_filter.shape != ():
        target = _apply_fourier_filter(target, target_filter)

    ft_target = jnp.fft.rfftn(target, s=fast_shape)
    ft_target2 = jnp.fft.rfftn(jnp.square(target), s=fast_shape)
    _n_obs, _inv_denominator, target = None, None, None

    unpadded_slice = tuple(slice(0, x) for x in template.shape)
    rot_buffer, mask_rot_buffer = jnp.zeros(fast_shape), jnp.zeros(fast_shape)
    if not rotate_mask:
        _n_obs = jnp.sum(template_mask)
        _inv_denominator = _reciprocal_target_std(
            ft_target=ft_target,
            ft_target2=ft_target2,
            template_mask=to_padded(mask_rot_buffer, template_mask, unpadded_slice),
            eps=eps,
            n_obs=_n_obs,
        )
        ft_target2 = None

    mask_scores = score_mask.shape != ()
    filter_template = template_filter.shape != ()
    bg_correction = background_template.shape != ()
    bg_scores = jnp.zeros(fast_shape) if bg_correction else 0

    _template_mask_rot = template_mask
    template_indices = be._index_grid(template.shape)
    center = be.divide(be.to_backend_array(template.shape) - 1, 2)

    def _sample_transform(ret, rotation_matrix):
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation_matrix, center=center
        )
        indices = be._transform_indices(template_indices, matrix)

        template_rot = be._interpolate(template, indices, order=1)
        n_obs, template_mask_rot = _n_obs, _template_mask_rot
        if rotate_mask:
            template_mask_rot = be._interpolate(template_mask, indices, order=1)
            n_obs = jnp.sum(template_mask_rot)

        if filter_template:
            template_rot = _apply_fourier_filter(template_rot, template_filter)
        template_rot = standardize(template_rot, template_mask_rot, n_obs)

        rot_pad = to_padded(rot_buffer, template_rot, unpadded_slice)

        inv_denominator = _inv_denominator
        if rotate_mask:
            mask_rot_pad = to_padded(mask_rot_buffer, template_mask_rot, unpadded_slice)
            inv_denominator = _reciprocal_target_std(
                ft_target=ft_target,
                ft_target2=ft_target2,
                template_mask=mask_rot_pad,
                n_obs=n_obs,
                eps=eps,
            )

        scores = _flcSphere_scoring(ft_target, rot_pad, inv_denominator)
        if mask_scores:
            scores = scores.at[:].multiply(score_mask)

        state, bg_scores, index = ret
        state = analyzer(state, scores, rotation_matrix, rotation_index=index)

        if bg_correction:
            template_rot = be._interpolate(background_template, indices, order=1)
            if filter_template:
                template_rot = _apply_fourier_filter(template_rot, template_filter)
            template_rot = standardize(template_rot, template_mask_rot, n_obs)

            rot_pad = to_padded(rot_buffer, template_rot, unpadded_slice)
            scores = _flcSphere_scoring(ft_target, rot_pad, inv_denominator)
            bg_scores = jnp.maximum(bg_scores, scores)

        return (state, bg_scores, index + 1), None

    (state, bg_scores, _), _ = lax.scan(
        _sample_transform, (analyzer.init_state(), bg_scores, 0), rotations
    )

    if bg_correction:
        if mask_scores:
            bg_scores = bg_scores.at[:].multiply(score_mask)
        bg_scores = bg_scores.at[:].add(-be.mean(bg_scores))
        state = analyzer.correct_background(state, bg_scores)

    return state
