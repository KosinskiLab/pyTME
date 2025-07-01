"""
Utility functions for jax backend.

Copyright (c) 2023-2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple
from functools import partial

import jax.numpy as jnp
from jax import pmap, lax, vmap

from ..types import BackendArray
from ..backends import backend as be
from ..matching_utils import normalize_template as _normalize_template


__all__ = ["scan"]


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
    :py:meth:`tme.matching_scores.flc_scoring`.
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
        rot_pad = be.topleft_pad(template_rot, fast_shape)
        mask_rot_pad = be.topleft_pad(template_mask_rot, fast_shape)

        scores = scoring_func(
            template=rot_pad,
            template_mask=mask_rot_pad,
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


# With full batching, we are comparing
# m targets (m, n, *d) against b templates (*d)
# this should be done on the level of the device
# the input to the scan_projections function should be (n, *d) and (*d+1)
# in the trivial case n = 1, but typically n > 1

# input are n tilts of dimension *d and a single template of *d+1

# Now it becomes tricky to avoid recomputation

# Compute FFT of target to avoid computation
# Split rotations according to jobs, to parallelize within the device
# Within each batch, scan over rotations

# Now we either push down with vmap or explicitly write out the comparison
# Advantage of vmap -> Less code potentially more readable
# Disavantage -> Might be less performant

# Reconciliate rotations data, return to get m, n, *d result.
# Fast shape needs to be of dimension d
# However, we typically compute it as the full shape of the score space

# we need -> vmapped scoring functions
# Consistent shape determination of the padding / ffts
# Merging of analyzer states (makes stuff easier actually)


@partial(vmap, in_axes=(0, 0, 0, None, None, None, None))
def _flc_scoring_batch(
    ft_target: BackendArray,  # vmapped (0)
    ft_target2: BackendArray,  # vmapped (0)
    template: BackendArray,  # broadcast (None)
    template_mask: BackendArray,  # broadcast (None)
    n_observations: BackendArray,  # broadcast (None)
    eps: float,  # broadcast (None)
    inv_denominator: BackendArray,  # dummy argument for unified interface
) -> BackendArray:
    return _flc_scoring(
        ft_target=ft_target,
        ft_target2=ft_target2,
        template=template,
        template_mask=template_mask,
        n_observations=n_observations,
        eps=eps,
    )


@partial(vmap, in_axes=(0, 0, 0, None, None, None, 0))
def _flcSphere_scoring_batch(
    ft_target: BackendArray,  # vmapped (0)
    ft_target2: BackendArray,  # vmapped (0) - dummy for unified interface
    template: BackendArray,  # broadcast (None)
    template_mask: BackendArray,  # broadcast (None) - dummy for unified interface
    n_observations: BackendArray,  # broadcast (None) - dummy for unified interface
    eps: float,  # broadcast (None) - dummy for unified interface
    inv_denominator: BackendArray,  # This is what we actually need
) -> BackendArray:
    return _flcSphere_scoring(
        template=template,
        ft_target=ft_target,
        inv_denominator=inv_denominator,
    )


@partial(vmap, in_axes=(0, 0, None, None, None))
def _reciprocal_target_std_batch(
    ft_target: BackendArray,
    ft_target2: BackendArray,
    template_mask: BackendArray,
    n_observations: float,
    eps: float,
) -> BackendArray:
    return _reciprocal_target_std(
        ft_target=ft_target,
        ft_target2=ft_target2,
        template_mask=template_mask,
        n_observations=n_observations,
        eps=eps,
    )


def _apply_fourier_filter_broadcast(
    arr: BackendArray, arr_filter: BackendArray
) -> BackendArray:
    arr_ft = jnp.fft.rfftn(arr, s=arr.shape)
    arr_ft = arr_ft * arr_filter
    return jnp.fft.irfftn(arr_ft, s=arr.shape)


@partial(
    pmap,
    in_axes=(0, None, None, None, None, None, None, None, None),
    static_broadcasted_argnums=[6, 7, 8],
)
# @partial(
#     vmap,
#     in_axes=(None, 0, 0, None, None, None, None, None, None),
# )
def scan_projections(
    target: BackendArray,
    template: BackendArray,
    template_mask: BackendArray,
    rotations: BackendArray,
    projection_filter: BackendArray,
    target_filter: BackendArray,
    fast_shape: Tuple[int],
    rotate_mask: bool,
    analyzer: object = None,
) -> Tuple[BackendArray, BackendArray]:
    from tme.projection import JAXProjector

    eps = jnp.finfo(template.dtype).resolution

    if hasattr(target_filter, "shape"):
        target = _apply_fourier_filter(target, target_filter)

    target_shape = target.shape
    axes = tuple(i + 1 for i in range(template.ndim - 1))
    ft_target = jnp.fft.rfftn(target, axes=axes)
    ft_target2 = jnp.fft.rfftn(jnp.square(target), axes=axes)

    template_projector = JAXProjector(template)
    template_mask_projector = JAXProjector(template_mask)

    mask_shape = fast_shape[1:]
    inv_denominator, target, scoring_func = None, None, _flc_scoring_batch
    if not rotate_mask:
        template_mask_proj = template_mask_projector(rotations[0])
        n_observations = jnp.sum(template_mask_proj)
        inv_denominator = _reciprocal_target_std_batch(
            ft_target,
            ft_target2,
            be.topleft_pad(template_mask_proj, mask_shape),
            n_observations,
            eps,
        )
        ft_target2, scoring_func = None, _flcSphere_scoring_batch

    _projection_filter_func = _identity
    if hasattr(projection_filter, "shape"):
        _projection_filter_func = _apply_fourier_filter_broadcast

    def _sample_transform(ret, rotation_matrix):
        state, index = ret
        template_rot = template_projector(rotation_matrix)
        template_mask_rot = template_mask_projector(rotation_matrix)
        n_observations = jnp.sum(template_mask_rot)

        # Projection filter to expand to target shape
        template_rot = _projection_filter_func(template_rot, projection_filter)
        template_rot = _normalize_template(
            template_rot, template_mask_rot, n_observations, axis=axes
        )
        template_rot = be.topleft_pad(template_rot, target_shape)
        template_mask_rot = be.topleft_pad(template_mask_rot, mask_shape)
        scores = scoring_func(
            ft_target,
            ft_target2,
            template_rot,
            template_mask_rot,
            n_observations,
            eps,
            inv_denominator,
        )
        state = analyzer(state, scores, rotation_matrix, rotation_index=index)
        return (state, index + 1), None

    state = analyzer.init_state()
    (state, index), _ = lax.scan(_sample_transform, (state, 0), rotations)
    return state[0], state[1]
    return state
