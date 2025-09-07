"""
Implements a range of cross-correlation coefficients.

Copyright (c) 2023-2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Callable, Tuple, Dict

import numpy as np

from .backends import backend as be
from .types import CallbackClass, BackendArray, shm_type
from .matching_utils import (
    conditional_execute,
    identity,
    standardize,
    to_padded,
)


def cc_setup(
    matching_data: type,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shm_handler: type,
    **kwargs,
) -> Dict:
    """
    Setup function for computing the unnormalized cross-correlation between
    ``target`` (f) and ``template`` (g)

    .. math::

        \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*).

    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    target_pad = be.topleft_pad(
        matching_data.target,
        matching_data._batch_shape(fast_shape, matching_data._template_batch),
    )
    axes = matching_data._batch_axis(matching_data._batch_mask)

    return {
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "ft_target": be.to_sharedarr(be.rfftn(target_pad, axes=axes), shm_handler),
        "inv_denominator": be.to_sharedarr(
            be.zeros(1, be._float_dtype) + 1, shm_handler
        ),
        "numerator": be.to_sharedarr(be.zeros(1, be._float_dtype), shm_handler),
    }


def lcc_setup(matching_data, **kwargs) -> Dict:
    """
    Setup function for computing the laplace cross-correlation between
    ``target`` (f) and ``template`` (g)

    .. math::

        \\mathcal{F}^{-1}(\\mathcal{F}(\\nabla^{2}f) \\cdot \\mathcal{F}(\\nabla^{2} g)^*)

    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    matching_data.target = matching_data.transform_target("laplace")
    matching_data.template = matching_data.transform_template("laplace")
    return cc_setup(matching_data=matching_data, **kwargs)


def corr_setup(
    matching_data,
    template_filter,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shm_handler: type,
    **kwargs,
) -> Dict:
    """
    Setup for computing a normalized cross-correlation between a
    ``target`` (f), a ``template`` (g) given  ``template_mask`` (m)

    .. math::

        \\frac{CC(f,g) - \\overline{g} \\cdot CC(f, m)}
        {(CC(f^2, m) - \\frac{CC(f, m)^2}{N_g}) \\cdot \\sigma_{g}},

    where

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*).

    Notes
    -----
    To be used with :py:meth:`corr_scoring`.

    References
    ----------
    .. [1]  Lewis P. J. Fast Normalized Cross-Correlation, Industrial Light and Magic.
    """
    template, template_mask = matching_data.template, matching_data.template_mask
    target_pad = be.topleft_pad(
        matching_data.target,
        matching_data._batch_shape(fast_shape, matching_data._template_batch),
    )
    data_axes = matching_data._batch_axis(matching_data._batch_mask)
    data_shape = tuple(fast_shape[i] for i in data_axes)

    ft_window = be.rfftn(be.topleft_pad(template_mask, fast_shape), axes=data_axes)

    ft_target = be.rfftn(be.square(target_pad), axes=data_axes)
    ft_target = be.multiply(ft_target, ft_window)
    denominator = be.irfftn(ft_target, s=data_shape, axes=data_axes)

    ft_target = be.rfftn(target_pad, axes=data_axes)
    ft_window = be.multiply(ft_target, ft_window)
    window_sum = be.irfftn(ft_window, s=data_shape, axes=data_axes)

    target_pad, ft_window = None, None

    # TODO: Factor in template_filter here
    if be.size(template_filter) != 1:
        warnings.warn(
            "CORR scores obtained with template_filter are not correctly scaled. "
            "Please use a different score or consider only relative peak heights."
        )
    axis = matching_data._batch_axis(matching_data._template_batch)
    n_obs = be.sum(
        be.astype(template_mask, be._overflow_safe_dtype), axis=axis, keepdims=True
    )
    template_mean = be.multiply(template, template_mask)
    template_mean = be.sum(template_mean, axis=axis, keepdims=True)
    template_mean = be.divide(template_mean, n_obs)
    template_ssd = be.square(template - template_mean) * template_mask
    template_ssd = be.sum(template_ssd, axis=axis, keepdims=True)

    template_volume = np.prod(
        tuple(
            int(x)
            for i, x in enumerate(template.shape)
            if matching_data._template_batch[i] == 0
        )
    )
    template = be.multiply(template, template_mask, out=template)

    numerator = be.multiply(window_sum, template_mean)
    window_sum = be.square(window_sum, out=window_sum)
    window_sum = be.divide(window_sum, template_volume, out=window_sum)
    denominator = be.subtract(denominator, window_sum, out=denominator)
    denominator = be.multiply(denominator, template_ssd, out=denominator)
    denominator = be.maximum(denominator, 0, out=denominator)
    denominator = be.sqrt(denominator, out=denominator)

    mask = denominator > be.eps(be._float_dtype)
    denominator = be.multiply(denominator, mask, out=denominator)
    denominator = be.add(denominator, ~mask, out=denominator)
    denominator = be.divide(1, denominator, out=denominator)
    denominator = be.multiply(denominator, mask, out=denominator)

    return {
        "template": be.to_sharedarr(template, shm_handler),
        "ft_target": be.to_sharedarr(ft_target, shm_handler),
        "inv_denominator": be.to_sharedarr(denominator, shm_handler),
        "numerator": be.to_sharedarr(numerator, shm_handler),
    }


def cam_setup(matching_data, **kwargs) -> Dict:
    """
    Like :py:meth:`corr_setup` but with standardized ``target`` and ``template``

    .. math::

        f' = \\frac{f - \\overline{f}}{\\sigma_f}.

    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    matching_data.target = matching_data.transform_target("standardize")
    matching_data.template = matching_data.transform_template("standardize")
    return flcSphericalMask_setup(matching_data=matching_data, **kwargs)


def ncc_setup(matching_data, **kwargs) -> Dict:
    matching_data.target = matching_data.transform_target("standardize")
    return cc_setup(matching_data=matching_data, **kwargs)


def flc_setup(
    matching_data,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shm_handler: type,
    **kwargs,
) -> Dict:
    """
    Setup function for :py:meth:`flc_scoring`.
    """
    target_pad = be.topleft_pad(
        matching_data.target,
        matching_data._batch_shape(fast_shape, matching_data._template_batch),
    )

    data_axes = matching_data._batch_axis(matching_data._batch_mask)

    ft_target = be.rfftn(target_pad, axes=data_axes)
    target_pad = be.square(target_pad, out=target_pad)
    ft_target2 = be.rfftn(target_pad, axes=data_axes)

    return {
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(matching_data.template_mask, shm_handler),
        "ft_target": be.to_sharedarr(ft_target, shm_handler),
        "ft_target2": be.to_sharedarr(ft_target2, shm_handler),
    }


def flcSphericalMask_setup(
    matching_data,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shm_handler: type,
    **kwargs,
) -> Dict:
    """
    Like :py:meth:`flc_setup` for rotation invariant masks

    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    template_mask = matching_data.template_mask
    axis = matching_data._batch_axis(matching_data._template_batch)
    n_obs = be.sum(
        be.astype(template_mask, be._overflow_safe_dtype), axis=axis, keepdims=True
    )

    target_pad = be.topleft_pad(
        matching_data.target,
        matching_data._batch_shape(fast_shape, matching_data._template_batch),
    )

    # Enable mask broadcasting
    _out_shape = tuple(
        y if i in axis else x
        for i, (x, y) in enumerate(zip(template_mask.shape, fast_shape))
    )
    template_mask_pad = be.topleft_pad(
        template_mask,
        matching_data._batch_shape(_out_shape, matching_data._target_batch),
    )

    data_axes = matching_data._batch_axis(matching_data._batch_mask)
    data_shape = tuple(fast_shape[i] for i in data_axes)

    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_template_mask = be.rfftn(template_mask_pad, s=data_shape, axes=data_axes)

    ft_target = be.rfftn(be.square(target_pad), axes=data_axes)
    ft_temp = be.multiply(ft_target, ft_template_mask, out=ft_temp)
    temp2 = be.irfftn(ft_temp, s=data_shape, axes=data_axes)

    ft_target = be.rfftn(target_pad, axes=data_axes)
    ft_temp = be.multiply(ft_target, ft_template_mask, out=ft_temp)
    temp = be.irfftn(ft_temp, s=data_shape, axes=data_axes)

    temp2 = be.norm_scores(1, temp2, temp, n_obs, be.eps(be._float_dtype), temp2)
    return {
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(template_mask, shm_handler),
        "ft_target": be.to_sharedarr(ft_target, shm_handler),
        "inv_denominator": be.to_sharedarr(temp2, shm_handler),
        "numerator": be.to_sharedarr(be.zeros(1, be._float_dtype), shm_handler),
    }


def mcc_setup(
    matching_data,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shm_handler: Callable,
    **kwargs,
) -> Dict:
    """
    Setup function for :py:meth:`mcc_scoring`.
    """
    target, target_mask = matching_data.target, matching_data.target_mask
    target = be.multiply(target, target_mask, out=target)

    ax = matching_data._batch_axis(matching_data._batch_mask)
    shape = matching_data._batch_shape(fast_shape, matching_data._template_batch)
    target = be.topleft_pad(target, shape)
    target_mask = be.topleft_pad(target_mask, shape)

    return {
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(matching_data.template_mask, shm_handler),
        "ft_target": be.to_sharedarr(be.rfftn(target, axes=ax), shm_handler),
        "ft_target2": be.to_sharedarr(
            be.rfftn(be.square(target), axes=ax), shm_handler
        ),
        "ft_target_mask": be.to_sharedarr(be.rfftn(target_mask, axes=ax), shm_handler),
    }


def corr_scoring(
    template: shm_type,
    template_filter: shm_type,
    ft_target: shm_type,
    inv_denominator: shm_type,
    numerator: shm_type,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: BackendArray,
    callback: CallbackClass,
    interpolation_order: int,
    template_mask: shm_type = None,
    score_mask: shm_type = None,
    template_background: shm_type = None,
) -> CallbackClass:
    """
    Calculates a normalized cross-correlation between a target f and a template g.

    .. math::

        (CC(f,g) - \\text{numerator}) \\cdot \\text{inv_denominator},

    where

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*).

    Parameters
    ----------
    template : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template data buffer, its shape and datatype.
    template_filter : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template filter data buffer, its shape and datatype.
    ft_target : Union[Tuple[type, tuple of ints, type], BackendArray]
        Fourier transformed target data buffer, its shape and datatype.
    inv_denominator : Union[Tuple[type, tuple of ints, type], BackendArray]
        Inverse denominator data buffer, its shape and datatype.
    numerator : Union[Tuple[type, tuple of ints, type], BackendArray]
        Numerator data buffer, its shape, and its datatype.
    fast_shape: tuple of ints
        Data shape for the forward Fourier transform.
    fast_ft_shape: tuple of ints
        Data shape for the inverse Fourier transform.
    rotations : BackendArray
        Rotation matrices to be sampled (n, d, d).
    callback : CallbackClass
        A callable for processing the result of each rotation.
    interpolation_order : int
        Spline order for template rotations.
    template_mask : Union[Tuple[type, tuple of ints, type], BackendArray], optional
        Template mask data buffer, its shape and datatype, None by default.
    score_mask : Union[Tuple[type, tuple of ints, type], BackendArray], optional
        Score mask data buffer, its shape and datatype, None by default.

    Returns
    -------
    CallbackClass
    """
    template = be.from_sharedarr(template)
    ft_target = be.from_sharedarr(ft_target)
    inv_denominator = be.from_sharedarr(inv_denominator)
    numerator = be.from_sharedarr(numerator)
    template_filter = be.from_sharedarr(template_filter)
    score_mask = be.from_sharedarr(score_mask)

    n_obs = None
    if template_mask is not None:
        template_mask = be.from_sharedarr(template_mask)
        n_obs = be.sum(template_mask) if template_mask is not None else None

    norm_template = conditional_execute(standardize, n_obs is not None)
    norm_sub = conditional_execute(be.subtract, numerator.shape != (1,))
    norm_mul = conditional_execute(be.multiply, inv_denominator.shape != (1))
    norm_mask = conditional_execute(be.multiply, score_mask.shape != (1,))

    arr = be.zeros(fast_shape, be._float_dtype)
    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)
    template_rot = be.zeros(template.shape, be._float_dtype)

    tmpl_filter_func = _create_filter_func(template.shape, template_filter)

    center = be.divide(be.to_backend_array(template.shape) - 1, 2)
    unpadded_slice = tuple(slice(0, stop) for stop in template.shape)

    background_correction = template_background is not None
    if background_correction:
        scores_alt, compute_norm = _setup_background_correction(
            fast_shape=fast_shape,
            template_background=template_background,
            rotation_buffer=template_rot,
            unpadded_slice=unpadded_slice,
            interpolation_order=interpolation_order,
            tmpl_filter_func=tmpl_filter_func,
            norm_template=norm_template,
        )

    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation, center=center, shape=template.shape
        )
        _ = be.rigid_transform(
            arr=template,
            rotation_matrix=matrix,
            out=template_rot,
            order=interpolation_order,
            cache=True,
            use_geometric_center=True,
        )

        template_rot = tmpl_filter_func(template_rot, ft_temp)
        template_rot = norm_template(template_rot, template_mask, n_obs)

        arr = to_padded(arr, template_rot, unpadded_slice)
        ft_temp = be.rfftn(arr, s=fast_shape, out=ft_temp)
        arr = _correlate_fts(ft_target, ft_temp, ft_temp, arr, fast_shape)

        arr = norm_sub(arr, numerator, out=arr)
        arr = norm_mul(arr, inv_denominator, out=arr)
        arr = norm_mask(arr, score_mask, out=arr)

        callback(arr, rotation_matrix=rotation)
        if background_correction:
            arr = compute_norm(arr, ft_target, ft_temp, matrix, template_mask, n_obs)
            arr = norm_sub(arr, numerator, out=arr)
            arr = norm_mul(arr, inv_denominator, out=arr)
            arr = norm_mask(arr, score_mask, out=arr)
            scores_alt = be.maximum(arr, scores_alt, out=scores_alt)

    if background_correction:
        scores_alt = norm_mask(scores_alt, score_mask, out=scores_alt)
        scores_alt = be.subtract(scores_alt, be.mean(scores_alt), out=scores_alt)
        callback.correct_background(scores_alt)

    return callback


def flc_scoring(
    template: shm_type,
    template_mask: shm_type,
    ft_target: shm_type,
    ft_target2: shm_type,
    template_filter: shm_type,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: BackendArray,
    callback: CallbackClass,
    interpolation_order: int,
    score_mask: shm_type = None,
    template_background: shm_type = None,
) -> CallbackClass:
    """
    Computes a normalized cross-correlation between ``target`` (f),
    ``template`` (g), and ``template_mask`` (m)

    .. math::

        \\frac{CC(f, \\frac{g*m - \\overline{g*m}}{\\sigma_{g*m}})}
        {N_m * \\sqrt{
            \\frac{CC(f^2, m)}{N_m} - (\\frac{CC(f, m)}{N_m})^2}
        },

    where

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*)

    and Nm is the sum of g.

    Parameters
    ----------
    template : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template data buffer, its shape and datatype.
    template_mask : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template mask data buffer, its shape and datatype.
    template_filter : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template filter data buffer, its shape and datatype.
    ft_target : Union[Tuple[type, tuple of ints, type], BackendArray]
        Fourier transformed target data buffer, its shape and datatype.
    ft_target2 : Union[Tuple[type, tuple of ints, type], BackendArray]
        Fourier transformed squared target data buffer, its shape and datatype.
    fast_shape : tuple of ints
        Data shape for the forward Fourier transform.
    fast_ft_shape : tuple of ints
        Data shape for the inverse Fourier transform.
    rotations : BackendArray
        Rotation matrices to be sampled (n, d, d).
    callback : CallbackClass
        A callable for processing the result of each rotation.
    interpolation_order : int
        Spline order for template rotations.

    Returns
    -------
    CallbackClass

    References
    ----------
    .. [1]  Hrabe T. et al, J. Struct. Biol. 178, 177 (2012).
    """
    template = be.from_sharedarr(template)
    template_mask = be.from_sharedarr(template_mask)
    ft_target = be.from_sharedarr(ft_target)
    ft_target2 = be.from_sharedarr(ft_target2)
    template_filter = be.from_sharedarr(template_filter)
    score_mask = be.from_sharedarr(score_mask)

    arr = be.zeros(fast_shape, be._float_dtype)
    temp = be.zeros(fast_shape, be._float_dtype)
    temp2 = be.zeros(fast_shape, be._float_dtype)
    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_denom = be.zeros(fast_ft_shape, be._complex_dtype)
    template_rot = be.zeros(template.shape, be._float_dtype)
    template_mask_rot = be.zeros(template.shape, be._float_dtype)

    tmpl_filter_func = _create_filter_func(template.shape, template_filter)
    norm_mask = conditional_execute(be.multiply, score_mask.shape != (1,))

    eps = be.eps(be._float_dtype)
    center = be.divide(be.to_backend_array(template.shape) - 1, 2)
    unpadded_slice = tuple(slice(0, stop) for stop in template.shape)

    background_correction = template_background is not None
    if background_correction:
        scores_alt, compute_norm = _setup_background_correction(
            fast_shape=fast_shape,
            template_background=template_background,
            rotation_buffer=template_rot,
            unpadded_slice=unpadded_slice,
            interpolation_order=interpolation_order,
            tmpl_filter_func=tmpl_filter_func,
            norm_template=standardize,
        )

    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation, center=center, shape=template.shape
        )
        _ = be.rigid_transform(
            arr=template,
            arr_mask=template_mask,
            rotation_matrix=matrix,
            out=template_rot,
            out_mask=template_mask_rot,
            order=interpolation_order,
            cache=True,
            use_geometric_center=True,
        )

        n_obs = be.sum(template_mask_rot)
        template_rot = tmpl_filter_func(template_rot, ft_temp)
        template_rot = standardize(template_rot, template_mask_rot, n_obs)

        arr = to_padded(arr, template_rot, unpadded_slice)
        temp = to_padded(temp, template_mask_rot, unpadded_slice)

        ft_temp = be.rfftn(temp, out=ft_temp, s=fast_shape)
        temp = _correlate_fts(ft_target, ft_temp, ft_denom, temp, fast_shape)
        temp2 = _correlate_fts(ft_target2, ft_temp, ft_denom, temp, fast_shape)

        ft_temp = be.rfftn(arr, out=ft_temp, s=fast_shape)
        arr = _correlate_fts(ft_target, ft_temp, ft_temp, arr, fast_shape)

        inv_sdev = be.norm_scores(1, temp2, temp, n_obs, eps, temp2)
        arr = be.multiply(arr, inv_sdev, out=arr)
        arr = norm_mask(arr, score_mask, out=arr)

        callback(arr, rotation_matrix=rotation)
        if background_correction:
            arr = compute_norm(arr, ft_target, ft_temp, matrix, template_mask, n_obs)
            arr = be.multiply(arr, inv_sdev, out=arr)
            scores_alt = be.maximum(arr, scores_alt, out=scores_alt)

    if background_correction:
        scores_alt = norm_mask(scores_alt, score_mask, out=scores_alt)
        scores_alt = be.subtract(scores_alt, be.mean(scores_alt), out=scores_alt)
        callback.correct_background(scores_alt)

    return callback


def ncc_scoring(
    template: shm_type,
    ft_target: shm_type,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: BackendArray,
    callback: CallbackClass,
    interpolation_order: int,
    template_filter: shm_type = None,
    score_mask: shm_type = None,
    template_background: shm_type = None,
    **kwargs,
) -> CallbackClass:
    template = be.from_sharedarr(template)
    ft_target = be.from_sharedarr(ft_target)
    score_mask = be.from_sharedarr(score_mask)
    template_filter = be.from_sharedarr(template_filter)

    arr = be.zeros(fast_shape, be._float_dtype)
    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)
    template_rot = be.zeros(template.shape, be._float_dtype)

    # Welford arrays for global statistics
    pixel_mean = be.zeros(fast_shape, be._float_dtype)
    pixel_M2 = be.zeros(fast_shape, be._float_dtype)

    tmpl_filter_func = _create_filter_func(template.shape, template_filter)
    norm_mask = conditional_execute(be.multiply, score_mask.shape != (1,))

    size = be.size(template)
    center = be.divide(be.to_backend_array(template.shape) - 1, 2)
    unpadded_slice = tuple(slice(0, stop) for stop in template.shape)
    n_angles = rotations.shape[0]

    # Scale forward transform by 1/n i.e. norm 'forward'
    ft_target = be.multiply(ft_target, 1 / be.size(arr))

    background_correction = template_background is not None
    for index in range(n_angles):
        arr = be.fill(arr, 0)
        rotation = rotations[index]
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation, center=center, shape=template.shape
        )

        be.rigid_transform(
            template,
            rotation_matrix=matrix,
            out=template_rot,
            order=interpolation_order,
            cache=True,
            use_geometric_center=True,
        )
        template_rot = tmpl_filter_func(template_rot, ft_temp)
        template_rot = standardize(template_rot, 1, size)

        arr = to_padded(arr, template_rot, unpadded_slice)
        ft_temp = be.rfftn(arr, s=fast_shape, norm="forward")
        ft_temp = be.multiply(ft_temp, ft_target, out=ft_temp)

        arr = be.irfftn(ft_temp, s=fast_shape, norm="forward")
        arr = norm_mask(arr, score_mask, out=arr)
        callback(arr, rotation_matrix=rotation)

        delta = be.subtract(arr, pixel_mean)
        pixel_mean = be.add(pixel_mean, be.divide(delta, index + 1), out=pixel_mean)
        delta2 = be.subtract(arr, pixel_mean)
        delta = be.multiply(delta, delta2, out=delta)
        pixel_M2 = be.add(pixel_M2, delta, out=pixel_M2)

    global_mean = be.mean(pixel_mean)
    pixel_variance = be.divide(pixel_M2, n_angles - 1)
    global_std = be.sqrt(be.mean(pixel_variance))

    callback.correct_background(global_mean, global_std)
    if background_correction:
        # Adapt units for local normalization
        pixel_mean = be.subtract(pixel_mean, global_mean, out=pixel_mean)
        pixel_mean = be.divide(pixel_mean, global_std, out=pixel_mean)

        pixel_std = be.sqrt(pixel_variance, out=pixel_variance)
        pixel_std = be.divide(pixel_std, global_std, out=pixel_variance)

        pixel_std = be.where(pixel_std > 1e-4, 1 / pixel_std, 0.0)
        callback.correct_background(pixel_mean, pixel_std)
    return callback


def mcc_scoring(
    template: shm_type,
    template_mask: shm_type,
    template_filter: shm_type,
    ft_target: shm_type,
    ft_target2: shm_type,
    ft_target_mask: shm_type,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: BackendArray,
    callback: CallbackClass,
    interpolation_order: int,
    overlap_ratio: float = 0.3,
    **kwargs,
) -> CallbackClass:
    """
    Computes a normalized cross-correlation score between ``target`` (f),
    ``template`` (g), ``template_mask`` (m) and ``target_mask`` (t)

    .. math::

        \\frac{
               CC(f, g) - \\frac{CC(f, m) \\cdot CC(t, g)}{CC(t, m)}
            }{
            \\sqrt{
                (CC(f ^ 2, m) - \\frac{CC(f, m) ^ 2}{CC(t, m)}) \\cdot
                (CC(t, g^2) - \\frac{CC(t, g) ^ 2}{CC(t, m)})
                }
            },

    where

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*).

    Parameters
    ----------
    template : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template data buffer, its shape and datatype.
    template_mask : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template mask data buffer, its shape and datatype.
    template_filter : Union[Tuple[type, tuple of ints, type], BackendArray]
        Template filter data buffer, its shape and datatype.
    ft_target : Union[Tuple[type, tuple of ints, type], BackendArray]
        Fourier transformed target data buffer, its shape and datatype.
    ft_target2 : Union[Tuple[type, tuple of ints, type], BackendArray]
        Fourier transformed squared target data buffer, its shape and datatype.
    ft_target_mask : Union[Tuple[type, tuple of ints, type], BackendArray]
        Fourier transformed target mask data buffer, its shape and datatype.
    fast_shape: tuple of ints
        Data shape for the forward Fourier transform.
    fast_ft_shape: tuple of ints
        Data shape for the inverse Fourier transform.
    rotations : BackendArray
        Rotation matrices to be sampled (n, d, d).
    callback : CallbackClass
        A callable for processing the result of each rotation.
    interpolation_order : int
        Spline order for template rotations.
    overlap_ratio : float, optional
        Required fractional mask overlap, 0.3 by default.

    Returns
    -------
    CallbackClass

    References
    ----------
    .. [1]  Masked FFT registration, Dirk Padfield, CVPR 2010 conference
    .. [2]  https://scikit-image.org/docs/stable/api/skimage.registration.html
    """
    float_dtype, complex_dtype = be._float_dtype, be._complex_dtype
    template = be.from_sharedarr(template)
    target_ft = be.from_sharedarr(ft_target)
    target_ft2 = be.from_sharedarr(ft_target2)
    template_mask = be.from_sharedarr(template_mask)
    target_mask_ft = be.from_sharedarr(ft_target_mask)
    template_filter = be.from_sharedarr(template_filter)

    axes = tuple(range(template.ndim))
    eps = be.eps(float_dtype)

    # Allocate score and process specific arrays
    template_rot = be.zeros(fast_shape, float_dtype)
    mask_overlap = be.zeros(fast_shape, float_dtype)
    numerator = be.zeros(fast_shape, float_dtype)
    temp = be.zeros(fast_shape, float_dtype)
    temp2 = be.zeros(fast_shape, float_dtype)
    temp3 = be.zeros(fast_shape, float_dtype)
    temp_ft = be.zeros(fast_ft_shape, complex_dtype)

    tmpl_filter_func = _create_filter_func(
        arr_shape=template.shape,
        template_filter=template_filter,
        arr_padded=True,
    )
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        template_rot = be.fill(template_rot, 0)
        temp = be.fill(temp, 0)
        be.rigid_transform(
            arr=template,
            arr_mask=template_mask,
            rotation_matrix=rotation,
            out=template_rot,
            out_mask=temp,
            use_geometric_center=True,
            order=interpolation_order,
            cache=True,
        )

        template_rot = tmpl_filter_func(template_rot, temp_ft)
        template_rot = standardize(template_rot, temp, be.sum(temp))

        temp_ft = be.rfftn(template_rot, out=temp_ft, s=fast_shape)
        temp2 = be.irfftn(target_mask_ft * temp_ft, out=temp2, s=fast_shape)
        numerator = be.irfftn(target_ft * temp_ft, out=numerator, s=fast_shape)

        # temp template_mask_rot | temp_ft template_mask_rot_ft
        # Calculate overlap of masks at every point in the convolution.
        # Locations with high overlap should not be taken into account.
        temp_ft = be.rfftn(temp, out=temp_ft, s=fast_shape)
        mask_overlap = be.irfftn(
            temp_ft * target_mask_ft, out=mask_overlap, s=fast_shape
        )
        be.maximum(mask_overlap, eps, out=mask_overlap)
        temp = be.irfftn(temp_ft * target_ft, out=temp, s=fast_shape)

        be.subtract(
            numerator,
            be.divide(be.multiply(temp, temp2), mask_overlap),
            out=numerator,
        )

        # temp_3 = fixed_denom
        be.multiply(temp_ft, target_ft2, out=temp_ft)
        temp3 = be.irfftn(temp_ft, out=temp3, s=fast_shape)
        be.subtract(temp3, be.divide(be.square(temp), mask_overlap), out=temp3)
        be.maximum(temp3, 0.0, out=temp3)

        # temp = moving_denom
        temp_ft = be.rfftn(be.square(template_rot), out=temp_ft, s=fast_shape)
        be.multiply(target_mask_ft, temp_ft, out=temp_ft)
        temp = be.irfftn(temp_ft, out=temp, s=fast_shape)

        be.subtract(temp, be.divide(be.square(temp2), mask_overlap), out=temp)
        be.maximum(temp, 0.0, out=temp)

        # temp_2 = denom
        be.multiply(temp3, temp, out=temp)
        be.sqrt(temp, out=temp2)

        # Pixels where `denom` is very small will introduce large
        # numbers after division. To get around this problem,
        # we zero-out problematic pixels.
        tol = 1e3 * eps * be.max(be.abs(temp2), axis=axes, keepdims=True)

        temp2[temp2 < tol] = 1
        temp = be.divide(numerator, temp2, out=temp)
        temp = be.clip(temp, a_min=-1, a_max=1, out=temp)

        # Apply overlap ratio threshold
        number_px_threshold = overlap_ratio * be.max(
            mask_overlap, axis=axes, keepdims=True
        )
        temp[mask_overlap < number_px_threshold] = 0.0
        callback(temp, rotation_matrix=rotation)

    return callback


def flc_scoring2(
    template: shm_type,
    template_mask: shm_type,
    ft_target: shm_type,
    ft_target2: shm_type,
    template_filter: shm_type,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: BackendArray,
    callback: CallbackClass,
    interpolation_order: int,
    score_mask: shm_type = None,
    template_background: shm_type = None,
) -> CallbackClass:
    template = be.from_sharedarr(template)
    template_mask = be.from_sharedarr(template_mask)
    ft_target = be.from_sharedarr(ft_target)
    ft_target2 = be.from_sharedarr(ft_target2)
    template_filter = be.from_sharedarr(template_filter)
    score_mask = be.from_sharedarr(score_mask)

    tar_batch, tmpl_batch = _get_batch_dim(ft_target, template)

    nd = len(fast_shape)
    sqz_slice = tuple(slice(0, 1) if i in tar_batch else slice(None) for i in range(nd))
    tmpl_subset = tuple(0 if i in tar_batch else slice(None) for i in range(nd))

    axes, shape, batched = None, fast_shape, len(tmpl_batch) > 0
    if len(tar_batch) or len(tmpl_batch):
        axes = tuple(i for i in range(nd) if i not in (*tar_batch, *tmpl_batch))
        shape = tuple(fast_shape[i] for i in axes)

    arr = be.zeros(fast_shape, be._float_dtype)
    temp = be.zeros(fast_shape, be._float_dtype)
    temp2 = be.zeros(fast_shape, be._float_dtype)
    ft_denom = be.zeros(fast_ft_shape, be._complex_dtype)

    tmp_sqz, arr_sqz, ft_temp = temp[sqz_slice], arr[sqz_slice], ft_denom[sqz_slice]

    tmpl_filter_func = _create_filter_func(
        arr_shape=template.shape,
        template_filter=template_filter,
        arr_padded=True,
    )
    norm_mask = conditional_execute(be.multiply, score_mask.shape != (1,))

    background_correction = template_background is not None
    if background_correction:
        scores_alt, compute_norm = _setup_background_correction(
            fast_shape=fast_shape,
            template_background=template_background,
            rotation_buffer=arr_sqz[tmpl_subset],
            unpadded_slice=tmpl_subset,
            interpolation_order=interpolation_order,
            tmpl_filter_func=tmpl_filter_func,
            norm_template=standardize,
        )

    eps = be.eps(be._float_dtype)
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        be.fill(arr, 0)
        be.fill(temp, 0)

        _, _ = be.rigid_transform(
            arr=template[tmpl_subset],
            arr_mask=template_mask[tmpl_subset],
            rotation_matrix=rotation,
            out=arr_sqz[tmpl_subset],
            out_mask=tmp_sqz[tmpl_subset],
            use_geometric_center=True,
            order=interpolation_order,
            cache=False,
            batched=batched,
        )

        n_obs = be.sum(tmp_sqz, axis=axes, keepdims=True)
        arr_norm = tmpl_filter_func(arr_sqz, ft_temp)
        arr_norm = standardize(arr_norm, tmp_sqz, n_obs, axis=axes)

        ft_temp = be.rfftn(tmp_sqz, out=ft_temp, axes=axes, s=shape)
        temp = _correlate_fts(ft_target, ft_temp, ft_denom, temp, shape, axes)
        temp2 = _correlate_fts(ft_target2, ft_temp, ft_denom, temp2, shape, axes)

        ft_temp = be.rfftn(arr_norm, out=ft_temp, axes=axes, s=shape)
        arr = _correlate_fts(ft_target, ft_temp, ft_denom, arr, shape, axes)

        inv_sdev = be.norm_scores(1, temp2, temp, n_obs, eps, temp2)
        arr = be.multiply(arr, inv_sdev, out=arr)
        arr = norm_mask(arr, score_mask, out=arr)

        callback(arr, rotation_matrix=rotation)
        if background_correction:
            arr = compute_norm(arr, ft_target, ft_temp, rotation, template_mask, n_obs)
            arr = be.multiply(arr, inv_sdev, out=arr)
            scores_alt = be.maximum(arr, scores_alt, out=scores_alt)

    if background_correction:
        scores_alt = norm_mask(scores_alt, score_mask, out=scores_alt)
        scores_alt = be.subtract(scores_alt, be.mean(scores_alt), out=scores_alt)
        callback.correct_background(scores_alt)

    return callback


def corr_scoring2(
    template: shm_type,
    template_filter: shm_type,
    ft_target: shm_type,
    inv_denominator: shm_type,
    numerator: shm_type,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: BackendArray,
    callback: CallbackClass,
    interpolation_order: int,
    target_filter: shm_type = None,
    template_mask: shm_type = None,
    score_mask: shm_type = None,
    template_background: shm_type = None,
) -> CallbackClass:
    template = be.from_sharedarr(template)
    ft_target = be.from_sharedarr(ft_target)
    inv_denominator = be.from_sharedarr(inv_denominator)
    numerator = be.from_sharedarr(numerator)
    template_filter = be.from_sharedarr(template_filter)
    score_mask = be.from_sharedarr(score_mask)

    tar_batch, tmpl_batch = _get_batch_dim(ft_target, template)

    nd = len(fast_shape)
    sqz_slice = tuple(slice(0, 1) if i in tar_batch else slice(None) for i in range(nd))
    tmpl_subset = tuple(0 if i in tar_batch else slice(None) for i in range(nd))

    axes, shape, batched = None, fast_shape, len(tmpl_batch) > 0
    if len(tar_batch) or len(tmpl_batch):
        axes = tuple(i for i in range(nd) if i not in (*tar_batch, *tmpl_batch))
        shape = tuple(fast_shape[i] for i in axes)

    unpadded_slice = tuple(
        slice(None) if i in (*tar_batch, *tmpl_batch) else slice(0, x)
        for i, x in enumerate(template.shape)
    )

    arr = be.zeros(fast_shape, be._float_dtype)
    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)
    arr_sqz, ft_sqz = arr[sqz_slice], ft_temp[sqz_slice]

    n_obs = None
    if template_mask is not None:
        template_mask = be.from_sharedarr(template_mask)
        n_obs = be.sum(template_mask, axis=axes, keepdims=True)

    norm_template = conditional_execute(standardize, n_obs is not None)
    norm_sub = conditional_execute(be.subtract, numerator.shape != (1,))
    norm_mul = conditional_execute(be.multiply, inv_denominator.shape != (1,))
    norm_mask = conditional_execute(be.multiply, score_mask.shape != (1,))

    template_filter_func = _create_filter_func(
        arr_shape=template.shape,
        template_filter=template_filter,
        arr_padded=True,
    )

    for index in range(rotations.shape[0]):
        be.fill(arr, 0)
        rotation = rotations[index]
        _, _ = be.rigid_transform(
            arr=template[tmpl_subset],
            rotation_matrix=rotation,
            out=arr_sqz[tmpl_subset],
            use_geometric_center=True,
            order=interpolation_order,
            cache=False,
            batched=batched,
        )
        arr_norm = template_filter_func(arr_sqz, ft_sqz)
        norm_template(arr_norm[unpadded_slice], template_mask, n_obs, axis=axes)

        ft_sqz = be.rfftn(arr_norm, out=ft_sqz, axes=axes, s=shape)
        arr = _correlate_fts(ft_target, ft_sqz, ft_temp, arr, shape, axes)

        arr = norm_sub(arr, numerator, out=arr)
        arr = norm_mul(arr, inv_denominator, out=arr)
        arr = norm_mask(arr, score_mask, out=arr)

        callback(arr, rotation_matrix=rotation)

    return callback


def _get_batch_dim(target, template):
    target_batch, template_batch = [], []
    for i in range(len(target.shape)):
        if target.shape[i] == 1 and template.shape[i] != 1:
            template_batch.append(i)
        if target.shape[i] != 1 and template.shape[i] == 1:
            target_batch.append(i)

    return target_batch, template_batch


def _correlate_fts(ft_tar, ft_tmpl, ft_buffer, real_buffer, fast_shape, axes=None):
    ft_buffer = be.multiply(ft_tar, ft_tmpl, out=ft_buffer)
    return be.irfftn(ft_buffer, out=real_buffer, s=fast_shape, axes=axes)


def _create_filter_func(
    arr_shape: Tuple[int],
    template_filter: BackendArray,
    arr_padded: bool = False,
    axes=None,
) -> Callable:
    """
    Configure template filtering function for Fourier transforms.

    Conceptually we distinguish between three cases. The base case
    is that both template and the corresponding filter have the same
    shape. Padding is used when the template filter is larger than
    the template, for instance to better resolve Fourier filters. Finally
    this function also handles the case when a filter is supposed to be
    broadcasted over the template batch dimension.

    Parameters
    ----------
    arr_shape : tuple of ints
        Shape of the array to be filtered.
    template_filter : BackendArray
        Precomputed filter to apply in the frequency domain.
    arr_padded : bool, optional
        Whether the input template is padded and will need to be cropped
        to arr_shape prior to filter applications. Defaults to False.
    axes : tuple of ints, optional
        Axes to perform Fourier transform over.

    Returns
    -------
    Callable
        Filter function with parameters template, ft_temp and template_filter.
    """
    filter_shape = template_filter.shape
    if filter_shape == (1,):
        return conditional_execute(identity, execute_operation=True)

    # Default case, all shapes are correctly matched
    def _apply_filter(template, ft_temp):
        ft_temp = be.rfftn(template, out=ft_temp, s=template.shape)
        ft_temp = be.multiply(ft_temp, template_filter, out=ft_temp)
        return be.irfftn(ft_temp, out=template, s=template.shape)

    if not arr_padded:
        return _apply_filter

    # Array is padded but filter is w.r.t to the original template
    real_subset = tuple(slice(0, x) for x in arr_shape)
    _template = be.zeros(arr_shape, be._float_dtype)
    _ft_temp = be.zeros(filter_shape, be._complex_dtype)

    def _apply_filter_subset(template, ft_temp):
        _template[:] = template[real_subset]
        template[real_subset] = _apply_filter(_template, _ft_temp)
        return template

    return _apply_filter_subset


def _setup_background_correction(
    fast_shape: Tuple[int],
    template_background: BackendArray,
    rotation_buffer: BackendArray,
    unpadded_slice: Tuple[slice],
    interpolation_order: int = 3,
    tmpl_filter_func: Callable = identity,
    norm_template: Callable = identity,
    axes=None,
    shape=None,
):
    scores_noise = be.zeros(fast_shape, be._float_dtype)
    template_background = be.from_sharedarr(template_background)

    fwd_shape = shape
    if shape is not None:
        fwd_shape = shape

    def compute_norm(arr, ft_target, ft_temp, matrix, template_mask, n_obs):
        _ = be.rigid_transform(
            arr=template_background,
            rotation_matrix=matrix,
            out=rotation_buffer,
            use_geometric_center=True,
            order=interpolation_order,
            cache=True,
        )
        template_rot = tmpl_filter_func(rotation_buffer, ft_temp)
        template_rot = norm_template(template_rot, template_mask, n_obs, axis=axes)

        arr = to_padded(arr, template_rot, unpadded_slice)
        ft_temp = be.rfftn(arr, out=ft_temp, axes=axes, s=fwd_shape)
        return _correlate_fts(ft_target, ft_temp, ft_temp, arr, fast_shape, axes)

    return scores_noise, compute_norm


MATCHING_EXHAUSTIVE_REGISTER = {
    "CC": (cc_setup, corr_scoring),
    "LCC": (lcc_setup, corr_scoring),
    "CORR": (corr_setup, corr_scoring),
    "CAM": (cam_setup, corr_scoring),
    # "NCC": (ncc_setup, ncc_scoring),
    "FLCSphericalMask": (flcSphericalMask_setup, corr_scoring),
    "FLC": (flc_setup, flc_scoring),
    "MCC": (mcc_setup, mcc_scoring),
    "batchFLCSphericalMask": (flcSphericalMask_setup, corr_scoring2),
    "batchFLC": (flc_setup, flc_scoring2),
}
