""" Implements a range of cross-correlation coefficients.

    Copyright (c) 2023-2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Callable, Tuple, Dict

import numpy as np
from scipy.ndimage import laplace

from .backends import backend as be
from .types import CallbackClass, BackendArray, shm_type
from .matching_utils import (
    conditional_execute,
    identity,
    normalize_template,
    _normalize_template_overflow_safe,
)


def _shape_match(shape1: Tuple[int], shape2: Tuple[int]) -> bool:
    """
    Determine whether ``shape1`` is equal to ``shape2``.

    Parameters
    ----------
    shape1, shape2 : tuple of ints
        Shapes to compare.

    Returns
    -------
    Bool
        ``shape1`` is equal to ``shape2``.
    """
    if len(shape1) != len(shape2):
        return False
    return shape1 == shape2


def _create_filter_func(
    fwd_shape: Tuple[int],
    inv_shape: Tuple[int],
    arr_shape: Tuple[int],
    arr_filter: BackendArray,
    arr_ft_shape: Tuple[int],
    inv_output_shape: Tuple[int],
    real_dtype: type,
    cmpl_dtype: type,
    fwd_axes=None,
    inv_axes=None,
    rfftn: Callable = None,
    irfftn: Callable = None,
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
    fwd_shape : tuple of ints
        Input shape of rfftn.
    inv_shape : tuple of ints
        Input shape of irfftn.
    arr_shape : tuple of ints
        Shape of the array to be filtered.
    arr_ft_shape : tuple of ints
        Shape of the Fourier transform of the array.
    arr_filter : BackendArray
        Precomputed filter to apply in the frequency domain.
    rfftn : Callable, optional
        Foward Fourier transform.
    irfftn : Callable, optional
        Inverse Fourier transform.

    Returns
    -------
    Callable
        Filter function with parameters template, ft_temp and template_filter.
    """
    if be.size(arr_filter) == 1:
        return conditional_execute(identity, identity, False)

    filter_shape = tuple(int(x) for x in arr_filter.shape)
    try:
        product_ft_shape = np.broadcast_shapes(arr_ft_shape, filter_shape)
    except ValueError:
        product_ft_shape, inv_output_shape = filter_shape, arr_shape

    rfft_valid = _shape_match(arr_shape, fwd_shape)
    rfft_valid = rfft_valid and _shape_match(product_ft_shape, inv_shape)
    rfft_valid = rfft_valid and rfftn is not None and irfftn is not None

    # FTTs were not or built for the wrong shape
    if not rfft_valid:
        _fwd_shape = arr_shape
        if all(x > y for x, y in zip(arr_shape, product_ft_shape)):
            _fwd_shape = fwd_shape

        rfftn, irfftn = be.build_fft(
            fwd_shape=_fwd_shape,
            inv_shape=product_ft_shape,
            real_dtype=real_dtype,
            cmpl_dtype=cmpl_dtype,
            inv_output_shape=inv_output_shape,
            fwd_axes=fwd_axes,
            inv_axes=inv_axes,
        )

    # Default case, all shapes are correctly matched
    def _apply_filter(template, ft_temp, template_filter):
        ft_temp = rfftn(template, ft_temp)
        ft_temp = be.multiply(ft_temp, template_filter, out=ft_temp)
        return irfftn(ft_temp, template)

    if not _shape_match(arr_ft_shape, filter_shape):
        real_subset = tuple(slice(0, x) for x in arr_shape)
        _template = be.zeros(arr_shape, be._float_dtype)
        _ft_temp = be.zeros(product_ft_shape, be._complex_dtype)

        # Arr is padded, filter is not
        def _apply_filter_subset(template, ft_temp, template_filter):
            # TODO: Benchmark this
            _template[:] = template[real_subset]
            template[real_subset] = _apply_filter(_template, _ft_temp, template_filter)
            return template

        # Filter application requires a broadcasting operation
        def _apply_filter_broadcast(template, ft_temp, template_filter):
            _ft_prod = rfftn(template, _ft_temp2)
            _ft_res = be.multiply(_ft_prod, template_filter, out=_ft_temp)
            return irfftn(_ft_res, _template)

        if any(x > y and y == 1 for x, y in zip(filter_shape, arr_ft_shape)):
            _template = be.zeros(inv_output_shape, be._float_dtype)
            _ft_temp2 = be.zeros((1, *product_ft_shape[1:]), be._complex_dtype)
            return _apply_filter_broadcast

        return _apply_filter_subset

    return _apply_filter


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

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "ft_target": be.to_sharedarr(be.rfftn(target_pad, axes=axes), shm_handler),
        "inv_denominator": be.to_sharedarr(
            be.zeros(1, be._float_dtype) + 1, shm_handler
        ),
        "numerator": be.to_sharedarr(be.zeros(1, be._float_dtype), shm_handler),
    }

    return ret


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
    target = be.to_numpy_array(matching_data._target)
    template = be.to_numpy_array(matching_data._template)

    subsets = matching_data._batch_iter(
        target.shape,
        tuple(1 if i in matching_data._target_dim else 0 for i in range(target.ndim)),
    )
    for subset in subsets:
        target[subset] = laplace(target[subset], mode="wrap")

    subsets = matching_data._batch_iter(
        template.shape,
        tuple(1 if i in matching_data._template_dim else 0 for i in range(target.ndim)),
    )
    for subset in subsets:
        template[subset] = laplace(template[subset], mode="wrap")

    matching_data._target = target
    matching_data._template = template

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

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(template, shm_handler),
        "ft_target": be.to_sharedarr(ft_target, shm_handler),
        "inv_denominator": be.to_sharedarr(denominator, shm_handler),
        "numerator": be.to_sharedarr(numerator, shm_handler),
    }

    return ret


def cam_setup(matching_data, **kwargs) -> Dict:
    """
    Like :py:meth:`corr_setup` but with standardized ``target``, ``template``

    .. math::

        f' = \\frac{f - \\overline{f}}{\\sigma_f}.

    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    template = matching_data._template
    axis = matching_data._batch_axis(matching_data._target_batch)
    matching_data._template = be.divide(
        be.subtract(template, be.mean(template, axis=axis, keepdims=True)),
        be.std(template, axis=axis, keepdims=True),
    )
    target = matching_data._target
    axis = matching_data._batch_axis(matching_data._template_batch)
    matching_data._target = be.divide(
        be.subtract(target, be.mean(target, axis=axis, keepdims=True)),
        be.std(target, axis=axis, keepdims=True),
    )
    return corr_setup(matching_data=matching_data, **kwargs)


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

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(matching_data.template_mask, shm_handler),
        "ft_target": be.to_sharedarr(ft_target, shm_handler),
        "ft_target2": be.to_sharedarr(ft_target2, shm_handler),
    }

    return ret


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
    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(template_mask, shm_handler),
        "ft_target": be.to_sharedarr(ft_target, shm_handler),
        "inv_denominator": be.to_sharedarr(temp2, shm_handler),
        "numerator": be.to_sharedarr(be.zeros(1, be._float_dtype), shm_handler),
    }

    return ret


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
    target = be.multiply(target, target_mask > 0, out=target)

    target = be.topleft_pad(
        target,
        matching_data._batch_shape(fast_shape, matching_data._template_batch),
    )
    target_mask = be.topleft_pad(
        target_mask,
        matching_data._batch_shape(fast_shape, matching_data._template_batch),
    )
    ax = matching_data._batch_axis(matching_data._batch_mask)

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(matching_data.template_mask, shm_handler),
        "ft_target": be.to_sharedarr(be.rfftn(target, axes=ax), shm_handler),
        "ft_target2": be.to_sharedarr(
            be.rfftn(be.square(target), axes=ax), shm_handler
        ),
        "ft_target_mask": be.to_sharedarr(be.rfftn(target_mask, axes=ax), shm_handler),
    }

    return ret


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

    Returns
    -------
    Optional[CallbackClass]
        ``callback`` if provided otherwise None.
    """
    template = be.from_sharedarr(template)
    ft_target = be.from_sharedarr(ft_target)
    inv_denominator = be.from_sharedarr(inv_denominator)
    numerator = be.from_sharedarr(numerator)
    template_filter = be.from_sharedarr(template_filter)

    norm_func, norm_template, mask_sum = normalize_template, False, 1
    if template_mask is not None:
        template_mask = be.from_sharedarr(template_mask)
        norm_template, mask_sum = True, be.sum(template_mask)
        if be.datatype_bytes(template_mask.dtype) == 2:
            norm_func = _normalize_template_overflow_safe
            mask_sum = be.sum(be.astype(template_mask, be._overflow_safe_dtype))

    callback_func = conditional_execute(callback, callback is not None)
    norm_template = conditional_execute(norm_func, norm_template)
    norm_numerator = conditional_execute(
        be.subtract, identity, _shape_match(numerator.shape, fast_shape)
    )
    norm_denominator = conditional_execute(
        be.multiply, identity, _shape_match(inv_denominator.shape, fast_shape)
    )

    arr = be.zeros(fast_shape, be._float_dtype)
    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)

    _fftargs = {
        "real_dtype": be._float_dtype,
        "cmpl_dtype": be._complex_dtype,
        "inv_output_shape": fast_shape,
        "fwd_axes": None,
        "inv_axes": None,
        "inv_shape": fast_ft_shape,
        "temp_fwd": arr,
    }

    _fftargs["fwd_shape"] = _fftargs["temp_fwd"].shape
    rfftn, irfftn = be.build_fft(temp_inv=ft_temp, **_fftargs)
    _ = _fftargs.pop("temp_fwd", None)

    template_filter_func = _create_filter_func(
        arr_shape=template.shape,
        arr_ft_shape=fast_ft_shape,
        arr_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
        **_fftargs,
    )

    unpadded_slice = tuple(slice(0, stop) for stop in template.shape)
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        arr = be.fill(arr, 0)
        arr, _ = be.rigid_transform(
            arr=template,
            rotation_matrix=rotation,
            out=arr,
            use_geometric_center=True,
            order=interpolation_order,
            cache=False,
        )
        arr = template_filter_func(arr, ft_temp, template_filter)
        norm_template(arr[unpadded_slice], template_mask, mask_sum)

        ft_temp = rfftn(arr, ft_temp)
        ft_temp = be.multiply(ft_target, ft_temp, out=ft_temp)
        arr = irfftn(ft_temp, arr)

        arr = norm_numerator(arr, numerator, out=arr)
        arr = norm_denominator(arr, inv_denominator, out=arr)
        callback_func(arr, rotation_matrix=rotation)

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
    callback_class_args : Dict
        Dictionary of arguments to be passed to ``callback``.
    interpolation_order : int
        Spline order for template rotations.

    References
    ----------
    .. [1]  Hrabe T. et al, J. Struct. Biol. 178, 177 (2012).
    """
    float_dtype, complex_dtype = be._float_dtype, be._complex_dtype
    template = be.from_sharedarr(template)
    template_mask = be.from_sharedarr(template_mask)
    ft_target = be.from_sharedarr(ft_target)
    ft_target2 = be.from_sharedarr(ft_target2)
    template_filter = be.from_sharedarr(template_filter)

    arr = be.zeros(fast_shape, float_dtype)
    temp = be.zeros(fast_shape, float_dtype)
    temp2 = be.zeros(fast_shape, float_dtype)
    ft_temp = be.zeros(fast_ft_shape, complex_dtype)
    ft_denom = be.zeros(fast_ft_shape, complex_dtype)

    _fftargs = {
        "real_dtype": be._float_dtype,
        "cmpl_dtype": be._complex_dtype,
        "inv_output_shape": fast_shape,
        "fwd_axes": None,
        "inv_axes": None,
        "inv_shape": fast_ft_shape,
        "temp_fwd": arr,
    }

    _fftargs["fwd_shape"] = _fftargs["temp_fwd"].shape
    rfftn, irfftn = be.build_fft(temp_inv=ft_temp, **_fftargs)
    _ = _fftargs.pop("temp_fwd", None)

    template_filter_func = _create_filter_func(
        arr_shape=template.shape,
        arr_ft_shape=fast_ft_shape,
        arr_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
        **_fftargs,
    )

    eps = be.eps(float_dtype)
    callback_func = conditional_execute(callback, callback is not None)
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        arr = be.fill(arr, 0)
        temp = be.fill(temp, 0)
        arr, temp = be.rigid_transform(
            arr=template,
            arr_mask=template_mask,
            rotation_matrix=rotation,
            out=arr,
            out_mask=temp,
            use_geometric_center=True,
            order=interpolation_order,
            cache=False,
        )

        n_obs = be.sum(temp)
        arr = template_filter_func(arr, ft_temp, template_filter)
        arr = normalize_template(arr, temp, n_obs, axis=None)

        ft_temp = rfftn(temp, ft_temp)
        ft_denom = be.multiply(ft_target, ft_temp, out=ft_denom)
        temp = irfftn(ft_denom, temp)
        ft_denom = be.multiply(ft_target2, ft_temp, out=ft_denom)
        temp2 = irfftn(ft_denom, temp2)

        ft_temp = rfftn(arr, ft_temp)
        ft_temp = be.multiply(ft_target, ft_temp, out=ft_temp)
        arr = irfftn(ft_temp, arr)

        arr = be.norm_scores(arr, temp2, temp, n_obs, eps, arr)
        callback_func(arr, rotation_matrix=rotation)

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

    _fftargs = {
        "real_dtype": be._float_dtype,
        "cmpl_dtype": be._complex_dtype,
        "inv_output_shape": fast_shape,
        "fwd_axes": None,
        "inv_axes": None,
        "inv_shape": fast_ft_shape,
        "temp_fwd": temp,
    }

    _fftargs["fwd_shape"] = _fftargs["temp_fwd"].shape
    rfftn, irfftn = be.build_fft(temp_inv=temp_ft, **_fftargs)
    _ = _fftargs.pop("temp_fwd", None)

    template_filter_func = _create_filter_func(
        arr_shape=template.shape,
        arr_ft_shape=fast_ft_shape,
        arr_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
        **_fftargs,
    )

    callback_func = conditional_execute(callback, callback is not None)
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
            cache=False,
        )

        template_filter_func(template_rot, temp_ft, template_filter)
        normalize_template(template_rot, temp, be.sum(temp))

        temp_ft = rfftn(template_rot, temp_ft)
        temp2 = irfftn(target_mask_ft * temp_ft, temp2)
        numerator = irfftn(target_ft * temp_ft, numerator)

        # temp template_mask_rot | temp_ft template_mask_rot_ft
        # Calculate overlap of masks at every point in the convolution.
        # Locations with high overlap should not be taken into account.
        temp_ft = rfftn(temp, temp_ft)
        mask_overlap = irfftn(temp_ft * target_mask_ft, mask_overlap)
        be.maximum(mask_overlap, eps, out=mask_overlap)
        temp = irfftn(temp_ft * target_ft, temp)

        be.subtract(
            numerator,
            be.divide(be.multiply(temp, temp2), mask_overlap),
            out=numerator,
        )

        # temp_3 = fixed_denom
        be.multiply(temp_ft, target_ft2, out=temp_ft)
        temp3 = irfftn(temp_ft, temp3)
        be.subtract(temp3, be.divide(be.square(temp), mask_overlap), out=temp3)
        be.maximum(temp3, 0.0, out=temp3)

        # temp = moving_denom
        temp_ft = rfftn(be.square(template_rot), temp_ft)
        be.multiply(target_mask_ft, temp_ft, out=temp_ft)
        temp = irfftn(temp_ft, temp)

        be.subtract(temp, be.divide(be.square(temp2), mask_overlap), out=temp)
        be.maximum(temp, 0.0, out=temp)

        # temp_2 = denom
        be.multiply(temp3, temp, out=temp)
        be.sqrt(temp, temp2)

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
        callback_func(temp, rotation_matrix=rotation)

    return callback


def _format_slice(shape, squeeze_axis):
    ret = tuple(
        slice(None) if i not in squeeze_axis else 0 for i, _ in enumerate(shape)
    )
    return ret


def _get_batch_dim(target, template):
    target_batch, template_batch = [], []
    for i in range(len(target.shape)):
        if target.shape[i] == 1 and template.shape[i] != 1:
            template_batch.append(i)
        if target.shape[i] != 1 and template.shape[i] == 1:
            target_batch.append(i)

    return target_batch, template_batch


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
) -> CallbackClass:
    callback_func = conditional_execute(callback, callback is not None)

    # Retrieve objects from shared memory
    template = be.from_sharedarr(template)
    template_mask = be.from_sharedarr(template_mask)
    ft_target = be.from_sharedarr(ft_target)
    ft_target2 = be.from_sharedarr(ft_target2)
    template_filter = be.from_sharedarr(template_filter)

    data_axes = None
    target_batch, template_batch = _get_batch_dim(ft_target, template)
    sqz_cmpl = tuple(1 if i in target_batch else x for i, x in enumerate(fast_ft_shape))
    sqz_slice = tuple(slice(0, 1) if x == 1 else slice(None) for x in sqz_cmpl)

    data_shape = fast_shape
    if len(target_batch) or len(template_batch):
        batch = (*target_batch, *template_batch)
        data_axes = tuple(i for i in range(len(fast_shape)) if i not in batch)
        data_shape = tuple(fast_shape[i] for i in data_axes)

    arr = be.zeros(fast_shape, be._float_dtype)
    temp = be.zeros(fast_shape, be._float_dtype)
    temp2 = be.zeros(fast_shape, be._float_dtype)
    ft_denom = be.zeros(fast_ft_shape, be._complex_dtype)

    tmp_sqz, arr_sqz, ft_temp = temp[sqz_slice], arr[sqz_slice], ft_denom[sqz_slice]
    if be.size(template_filter) != 1:
        ret_shape = np.broadcast_shapes(
            sqz_cmpl, tuple(int(x) for x in template_filter.shape)
        )
        ft_temp = be.zeros(ret_shape, be._complex_dtype)

    _fftargs = {
        "real_dtype": be._float_dtype,
        "cmpl_dtype": be._complex_dtype,
        "inv_output_shape": fast_shape,
        "fwd_axes": data_axes,
        "inv_axes": data_axes,
        "inv_shape": fast_ft_shape,
        "temp_fwd": arr_sqz if _shape_match(ft_temp.shape, sqz_cmpl) else arr,
    }

    # build_fft ignores fwd_shape if temp_fwd is given and serves only for bookkeeping
    _fftargs["fwd_shape"] = _fftargs["temp_fwd"].shape
    rfftn, irfftn = be.build_fft(temp_inv=ft_denom, **_fftargs)
    _ = _fftargs.pop("temp_fwd", None)

    template_filter_func = _create_filter_func(
        arr_shape=template.shape,
        arr_ft_shape=sqz_cmpl,
        arr_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
        **_fftargs,
    )

    eps = be.eps(be._float_dtype)
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        be.fill(arr, 0)
        be.fill(temp, 0)
        arr_sqz, tmp_sqz = be.rigid_transform(
            arr=template,
            arr_mask=template_mask,
            rotation_matrix=rotation,
            out=arr_sqz,
            out_mask=tmp_sqz,
            use_geometric_center=True,
            order=interpolation_order,
            cache=False,
        )
        n_obs = be.sum(tmp_sqz, axis=data_axes, keepdims=True)
        arr_norm = template_filter_func(arr_sqz, ft_temp, template_filter)
        arr_norm = normalize_template(arr_norm, tmp_sqz, n_obs, axis=data_axes)

        ft_temp = be.rfftn(tmp_sqz, ft_temp, axes=data_axes)
        ft_denom = be.multiply(ft_target, ft_temp, out=ft_denom)
        temp = be.irfftn(ft_denom, temp, axes=data_axes, s=data_shape)
        ft_denom = be.multiply(ft_target2, ft_temp, out=ft_denom)
        temp2 = be.irfftn(ft_denom, temp2, axes=data_axes, s=data_shape)

        ft_temp = rfftn(arr_norm, ft_denom)
        ft_denom = be.multiply(ft_target, ft_temp, out=ft_denom)
        arr = irfftn(ft_denom, arr)

        be.norm_scores(arr, temp2, temp, n_obs, eps, arr)
        callback_func(arr, rotation_matrix=rotation)

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
) -> CallbackClass:
    template = be.from_sharedarr(template)
    ft_target = be.from_sharedarr(ft_target)
    inv_denominator = be.from_sharedarr(inv_denominator)
    numerator = be.from_sharedarr(numerator)
    template_filter = be.from_sharedarr(template_filter)

    data_axes = None
    target_batch, template_batch = _get_batch_dim(ft_target, template)
    sqz_cmpl = tuple(1 if i in target_batch else x for i, x in enumerate(fast_ft_shape))
    sqz_slice = tuple(slice(0, 1) if x == 1 else slice(None) for x in sqz_cmpl)
    unpadded_slice = tuple(slice(0, stop) for stop in template.shape)
    if len(target_batch) or len(template_batch):
        batch = (*target_batch, *template_batch)
        data_axes = tuple(i for i in range(len(fast_shape)) if i not in batch)
        unpadded_slice = tuple(
            slice(None) if i in batch else slice(0, x)
            for i, x in enumerate(template.shape)
        )

    arr = be.zeros(fast_shape, be._float_dtype)
    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)
    arr_sqz, ft_sqz = arr[sqz_slice], ft_temp[sqz_slice]

    if be.size(template_filter) != 1:
        # The filter could be w.r.t the unpadded template
        ret_shape = tuple(
            int(x * y) if x == 1 or y == 1 else y
            for x, y in zip(sqz_cmpl, template_filter.shape)
        )
        ft_sqz = be.zeros(ret_shape, be._complex_dtype)

    norm_func, norm_template, mask_sum = normalize_template, False, 1
    if template_mask is not None:
        template_mask = be.from_sharedarr(template_mask)
        norm_template, mask_sum = True, be.sum(
            be.astype(template_mask, be._overflow_safe_dtype),
            axis=data_axes,
            keepdims=True,
        )
        if be.datatype_bytes(template_mask.dtype) == 2:
            norm_func = _normalize_template_overflow_safe

    callback_func = conditional_execute(callback, callback is not None)
    norm_template = conditional_execute(norm_func, norm_template)
    norm_numerator = conditional_execute(
        be.subtract, identity, _shape_match(numerator.shape, fast_shape)
    )
    norm_denominator = conditional_execute(
        be.multiply, identity, _shape_match(inv_denominator.shape, fast_shape)
    )

    _fftargs = {
        "real_dtype": be._float_dtype,
        "cmpl_dtype": be._complex_dtype,
        "fwd_axes": data_axes,
        "inv_axes": data_axes,
        "inv_shape": fast_ft_shape,
        "inv_output_shape": fast_shape,
        "temp_fwd": arr_sqz if _shape_match(ft_sqz.shape, sqz_cmpl) else arr,
    }

    # build_fft ignores fwd_shape if temp_fwd is given and serves only for bookkeeping
    _fftargs["fwd_shape"] = _fftargs["temp_fwd"].shape
    rfftn, irfftn = be.build_fft(temp_inv=ft_temp, **_fftargs)
    _ = _fftargs.pop("temp_fwd", None)

    template_filter_func = _create_filter_func(
        arr_shape=template.shape,
        arr_ft_shape=sqz_cmpl,
        arr_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
        **_fftargs,
    )

    for index in range(rotations.shape[0]):
        be.fill(arr, 0)
        rotation = rotations[index]
        arr_sqz, _ = be.rigid_transform(
            arr=template,
            rotation_matrix=rotation,
            out=arr_sqz,
            use_geometric_center=True,
            order=interpolation_order,
            cache=False,
        )
        arr_norm = template_filter_func(arr_sqz, ft_sqz, template_filter)
        norm_template(arr_norm[unpadded_slice], template_mask, mask_sum, axis=data_axes)

        ft_sqz = rfftn(arr_norm, ft_sqz)
        ft_temp = be.multiply(ft_target, ft_sqz, out=ft_temp)
        arr = irfftn(ft_temp, arr)

        arr = norm_numerator(arr, numerator, out=arr)
        arr = norm_denominator(arr, inv_denominator, out=arr)
        callback_func(arr, rotation_matrix=rotation)

    return callback


MATCHING_EXHAUSTIVE_REGISTER = {
    "CC": (cc_setup, corr_scoring),
    "LCC": (lcc_setup, corr_scoring),
    "CORR": (corr_setup, corr_scoring),
    "CAM": (cam_setup, corr_scoring),
    "FLCSphericalMask": (flcSphericalMask_setup, corr_scoring),
    "FLC": (flc_setup, flc_scoring),
    "MCC": (mcc_setup, mcc_scoring),
    "batchFLCSpherical": (flcSphericalMask_setup, corr_scoring2),
    "batchFLC": (flc_setup, flc_scoring2),
}
