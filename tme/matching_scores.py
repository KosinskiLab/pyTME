""" Implements a range of cross-correlation coefficients.

    Copyright (c) 2023-2024 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import warnings
from typing import Callable, Tuple, Dict, Optional

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


def _setup_template_filtering(
    forward_ft_shape: Tuple[int],
    inverse_ft_shape: Tuple[int],
    template_shape: Tuple[int],
    template_filter: BackendArray,
    rfftn: Callable = None,
    irfftn: Callable = None,
) -> Callable:
    """
    Configure template filtering function for Fourier transforms.

    Parameters
    ----------
    forward_ft_shape : tuple of ints
        Shape for the forward Fourier transform.
    inverse_ft_shape : tuple of ints
        Shape for the inverse Fourier transform.
    template_shape : tuple of ints
        Shape of the template to be filtered.
    template_filter : BackendArray
        Precomputed filter to apply in the frequency domain.
    rfftn : Callable, optional
        Real-to-complex FFT function.
    irfftn : Callable, optional
        Complex-to-real inverse FFT function.

    Returns
    -------
    Callable
        Filter function with parameters template, ft_temp and template_filter.

    Notes
    -----
        If the shape of template_filter does not match inverse_ft_shape
        the template is assumed to be padded and cropped back to template_shape
        prior to filter application.
    """
    if be.size(template_filter) == 1:
        return conditional_execute(identity, identity, False)

    shape_mismatch = False
    if not _shape_match(template_filter.shape, inverse_ft_shape):
        shape_mismatch = True
        forward_ft_shape = template_shape
        inverse_ft_shape = template_filter.shape

    if rfftn is not None and irfftn is not None:
        rfftn, irfftn = be.build_fft(
            fast_shape=forward_ft_shape,
            fast_ft_shape=inverse_ft_shape,
            real_dtype=be._float_dtype,
            complex_dtype=be._complex_dtype,
            inverse_fast_shape=forward_ft_shape,
        )

    # Default case, all shapes are correctly matched
    def _apply_template_filter(template, ft_temp, template_filter):
        ft_temp = rfftn(template, ft_temp)
        ft_temp = be.multiply(ft_temp, template_filter, out=ft_temp)
        return irfftn(ft_temp, template)

    # Template is padded, filter is not. Crop and assign for continuous arrays
    if shape_mismatch:
        real_subset = tuple(slice(0, x) for x in forward_ft_shape)
        _template = be.zeros(forward_ft_shape, be._float_dtype)
        _ft_temp = be.zeros(inverse_ft_shape, be._complex_dtype)

        def _apply_filter_shape_mismatch(template, ft_temp, template_filter):
            _template[:] = template[real_subset]
            return _apply_template_filter(_template, _ft_temp, template_filter)

        return _apply_filter_shape_mismatch

    return _apply_template_filter


def cc_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: BackendArray,
    target: BackendArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shared_memory_handler: type,
    **kwargs,
) -> Dict:
    """
    Setup function for comuting a unnormalized cross-correlation between
    ``target`` (f) and ``template`` (g)

    .. math::

        \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*).


    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    target_pad_ft = be.zeros(fast_ft_shape, be._complex_dtype)
    target_pad_ft = rfftn(be.topleft_pad(target, fast_shape), target_pad_ft)
    numerator = be.zeros(1, be._float_dtype)
    inv_denominator = be.zeros(1, be._float_dtype) + 1

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(template, shared_memory_handler),
        "ft_target": be.to_sharedarr(target_pad_ft, shared_memory_handler),
        "inv_denominator": be.to_sharedarr(inv_denominator, shared_memory_handler),
        "numerator": be.to_sharedarr(numerator, shared_memory_handler),
    }

    return ret


def lcc_setup(target: BackendArray, template: BackendArray, **kwargs) -> Dict:
    """
    Setup function for computing a laplace cross-correlation between
    ``target`` (f) and ``template`` (g)

    .. math::

        \\mathcal{F}^{-1}(\\mathcal{F}(\\nabla^{2}f) \\cdot \\mathcal{F}(\\nabla^{2} g)^*)


    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    target, template = be.to_numpy_array(target), be.to_numpy_array(template)
    kwargs["target"] = be.to_backend_array(laplace(target, mode="wrap"))
    kwargs["template"] = be.to_backend_array(laplace(template, mode="wrap"))
    return cc_setup(**kwargs)


def corr_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: BackendArray,
    template_mask: BackendArray,
    template_filter: BackendArray,
    target: BackendArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shared_memory_handler: type,
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
    target_pad = be.topleft_pad(target, fast_shape)

    # The exact composition of the denominator is debatable
    # scikit-image match_template multiplies the running sum of the target
    # with a scaling factor derived from the template. This is probably appropriate
    # in pattern matching situations where the template exists in the target
    ft_window = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_window = rfftn(be.topleft_pad(template_mask, fast_shape), ft_window)
    ft_target = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_target2 = be.zeros(fast_ft_shape, be._complex_dtype)
    denominator = be.zeros(fast_shape, be._float_dtype)
    window_sum = be.zeros(fast_shape, be._float_dtype)

    ft_target = rfftn(target_pad, ft_target)
    ft_target2 = rfftn(be.square(target_pad), ft_target2)
    ft_target2 = be.multiply(ft_target2, ft_window, out=ft_target2)
    denominator = irfftn(ft_target2, denominator)
    ft_window = be.multiply(ft_target, ft_window, out=ft_window)
    window_sum = irfftn(ft_window, window_sum)

    target_pad, ft_target2, ft_window = None, None, None

    # TODO: Factor in template_filter here
    if be.size(template_filter) != 1:
        warnings.warn(
            "CORR scores obtained with template_filter are not correctly scaled. "
            "Please use a different score or consider only relative peak heights."
        )
    n_obs, norm_func = be.sum(template_mask), normalize_template
    if be.datatype_bytes(template_mask.dtype) == 2:
        norm_func = _normalize_template_overflow_safe
        n_obs = be.sum(be.astype(template_mask, be._overflow_safe_dtype))

    template = norm_func(template, template_mask, n_obs)
    template_mean = be.sum(be.multiply(template, template_mask))
    template_mean = be.divide(template_mean, n_obs)
    template_ssd = be.sum(be.square(template - template_mean) * template_mask)
    template_volume = np.prod(tuple(int(x) for x in template.shape))
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
        "template": be.to_sharedarr(template, shared_memory_handler),
        "ft_target": be.to_sharedarr(ft_target, shared_memory_handler),
        "inv_denominator": be.to_sharedarr(denominator, shared_memory_handler),
        "numerator": be.to_sharedarr(numerator, shared_memory_handler),
    }

    return ret


def cam_setup(template: BackendArray, target: BackendArray, **kwargs) -> Dict:
    """
    Like :py:meth:`corr_setup` but with standardized ``target``, ``template``

    .. math::

        f' = \\frac{f - \\overline{f}}{\\sigma_f}.

    Notes
    -----
    To be used with :py:meth:`corr_scoring`.
    """
    template = (template - be.mean(template)) / be.std(template)
    target = (target - be.mean(target)) / be.std(target)
    return corr_setup(template=template, target=target, **kwargs)


def flc_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: BackendArray,
    template_mask: BackendArray,
    target: BackendArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shared_memory_handler: type,
    **kwargs,
) -> Dict:
    """
    Setup function for :py:meth:`flc_scoring`.
    """
    target_pad = be.topleft_pad(target, fast_shape)
    ft_target = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_target2 = be.zeros(fast_ft_shape, be._complex_dtype)

    ft_target = rfftn(target_pad, ft_target)
    target_pad = be.square(target_pad, out=target_pad)
    ft_target2 = rfftn(target_pad, ft_target2)
    template = normalize_template(template, template_mask, be.sum(template_mask))

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(template, shared_memory_handler),
        "template_mask": be.to_sharedarr(template_mask, shared_memory_handler),
        "ft_target": be.to_sharedarr(ft_target, shared_memory_handler),
        "ft_target2": be.to_sharedarr(ft_target2, shared_memory_handler),
    }

    return ret


def flcSphericalMask_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: BackendArray,
    template_mask: BackendArray,
    target: BackendArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shared_memory_handler: type,
    **kwargs,
) -> Dict:
    """
    Setup for :py:meth:`corr_scoring`, like :py:meth:`flc_setup` but for rotation
    invariant masks.
    """
    n_obs, norm_func = be.sum(template_mask), normalize_template
    if be.datatype_bytes(template_mask.dtype) == 2:
        norm_func = _normalize_template_overflow_safe
        n_obs = be.sum(be.astype(template_mask, be._overflow_safe_dtype))

    target_pad = be.topleft_pad(target, fast_shape)
    temp = be.zeros(fast_shape, be._float_dtype)
    temp2 = be.zeros(fast_shape, be._float_dtype)
    numerator = be.zeros(1, be._float_dtype)
    ft_target = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_template_mask = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_temp = be.zeros(fast_ft_shape, be._complex_dtype)

    template = norm_func(template, template_mask, n_obs)
    ft_template_mask = rfftn(
        be.topleft_pad(template_mask, fast_shape), ft_template_mask
    )

    # E(X^2) - E(X)^2
    ft_target = rfftn(be.square(target_pad), ft_target)
    ft_temp = be.multiply(ft_target, ft_template_mask, out=ft_temp)
    temp2 = irfftn(ft_temp, temp2)
    temp2 = be.divide(temp2, n_obs, out=temp2)

    ft_target = rfftn(target_pad, ft_target)
    ft_temp = be.multiply(ft_target, ft_template_mask, out=ft_temp)
    temp = irfftn(ft_temp, temp)
    temp = be.divide(temp, n_obs, out=temp)
    temp = be.square(temp, out=temp)

    temp = be.subtract(temp2, temp, out=temp)
    temp = be.maximum(temp, 0.0, out=temp)
    temp = be.sqrt(temp, out=temp)

    # Avoide divide by zero warnings
    mask = temp > be.eps(be._float_dtype)
    temp = be.multiply(temp, mask * n_obs, out=temp)
    temp = be.add(temp, ~mask, out=temp)
    temp2 = be.divide(1, temp, out=temp)
    temp2 = be.multiply(temp2, mask, out=temp2)

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(template, shared_memory_handler),
        "template_mask": be.to_sharedarr(template_mask, shared_memory_handler),
        "ft_target": be.to_sharedarr(ft_target, shared_memory_handler),
        "inv_denominator": be.to_sharedarr(temp2, shared_memory_handler),
        "numerator": be.to_sharedarr(numerator, shared_memory_handler),
    }

    return ret


def mcc_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: BackendArray,
    template_mask: BackendArray,
    target: BackendArray,
    target_mask: BackendArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    shared_memory_handler: Callable,
    **kwargs,
) -> Dict:
    """
    Setup function for :py:meth:`mcc_scoring`.
    """
    target = be.multiply(target, target_mask > 0, out=target)
    target_pad = be.topleft_pad(target, fast_shape)

    ft_target = be.zeros(fast_ft_shape, be._complex_dtype)
    ft_target2 = be.zeros(fast_ft_shape, be._complex_dtype)
    target_mask_ft = be.zeros(fast_ft_shape, be._complex_dtype)

    ft_target = rfftn(target_pad, ft_target)
    ft_target2 = rfftn(be.square(target_pad), ft_target2)
    target_mask_ft = rfftn(be.topleft_pad(target_mask, fast_shape), target_mask_ft)

    ret = {
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "template": be.to_sharedarr(template, shared_memory_handler),
        "template_mask": be.to_sharedarr(template_mask, shared_memory_handler),
        "ft_target": be.to_sharedarr(ft_target, shared_memory_handler),
        "ft_target2": be.to_sharedarr(ft_target2, shared_memory_handler),
        "ft_target_mask": be.to_sharedarr(target_mask_ft, shared_memory_handler),
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
) -> Optional[CallbackClass]:
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
    rfftn, irfftn = be.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=be._float_dtype,
        complex_dtype=be._complex_dtype,
        temp_real=arr,
        temp_fft=ft_temp,
    )

    template_filter_func = _setup_template_filtering(
        forward_ft_shape=fast_shape,
        inverse_ft_shape=fast_ft_shape,
        template_shape=template.shape,
        template_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
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
            cache=True,
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
) -> Optional[CallbackClass]:
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

    Returns
    -------
    Optional[CallbackClass]
        ``callback`` if provided otherwise None.

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

    rfftn, irfftn = be.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=float_dtype,
        complex_dtype=complex_dtype,
        temp_real=arr,
        temp_fft=ft_temp,
    )

    template_filter_func = _setup_template_filtering(
        forward_ft_shape=fast_shape,
        inverse_ft_shape=fast_ft_shape,
        template_shape=template.shape,
        template_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
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
            rotation_matrix=rotations[index],
            out=arr,
            out_mask=temp,
            use_geometric_center=True,
            order=interpolation_order,
            cache=True,
        )

        n_obs = be.sum(temp)
        arr = template_filter_func(arr, ft_temp, template_filter)
        arr = normalize_template(arr, temp, n_obs)

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

    rfftn, irfftn = be.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=float_dtype,
        complex_dtype=complex_dtype,
        temp_real=numerator,
        temp_fft=temp_ft,
    )

    template_filter_func = _setup_template_filtering(
        forward_ft_shape=fast_shape,
        inverse_ft_shape=fast_ft_shape,
        template_shape=template.shape,
        template_filter=template_filter,
        rfftn=rfftn,
        irfftn=irfftn,
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
            cache=True,
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
        be.divide(numerator, temp2, out=temp)
        be.clip(temp, a_min=-1, a_max=1, out=temp)

        # Apply overlap ratio threshold
        number_px_threshold = overlap_ratio * be.max(
            mask_overlap, axis=axes, keepdims=True
        )
        temp[mask_overlap < number_px_threshold] = 0.0
        callback_func(temp, rotation_matrix=rotation)

    return callback


MATCHING_EXHAUSTIVE_REGISTER = {
    "CC": (cc_setup, corr_scoring),
    "LCC": (lcc_setup, corr_scoring),
    "CORR": (corr_setup, corr_scoring),
    "CAM": (cam_setup, corr_scoring),
    "FLCSphericalMask": (flcSphericalMask_setup, corr_scoring),
    "FLC": (flc_setup, flc_scoring),
    "MCC": (mcc_setup, mcc_scoring),
}
