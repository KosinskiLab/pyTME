"""
Implements a range of cross-correlation coefficients.

Copyright (c) 2023-2024 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Callable, Tuple, Dict

from .backends import backend as be
from .types import CallbackClass, BackendArray, shm_type
from .matching_utils import conditional_execute, identity, standardize, to_padded


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
    pad_shape, axes, *_ = matching_data._batch_shape(fast_shape)
    target_pad = be.topleft_pad(
        matching_data._to_full_batch(matching_data.target), pad_shape
    )

    return {
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "ft_target": be.to_sharedarr(be.rfftn(target_pad, axes=axes), shm_handler),
        "inv_denominator": be.to_sharedarr(be.zeros(1, be._float) + 1, shm_handler),
        "numerator": be.to_sharedarr(be.zeros(1, be._float), shm_handler),
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


def cam_setup(matching_data, **kwargs) -> Dict:
    """
    Like :py:meth:`flcSphericalMask_setup` but with standardized ``target`` and ``template``

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
    pad_shape, axes, *_ = matching_data._batch_shape(fast_shape)
    target_pad = be.topleft_pad(
        matching_data._to_full_batch(matching_data.target), pad_shape
    )

    ft_target = be.rfftn(target_pad, axes=axes)
    target_pad = be.square(target_pad, out=target_pad)
    ft_target2 = be.rfftn(target_pad, axes=axes)

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
    pad_shape, *_ = matching_data._batch_shape(fast_shape)
    target_pad = be.topleft_pad(
        matching_data._to_full_batch(matching_data.target), pad_shape
    )

    template_mask = matching_data.template_mask
    pad_shape, axes, axis = matching_data._batch_shape(fast_shape, target=False)
    template_mask_pad = be.topleft_pad(
        matching_data._to_full_batch(template_mask, target=False), pad_shape
    )

    data_shape = tuple(fast_shape[i] for i in axes)
    ft_temp = be.zeros(fast_ft_shape, be._complex)
    ft_template_mask = be.rfftn(template_mask_pad, s=data_shape, axes=axes)

    ft_target = be.rfftn(be.square(target_pad), axes=axes)
    ft_temp = be.multiply(ft_target, ft_template_mask, out=ft_temp)
    temp2 = be.irfftn(ft_temp, s=data_shape, axes=axes)

    ft_target = be.rfftn(target_pad, axes=axes)
    ft_temp = be.multiply(ft_target, ft_template_mask, out=ft_temp)
    temp = be.irfftn(ft_temp, s=data_shape, axes=axes)

    n_obs = be.sum(template_mask, axis=axis, keepdims=True)
    temp2 = be.norm_scores(1, temp2, temp, n_obs, be.eps(be._float), temp2)
    return {
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(template_mask, shm_handler),
        "ft_target": be.to_sharedarr(ft_target, shm_handler),
        "inv_denominator": be.to_sharedarr(temp2, shm_handler),
        "numerator": be.to_sharedarr(be.zeros(1, be._float), shm_handler),
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
    target = matching_data._to_full_batch(matching_data.target)
    target_mask = matching_data._to_full_batch(matching_data.target_mask)
    target = be.multiply(target, target_mask, out=target)

    pad_shape, x, *_ = matching_data._batch_shape(fast_shape)
    target = be.topleft_pad(target, pad_shape)
    target_mask = be.topleft_pad(target_mask, pad_shape)

    return {
        "template": be.to_sharedarr(matching_data.template, shm_handler),
        "template_mask": be.to_sharedarr(matching_data.template_mask, shm_handler),
        "ft_target": be.to_sharedarr(be.rfftn(target, axes=x), shm_handler),
        "ft_target2": be.to_sharedarr(be.rfftn(be.square(target), axes=x), shm_handler),
        "ft_target_mask": be.to_sharedarr(be.rfftn(target_mask, axes=x), shm_handler),
    }


def _scoring_buffers(
    template, ft_target, fast_shape, fast_ft_shape, template_filter, score_mask
):
    """Compute batch dimensions and allocate common scoring buffers.

    Returns
    -------
    tuple
        (batched, tmpl_axes, out_axes, spatial,
         tmpl_rot, arr, ft_denom, tmpl_rot_pad, ft_tmpl,
         rshape, center, tmpl_filter_func, norm_mask, top_slice)
    """
    batched = ft_target.ndim != template.ndim
    tb, ob = int(batched), 2 * int(batched)

    tmpl_axes, out_axes, spatial = None, None, fast_shape
    if batched:
        tmpl_axes = tuple(range(tb, template.ndim))
        out_axes = tuple(range(ob, len(fast_shape)))
        spatial = tuple(fast_shape[i] for i in out_axes)

    tmpl_rot = be.zeros(template.shape, be._float)
    arr = be.zeros(fast_shape, be._float)
    ft_denom = be.zeros(fast_ft_shape, be._complex)

    tmpl_rot_pad, reduced_ft = arr, fast_ft_shape
    if batched:
        reduced = (1,) + template.shape[:tb] + spatial
        reduced_ft = (
            (1,) + template.shape[:tb] + tuple(fast_ft_shape[i] for i in out_axes)
        )
        tmpl_rot_pad = be.zeros(reduced, be._float)
    ft_tmpl = be.zeros(reduced_ft, be._complex)

    rshape = template.shape[tb:]
    center = be.divide(be.to_backend_array(rshape) - 1, 2)
    tmpl_filter_func = _create_filter_func(
        template.shape, template_filter, axes=tmpl_axes
    )
    norm_mask = conditional_execute(be.multiply, score_mask.shape != (1,))
    top_slice = (slice(None),) * ob + tuple(slice(0, s) for s in rshape)

    return (
        batched,
        tmpl_axes,
        out_axes,
        spatial,
        tmpl_rot,
        arr,
        ft_denom,
        tmpl_rot_pad,
        ft_tmpl,
        rshape,
        center,
        tmpl_filter_func,
        norm_mask,
        top_slice,
    )


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

    (
        batched,
        tmpl_axes,
        out_axes,
        spatial,
        tmpl_rot,
        arr,
        ft_denom,
        tmpl_rot_pad,
        ft_tmpl,
        rshape,
        center,
        tmpl_filter_func,
        norm_mask,
        top_slice,
    ) = _scoring_buffers(
        template,
        ft_target,
        fast_shape,
        fast_ft_shape,
        template_filter,
        score_mask,
    )
    size = 1
    for s in rshape:
        size *= s
    n_spatial = 1
    for s in spatial:
        n_spatial *= s

    ft_target = be.multiply(ft_target, 1 / n_spatial**0.5)
    padded_ft_scale = (n_spatial / size) ** 0.5
    n_angles = rotations.shape[0]

    background_correction = template_background is not None
    if background_correction:
        scores_alt = be.zeros(fast_shape, be._float)
        compute_norm = _setup_background_correction(
            template_background=template_background,
            rotation_buffer=tmpl_rot,
            pad_buffer=tmpl_rot_pad,
            ft_buffer=ft_tmpl,
            unpadded_slice=top_slice,
            interpolation_order=interpolation_order,
            tmpl_filter_func=tmpl_filter_func,
            norm_template=lambda t, _m, _n, axis=None: standardize(
                t, 1, size, axis=axis
            ),
            axes=out_axes,
            shape=spatial,
        )

    for index in range(n_angles):
        rotation = rotations[index]
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation, center=center, shape=rshape, batched=batched
        )
        _ = be.rigid_transform(
            arr=template,
            matrix=matrix,
            out=tmpl_rot,
            order=interpolation_order,
            cache=not batched,
        )
        tmpl_rot = tmpl_filter_func(tmpl_rot)
        tmpl_rot = standardize(tmpl_rot, 1, size, axis=tmpl_axes)
        tmpl_rot_pad = to_padded(tmpl_rot_pad, tmpl_rot, top_slice)

        # Rescale the template FT to variance N
        ft_tmpl = be.rfftn(tmpl_rot_pad, out=ft_tmpl, axes=out_axes, s=spatial)
        ft_tmpl = be.multiply(ft_tmpl, padded_ft_scale, out=ft_tmpl)

        # Since ft_target has variance 1, the product will have variance N, yielding
        # a cross correlation score with a variance of 1 after normalization
        arr = _correlate_fts(ft_target, ft_tmpl, ft_denom, arr, spatial, out_axes)
        arr = norm_mask(arr, score_mask, out=arr)

        callback(arr, rotation_matrix=rotation)
        if background_correction:
            arr = compute_norm(arr, ft_target, ft_denom, matrix, None, None)
            arr = be.multiply(arr, padded_ft_scale, out=arr)
            arr = norm_mask(arr, score_mask, out=arr)
            scores_alt = be.maximum(arr, scores_alt, out=scores_alt)

    if background_correction:
        scores_alt = be.subtract(scores_alt, be.mean(scores_alt), out=scores_alt)
        callback.correct_background(scores_alt)

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

    Notes
    -----
    Both target and template can carry a leading batch dimension. The setup
    function prepads the target with a singleton template-batch dim so
    that broadcasting handles the combination naturally:

        ft_target  (b, 1, *ft_d) * ft_tmpl (n, *ft_d)
    """
    float_dtype, complex_dtype = be._float, be._complex
    template = be.from_sharedarr(template)
    target_ft = be.from_sharedarr(ft_target)
    target_ft2 = be.from_sharedarr(ft_target2)
    template_mask = be.from_sharedarr(template_mask)
    target_mask_ft = be.from_sharedarr(ft_target_mask)
    template_filter = be.from_sharedarr(template_filter)

    batched = target_ft.ndim != template.ndim
    tb, ob = int(batched), 2 * int(batched)

    tmpl_axes, out_axes, spatial = None, None, fast_shape
    if batched:
        tmpl_axes = tuple(range(tb, template.ndim))
        out_axes = tuple(range(ob, len(fast_shape)))
        spatial = tuple(fast_shape[i] for i in out_axes)

    eps = be.eps(float_dtype)

    # Template-space buffers
    tmpl_rot = be.zeros(template.shape, float_dtype)
    mask_rot = be.zeros(template.shape, float_dtype)

    # Output-space buffers
    template_rot_pad = be.zeros(fast_shape, float_dtype)
    mask_overlap = be.zeros(fast_shape, float_dtype)
    numerator = be.zeros(fast_shape, float_dtype)
    temp = be.zeros(fast_shape, float_dtype)
    temp2 = be.zeros(fast_shape, float_dtype)
    temp3 = be.zeros(fast_shape, float_dtype)
    temp_ft = be.zeros(fast_ft_shape, complex_dtype)

    # Padded rotation buffers, reduced shape for batched to avoid redundant FFTs
    tmpl_rot_pad, mask_rot_pad = template_rot_pad, template_rot_pad
    reduced_ft = fast_ft_shape
    if batched:
        reduced = (1,) + template.shape[:tb] + spatial
        reduced_ft = (
            (1,) + template.shape[:tb] + tuple(fast_ft_shape[i] for i in out_axes)
        )
        tmpl_rot_pad = be.zeros(reduced, float_dtype)
        mask_rot_pad = be.zeros(reduced, float_dtype)
    ft_tmpl = be.zeros(reduced_ft, complex_dtype)

    rshape = template.shape[tb:]
    center = be.divide(be.to_backend_array(rshape) - 1, 2)
    tmpl_filter_func = _create_filter_func(
        arr_shape=template.shape,
        template_filter=template_filter,
        axes=tmpl_axes,
    )
    top_slice = (slice(None),) * ob + tuple(slice(0, s) for s in rshape)

    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation, center=center, shape=rshape, batched=batched
        )
        be.rigid_transform(
            arr=template,
            arr_mask=template_mask,
            matrix=matrix,
            out=tmpl_rot,
            out_mask=mask_rot,
            order=interpolation_order,
            cache=not batched,
        )

        tmpl_rot = tmpl_filter_func(tmpl_rot)
        tmpl_rot = standardize(
            tmpl_rot,
            mask_rot,
            be.sum(mask_rot, axis=tmpl_axes, keepdims=True),
            axis=tmpl_axes,
        )

        # FT of rotated standardized template
        tmpl_rot_pad = to_padded(tmpl_rot_pad, tmpl_rot, top_slice)
        ft_tmpl = be.rfftn(tmpl_rot_pad, out=ft_tmpl, axes=out_axes, s=spatial)
        temp2 = _correlate_fts(
            target_mask_ft, ft_tmpl, temp_ft, temp2, spatial, out_axes
        )
        numerator = _correlate_fts(
            target_ft, ft_tmpl, temp_ft, numerator, spatial, out_axes
        )

        # FT of rotated mask
        mask_rot_pad = to_padded(mask_rot_pad, mask_rot, top_slice)
        ft_tmpl = be.rfftn(mask_rot_pad, out=ft_tmpl, axes=out_axes, s=spatial)
        mask_overlap = _correlate_fts(
            ft_tmpl, target_mask_ft, temp_ft, mask_overlap, spatial, out_axes
        )
        be.maximum(mask_overlap, eps, out=mask_overlap)
        temp = _correlate_fts(ft_tmpl, target_ft, temp_ft, temp, spatial, out_axes)

        be.subtract(
            numerator,
            be.divide(be.multiply(temp, temp2), mask_overlap),
            out=numerator,
        )

        # fixed_denom
        be.multiply(target_ft2, ft_tmpl, out=temp_ft)
        temp3 = be.irfftn(temp_ft, out=temp3, s=spatial, axes=out_axes)
        be.subtract(temp3, be.divide(be.square(temp), mask_overlap), out=temp3)
        be.maximum(temp3, 0.0, out=temp3)

        # moving_denom
        ft_tmpl = be.rfftn(
            to_padded(tmpl_rot_pad, be.square(tmpl_rot), top_slice),
            out=ft_tmpl,
            axes=out_axes,
            s=spatial,
        )
        be.multiply(target_mask_ft, ft_tmpl, out=temp_ft)
        temp = be.irfftn(temp_ft, out=temp, s=spatial, axes=out_axes)

        be.subtract(temp, be.divide(be.square(temp2), mask_overlap), out=temp)
        be.maximum(temp, 0.0, out=temp)

        # denom
        be.multiply(temp3, temp, out=temp)
        be.sqrt(temp, out=temp2)

        tol = 1e3 * eps * be.max(be.abs(temp2), axis=out_axes, keepdims=True)
        temp2[temp2 < tol] = 1
        temp = be.divide(numerator, temp2, out=temp)
        temp = be.clip(temp, a_min=-1, a_max=1, out=temp)

        number_px_threshold = overlap_ratio * be.max(
            mask_overlap, axis=out_axes, keepdims=True
        )
        temp[mask_overlap < number_px_threshold] = 0.0
        callback(temp, rotation_matrix=rotation)

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
    **kwargs,
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

    Notes
    -----
    Both target and template can carry a leading batch dimension. The setup
    function prepads the target with a singleton template-batch dim so
    that broadcasting handles the combination naturally:

        ft_target  (b, 1, *ft_d) * ft_tmpl (n, *ft_d)
    """
    template = be.from_sharedarr(template)
    template_mask = be.from_sharedarr(template_mask)
    ft_target = be.from_sharedarr(ft_target)
    ft_target2 = be.from_sharedarr(ft_target2)
    template_filter = be.from_sharedarr(template_filter)
    score_mask = be.from_sharedarr(score_mask)

    (
        batched,
        tmpl_axes,
        out_axes,
        spatial,
        tmpl_rot,
        arr,
        ft_denom,
        tmpl_rot_pad,
        ft_tmpl,
        rshape,
        center,
        tmpl_filter_func,
        norm_mask,
        top_slice,
    ) = _scoring_buffers(
        template,
        ft_target,
        fast_shape,
        fast_ft_shape,
        template_filter,
        score_mask,
    )
    mask_rot = be.zeros(template.shape, be._float)
    temp = be.zeros(fast_shape, be._float)
    temp2 = be.zeros(fast_shape, be._float)

    background_correction = template_background is not None
    if background_correction:
        scores_alt = be.zeros(fast_shape, be._float)
        compute_norm = _setup_background_correction(
            template_background=template_background,
            rotation_buffer=tmpl_rot,
            pad_buffer=tmpl_rot_pad,
            ft_buffer=ft_tmpl,
            unpadded_slice=top_slice,
            interpolation_order=interpolation_order,
            tmpl_filter_func=tmpl_filter_func,
            norm_template=standardize,
            tmpl_axes=tmpl_axes,
            axes=out_axes,
            shape=spatial,
        )

    eps = be.eps(be._float)
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation, center=center, shape=rshape, batched=batched
        )
        _, _ = be.rigid_transform(
            arr=template,
            arr_mask=template_mask,
            matrix=matrix,
            out=tmpl_rot,
            out_mask=mask_rot,
            order=interpolation_order,
            cache=not batched,
        )
        n_obs = be.sum(mask_rot, axis=tmpl_axes, keepdims=True)
        tmpl_rot = tmpl_filter_func(tmpl_rot)
        tmpl_rot = standardize(tmpl_rot, mask_rot, n_obs, axis=tmpl_axes)

        tmpl_rot_pad = to_padded(tmpl_rot_pad, mask_rot, top_slice)
        ft_tmpl = be.rfftn(tmpl_rot_pad, out=ft_tmpl, axes=out_axes, s=spatial)
        temp = _correlate_fts(ft_target, ft_tmpl, ft_denom, temp, spatial, out_axes)
        temp2 = _correlate_fts(ft_target2, ft_tmpl, ft_denom, temp2, spatial, out_axes)

        tmpl_rot_pad = to_padded(tmpl_rot_pad, tmpl_rot, top_slice)
        ft_tmpl = be.rfftn(tmpl_rot_pad, out=ft_tmpl, axes=out_axes, s=spatial)
        arr = _correlate_fts(ft_target, ft_tmpl, ft_denom, arr, spatial, out_axes)

        inv_sdev = be.norm_scores(1, temp2, temp, n_obs, eps, temp2)
        arr = be.multiply(arr, inv_sdev, out=arr)
        arr = norm_mask(arr, score_mask, out=arr)

        callback(arr, rotation_matrix=rotation)
        if background_correction:
            arr = compute_norm(arr, ft_target, ft_denom, matrix, mask_rot, n_obs)
            arr = be.multiply(arr, inv_sdev, out=arr)
            scores_alt = be.maximum(arr, scores_alt, out=scores_alt)

    if background_correction:
        scores_alt = norm_mask(scores_alt, score_mask, out=scores_alt)
        scores_alt = be.subtract(scores_alt, be.mean(scores_alt), out=scores_alt)
        callback.correct_background(scores_alt)

    return callback


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
    **kwargs,
) -> CallbackClass:
    template = be.from_sharedarr(template)
    ft_target = be.from_sharedarr(ft_target)
    inv_denominator = be.from_sharedarr(inv_denominator)
    numerator = be.from_sharedarr(numerator)
    template_filter = be.from_sharedarr(template_filter)
    score_mask = be.from_sharedarr(score_mask)

    (
        batched,
        tmpl_axes,
        out_axes,
        spatial,
        tmpl_rot,
        arr,
        ft_denom,
        tmpl_rot_pad,
        ft_tmpl,
        rshape,
        center,
        tmpl_filter_func,
        norm_mask,
        top_slice,
    ) = _scoring_buffers(
        template,
        ft_target,
        fast_shape,
        fast_ft_shape,
        template_filter,
        score_mask,
    )
    n_obs = None
    if template_mask is not None:
        template_mask = be.from_sharedarr(template_mask)
        n_obs = be.sum(template_mask, axis=tmpl_axes, keepdims=True)

    norm_template = conditional_execute(standardize, n_obs is not None)
    norm_sub = conditional_execute(be.subtract, numerator.shape != (1,))
    norm_mul = conditional_execute(be.multiply, inv_denominator.shape != (1,))

    background_correction = template_background is not None
    if background_correction:
        scores_alt = be.zeros(fast_shape, be._float)
        compute_norm = _setup_background_correction(
            template_background=template_background,
            rotation_buffer=tmpl_rot,
            pad_buffer=tmpl_rot_pad,
            ft_buffer=ft_tmpl,
            unpadded_slice=top_slice,
            interpolation_order=interpolation_order,
            tmpl_filter_func=tmpl_filter_func,
            norm_template=norm_template,
            axes=out_axes,
            shape=spatial,
        )

    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        matrix = be._build_transform_matrix(
            rotation_matrix=rotation, center=center, shape=rshape, batched=batched
        )
        _ = be.rigid_transform(
            arr=template,
            matrix=matrix,
            out=tmpl_rot,
            order=interpolation_order,
            cache=not batched,
        )

        tmpl_rot = tmpl_filter_func(tmpl_rot)
        tmpl_rot = norm_template(tmpl_rot, template_mask, n_obs, axis=tmpl_axes)

        tmpl_rot_pad = to_padded(tmpl_rot_pad, tmpl_rot, top_slice)
        ft_tmpl = be.rfftn(tmpl_rot_pad, out=ft_tmpl, axes=out_axes, s=spatial)
        arr = _correlate_fts(ft_target, ft_tmpl, ft_denom, arr, spatial, out_axes)

        arr = norm_sub(arr, numerator, out=arr)
        arr = norm_mul(arr, inv_denominator, out=arr)
        arr = norm_mask(arr, score_mask, out=arr)

        callback(arr, rotation_matrix=rotation)
        if background_correction:
            arr = compute_norm(arr, ft_target, ft_denom, matrix, template_mask, n_obs)
            arr = norm_sub(arr, numerator, out=arr)
            arr = norm_mul(arr, inv_denominator, out=arr)
            arr = norm_mask(arr, score_mask, out=arr)
            scores_alt = be.maximum(arr, scores_alt, out=scores_alt)

    if background_correction:
        scores_alt = norm_mask(scores_alt, score_mask, out=scores_alt)
        scores_alt = be.subtract(scores_alt, be.mean(scores_alt), out=scores_alt)
        callback.correct_background(scores_alt)

    return callback


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
    def _apply_filter(template, ft_temp=None):
        s = (
            tuple(template.shape[a] for a in axes)
            if axes is not None
            else template.shape
        )
        ft_temp = be.rfftn(template, out=ft_temp, s=s, axes=axes)
        ft_temp = be.multiply(ft_temp, template_filter, out=ft_temp)
        return be.irfftn(ft_temp, out=template, s=s, axes=axes)

    if not arr_padded:
        return _apply_filter

    # Array is padded but filter is w.r.t to the original template
    real_subset = tuple(slice(0, x) for x in arr_shape)
    _template = be.zeros(arr_shape, be._float)
    _ft_temp = be.zeros(filter_shape, be._complex)

    def _apply_filter_subset(template, ft_temp):
        _template[:] = template[real_subset]
        template[real_subset] = _apply_filter(_template, _ft_temp)
        return template

    return _apply_filter_subset


def _setup_background_correction(
    template_background: BackendArray,
    rotation_buffer: BackendArray,
    pad_buffer: BackendArray,
    ft_buffer: BackendArray,
    unpadded_slice: Tuple[slice],
    interpolation_order: int = 3,
    tmpl_filter_func: Callable = identity,
    norm_template: Callable = identity,
    tmpl_axes=None,
    axes=None,
    shape=None,
) -> Callable:
    template_background = be.from_sharedarr(template_background)

    def compute_norm(arr, ft_target, ft_denom, matrix, template_mask, n_obs):
        _ = be.rigid_transform(
            arr=template_background,
            matrix=matrix,
            out=rotation_buffer,
            order=interpolation_order,
            cache=True,
        )
        template_rot = tmpl_filter_func(rotation_buffer)
        template_rot = norm_template(template_rot, template_mask, n_obs, axis=tmpl_axes)
        _pad = to_padded(pad_buffer, template_rot, unpadded_slice)
        _ft = be.rfftn(_pad, out=ft_buffer, axes=axes, s=shape)
        return _correlate_fts(ft_target, _ft, ft_denom, arr, shape, axes)

    return compute_norm


MATCHING_EXHAUSTIVE_REGISTER = {
    "CC": (cc_setup, corr_scoring),
    "LCC": (lcc_setup, corr_scoring),
    "CAM": (cam_setup, corr_scoring),
    "NCC": (ncc_setup, ncc_scoring),
    "FLCSphericalMask": (flcSphericalMask_setup, corr_scoring),
    "FLC": (flc_setup, flc_scoring),
    "MCC": (mcc_setup, mcc_scoring),
}
