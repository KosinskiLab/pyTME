"""
Background correction methods for template matching scores.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Callable, Tuple, NamedTuple

from .backends import backend as be
from .types import BackendArray, shm_type
from .matching_utils import identity, to_padded


class NoiseTemplateState(NamedTuple):
    """Accumulated MIP of background-template scores."""

    scores_alt: BackendArray


class FlatFieldState(NamedTuple):
    pixel_mean: BackendArray
    pixel_M2: BackendArray
    pixel_n: BackendArray


def _build_noise_template_corrector(
    fast_shape: Tuple[int],
    template_background: shm_type,
    ft_target: BackendArray,
    ft_denom: BackendArray,
    rotation_buffer: BackendArray,
    pad_buffer: BackendArray,
    ft_buffer: BackendArray,
    unpadded_slice: Tuple[slice],
    interpolation_order: int = 3,
    tmpl_filter_func: Callable = identity,
    norm_template: Callable = identity,
    normalize_fn: Callable = None,
    tmpl_axes: Tuple[int] = None,
    axes: Tuple[int] = None,
    shape: Tuple[int] = None,
):
    """Build a noise-template background corrector.

    Returns ``(state, update, finalize)`` where *state* is the initial
    :class:`NoiseTemplateState`, *update* is called once per rotation,
    and *finalize* applies the correction to the callback after all
    rotations have been processed.

    Configuration (shared buffers, the background template, filter and
    normalisation functions) is captured in the returned closures.
    Only the MIP accumulator travels as explicit state.

    Parameters
    ----------
    fast_shape : tuple of int
        Shape of the output score array.
    template_background : shm_type
        Background (phase-scrambled) template, or its shared-memory
        descriptor.
    ft_target : BackendArray
        Fourier-transformed target (pre-computed in the setup function
        of the scoring method).
    ft_denom : BackendArray
        Pre-allocated buffer for the element-wise Fourier product.
    rotation_buffer : BackendArray
        Pre-allocated buffer for the rotated background template
        (same shape as the unpadded template).
    pad_buffer : BackendArray
        Pre-allocated buffer for the zero-padded rotated background
        template.
    ft_buffer : BackendArray
        Pre-allocated buffer for the Fourier transform of the padded
        background template.
    unpadded_slice : tuple of slice
        Index expression that maps the unpadded template into
        *pad_buffer*.
    interpolation_order : int, optional
        Spline interpolation order for rotating the background
        template.  Defaults to 3.
    tmpl_filter_func : callable, optional
        Filter applied to the rotated background template (e.g.
        Fourier-space weighting).  Defaults to :func:`identity`.
    norm_template : callable, optional
        Normalisation applied to the rotated, filtered background
        template (e.g. :func:`standardize`).  Defaults to
        :func:`identity`.
    normalize_fn : callable, optional
        Score-level normalisation applied to the background CC before
        the MIP update (e.g. ``inv_denominator`` scaling).  When the
        normalisation depends on per-rotation quantities (such as
        ``inv_sdev`` in FLC scoring), pass ``None`` here and supply the
        function via the *normalize_fn* keyword of *update* instead.
    tmpl_axes : tuple of int, optional
        Template axes for normalisation.
    axes : tuple of int, optional
        FFT axes.
    shape : tuple of int, optional
        Real-space shape for the inverse FFT.
    """
    template_background = be.from_sharedarr(template_background)

    def _compute_norm(arr, matrix, template_mask, n_obs):
        _ = be.rigid_transform(
            arr=template_background,
            matrix=matrix,
            out=rotation_buffer,
            order=interpolation_order,
            cache=True,
        )
        tmpl_rot = tmpl_filter_func(rotation_buffer)
        tmpl_rot = norm_template(tmpl_rot, template_mask, n_obs, axis=tmpl_axes)
        _pad = to_padded(pad_buffer, tmpl_rot, unpadded_slice)
        _ft = be.rfftn(_pad, out=ft_buffer, axes=axes, s=shape)
        ft_denom_local = be.multiply(ft_target, _ft, out=ft_denom)
        return be.irfftn(ft_denom_local, out=arr, s=shape, axes=axes)

    default_normalize = normalize_fn

    def update(
        state,
        arr,
        index,
        *,
        matrix,
        template_mask=None,
        n_obs=None,
        normalize_fn=None,
        **kwargs
    ):
        """Compute background CC for this rotation and update MIP.

        Parameters
        ----------
        state : NoiseTemplateState
        arr : BackendArray
            Score array (used as output buffer for the background CC).
        index : int
            Zero-based rotation index (unused, accepted for interface
            compatibility).
        matrix : BackendArray
            Rotation/transform matrix for the current angle.
        template_mask : BackendArray, optional
            Rotated template mask (needed by scores that normalise the
            background template with a mask, e.g. FLC).
        n_obs : BackendArray, optional
            Number of observations (sum of *template_mask*).
        normalize_fn : callable, optional
            Per-rotation normalisation override.  Takes precedence over
            the *normalize_fn* supplied at construction time.
        """
        arr = _compute_norm(arr, matrix, template_mask, n_obs)
        norm = normalize_fn or default_normalize
        if norm is not None:
            arr = norm(arr)
        scores_alt = be.maximum(arr, state.scores_alt)
        return NoiseTemplateState(scores_alt)

    def finalize(state, callback, n_angles):
        """Subtract background mean and apply correction to *callback*."""
        scores_alt = be.subtract(state.scores_alt, be.mean(state.scores_alt))
        callback.correct_background(scores_alt)

    state = NoiseTemplateState(be.zeros(fast_shape, be._float))
    return state, update, finalize


def _build_flat_field_corrector(fast_shape):
    """Build a flat-fielding corrector.

    Parameters
    ----------
    fast_shape : tuple of int
        Shape of the output score array.
    """

    def update(state, arr, index, **kwargs):
        new_n = state.pixel_n + 1
        delta = be.subtract(arr, state.pixel_mean)
        new_mean = be.add(state.pixel_mean, be.divide(delta, be.maximum(new_n, 1)))
        delta2 = be.subtract(arr, new_mean)
        new_M2 = be.add(state.pixel_M2, be.multiply(delta, delta2))
        return FlatFieldState(new_mean, new_M2, new_n)

    def finalize(state, callback, n_angles):
        safe_n = be.maximum(state.pixel_n - 1, 1)
        pixel_variance = be.divide(state.pixel_M2, safe_n)

        global_mean = be.mean(state.pixel_mean)
        global_std = be.sqrt(be.mean(pixel_variance))
        callback.correct_background(global_mean, 1.0 / global_std)

        local_mean = be.divide(be.subtract(state.pixel_mean, global_mean), global_std)
        local_std = be.divide(be.sqrt(pixel_variance), global_std)

        # Tikhonov-regularised denominator. Since global_std = 1, eps is relative
        eps = 1.0
        inv_local_std = be.divide(1.0, be.sqrt(be.add(be.square(local_std), eps * eps)))

        callback.correct_background(local_mean, inv_local_std)

    state = FlatFieldState(
        be.zeros(fast_shape, be._float),
        be.zeros(fast_shape, be._float),
        be.zeros(fast_shape, be._float),
    )
    return state, update, finalize


def _build_null_corrector():
    """Return a no-op corrector triple."""
    state = ()
    return state, lambda s, *a, **kw: s, lambda s, *a, **kw: None


def setup_background_corrector(method, fast_shape, **kwargs):
    """Create a background corrector for use inside a scoring function.

    Correction methods expose the same functional interface so that scoring
    functions can remain agnostic to the correction strategy

        state, update, finalize = setup_background_corrector(...)

        for index in range(n_angles):
            # ... compute and normalise arr ...
            callback(arr, rotation_matrix=rotation)
            state = update(state, arr, index, matrix=matrix)

        finalize(state, callback, n_angles)

    Parameters
    ----------
    method : str {None, "phase-scrambling", "flat-fielding"}
        Correction strategy.  ``None`` returns a no-op corrector.
    fast_shape : tuple of int
        Shape of the output score array.
    **kwargs
        Forwarded to the selected builder.  See
        :func:`_build_noise_template_corrector` for the noise-template
        parameters and :func:`_build_flat_field_corrector` for flat
        fielding (which needs no extra arguments beyond *fast_shape*).

    Returns
    -------
    state : NamedTuple or tuple
        Initial corrector state.
    update : callable
        ``(state, arr, index, **kw) -> state`` — called once per
        rotation, after the main scores have been sent to the callback.
        Extra keyword arguments (``matrix``, ``template_mask``,
        ``n_obs``, ``normalize_fn``) are consumed by the noise-template
        corrector and ignored by flat fielding.
    finalize : callable
        ``(state, callback, n_angles) -> None`` — called after the
        rotation loop to apply the correction.
    """
    if method == "phase-scrambling":
        return _build_noise_template_corrector(fast_shape=fast_shape, **kwargs)
    if method == "flat-fielding":
        return _build_flat_field_corrector(fast_shape=fast_shape)
    return _build_null_corrector()
