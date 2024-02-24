""" Implements cross-correlation based template matching using different metrics.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import sys
import warnings
from copy import deepcopy
from itertools import product
from typing import Callable, Tuple, Dict
from functools import wraps
from joblib import Parallel, delayed
from multiprocessing.managers import SharedMemoryManager

import numpy as np
from scipy.ndimage import laplace

from .analyzer import MaxScoreOverRotations
from .matching_utils import (
    apply_convolution_mode,
    handle_traceback,
    split_numpy_array_slices,
    conditional_execute,
)
from .matching_memory import MatchingMemoryUsage, MATCHING_MEMORY_REGISTRY
from .preprocessor import Preprocessor
from .matching_data import MatchingData
from .backends import backend
from .types import NDArray, CallbackClass

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["PYFFTW_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _run_inner(backend_name, backend_args, **kwargs):
    from tme.backends import backend

    backend.change_backend(backend_name, **backend_args)
    return scan(**kwargs)


def cc_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: NDArray,
    target: NDArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    real_dtype: type,
    complex_dtype: type,
    shared_memory_handler: Callable,
    callback_class: Callable,
    callback_class_args: Dict,
    **kwargs,
) -> Dict:
    """
    Setup to compute the cross-correlation between a target f and template g
    defined as:

    .. math::

        \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*)


    See Also
    --------
    :py:meth:`corr_scoring`
    :py:class:`tme.matching_optimization.CrossCorrelation`
    """
    target_shape = target.shape
    target_pad = backend.topleft_pad(target, fast_shape)
    target_pad_ft = backend.preallocate_array(fast_ft_shape, complex_dtype)

    rfftn(target_pad, target_pad_ft)

    target_ft_out = backend.arr_to_sharedarr(
        arr=target_pad_ft, shared_memory_handler=shared_memory_handler
    )

    template_out = backend.arr_to_sharedarr(
        arr=template, shared_memory_handler=shared_memory_handler
    )
    inv_denominator_buffer = backend.arr_to_sharedarr(
        arr=backend.preallocate_array(1, real_dtype) + 1,
        shared_memory_handler=shared_memory_handler,
    )
    numerator2_buffer = backend.arr_to_sharedarr(
        arr=backend.preallocate_array(1, real_dtype),
        shared_memory_handler=shared_memory_handler,
    )

    target_ft_tuple = (target_ft_out, fast_ft_shape, complex_dtype)
    template_tuple = (template_out, template.shape, real_dtype)

    inv_denominator_tuple = (inv_denominator_buffer, (1,), real_dtype)
    numerator2_tuple = (numerator2_buffer, (1,), real_dtype)

    ret = {
        "template": template_tuple,
        "ft_target": target_ft_tuple,
        "inv_denominator": inv_denominator_tuple,
        "numerator2": numerator2_tuple,
        "targetshape": target_shape,
        "templateshape": template.shape,
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "real_dtype": real_dtype,
        "complex_dtype": complex_dtype,
        "callback_class": callback_class,
        "callback_class_args": callback_class_args,
        "use_memmap": kwargs.get("use_memmap", False),
        "temp_dir": kwargs.get("temp_dir", None),
    }

    return ret


def lcc_setup(**kwargs) -> Dict:
    """
    Setup to compute the cross-correlation between a laplace transformed target f
    and laplace transformed template g defined as:

    .. math::

        \\mathcal{F}^{-1}(\\mathcal{F}(\\nabla^{2}f) \\cdot \\mathcal{F}(\\nabla^{2} g)^*)


    See Also
    --------
    :py:meth:`corr_scoring`
    :py:class:`tme.matching_optimization.LaplaceCrossCorrelation`
    """
    kwargs["target"] = laplace(kwargs["target"], mode="wrap")
    kwargs["template"] = laplace(kwargs["template"], mode="wrap")
    return cc_setup(**kwargs)


def corr_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: NDArray,
    template_mask: NDArray,
    target: NDArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    real_dtype: type,
    complex_dtype: type,
    shared_memory_handler: Callable,
    callback_class: Callable,
    callback_class_args: Dict,
    **kwargs,
) -> Dict:
    """
    Setup to compute a normalized cross-correlation between a target f and a template g.

    .. math::

        \\frac{CC(f,g) - \\overline{g} \\cdot CC(f, m)}
        {(CC(f^2, m) - \\frac{CC(f, m)^2}{N_g}) \\cdot \\sigma_{g}}

    Where:

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*)

    and m is a mask with the same dimension as the template filled with ones.

    References
    ----------
    .. [1]  J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
            and Magic.

    See Also
    --------
    :py:meth:`corr_scoring`
    :py:class:`tme.matching_optimization.NormalizedCrossCorrelation`.
    """
    target_pad = backend.topleft_pad(target, fast_shape)

    # The exact composition of the denominator is debatable
    # scikit-image match_template multiplies the running sum of the target
    # with a scaling factor derived from the template. This is probably appropriate
    # in pattern matching situations where the template exists in the target
    window_template = backend.topleft_pad(template_mask, fast_shape)
    ft_window_template = backend.preallocate_array(fast_ft_shape, complex_dtype)
    rfftn(window_template, ft_window_template)
    window_template = None

    # Target and squared target window sums
    ft_target = backend.preallocate_array(fast_ft_shape, complex_dtype)
    ft_target2 = backend.preallocate_array(fast_ft_shape, complex_dtype)
    denominator = backend.preallocate_array(fast_shape, real_dtype)
    target_window_sum = backend.preallocate_array(fast_shape, real_dtype)
    rfftn(target_pad, ft_target)

    rfftn(backend.square(target_pad), ft_target2)
    backend.multiply(ft_target2, ft_window_template, out=ft_target2)
    irfftn(ft_target2, denominator)

    backend.multiply(ft_target, ft_window_template, out=ft_window_template)
    irfftn(ft_window_template, target_window_sum)

    target_pad, ft_target2, ft_window_template = None, None, None

    # Normalizing constants
    template_mean = backend.mean(template)
    template_volume = np.prod(template.shape)
    template_ssd = backend.sum(
        backend.square(backend.subtract(template, template_mean))
    )

    # Final numerator is score - numerator2
    numerator2 = backend.multiply(target_window_sum, template_mean)

    # Compute denominator
    backend.multiply(target_window_sum, target_window_sum, out=target_window_sum)
    backend.divide(target_window_sum, template_volume, out=target_window_sum)

    backend.subtract(denominator, target_window_sum, out=denominator)
    backend.multiply(denominator, template_ssd, out=denominator)
    backend.maximum(denominator, 0, out=denominator)
    backend.sqrt(denominator, out=denominator)
    target_window_sum = None

    # Invert denominator to compute final score as product
    denominator_mask = denominator > backend.eps(denominator.dtype)
    inv_denominator = backend.preallocate_array(fast_shape, real_dtype)
    inv_denominator[denominator_mask] = 1 / denominator[denominator_mask]

    # Convert arrays used in subsequent fitting to SharedMemory objects
    template_buffer = backend.arr_to_sharedarr(
        arr=template, shared_memory_handler=shared_memory_handler
    )
    target_ft_buffer = backend.arr_to_sharedarr(
        arr=ft_target, shared_memory_handler=shared_memory_handler
    )
    inv_denominator_buffer = backend.arr_to_sharedarr(
        arr=inv_denominator, shared_memory_handler=shared_memory_handler
    )
    numerator2_buffer = backend.arr_to_sharedarr(
        arr=numerator2, shared_memory_handler=shared_memory_handler
    )

    template_tuple = (template_buffer, deepcopy(template.shape), real_dtype)
    target_ft_tuple = (target_ft_buffer, fast_ft_shape, complex_dtype)

    inv_denominator_tuple = (inv_denominator_buffer, fast_shape, real_dtype)
    numerator2_tuple = (numerator2_buffer, fast_shape, real_dtype)

    ft_target, inv_denominator, numerator2 = None, None, None

    ret = {
        "template": template_tuple,
        "ft_target": target_ft_tuple,
        "inv_denominator": inv_denominator_tuple,
        "numerator2": numerator2_tuple,
        "targetshape": deepcopy(target.shape),
        "templateshape": deepcopy(template.shape),
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "real_dtype": real_dtype,
        "complex_dtype": complex_dtype,
        "callback_class": callback_class,
        "callback_class_args": callback_class_args,
        "template_mean": kwargs.get("template_mean", template_mean),
    }

    return ret


def cam_setup(**kwargs):
    """
    Setup to compute a normalized cross-correlation between a target f and a template g
    over their means. In practice this can be expressed like the cross-correlation
    CORR defined in :py:meth:`corr_scoring`, so that:

    Notes
    -----

    .. math::

        \\text{CORR}(f-\\overline{f}, g - \\overline{g})

    Where

    .. math::

        \\overline{f}, \\overline{g}

    are the mean of f and g respectively.

    References
    ----------
    .. [1]  J. P. Lewis, "Fast Normalized Cross-Correlation", Industrial Light
            and Magic.

    See Also
    --------
    :py:meth:`corr_scoring`.
    :py:class:`tme.matching_optimization.NormalizedCrossCorrelationMean`.
    """
    kwargs["template"] = kwargs["template"] - kwargs["template"].mean()
    return corr_setup(**kwargs)


def flc_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: NDArray,
    template_mask: NDArray,
    target: NDArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    real_dtype: type,
    complex_dtype: type,
    shared_memory_handler: Callable,
    callback_class: Callable,
    callback_class_args: Dict,
    **kwargs,
) -> Dict:
    """
    Setup to compute a normalized cross-correlation score of a target f a template g
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

    References
    ----------
    .. [1]  W. Wan, S. Khavnekar, J. Wagner, P. Erdmann, and W. Baumeister
            Microsc. Microanal. 26, 2516 (2020)
    .. [2]  T. Hrabe, Y. Chen, S. Pfeffer, L. Kuhn Cuellar, A.-V. Mangold,
            and F. Förster, J. Struct. Biol. 178, 177 (2012).

    See Also
    --------
    :py:meth:`flc_scoring`
    """
    target_pad = backend.topleft_pad(target, fast_shape)

    # Target and squared target window sums
    ft_target = backend.preallocate_array(fast_ft_shape, complex_dtype)
    ft_target2 = backend.preallocate_array(fast_ft_shape, complex_dtype)
    rfftn(target_pad, ft_target)
    rfftn(backend.square(target_pad), ft_target2)

    # Convert arrays used in subsequent fitting to SharedMemory objects
    ft_target = backend.arr_to_sharedarr(
        arr=ft_target, shared_memory_handler=shared_memory_handler
    )
    ft_target2 = backend.arr_to_sharedarr(
        arr=ft_target2, shared_memory_handler=shared_memory_handler
    )

    template_mask = template_mask > 0
    template_mean = backend.mean(template[template_mask])
    template_std = backend.std(template[template_mask])
    template_mask = backend.astype(template_mask, real_dtype)

    backend.divide(template - template_mean, template_std, out=template)
    backend.multiply(template, template_mask, out=template)

    template_buffer = backend.arr_to_sharedarr(
        arr=template, shared_memory_handler=shared_memory_handler
    )
    template_mask_buffer = backend.arr_to_sharedarr(
        arr=template_mask, shared_memory_handler=shared_memory_handler
    )

    template_tuple = (template_buffer, template.shape, real_dtype)
    template_mask_tuple = (template_mask_buffer, template_mask.shape, real_dtype)

    target_ft_tuple = (ft_target, fast_ft_shape, complex_dtype)
    target_ft2_tuple = (ft_target2, fast_ft_shape, complex_dtype)

    ret = {
        "template": template_tuple,
        "template_mask": template_mask_tuple,
        "ft_target": target_ft_tuple,
        "ft_target2": target_ft2_tuple,
        "targetshape": target.shape,
        "templateshape": template.shape,
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "real_dtype": real_dtype,
        "complex_dtype": complex_dtype,
        "callback_class": callback_class,
        "callback_class_args": callback_class_args,
    }

    return ret


def flcSphericalMask_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: NDArray,
    template_mask: NDArray,
    target: NDArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    real_dtype: type,
    complex_dtype: type,
    shared_memory_handler: Callable,
    callback_class: Callable,
    callback_class_args: Dict,
    **kwargs,
) -> Dict:
    """
    Like :py:meth:`flc_setup` but for rotation invariant masks. In such cases
    the score can be computed quicker by not computing the fourier transforms
    of the mask for each rotation.

    .. math::

        \\frac{CC(f, \\frac{g*m - \\overline{g*m}}{\\sigma_{g*m}})}
        {N_m * \\sqrt{
            \\frac{CC(f^2, m)}{N_m} - (\\frac{CC(f, m)}{N_m})^2}
        }

    Where:

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*)

    and Nm is the number of voxels within the template mask m.

    References
    ----------
    .. [1]  W. Wan, S. Khavnekar, J. Wagner, P. Erdmann, and W. Baumeister
            Microsc. Microanal. 26, 2516 (2020)
    .. [2]  T. Hrabe, Y. Chen, S. Pfeffer, L. Kuhn Cuellar, A.-V. Mangold,
            and F. Förster, J. Struct. Biol. 178, 177 (2012).

    See Also
    --------
    :py:meth:`corr_scoring`
    """
    target_pad = backend.topleft_pad(target, fast_shape)
    template_mask_pad = backend.topleft_pad(template_mask, fast_shape)

    # Target and squared target window sums
    ft_target = backend.preallocate_array(fast_ft_shape, complex_dtype)
    ft_template_mask = backend.preallocate_array(fast_ft_shape, complex_dtype)
    ft_temp = backend.preallocate_array(fast_ft_shape, complex_dtype)

    temp = backend.preallocate_array(fast_shape, real_dtype)
    temp2 = backend.preallocate_array(fast_shape, real_dtype)
    numerator2 = backend.preallocate_array(1, real_dtype)

    eps = backend.eps(real_dtype)
    n_observations = backend.sum(template_mask_pad > np.exp(-2))
    rfftn(template_mask_pad, ft_template_mask)

    # Denominator E(X^2) - E(X)^2
    rfftn(backend.square(target_pad), ft_target)
    backend.multiply(ft_target, ft_template_mask, out=ft_temp)
    irfftn(ft_temp, temp2)
    backend.divide(temp2, n_observations, out=temp2)

    rfftn(target_pad, ft_target)
    backend.multiply(ft_target, ft_template_mask, out=ft_temp)
    irfftn(ft_temp, temp)
    backend.divide(temp, n_observations, out=temp)
    backend.square(temp, out=temp)

    backend.subtract(temp2, temp, out=temp)

    backend.maximum(temp, 0.0, out=temp)
    backend.sqrt(temp, out=temp)
    backend.multiply(temp, n_observations, out=temp)

    tol = 1e3 * eps * backend.max(backend.abs(temp))
    nonzero_indices = temp > tol

    backend.fill(temp2, 0)
    temp2[nonzero_indices] = 1 / temp[nonzero_indices]

    template_mask = template_mask > np.exp(-2)
    template_mean = backend.mean(template[template_mask])
    template_std = backend.std(template[template_mask])

    template = backend.divide(backend.subtract(template, template_mean), template_std)
    backend.multiply(template, template_mask, out=template)

    # Convert arrays used in subsequent fitting to SharedMemory objects
    template_buffer = backend.arr_to_sharedarr(
        arr=template, shared_memory_handler=shared_memory_handler
    )
    target_ft_buffer = backend.arr_to_sharedarr(
        arr=ft_target, shared_memory_handler=shared_memory_handler
    )
    inv_denominator_buffer = backend.arr_to_sharedarr(
        arr=temp2, shared_memory_handler=shared_memory_handler
    )
    numerator2_buffer = backend.arr_to_sharedarr(
        arr=numerator2, shared_memory_handler=shared_memory_handler
    )

    template_tuple = (template_buffer, template.shape, real_dtype)
    target_ft_tuple = (target_ft_buffer, fast_ft_shape, complex_dtype)

    inv_denominator_tuple = (inv_denominator_buffer, fast_shape, real_dtype)
    numerator2_tuple = (numerator2_buffer, (1,), real_dtype)

    ret = {
        "template": template_tuple,
        "ft_target": target_ft_tuple,
        "inv_denominator": inv_denominator_tuple,
        "numerator2": numerator2_tuple,
        "targetshape": target.shape,
        "templateshape": template.shape,
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "real_dtype": real_dtype,
        "complex_dtype": complex_dtype,
        "callback_class": callback_class,
        "callback_class_args": callback_class_args,
    }

    return ret


def mcc_setup(
    rfftn: Callable,
    irfftn: Callable,
    template: NDArray,
    template_mask: NDArray,
    target: NDArray,
    target_mask: NDArray,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    real_dtype: type,
    complex_dtype: type,
    shared_memory_handler: Callable,
    callback_class: Callable,
    callback_class_args: Dict,
    **kwargs,
) -> Dict:
    """
    Setup to compute a normalized cross-correlation score with masks
    for both template and target.

    .. math::

        \\frac{
               CC(f, g) - \\frac{CC(f, m) \\cdot CC(t, g)}{CC(t, m)}
            }{
            \\sqrt{
                (CC(f ^ 2, m) - \\frac{CC(f, m) ^ 2}{CC(t, m)}) \\cdot
                (CC(t, g^2) - \\frac{CC(t, g) ^ 2}{CC(t, m)})
                }
            }

    Where:

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*)


    References
    ----------
    .. [1]  Masked FFT registration, Dirk Padfield, CVPR 2010 conference

    See Also
    --------
    :py:meth:`mcc_scoring`
    :py:class:`tme.matching_optimization.MaskedCrossCorrelation`
    """
    target = backend.multiply(target, target_mask > 0, out=target)

    target_pad = backend.topleft_pad(target, fast_shape)
    target_mask_pad = backend.topleft_pad(target_mask, fast_shape)

    target_ft = backend.preallocate_array(fast_ft_shape, complex_dtype)
    rfftn(target_pad, target_ft)
    target_ft_buffer = backend.arr_to_sharedarr(
        arr=target_ft, shared_memory_handler=shared_memory_handler
    )

    target_ft2 = backend.preallocate_array(fast_ft_shape, complex_dtype)
    rfftn(backend.square(target_pad), target_ft2)
    target_ft2_buffer = backend.arr_to_sharedarr(
        arr=target_ft2, shared_memory_handler=shared_memory_handler
    )

    target_mask_ft = backend.preallocate_array(fast_ft_shape, complex_dtype)
    rfftn(target_mask_pad, target_mask_ft)
    target_mask_ft_buffer = backend.arr_to_sharedarr(
        arr=target_mask_ft, shared_memory_handler=shared_memory_handler
    )

    template_buffer = backend.arr_to_sharedarr(
        arr=template, shared_memory_handler=shared_memory_handler
    )
    template_mask_buffer = backend.arr_to_sharedarr(
        arr=template_mask, shared_memory_handler=shared_memory_handler
    )

    template_tuple = (template_buffer, template.shape, real_dtype)
    template_mask_tuple = (template_mask_buffer, template.shape, real_dtype)

    target_ft_tuple = (target_ft_buffer, fast_ft_shape, complex_dtype)
    target_ft2_tuple = (target_ft2_buffer, fast_ft_shape, complex_dtype)
    target_mask_ft_tuple = (target_mask_ft_buffer, fast_ft_shape, complex_dtype)

    ret = {
        "template": template_tuple,
        "template_mask": template_mask_tuple,
        "ft_target": target_ft_tuple,
        "ft_target2": target_ft2_tuple,
        "ft_target_mask": target_mask_ft_tuple,
        "targetshape": target.shape,
        "templateshape": template.shape,
        "fast_shape": fast_shape,
        "fast_ft_shape": fast_ft_shape,
        "real_dtype": real_dtype,
        "complex_dtype": complex_dtype,
        "callback_class": callback_class,
        "callback_class_args": callback_class_args,
    }

    return ret


def corr_scoring(
    template: Tuple[type, Tuple[int], type],
    ft_target: Tuple[type, Tuple[int], type],
    inv_denominator: Tuple[type, Tuple[int], type],
    numerator2: Tuple[type, Tuple[int], type],
    template_filter: Tuple[type, Tuple[int], type],
    targetshape: Tuple[int],
    templateshape: Tuple[int],
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: NDArray,
    real_dtype: type,
    complex_dtype: type,
    callback_class: CallbackClass,
    callback_class_args: Dict,
    interpolation_order: int,
    convolution_mode: str = "full",
    **kwargs,
) -> CallbackClass:
    """
    Calculates a normalized cross-correlation between a target f and a template g.

    .. math::

        (CC(f,g) - numerator2) \\cdot inv\\_denominator

    Parameters
    ----------
    template : Tuple[type, Tuple[int], type]
        Tuple containing a pointer to the template data, its shape, and its datatype.
    ft_target : Tuple[type, Tuple[int], type]
        Tuple containing a pointer to the fourier tranform of the target,
        its shape, and its datatype.
    inv_denominator : Tuple[type, Tuple[int], type]
        Tuple containing a pointer to the inverse denominator data, its shape, and its
        datatype.
    numerator2 : Tuple[type, Tuple[int], type]
        Tuple containing a pointer to the numerator2 data, its shape, and its datatype.
    targetshape : Tuple[int]
        The shape of the target.
    templateshape : Tuple[int]
        The shape of the template.
    fast_shape : Tuple[int]
        The shape for fast Fourier transform.
    fast_ft_shape : Tuple[int]
        The shape for fast Fourier transform of the target.
    rotations : NDArray
        Array containing the rotation matrices to be applied on the template.
    real_dtype : type
        Data type for the real part of the array.
    complex_dtype : type
        Data type for the complex part of the array.
    callback_class : CallbackClass
        A callable class or function for processing the results after each
        rotation.
    callback_class_args : Dict
        Dictionary of arguments to be passed to the callback class if it's
        instantiable.
    interpolation_order : int
        The order of interpolation to be used while rotating the template.
    convolution_mode : str, optional
        Mode to use for convolution, default is "full".
    **kwargs :
        Additional arguments to be passed to the function.

    Returns
    -------
    CallbackClass
        If callback_class was provided an instance of callback_class otherwise None.

    See Also
    --------
    :py:meth:`cc_setup`
    :py:meth:`corr_setup`
    :py:meth:`cam_setup`
    :py:meth:`flcSphericalMask_setup`
    """
    template_buffer, template_shape, template_dtype = template
    ft_target_buffer, ft_target_shape, ft_target_dtype = ft_target
    inv_denominator_buffer, inv_denominator_pointer_shape, _ = inv_denominator
    numerator2_buffer, numerator2_shape, _ = numerator2
    filter_buffer, filter_shape, filter_dtype = template_filter

    if callback_class is not None and isinstance(callback_class, type):
        callback = callback_class(**callback_class_args)
    elif not isinstance(callback_class, type):
        callback = callback_class

    # Retrieve objects from shared memory
    template = backend.sharedarr_to_arr(template_shape, template_dtype, template_buffer)
    ft_target = backend.sharedarr_to_arr(
        ft_target_shape, ft_target_dtype, ft_target_buffer
    )
    inv_denominator = backend.sharedarr_to_arr(
        inv_denominator_pointer_shape, template_dtype, inv_denominator_buffer
    )
    numerator2 = backend.sharedarr_to_arr(
        numerator2_shape, template_dtype, numerator2_buffer
    )
    template_filter = backend.sharedarr_to_arr(
        filter_shape, filter_dtype, filter_buffer
    )

    arr = backend.preallocate_array(fast_shape, real_dtype)
    ft_temp = backend.preallocate_array(fast_ft_shape, complex_dtype)

    rfftn, irfftn = backend.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        fftargs=kwargs.get("fftargs", {}),
        temp_real=arr,
        temp_fft=ft_temp,
    )

    norm_numerator = (backend.sum(numerator2) != 0) & (backend.size(numerator2) != 1)
    norm_denominator = (backend.sum(inv_denominator) != 1) & (
        backend.size(inv_denominator) != 1
    )
    filter_template = backend.size(template_filter) != 0

    norm_func_numerator = conditional_execute(backend.subtract, norm_numerator)
    norm_func_denominator = conditional_execute(backend.multiply, norm_denominator)
    template_filter_func = conditional_execute(backend.multiply, filter_template)

    axis = tuple(range(arr.ndim))
    fourier_shift = callback_class_args.get("fourier_shift", backend.zeros(arr.ndim))
    fourier_shift_scores = backend.sum(fourier_shift != 0) != 0

    template_sum = backend.sum(template)
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        backend.fill(arr, 0)
        backend.rotate_array(
            arr=template,
            rotation_matrix=rotation,
            out=arr,
            use_geometric_center=False,
            order=interpolation_order,
        )
        rotation_norm = template_sum / backend.sum(arr)
        backend.multiply(arr, rotation_norm, out=arr)

        rfftn(arr, ft_temp)
        template_filter_func(ft_temp, template_filter, out=ft_temp)

        backend.multiply(ft_target, ft_temp, out=ft_temp)
        irfftn(ft_temp, arr)

        norm_func_numerator(arr, numerator2, out=arr)
        norm_func_denominator(arr, inv_denominator, out=arr)

        if fourier_shift_scores:
            arr = backend.roll(arr, shift=fourier_shift, axis=axis)

        score = apply_convolution_mode(
            arr, convolution_mode=convolution_mode, s1=targetshape, s2=templateshape
        )

        if callback_class is not None:
            callback(
                score,
                rotation_matrix=rotation,
                rotation_index=index,
                **callback_class_args,
            )

    return callback


def flc_scoring(
    template: Tuple[type, Tuple[int], type],
    template_mask: Tuple[type, Tuple[int], type],
    ft_target: Tuple[type, Tuple[int], type],
    ft_target2: Tuple[type, Tuple[int], type],
    template_filter: Tuple[type, Tuple[int], type],
    targetshape: Tuple[int],
    templateshape: Tuple[int],
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: NDArray,
    real_dtype: type,
    complex_dtype: type,
    callback_class: CallbackClass,
    callback_class_args: Dict,
    interpolation_order: int,
    **kwargs,
) -> CallbackClass:
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

    References
    ----------
    .. [1]  W. Wan, S. Khavnekar, J. Wagner, P. Erdmann, and W. Baumeister
            Microsc. Microanal. 26, 2516 (2020)
    .. [2]  T. Hrabe, Y. Chen, S. Pfeffer, L. Kuhn Cuellar, A.-V. Mangold,
            and F. Förster, J. Struct. Biol. 178, 177 (2012).
    """
    template_buffer, template_shape, template_dtype = template
    template_mask_buffer, *_ = template_mask
    filter_buffer, filter_shape, filter_dtype = template_filter

    ft_target_buffer, ft_target_shape, ft_target_dtype = ft_target
    ft_target2_buffer, *_ = ft_target2

    if callback_class is not None and isinstance(callback_class, type):
        callback = callback_class(**callback_class_args)
    elif not isinstance(callback_class, type):
        callback = callback_class

    # Retrieve objects from shared memory
    template = backend.sharedarr_to_arr(template_shape, template_dtype, template_buffer)
    template_mask = backend.sharedarr_to_arr(
        template_shape, template_dtype, template_mask_buffer
    )
    ft_target = backend.sharedarr_to_arr(
        ft_target_shape, ft_target_dtype, ft_target_buffer
    )
    ft_target2 = backend.sharedarr_to_arr(
        ft_target_shape, ft_target_dtype, ft_target2_buffer
    )
    template_filter = backend.sharedarr_to_arr(
        filter_shape, filter_dtype, filter_buffer
    )

    arr = backend.preallocate_array(fast_shape, real_dtype)
    temp = backend.preallocate_array(fast_shape, real_dtype)
    temp2 = backend.preallocate_array(fast_shape, real_dtype)

    ft_temp = backend.preallocate_array(fast_ft_shape, complex_dtype)
    ft_denom = backend.preallocate_array(fast_ft_shape, complex_dtype)

    rfftn, irfftn = backend.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        fftargs=kwargs.get("fftargs", {}),
        temp_real=arr,
        temp_fft=ft_temp,
    )
    eps = backend.eps(real_dtype)
    filter_template = backend.size(template_filter) != 0
    template_filter_func = conditional_execute(backend.multiply, filter_template)

    axis = tuple(range(arr.ndim))
    fourier_shift = callback_class_args.get("fourier_shift", backend.zeros(arr.ndim))
    fourier_shift_scores = backend.sum(fourier_shift != 0) != 0

    template_sum = backend.sum(template)
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        backend.fill(arr, 0)
        backend.fill(temp, 0)
        backend.rotate_array(
            arr=template,
            arr_mask=template_mask,
            rotation_matrix=rotation,
            out=arr,
            out_mask=temp,
            use_geometric_center=False,
            order=interpolation_order,
        )
        rotation_norm = template_sum / backend.sum(arr)
        backend.multiply(arr, rotation_norm, out=arr)
        n_observations = backend.sum(temp)

        rfftn(temp, ft_temp)

        backend.multiply(ft_target, ft_temp, out=ft_denom)
        irfftn(ft_denom, temp)
        backend.divide(temp, n_observations, out=temp)
        backend.square(temp, out=temp)

        backend.multiply(ft_target2, ft_temp, out=ft_denom)
        irfftn(ft_denom, temp2)
        backend.divide(temp2, n_observations, out=temp2)

        backend.subtract(temp2, temp, out=temp)
        backend.maximum(temp, 0.0, out=temp)
        backend.sqrt(temp, out=temp)
        backend.multiply(temp, n_observations, out=temp)

        rfftn(arr, ft_temp)
        template_filter_func(ft_temp, template_filter, out=ft_temp)
        backend.multiply(ft_target, ft_temp, out=ft_temp)
        irfftn(ft_temp, arr)

        tol = tol = 1e3 * eps * backend.max(backend.abs(temp))
        nonzero_indices = temp > tol
        backend.fill(temp2, 0)
        temp2[nonzero_indices] = arr[nonzero_indices] / temp[nonzero_indices]

        convolution_mode = kwargs.get("convolution_mode", "full")

        if fourier_shift_scores:
            temp2 = backend.roll(temp2, shift=fourier_shift, axis=axis)

        score = apply_convolution_mode(
            temp2, convolution_mode=convolution_mode, s1=targetshape, s2=templateshape
        )

        if callback_class is not None:
            callback(
                score,
                rotation_matrix=rotation,
                rotation_index=index,
                **callback_class_args,
            )

    return callback


def mcc_scoring(
    template: Tuple[type, Tuple[int], type],
    template_mask: Tuple[type, Tuple[int], type],
    ft_target: Tuple[type, Tuple[int], type],
    ft_target2: Tuple[type, Tuple[int], type],
    ft_target_mask: Tuple[type, Tuple[int], type],
    template_filter: Tuple[type, Tuple[int], type],
    targetshape: Tuple[int],
    templateshape: Tuple[int],
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    rotations: NDArray,
    real_dtype: type,
    complex_dtype: type,
    callback_class: CallbackClass,
    callback_class_args: type,
    interpolation_order: int,
    overlap_ratio: float = 0.3,
    **kwargs,
) -> CallbackClass:
    """
    Computes a cross-correlation score with masks for both template and target.

    .. math::

        \\frac{
               CC(f, g) - \\frac{CC(f, m) \\cdot CC(t, g)}{CC(t, m)}
            }{
            \\sqrt{
                (CC(f ^ 2, m) - \\frac{CC(f, m) ^ 2}{CC(t, m)}) \\cdot
                (CC(t, g^2) - \\frac{CC(t, g) ^ 2}{CC(t, m)})
                }
            }

    Where:

    .. math::

        CC(f,g) = \\mathcal{F}^{-1}(\\mathcal{F}(f) \\cdot \\mathcal{F}(g)^*)


    References
    ----------
    .. [1]  Masked FFT registration, Dirk Padfield, CVPR 2010 conference
    .. [2]  https://scikit-image.org/docs/stable/api/skimage.registration.html

    See Also
    --------
    :py:class:`tme.matching_optimization.MaskedCrossCorrelation`
    """
    template_buffer, template_shape, template_dtype = template
    ft_target_buffer, ft_target_shape, ft_target_dtype = ft_target
    ft_target2_buffer, ft_target_shape, ft_target_dtype = ft_target2
    template_mask_buffer, _, _ = template
    ft_target_mask_buffer, _, _ = ft_target
    filter_buffer, filter_shape, filter_dtype = template_filter

    if callback_class is not None and isinstance(callback_class, type):
        callback = callback_class(**callback_class_args)
    elif not isinstance(callback_class, type):
        callback = callback_class

    # Retrieve objects from shared memory
    template = backend.sharedarr_to_arr(template_shape, template_dtype, template_buffer)
    target_ft = backend.sharedarr_to_arr(
        ft_target_shape, ft_target_dtype, ft_target_buffer
    )
    target_ft2 = backend.sharedarr_to_arr(
        ft_target_shape, ft_target_dtype, ft_target2_buffer
    )
    template_mask = backend.sharedarr_to_arr(
        template_shape, template_dtype, template_mask_buffer
    )
    target_mask_ft = backend.sharedarr_to_arr(
        ft_target_shape, ft_target_dtype, ft_target_mask_buffer
    )
    template_filter = backend.sharedarr_to_arr(
        filter_shape, filter_dtype, filter_buffer
    )

    axes = tuple(range(template.ndim))
    eps = backend.eps(real_dtype)

    # Allocate score and process specific arrays
    template_rot = backend.preallocate_array(fast_shape, real_dtype)
    mask_overlap = backend.preallocate_array(fast_shape, real_dtype)
    numerator = backend.preallocate_array(fast_shape, real_dtype)

    temp = backend.preallocate_array(fast_shape, real_dtype)
    temp2 = backend.preallocate_array(fast_shape, real_dtype)
    temp3 = backend.preallocate_array(fast_shape, real_dtype)
    temp_ft = backend.preallocate_array(fast_ft_shape, complex_dtype)

    rfftn, irfftn = backend.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=real_dtype,
        complex_dtype=complex_dtype,
        fftargs=kwargs.get("fftargs", {}),
        temp_real=numerator,
        temp_fft=temp_ft,
    )

    filter_template = backend.size(template_filter) != 0
    template_filter_func = conditional_execute(backend.multiply, filter_template)

    axis = tuple(range(template.ndim))
    fourier_shift = callback_class_args.get(
        "fourier_shift", backend.zeros(template.ndim)
    )
    fourier_shift_scores = backend.sum(fourier_shift != 0) != 0

    # Calculate scores across all rotations
    for index in range(rotations.shape[0]):
        rotation = rotations[index]
        backend.fill(template_rot, 0)
        backend.fill(temp, 0)

        backend.rotate_array(
            arr=template,
            arr_mask=template_mask,
            rotation_matrix=rotation,
            out=template_rot,
            out_mask=temp,
            use_geometric_center=False,
            order=interpolation_order,
        )

        backend.multiply(template_rot, temp > 0, out=template_rot)

        # template_rot_ft
        rfftn(template_rot, temp_ft)
        template_filter_func(temp_ft, template_filter, out=temp_ft)
        irfftn(target_mask_ft * temp_ft, temp2)
        irfftn(target_ft * temp_ft, numerator)

        # temp template_mask_rot | temp_ft template_mask_rot_ft
        # Calculate overlap of masks at every point in the convolution.
        # Locations with high overlap should not be taken into account.
        rfftn(temp, temp_ft)
        irfftn(temp_ft * target_mask_ft, mask_overlap)
        mask_overlap[:] = np.round(mask_overlap)
        mask_overlap[:] = np.maximum(mask_overlap, eps)
        irfftn(temp_ft * target_ft, temp)

        backend.subtract(
            numerator,
            backend.divide(backend.multiply(temp, temp2), mask_overlap),
            out=numerator,
        )

        # temp_3 = fixed_denom
        backend.multiply(temp_ft, target_ft2, out=temp_ft)
        irfftn(temp_ft, temp3)
        backend.subtract(
            temp3, backend.divide(backend.square(temp), mask_overlap), out=temp3
        )
        backend.maximum(temp3, 0.0, out=temp3)

        # temp = moving_denom
        rfftn(backend.square(template_rot), temp_ft)
        backend.multiply(target_mask_ft, temp_ft, out=temp_ft)
        irfftn(temp_ft, temp)

        backend.subtract(
            temp, backend.divide(backend.square(temp2), mask_overlap), out=temp
        )
        backend.maximum(temp, 0.0, out=temp)

        # temp_2 = denom
        backend.multiply(temp3, temp, out=temp)
        backend.sqrt(temp, temp2)

        # Pixels where `denom` is very small will introduce large
        # numbers after division. To get around this problem,
        # we zero-out problematic pixels.
        tol = 1e3 * eps * backend.max(backend.abs(temp2), axis=axes, keepdims=True)
        nonzero_indices = temp2 > tol

        backend.fill(temp, 0)
        temp[nonzero_indices] = numerator[nonzero_indices] / temp2[nonzero_indices]
        backend.clip(temp, a_min=-1, a_max=1, out=temp)

        # Apply overlap ratio threshold
        number_px_threshold = overlap_ratio * backend.max(
            mask_overlap, axis=axes, keepdims=True
        )
        temp[mask_overlap < number_px_threshold] = 0.0
        convolution_mode = kwargs.get("convolution_mode", "full")

        if fourier_shift_scores:
            temp = backend.roll(temp, shift=fourier_shift, axis=axis)

        score = apply_convolution_mode(
            temp, convolution_mode=convolution_mode, s1=targetshape, s2=templateshape
        )
        if callback_class is not None:
            callback(
                score,
                rotation_matrix=rotation,
                rotation_index=index,
                **callback_class_args,
            )

    return callback


def device_memory_handler(func: Callable):
    """Decorator function providing SharedMemory Handler."""

    @wraps(func)
    def inner_function(*args, **kwargs):
        return_value = None
        last_type, last_value, last_traceback = sys.exc_info()
        try:
            with SharedMemoryManager() as smh:
                with backend.set_device(kwargs.get("gpu_index", 0)):
                    return_value = func(shared_memory_handler=smh, *args, **kwargs)
        except Exception as e:
            print(e)
            last_type, last_value, last_traceback = sys.exc_info()
        finally:
            handle_traceback(last_type, last_value, last_traceback)
        return return_value

    return inner_function


@device_memory_handler
def scan(
    matching_data: MatchingData,
    matching_setup: Callable,
    matching_score: Callable,
    n_jobs: int = 4,
    callback_class: CallbackClass = None,
    callback_class_args: Dict = {},
    fftargs: Dict = {},
    pad_fourier: bool = True,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    **kwargs,
) -> Tuple:
    """
    Perform template matching between target and template and sample
    different rotations of template.

    Parameters
    ----------
    matching_data : MatchingData
        Template matching data.
    matching_setup : Callable
        Function pointer to setup function.
    matching_score : Callable
        Function pointer to scoring function.
    n_jobs : int, optional
        Number of parallel jobs. Default is 4.
    callback_class : type, optional
        Analyzer class pointer to operate on computed scores.
    callback_class_args : dict, optional
        Arguments passed to the callback_class. Default is an empty dictionary.
    fftargs : dict, optional
        Arguments for the FFT operations. Default is an empty dictionary.
    pad_fourier: bool, optional
        Whether to pad target and template to the full convolution shape.
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        How many jobs should be processed by a single callback_class instance,
        if ones is provided.
    **kwargs : various
        Additional arguments.

    Returns
    -------
    Tuple
        The merged results from callback_class if provided otherwise None.
    """
    shape_diff = backend.subtract(
        matching_data._target.shape, matching_data._template.shape
    )
    if backend.sum(shape_diff < 0) and not pad_fourier:
        warnings.warn(
            "Target is larger than template and Fourier padding is turned off."
            " This can lead to shifted results. You can swap template and target, or"
            " zero-pad the target."
        )

    matching_data.to_backend()

    fast_shape, fast_ft_shape, fourier_shift = matching_data.fourier_padding(
        pad_fourier=pad_fourier
    )

    callback_class_args["fourier_shift"] = fourier_shift
    rfftn, irfftn = backend.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=matching_data._default_dtype,
        complex_dtype=matching_data._complex_dtype,
        fftargs=fftargs,
    )
    setup = matching_setup(
        rfftn=rfftn,
        irfftn=irfftn,
        template=matching_data.template,
        template_mask=matching_data.template_mask,
        target=matching_data.target,
        target_mask=matching_data.target_mask,
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=matching_data._default_dtype,
        complex_dtype=matching_data._complex_dtype,
        callback_class=callback_class,
        callback_class_args=callback_class_args,
        **kwargs,
    )
    rfftn, irfftn = None, None

    template_filter, preprocessor = None, Preprocessor()
    for method, parameters in matching_data.template_filter.items():
        parameters["shape"] = fast_shape
        parameters["omit_negative_frequencies"] = True
        out = preprocessor.apply_method(method=method, parameters=parameters)
        if template_filter is None:
            template_filter = out
        np.multiply(template_filter, out, out=template_filter)

    if template_filter is None:
        template_filter = backend.full(
            shape=(1,), fill_value=1, dtype=backend._default_dtype
        )
    else:
        template_filter = backend.to_backend_array(template_filter)

    template_filter = backend.astype(template_filter, backend._default_dtype)
    template_filter_buffer = backend.arr_to_sharedarr(
        arr=template_filter,
        shared_memory_handler=kwargs.get("shared_memory_handler", None),
    )
    setup["template_filter"] = (
        template_filter_buffer,
        template_filter.shape,
        template_filter.dtype,
    )

    callback_class_args["translation_offset"] = backend.astype(
        matching_data._translation_offset, int
    )
    callback_class_args["thread_safe"] = n_jobs > 1
    callback_class_args["gpu_index"] = kwargs.get("gpu_index", -1)

    n_callback_classes = max(n_jobs // jobs_per_callback_class, 1)
    callback_class = setup.pop("callback_class", callback_class)
    callback_class_args = setup.pop("callback_class_args", callback_class_args)
    callback_classes = [callback_class for _ in range(n_callback_classes)]
    if callback_class == MaxScoreOverRotations:
        score_space_shape = backend.subtract(
            matching_data.target.shape,
            matching_data._target_pad,
        )
        callback_classes = [
            class_name(
                score_space_shape=score_space_shape,
                score_space_dtype=matching_data._default_dtype,
                shared_memory_handler=kwargs.get("shared_memory_handler", None),
                rotation_space_dtype=backend._default_dtype_int,
                **callback_class_args,
            )
            for class_name in callback_classes
        ]

    matching_data._target, matching_data._template = None, None
    matching_data._target_mask, matching_data._template_mask = None, None

    setup["fftargs"] = fftargs.copy()
    convolution_mode = "same"
    if backend.sum(backend.to_backend_array(matching_data._target_pad)) > 0:
        convolution_mode = "valid"
    setup["convolution_mode"] = convolution_mode
    setup["interpolation_order"] = interpolation_order
    rotation_list = matching_data._split_rotations_on_jobs(n_jobs)

    backend.free_cache()

    def _run_scoring(backend_name, backend_args, rotations, **kwargs):
        from tme.backends import backend

        backend.change_backend(backend_name, **backend_args)
        return matching_score(rotations=rotations, **kwargs)

    callbacks = Parallel(n_jobs=n_jobs)(
        delayed(_run_scoring)(
            backend_name=backend._backend_name,
            backend_args=backend._backend_args,
            rotations=rotation,
            callback_class=callback_classes[index % n_callback_classes],
            callback_class_args=callback_class_args,
            **setup,
        )
        for index, rotation in enumerate(rotation_list)
    )

    callbacks = [
        tuple(callback)
        for callback in callbacks[0:n_callback_classes]
        if callback is not None
    ]
    backend.free_cache()

    merged_callback = None
    if callback_class is not None:
        merged_callback = callback_class.merge(
            callbacks,
            **callback_class_args,
            score_indices=matching_data.indices,
            inner_merge=True,
        )

    return merged_callback


def scan_subsets(
    matching_data: MatchingData,
    matching_score: Callable,
    matching_setup: Callable,
    callback_class: CallbackClass = None,
    callback_class_args: Dict = {},
    job_schedule: Tuple[int] = (1, 1),
    target_splits: Dict = {},
    template_splits: Dict = {},
    pad_target_edges: bool = False,
    pad_fourier: bool = True,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    **kwargs,
) -> Tuple:
    """
    Wrapper around :py:meth:`scan` that supports template matching on splits
    of template and target.

    Parameters
    ----------
    matching_data : MatchingData
        Template matching data.
    matching_func : type
        Function pointer to setup function.
    matching_score : type
        Function pointer to scoring function.
    callback_class : type, optional
        Analyzer class pointer to operate on computed scores.
    callback_class_args : dict, optional
        Arguments passed to the callback_class. Default is an empty dictionary.
    job_schedule : tuple of int, optional
        Schedule of jobs. Default is (1, 1).
    target_splits : dict, optional
        Splits for target. Default is an empty dictionary, i.e. no splits
    template_splits : dict, optional
        Splits for template. Default is an empty dictionary, i.e. no splits.
    pad_target_edges : bool, optional
        Whether to pad the target boundaries by half the template shape
        along each axis.
    pad_fourier: bool, optional
        Whether to pad target and template to the full convolution shape.
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        How many jobs should be processed by a single callback_class instance,
        if ones is provided.
    **kwargs : various
        Additional arguments.

    Notes
    -----
        Objects in matching_data might be destroyed during computation.

    Returns
    -------
    Tuple
        The merged results from callback_class if provided otherwise None.
    """
    target_splits = split_numpy_array_slices(
        matching_data._target.shape, splits=target_splits
    )
    template_splits = split_numpy_array_slices(
        matching_data._template.shape, splits=template_splits
    )
    target_pad = matching_data.target_padding(pad_target=pad_target_edges)

    outer_jobs, inner_jobs = job_schedule
    results = Parallel(n_jobs=outer_jobs)(
        delayed(_run_inner)(
            backend_name=backend._backend_name,
            backend_args=backend._backend_args,
            matching_data=matching_data.subset_by_slice(
                target_slice=target_split,
                target_pad=target_pad,
                template_slice=template_split,
            ),
            matching_score=matching_score,
            matching_setup=matching_setup,
            n_jobs=inner_jobs,
            callback_class=callback_class,
            callback_class_args=callback_class_args,
            interpolation_order=interpolation_order,
            pad_fourier=pad_fourier,
            gpu_index=index % outer_jobs,
            **kwargs,
        )
        for index, (target_split, template_split) in enumerate(
            product(target_splits, template_splits)
        )
    )

    matching_data._target, matching_data._template = None, None
    matching_data._target_mask, matching_data._template_mask = None, None

    if callback_class is not None:
        candidates = callback_class.merge(
            results, **callback_class_args, inner_merge=False
        )
        return candidates


MATCHING_EXHAUSTIVE_REGISTER = {
    "CC": (cc_setup, corr_scoring),
    "LCC": (lcc_setup, corr_scoring),
    "CORR": (corr_setup, corr_scoring),
    "CAM": (cam_setup, corr_scoring),
    "FLCSphericalMask": (flcSphericalMask_setup, corr_scoring),
    "FLC": (flc_setup, flc_scoring),
    "MCC": (mcc_setup, mcc_scoring),
}


def register_matching_exhaustive(
    matching: str,
    matching_setup: Callable,
    matching_scoring: Callable,
    memory_class: MatchingMemoryUsage,
) -> None:
    """
    Registers a new matching scheme.

    Parameters
    ----------
    matching : str
        Name of the matching method.
    matching_setup : Callable
        The setup function associated with the name.
    matching_scoring : Callable
        The scoring function associated with the name.
    memory_class : MatchingMemoryUsage
        The custom memory estimation class extending
        :py:class:`tme.matching_memory.MatchingMemoryUsage`.

    Raises
    ------
    ValueError
        If a function with the name ``matching`` already exists in the registry.
    ValueError
        If ``memory_class`` is not a subclass of
        :py:class:`tme.matching_memory.MatchingMemoryUsage`.
    """

    if matching in MATCHING_EXHAUSTIVE_REGISTER:
        raise ValueError(f"A method with name '{matching}' is already registered.")
    if not issubclass(memory_class, MatchingMemoryUsage):
        raise ValueError(f"{memory_class} is not a subclass of {MatchingMemoryUsage}.")

    MATCHING_EXHAUSTIVE_REGISTER[matching] = (matching_setup, matching_scoring)
    MATCHING_MEMORY_REGISTRY[matching] = memory_class
