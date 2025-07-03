"""
Implements cross-correlation based template matching using different metrics.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import sys
import warnings
from math import prod
from functools import wraps
from itertools import product
from typing import Callable, Tuple, Dict, Optional

from joblib import Parallel, delayed
from multiprocessing.managers import SharedMemoryManager

from .filters import Compose
from .backends import backend as be
from .matching_utils import split_shape
from .types import CallbackClass, MatchingData
from .analyzer.proxy import SharedAnalyzerProxy
from .matching_scores import MATCHING_EXHAUSTIVE_REGISTER
from .memory import MatchingMemoryUsage, MATCHING_MEMORY_REGISTRY


def _wrap_backend(func):
    @wraps(func)
    def wrapper(*args, backend_name: str, backend_args: Dict, **kwargs):
        from tme.backends import backend as be

        be.change_backend(backend_name, **backend_args)
        return func(*args, **kwargs)

    return wrapper


def _setup_template_filter_apply_target_filter(
    matching_data: MatchingData,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    pad_template_filter: bool = True,
):
    target_filter = None
    backend_arr = type(be.zeros((1), dtype=be._float_dtype))
    template_filter = be.full(shape=(1,), fill_value=1, dtype=be._float_dtype)
    if isinstance(matching_data.template_filter, backend_arr):
        template_filter = matching_data.template_filter

    if isinstance(matching_data.target_filter, backend_arr):
        target_filter = matching_data.target_filter

    filter_template = isinstance(matching_data.template_filter, Compose)
    filter_target = isinstance(matching_data.target_filter, Compose)

    # For now assume user-supplied template_filter is correctly padded
    if filter_target is None and target_filter is None:
        return template_filter

    cmpl_template_shape_full, batch_mask = fast_ft_shape, matching_data._batch_mask
    real_shape = matching_data._batch_shape(fast_shape, batch_mask, keepdims=False)
    cmpl_shape = matching_data._batch_shape(fast_ft_shape, batch_mask, keepdims=True)

    real_template_shape, cmpl_template_shape = real_shape, cmpl_shape
    cmpl_template_shape_full = matching_data._batch_shape(
        fast_ft_shape, matching_data._target_batch, keepdims=True
    )
    cmpl_target_shape_full = matching_data._batch_shape(
        fast_ft_shape, matching_data._template_batch, keepdims=True
    )
    if filter_template and not pad_template_filter:
        out_shape = matching_data._output_template_shape
        real_template_shape = matching_data._batch_shape(
            out_shape, batch_mask, keepdims=False
        )
        cmpl_template_shape = list(
            matching_data._batch_shape(out_shape, batch_mask, keepdims=True)
        )
        cmpl_template_shape_full = list(out_shape)
        cmpl_template_shape[-1] = cmpl_template_shape[-1] // 2 + 1
        cmpl_template_shape_full[-1] = cmpl_template_shape_full[-1] // 2 + 1

    # Setup composable filters
    target_temp = be.topleft_pad(matching_data.target, fast_shape)
    target_temp_ft = be.rfftn(target_temp)
    filter_kwargs = {
        "return_real_fourier": True,
        "shape_is_real_fourier": False,
        "data_rfft": target_temp_ft,
        "batch_dimension": matching_data._target_dim,
    }

    if filter_template:
        template_filter = matching_data.template_filter(
            shape=real_template_shape, **filter_kwargs
        )["data"]
        template_filter_size = int(be.size(template_filter))

        if template_filter_size == prod(cmpl_template_shape_full):
            cmpl_template_shape = cmpl_template_shape_full
        elif template_filter_size == prod(cmpl_shape):
            cmpl_template_shape = cmpl_shape
        template_filter = be.reshape(template_filter, cmpl_template_shape)

    if filter_target:
        target_filter = matching_data.target_filter(
            shape=real_shape, weight_type=None, **filter_kwargs
        )["data"]
        if int(be.size(target_filter)) == prod(cmpl_target_shape_full):
            cmpl_shape = cmpl_target_shape_full

        target_filter = be.reshape(target_filter, cmpl_shape)
        target_temp_ft = be.multiply(target_temp_ft, target_filter, out=target_temp_ft)

        target_temp = be.irfftn(target_temp_ft, s=target_temp.shape)
        matching_data._target = be.topleft_pad(target_temp, matching_data.target.shape)

    return be.astype(be.to_backend_array(template_filter), be._float_dtype)


def device_memory_handler(func: Callable):
    """Decorator function providing SharedMemory Handler."""

    @wraps(func)
    def inner_function(*args, **kwargs):
        return_value = None
        last_type, last_value, last_traceback = sys.exc_info()
        try:
            with SharedMemoryManager() as smh:
                gpu_index = kwargs.pop("gpu_index") if "gpu_index" in kwargs else 0
                with be.set_device(gpu_index):
                    return_value = func(shm_handler=smh, *args, **kwargs)
        except Exception:
            last_type, last_value, last_traceback = sys.exc_info()
        finally:
            if last_type is not None:
                raise last_value.with_traceback(last_traceback)
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
    pad_target: bool = True,
    pad_template_filter: bool = True,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    shm_handler=None,
    target_slice=None,
    template_slice=None,
) -> Optional[Tuple]:
    """
    Run template matching.

    .. warning:: ``matching_data`` might be altered or destroyed during computation.

    Parameters
    ----------
    matching_data : :py:class:`tme.matching_data.MatchingData`
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
    pad_target: bool, optional
        Whether to pad target to the full convolution shape.
    pad_template_filter: bool, optional
        Whether to pad potential template filters to the full convolution shape.
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        Number of jobs a callback_class instance is shared between, 8 by default.
    shm_handler : type, optional
        Manager for shared memory objects, None by default.

    Returns
    -------
    Optional[Tuple]
        The merged results from callback_class if provided otherwise None.

    Examples
    --------
    Schematically, :py:meth:`scan` is identical to :py:meth:`scan_subsets`,
    with the distinction that the objects contained in ``matching_data`` are not
    split and the search is only parallelized over angles.
    Assuming you have followed the example in :py:meth:`scan_subsets`, :py:meth:`scan`
    can be invoked like so

    >>> from tme.matching_exhaustive import scan
    >>> results = scan(
    >>>    matching_data=matching_data,
    >>>    matching_score=matching_score,
    >>>    matching_setup=matching_setup,
    >>>    callback_class=callback_class,
    >>>    callback_class_args=callback_class_args,
    >>> )

    """
    matching_data, translation_offset = matching_data.subset_by_slice(
        target_slice=target_slice,
        template_slice=template_slice,
        target_pad=matching_data.target_padding(pad_target=pad_target),
    )

    matching_data.to_backend()
    template_shape = matching_data._batch_shape(
        matching_data.template.shape, matching_data._template_batch
    )
    conv, fwd, inv, shift = matching_data.fourier_padding(pad_target=pad_target)

    template_filter = _setup_template_filter_apply_target_filter(
        matching_data=matching_data,
        fast_shape=fwd,
        fast_ft_shape=inv,
        pad_template_filter=pad_template_filter,
    )

    default_callback_args = {
        "shape": fwd,
        "offset": translation_offset,
        "fourier_shift": shift,
        "fast_shape": fwd,
        "targetshape": matching_data._output_shape,
        "templateshape": template_shape,
        "convolution_shape": conv,
        "thread_safe": n_jobs > 1,
        "convolution_mode": "valid" if pad_target else "same",
        "shm_handler": shm_handler,
        "only_unique_rotations": True,
        "aggregate_axis": matching_data._batch_axis(matching_data._batch_mask),
        "n_rotations": matching_data.rotations.shape[0],
    }
    callback_class_args["inversion_mapping"] = n_jobs == 1
    default_callback_args.update(callback_class_args)

    setup = matching_setup(
        matching_data=matching_data,
        template_filter=template_filter,
        fast_shape=fwd,
        fast_ft_shape=inv,
        shm_handler=shm_handler,
    )
    setup["interpolation_order"] = interpolation_order
    setup["template_filter"] = be.to_sharedarr(template_filter, shm_handler)

    matching_data._free_data()
    be.free_cache()

    n_callback_classes = max(n_jobs // jobs_per_callback_class, 1)
    callback_classes = [
        (
            SharedAnalyzerProxy(
                callback_class,
                default_callback_args,
                shm_handler=shm_handler if n_jobs > 1 else None,
            )
            if callback_class
            else None
        )
        for _ in range(n_callback_classes)
    ]
    ret = Parallel(n_jobs=n_jobs)(
        delayed(_wrap_backend(matching_score))(
            backend_name=be._backend_name,
            backend_args=be._backend_args,
            rotations=rotation,
            callback=callback_classes[index % n_callback_classes],
            **setup,
        )
        for index, rotation in enumerate(matching_data._split_rotations_on_jobs(n_jobs))
    )
    callbacks = [
        callback.result(**default_callback_args)
        for callback in ret[:n_callback_classes]
        if callback
    ]
    be.free_cache()

    if callback_class:
        ret = callback_class.merge(callbacks, **default_callback_args)
    return ret


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
    pad_template_filter: bool = True,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    backend_name: str = None,
    backend_args: Dict = {},
    verbose: bool = False,
    **kwargs,
) -> Optional[Tuple]:
    """
    Wrapper around :py:meth:`scan` that supports matching on splits
    of ``matching_data``.

    Parameters
    ----------
    matching_data : :py:class:`tme.matching_data.MatchingData`
        MatchingData instance containing relevant data.
    matching_setup : type
        Function pointer to setup function.
    matching_score : type
        Function pointer to scoring function.
    callback_class : type, optional
        Analyzer class pointer to operate on computed scores.
    callback_class_args : dict, optional
        Arguments passed to the callback_class. Default is an empty dictionary.
    job_schedule : tuple of int, optional
        Job scheduling scheme, default is (1, 1). First value corresponds
        to the number of splits that are processed in parallel, the second
        to the number of angles evaluated in parallel on each split.
    target_splits : dict, optional
        Splits for target. Default is an empty dictionary, i.e. no splits.
        See :py:meth:`tme.matching_utils.compute_parallelization_schedule`.
    template_splits : dict, optional
        Splits for template. Default is an empty dictionary, i.e. no splits.
        See :py:meth:`tme.matching_utils.compute_parallelization_schedule`.
    pad_target_edges : bool, optional
        Pad the target boundaries to avoid edge effects.
    pad_template_filter: bool, optional
        Whether to pad potential template filters to the full convolution shape.
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        How many jobs should be processed by a single callback_class instance,
        if ones is provided.
    verbose : bool, optional
        Indicate matching progress.

    Returns
    -------
    Optional[Tuple]
        The merged results from callback_class if provided otherwise None.

    Examples
    --------
    All data relevant to template matching will be contained in ``matching_data``, which
    is a :py:class:`tme.matching_data.MatchingData` instance and can be created like so

    >>> import numpy as np
    >>> from tme.matching_data import MatchingData
    >>> from tme.matching_utils import get_rotation_matrices
    >>> target = np.random.rand(50,40,60)
    >>> template = target[15:25, 10:20, 30:40]
    >>> matching_data = MatchingData(target, template)
    >>> matching_data.rotations = get_rotation_matrices(
    >>>    angular_sampling=60, dim=target.ndim
    >>> )

    The template matching procedure is determined by ``matching_setup`` and
    ``matching_score``, which are unique to each score. In the following,
    we will be using the `FLCSphericalMask` score, which is composed of
    :py:meth:`tme.matching_scores.flcSphericalMask_setup` and
    :py:meth:`tme.matching_scores.corr_scoring`

    >>> from tme.matching_exhaustive import MATCHING_EXHAUSTIVE_REGISTER
    >>> funcs = MATCHING_EXHAUSTIVE_REGISTER.get("FLCSphericalMask")
    >>> matching_setup, matching_score = funcs

    Computed scores are flexibly analyzed by being passed through an analyzer. In the
    following, we will use :py:class:`tme.analyzer.MaxScoreOverRotations` to
    aggregate sores over rotations

    >>> from tme.analyzer import MaxScoreOverRotations
    >>> callback_class = MaxScoreOverRotations
    >>> callback_class_args = {"score_threshold" : 0}

    In case the entire template matching problem does not fit into memory, we can
    determine the splitting procedure. In this case, we halv the first axis of the target
    once. Splitting and ``job_schedule`` is typically computed using
    :py:meth:`tme.matching_utils.compute_parallelization_schedule`.

    >>> target_splits = {0 : 1}

    Finally, we can perform template matching. Note that the data
    contained in ``matching_data`` will be destroyed when running the following

    >>> from tme.matching_exhaustive import scan_subsets
    >>> results = scan_subsets(
    >>>    matching_data=matching_data,
    >>>    matching_score=matching_score,
    >>>    matching_setup=matching_setup,
    >>>    callback_class=callback_class,
    >>>    callback_class_args=callback_class_args,
    >>>    target_splits=target_splits,
    >>> )

    The ``results`` tuple contains the output of the chosen analyzer.

    See Also
    --------
    :py:meth:`tme.matching_utils.compute_parallelization_schedule`
    """
    template_splits = split_shape(matching_data._template.shape, splits=template_splits)
    target_splits = split_shape(matching_data._target.shape, splits=target_splits)
    if (len(target_splits) > 1) and not pad_target_edges:
        warnings.warn(
            "Target splitting without padding target edges leads to unreliable "
            "similarity estimates around the split border."
        )
    splits = tuple(product(target_splits, template_splits))

    outer_jobs, inner_jobs = job_schedule
    if be._backend_name == "jax":
        func = be.scan

        corr_scoring = MATCHING_EXHAUSTIVE_REGISTER.get("CORR", (None, None))[1]
        results = func(
            matching_data=matching_data,
            splits=splits,
            n_jobs=outer_jobs,
            rotate_mask=matching_score != corr_scoring,
            callback_class=callback_class,
            callback_class_args=callback_class_args,
        )
    else:
        results = Parallel(n_jobs=outer_jobs, verbose=verbose)(
            [
                delayed(_wrap_backend(scan))(
                    backend_name=be._backend_name,
                    backend_args=be._backend_args,
                    matching_data=matching_data,
                    matching_score=matching_score,
                    matching_setup=matching_setup,
                    n_jobs=inner_jobs,
                    callback_class=callback_class,
                    callback_class_args=callback_class_args,
                    interpolation_order=interpolation_order,
                    pad_target=pad_target_edges,
                    gpu_index=index % outer_jobs,
                    pad_template_filter=pad_template_filter,
                    target_slice=target_split,
                    template_slice=template_split,
                )
                for index, (target_split, template_split) in enumerate(splits)
            ]
        )
    matching_data._free_data()
    if callback_class is not None:
        return callback_class.merge(results, **callback_class_args)
    return None


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
        Corresponding setup function.
    matching_scoring : Callable
        Corresponing scoring function.
    memory_class : MatchingMemoryUsage
        Child of :py:class:`tme.memory.MatchingMemoryUsage`.

    Raises
    ------
    ValueError
        If a function with the name ``matching`` already exists in the registry, or
        if ``memory_class`` is no child of :py:class:`tme.memory.MatchingMemoryUsage`.
    """

    if matching in MATCHING_EXHAUSTIVE_REGISTER:
        raise ValueError(f"A method with name '{matching}' is already registered.")
    if not issubclass(memory_class, MatchingMemoryUsage):
        raise ValueError(f"{memory_class} is not a subclass of {MatchingMemoryUsage}.")

    MATCHING_EXHAUSTIVE_REGISTER[matching] = (matching_setup, matching_scoring)
    MATCHING_MEMORY_REGISTRY[matching] = memory_class
