"""
Implements cross-correlation based template matching using different metrics.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import sys
import warnings
from functools import wraps
from itertools import product
from typing import Callable, Tuple, Dict, Optional

from joblib import Parallel, delayed
from multiprocessing.managers import SharedMemoryManager

from .backends import backend as be
from .matching_utils import split_shape, setup_filter
from .types import CallbackClass, MatchingData
from .analyzer.proxy import SharedAnalyzerProxy
from .matching_scores import MATCHING_EXHAUSTIVE_REGISTER
from .memory import MatchingMemoryUsage, MATCHING_MEMORY_REGISTRY

__all__ = ["match_exhaustive"]


def _wrap_backend(func):
    @wraps(func)
    def wrapper(*args, backend_name: str, backend_args: Dict, **kwargs):
        from tme.backends import backend as be

        be.change_backend(backend_name, **backend_args)
        return func(*args, **kwargs)

    return wrapper


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
def _match_exhaustive(
    matching_data: MatchingData,
    matching_setup: Callable,
    matching_score: Callable,
    callback_class: CallbackClass,
    callback_class_args: Dict = {},
    n_jobs: int = 4,
    pad_target: bool = True,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    shm_handler=None,
    target_slice=None,
    template_slice=None,
    background_correction: str = None,
    **kwargs,
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
    callback_class : type
        Analyzer class pointer to operate on computed scores.
    callback_class_args : dict, optional
        Arguments passed to the callback_class. Default is an empty dictionary.
    pad_target: bool, optional
        Whether to pad target to the full convolution shape.
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        Number of jobs a callback_class instance is shared between, 8 by default.
    shm_handler : type, optional
        Manager for shared memory objects, None by default.
    target_slice : tuple of slice, optional
        Target subset to process.
    template_slice : tuple of slice, optional
        Template subset to process.
    background_correction : str, optional
        Background correctoin use use. Supported methods are 'phase-scrambling'.

    Returns
    -------
    Optional[Tuple]
        The merged results from callback_class if provided otherwise None.

    Notes
    -----
    Schematically, this function is identical to :py:meth:`match_exhaustive`,
    with the distinction that the objects contained in ``matching_data`` are not
    split and the search is only parallelized over angles.
    """
    matching_data, translation_offset = matching_data.subset_by_slice(
        target_slice=target_slice,
        template_slice=template_slice,
        target_pad=matching_data.target_padding(pad_target=pad_target),
    )

    matching_data.to_backend()
    template_shape = matching_data._batch_shape(
        matching_data._template.shape, matching_data._template_batch
    )
    conv, fwd, inv, shift = matching_data.fourier_padding()

    # Mask invalid scores from padding to not skew score statistics
    score_mask = be.full(shape=(1,), fill_value=1, dtype=bool)
    if pad_target:
        score_mask = matching_data._score_mask(fwd, shift)

    template_filter, _ = setup_filter(
        matching_data=matching_data,
        fast_shape=fwd,
        fast_ft_shape=inv,
        pad_template_filter=False,
        apply_target_filter=True,
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
        "aggregate_axis": matching_data._batch_axis(matching_data._batch_mask),
        "n_rotations": matching_data.rotations.shape[0],
        "inversion_mapping": n_jobs == 1,
    }
    default_callback_args.update(callback_class_args)

    setup = matching_setup(
        matching_data=matching_data,
        template_filter=template_filter,
        fast_shape=fwd,
        fast_ft_shape=inv,
        shm_handler=shm_handler,
    )

    if background_correction == "phase-scrambling":
        # Use getter to make sure template is reversed correctly
        matching_data.template = matching_data.transform_template("phase_randomization")
        setup["template_background"] = be.to_sharedarr(matching_data.template)

    matching_data.free()
    if not callback_class.shareable:
        jobs_per_callback_class = 1

    n_callback_classes = max(n_jobs // jobs_per_callback_class, 1)
    callback_classes = [
        SharedAnalyzerProxy(
            callback_class,
            default_callback_args,
            shm_handler=shm_handler if n_jobs > 1 else None,
        )
        for _ in range(n_callback_classes)
    ]
    ret = Parallel(n_jobs=n_jobs)(
        delayed(_wrap_backend(matching_score))(
            backend_name=be._backend_name,
            backend_args=be._backend_args,
            fast_shape=fwd,
            fast_ft_shape=inv,
            rotations=rotation,
            callback=callback_classes[index % n_callback_classes],
            interpolation_order=interpolation_order,
            template_filter=be.to_sharedarr(template_filter, shm_handler),
            score_mask=be.to_sharedarr(score_mask, shm_handler),
            **setup,
        )
        for index, rotation in enumerate(matching_data._split_rotations_on_jobs(n_jobs))
    )
    be.free_cache()

    # Background correction creates individual non-shared arrays
    if background_correction is None:
        ret = ret[:n_callback_classes]
    callbacks = [x.result(**default_callback_args) for x in ret]
    return callback_class.merge(callbacks, **default_callback_args)


def match_exhaustive(
    matching_data: MatchingData,
    matching_score: Callable,
    matching_setup: Callable,
    callback_class: CallbackClass,
    callback_class_args: Dict = {},
    job_schedule: Tuple[int] = (1, 1),
    target_splits: Dict = {},
    template_splits: Dict = {},
    pad_target_edges: bool = False,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    backend_name: str = None,
    backend_args: Dict = {},
    verbose: bool = False,
    background_correction: str = None,
    **kwargs,
) -> Optional[Tuple]:
    """
    Run exhaustive template matching over all translations and a subset of rotations
    specified in `matching_data`.

    Parameters
    ----------
    matching_data : :py:class:`tme.matching_data.MatchingData`
        MatchingData instance containing relevant data.
    matching_setup : type
        Function pointer to setup function.
    matching_score : type
        Function pointer to scoring function.
    callback_class : type
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
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        How many jobs should be processed by a single callback_class instance,
        if ones is provided.
    verbose : bool, optional
        Indicate matching progress, defaults to False.
    background_correction : str, optional
        Background correctoin use use. Supported methods are 'phase-scrambling'.

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
    >>> from tme.rotations import get_rotation_matrices
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

    >>> from tme.matching_exhaustive import match_exhaustive
    >>> results = match_exhaustive(
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
    if background_correction not in (None, "phase-scrambling"):
        raise ValueError(
            "Argument background_correction can be either None or "
            f"'phase-scrambling', got {background_correction}."
        )

    template_splits = split_shape(matching_data._template.shape, splits=template_splits)
    target_splits = split_shape(matching_data._target.shape, splits=target_splits)
    if (len(target_splits) > 1) and not pad_target_edges:
        warnings.warn(
            "Target splitting without padding target edges leads to unreliable "
            "similarity estimates around the split border."
        )
    splits = tuple(product(target_splits, template_splits))

    kwargs = {
        "match_projection": kwargs.get("match_projection", False),
        "matching_data": matching_data,
        "callback_class": callback_class,
        "callback_class_args": callback_class_args,
    }
    outer_jobs, inner_jobs = job_schedule
    if be._backend_name == "jax":
        score = MATCHING_EXHAUSTIVE_REGISTER.get("FLC", (None, None))[1]
        results = be.scan(
            splits=splits,
            n_jobs=outer_jobs,
            rotate_mask=matching_score == score,
            background_correction=background_correction,
            **kwargs,
        )
    else:
        results = Parallel(n_jobs=outer_jobs, verbose=verbose)(
            [
                delayed(_wrap_backend(_match_exhaustive))(
                    backend_name=be._backend_name,
                    backend_args=be._backend_args,
                    matching_score=matching_score,
                    matching_setup=matching_setup,
                    n_jobs=inner_jobs,
                    interpolation_order=interpolation_order,
                    pad_target=pad_target_edges,
                    gpu_index=index % outer_jobs,
                    target_slice=target_split,
                    template_slice=template_split,
                    background_correction=background_correction,
                    **kwargs,
                )
                for index, (target_split, template_split) in enumerate(splits)
            ]
        )
    return callback_class.merge(results, **callback_class_args)


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


def scan(*args, **kwargs):
    warnings.warn(
        "Using scan directly is deprecated and will raise an error "
        "in future releases. Please use match_exhaustive instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return _match_exhaustive(*args, **kwargs)


def scan_subsets(*args, **kwargs):
    warnings.warn(
        "Using scan_subsets directly is deprecated and will raise an error "
        "in future releases. Please use match_exhaustive instead.",
        DeprecationWarning,
    )
    return match_exhaustive(*args, **kwargs)
