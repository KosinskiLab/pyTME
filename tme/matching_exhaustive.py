""" Implements cross-correlation based template matching using different metrics.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import sys
import warnings
import traceback
from functools import wraps
from itertools import product
from typing import Callable, Tuple, Dict, Optional

from joblib import Parallel, delayed
from multiprocessing.managers import SharedMemoryManager

from .backends import backend as be
from .preprocessing import Compose
from .matching_utils import split_shape
from .matching_scores import MATCHING_EXHAUSTIVE_REGISTER
from .types import CallbackClass, MatchingData
from .memory import MatchingMemoryUsage, MATCHING_MEMORY_REGISTRY


def _handle_traceback(last_type, last_value, last_traceback):
    """
    Handle sys.exc_info().

    Parameters
    ----------
    last_type : type
        The type of the last exception.
    last_value :
        The value of the last exception.
    last_traceback : traceback
        Traceback call stack at the point where the Exception occured.

    Raises
    ------
    Exception
        Re-raises the last exception.
    """
    if last_type is None:
        return None
    traceback.print_tb(last_traceback)
    raise Exception(last_value)


def _wrap_backend(func):
    @wraps(func)
    def wrapper(*args, backend_name: str, backend_args: Dict, **kwargs):
        from tme.backends import backend as be

        be.change_backend(backend_name, **backend_args)
        return func(*args, **kwargs)

    return wrapper


def _setup_template_filter_apply_target_filter(
    matching_data: MatchingData,
    rfftn: Callable,
    irfftn: Callable,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    pad_template_filter: bool = True,
):
    filter_template = isinstance(matching_data.template_filter, Compose)
    filter_target = isinstance(matching_data.target_filter, Compose)

    template_filter = be.full(shape=(1,), fill_value=1, dtype=be._float_dtype)

    if not filter_template and not filter_target:
        return template_filter

    inv_mask = be.subtract(1, be.to_backend_array(matching_data._batch_mask))
    filter_shape = be.multiply(be.to_backend_array(fast_ft_shape), inv_mask)
    filter_shape = tuple(int(x) if x != 0 else 1 for x in filter_shape)
    fast_shape = be.multiply(be.to_backend_array(fast_shape), inv_mask)
    fast_shape = tuple(int(x) for x in fast_shape if x != 0)

    fastt_shape, fastt_ft_shape = fast_shape, filter_shape
    if filter_template and not pad_template_filter:
        # FFT shape acrobatics for faster filter application
        # _, fastt_shape, _, _ = matching_data._fourier_padding(
        #     target_shape=be.to_numpy_array(matching_data._template.shape),
        #     template_shape=be.to_numpy_array(
        #         [1 for _ in matching_data._template.shape]
        #     ),
        #     pad_fourier=False,
        # )
        fastt_shape = matching_data._template.shape
        # matching_data.template = be.reverse(
        #     be.topleft_pad(matching_data.template, fastt_shape)
        # )
        # matching_data.template_mask = be.reverse(
        #     be.topleft_pad(matching_data.template_mask, fastt_shape)
        # )
        matching_data._set_matching_dimension(
            target_dims=matching_data._target_dims,
            template_dims=matching_data._template_dims,
        )
        fastt_ft_shape = [int(x) for x in matching_data._output_template_shape]
        fastt_ft_shape[-1] = fastt_ft_shape[-1] // 2 + 1

    target_temp = be.topleft_pad(matching_data.target, fast_shape)
    target_temp_ft = be.zeros(fast_ft_shape, be._complex_dtype)
    target_temp_ft = rfftn(target_temp, target_temp_ft)
    if filter_template:
        template_filter = matching_data.template_filter(
            shape=fastt_shape,
            return_real_fourier=True,
            shape_is_real_fourier=False,
            data_rfft=target_temp_ft,
            batch_dimension=matching_data._target_dims,
        )["data"]
        template_filter = be.reshape(template_filter, fastt_ft_shape)

    if filter_target:
        target_filter = matching_data.target_filter(
            shape=fast_shape,
            return_real_fourier=True,
            shape_is_real_fourier=False,
            data_rfft=target_temp_ft,
            weight_type=None,
            batch_dimension=matching_data._target_dims,
        )["data"]
        target_filter = be.reshape(target_filter, filter_shape)
        target_temp_ft = be.multiply(target_temp_ft, target_filter, out=target_temp_ft)

        target_temp = irfftn(target_temp_ft, target_temp)
        matching_data._target = be.topleft_pad(target_temp, matching_data.target.shape)

    return template_filter


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
                    return_value = func(shared_memory_handler=smh, *args, **kwargs)
        except Exception as e:
            print(e)
            last_type, last_value, last_traceback = sys.exc_info()
        finally:
            _handle_traceback(last_type, last_value, last_traceback)
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
    pad_fourier: bool = True,
    pad_template_filter: bool = True,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    shared_memory_handler=None,
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
    pad_fourier: bool, optional
        Whether to pad target and template to the full convolution shape.
    pad_template_filter: bool, optional
        Whether to pad potential template filters to the full convolution shape.
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        How many jobs should be processed by a single callback_class instance,
        if one is provided.
    shared_memory_handler : type, optional
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
    >>>    matching_data = matching_data,
    >>>    matching_score = matching_score,
    >>>    matching_setup = matching_setup,
    >>>    callback_class = callback_class,
    >>>    callback_class_args = callback_class_args,
    >>> )

    """
    matching_data.to_backend()
    (
        conv_shape,
        fast_shape,
        fast_ft_shape,
        fourier_shift,
    ) = matching_data.fourier_padding(pad_fourier=pad_fourier)
    template_shape = matching_data.template.shape

    rfftn, irfftn = be.build_fft(
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        real_dtype=be._float_dtype,
        complex_dtype=be._complex_dtype,
    )
    template_filter = _setup_template_filter_apply_target_filter(
        matching_data=matching_data,
        rfftn=rfftn,
        irfftn=irfftn,
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        pad_template_filter=pad_template_filter,
    )
    template_filter = be.astype(be.to_backend_array(template_filter), be._float_dtype)

    setup = matching_setup(
        rfftn=rfftn,
        irfftn=irfftn,
        template=matching_data.template,
        template_filter=template_filter,
        template_mask=matching_data.template_mask,
        target=matching_data.target,
        target_mask=matching_data.target_mask,
        fast_shape=fast_shape,
        fast_ft_shape=fast_ft_shape,
        shared_memory_handler=shared_memory_handler,
    )
    rfftn, irfftn = None, None
    setup["interpolation_order"] = interpolation_order
    setup["template_filter"] = be.to_sharedarr(template_filter, shared_memory_handler)

    offset = be.to_backend_array(matching_data._translation_offset)
    convmode = "valid" if getattr(matching_data, "_is_padded", False) else "same"
    default_callback_args = {
        "offset": be.astype(offset, be._int_dtype),
        "thread_safe": n_jobs > 1,
        "fourier_shift": fourier_shift,
        "convolution_mode": convmode,
        "targetshape": matching_data.target.shape,
        "templateshape": template_shape,
        "convolution_shape": conv_shape,
        "fast_shape": fast_shape,
        "indices": getattr(matching_data, "indices", None),
        "shared_memory_handler": shared_memory_handler,
        "only_unique_rotations": True,
    }
    default_callback_args.update(callback_class_args)

    matching_data._free_data()
    be.free_cache()

    # For highly parallel jobs, blocking in certain analyzers becomes a bottleneck
    if getattr(callback_class, "shared", True):
        jobs_per_callback_class = 1
    n_callback_classes = max(n_jobs // jobs_per_callback_class, 1)
    callback_classes = [
        callback_class(
            shape=fast_shape,
            **default_callback_args,
        )
        if callback_class is not None
        else None
        for _ in range(n_callback_classes)
    ]
    callbacks = Parallel(n_jobs=n_jobs)(
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
        tuple(callback._postprocess(**default_callback_args))
        for callback in callbacks
        if callback is not None
    ]
    be.free_cache()

    if callback_class is not None:
        return callback_class.merge(callbacks, **default_callback_args)
    return None


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
    pad_template_filter: bool = True,
    interpolation_order: int = 3,
    jobs_per_callback_class: int = 8,
    backend_name: str = None,
    backend_args: Dict = {},
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
        Whether to pad the target boundaries by half the template shape
        along each axis.
    pad_fourier: bool, optional
        Whether to pad target and template to the full convolution shape.
    pad_template_filter: bool, optional
        Whether to pad potential template filters to the full convolution shape.
    interpolation_order : int, optional
        Order of spline interpolation for rotations.
    jobs_per_callback_class : int, optional
        How many jobs should be processed by a single callback_class instance,
        if ones is provided.

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
    >>>    angular_sampling = 60, dim = target.ndim
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
    >>>    matching_data = matching_data,
    >>>    matching_score = matching_score,
    >>>    matching_setup = matching_setup,
    >>>    callback_class = callback_class,
    >>>    callback_class_args = callback_class_args,
    >>>    target_splits = target_splits,
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
    target_pad = matching_data.target_padding(pad_target=pad_target_edges)
    if hasattr(be, "scan"):
        corr_scoring = MATCHING_EXHAUSTIVE_REGISTER.get("CORR", (None, None))[1]
        results = be.scan(
            matching_data=matching_data,
            splits=splits,
            n_jobs=outer_jobs,
            rotate_mask=matching_score != corr_scoring,
            callback_class=callback_class,
        )
    else:
        results = Parallel(n_jobs=outer_jobs)(
            delayed(_wrap_backend(scan))(
                backend_name=be._backend_name,
                backend_args=be._backend_args,
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
                pad_template_filter=pad_template_filter,
            )
            for index, (target_split, template_split) in enumerate(splits)
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
