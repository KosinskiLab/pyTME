"""
Utility functions for template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
import warnings
from tempfile import mkstemp
from itertools import product
from typing import Tuple, Dict, Callable, Generator

import numpy as np

from .backends import backend as be
from .types import NDArray, BackendArray, MatchingData


def copy_docstring(source_func, append: bool = True):
    """Decorator to copy docstring from source function."""

    def decorator(target_func):
        base_doc = source_func.__doc__ or ""
        if append and target_func.__doc__:
            target_func.__doc__ = base_doc + "\n\n" + target_func.__doc__
        else:
            target_func.__doc__ = base_doc
        return target_func

    return decorator


def to_padded(buffer, data, unpadded_slice):
    buffer = be.fill(buffer, 0)
    return be.at(buffer, unpadded_slice, data)


def identity(arr, *args, **kwargs):
    return arr


def conditional_execute(
    func: Callable,
    execute_operation: bool = False,
    alt_func: Callable = identity,
) -> Callable:
    """
    Return the given function or alternative function based on execute_operation.

    Parameters
    ----------
    func : Callable
        Callable.
    alt_func : Callable
        Callable to return if ``execute_operation`` is False, identity by default.
    execute_operation : bool
        Whether to return ``func`` or a ``alt_func`` function.

    Returns
    -------
    Callable
        ``func`` if ``execute_operation`` else ``alt_func``.
    """

    return func if execute_operation else alt_func


def standardize(
    template: BackendArray, mask: BackendArray, n_observations: float, axis=None
) -> BackendArray:
    """
    Standardizes ``template`` to zero mean and unit standard deviation in ``mask``.

    .. warning:: ``template`` is modified during the operation.

    Parameters
    ----------
    template : BackendArray
        Input data.
    mask : BackendArray
        Mask of the same shape as ``template``.
    n_observations : float
        Sum of mask elements.
    axis : tuple of floats, optional
        Axis to normalize over, all axis by default.

    Returns
    -------
    BackendArray
        Standardized input data.

    References
    ----------
    .. [1]  Hrabe T. et al, J. Struct. Biol. 178, 177 (2012).
    """
    masked_mean = be.sum(be.multiply(template, mask), axis=axis, keepdims=True)
    masked_mean = be.divide(masked_mean, n_observations)
    masked_std = be.sum(
        be.multiply(be.square(template), mask), axis=axis, keepdims=True
    )
    masked_std = be.subtract(masked_std / n_observations, be.square(masked_mean))
    masked_std = be.sqrt(be.maximum(masked_std, 0))

    template = be.subtract(template, masked_mean, out=template)
    template = be.divide(template, masked_std, out=template)
    return be.multiply(template, mask, out=template)


def _standardize_safe(
    template: BackendArray, mask: BackendArray, n_observations: float, axis=None
) -> BackendArray:
    """Overflow-safe version of standardize using higher precision arithmetic."""
    _template = be.astype(template, be._overflow_safe_dtype)
    _mask = be.astype(mask, be._overflow_safe_dtype)
    standardize(
        template=_template, mask=_mask, n_observations=n_observations, axis=axis
    )
    template[:] = be.astype(_template, template.dtype)
    return template


def generate_tempfile_name(suffix: str = None, tmpdir: str = None) -> str:
    """
    Returns the path to a temporary file with given suffix. If defined. the
    environment variable TMPDIR is used as base.

    Parameters
    ----------
    suffix : str, optional
        File suffix. By default the file has no suffix.
    tmpdir : str, optional
        Directory the file is created in. Defaults to the system temp location
        honoring TMPDIR.

    Returns
    -------
    str
        The generated filename
    """
    fd, path = mkstemp(suffix=suffix, dir=tmpdir)
    os.close(fd)
    return path


def array_to_memmap(arr: NDArray, filename: str = None, mode: str = "r") -> np.memmap:
    """
    Converts a obj:`numpy.ndarray` to a obj:`numpy.memmap`.

    Parameters
    ----------
    arr : obj:`numpy.ndarray`
        Input data.
    filename : str, optional
        Path to new memmap, :py:meth:`generate_tempfile_name` is used by default.
    mode : str, optional
        Mode to open the returned memmap object in, defautls to 'r'.

    Returns
    -------
    obj:`numpy.memmap`
        Memmaped array in reading mode.
    """
    if filename is None:
        filename = generate_tempfile_name()

    arr.tofile(filename)
    return np.memmap(filename, mode=mode, dtype=arr.dtype, shape=arr.shape)


def memmap_to_array(arr: NDArray) -> NDArray:
    """
    Convert a obj:`numpy.memmap` to a obj:`numpy.ndarray` and delete the memmap.

    Parameters
    ----------
    arr : obj:`numpy.memmap`
        Input data.

    Returns
    -------
    obj:`numpy.ndarray`
        In-memory version of ``arr``.
    """
    if isinstance(arr, np.memmap):
        memmap_filepath = arr.filename
        ret = np.array(arr)
        # Windows refuses to remove a file while its mmap handle is open.
        arr._mmap.close()
        del arr
        os.remove(memmap_filepath)
        arr = ret
    return arr


def center_slice(current_shape: Tuple[int], new_shape: Tuple[int]) -> Tuple[slice]:
    """Extract the center slice of ``current_shape`` to retrieve ``new_shape``."""
    new_shape = tuple(int(x) for x in new_shape)
    current_shape = tuple(int(x) for x in current_shape)
    starts = tuple((x - y) // 2 for x, y in zip(current_shape, new_shape))
    stops = tuple(sum(stop) for stop in zip(starts, new_shape))
    box = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    return box


def apply_convolution_mode(
    arr: BackendArray,
    convolution_mode: str,
    s1: Tuple[int],
    s2: Tuple[int],
    convolution_shape: Tuple[int] = None,
) -> BackendArray:
    """
    Applies convolution_mode to ``arr``.

    Parameters
    ----------
    arr : BackendArray
        Array containing convolution result of arrays with shape s1 and s2.
    convolution_mode : str
        Analogous to mode in obj:`scipy.signal.convolve`:

        +---------+----------------------------------------------------------+
        | 'full'  | returns full template matching result of the inputs.     |
        +---------+----------------------------------------------------------+
        | 'valid' | returns elements that do not rely on zero-padding..      |
        +---------+----------------------------------------------------------+
        | 'same'  | output is the same size as s1.                           |
        +---------+----------------------------------------------------------+
    s1 : tuple of ints
        Tuple of integers corresponding to shape of convolution array 1.
    s2 : tuple of ints
        Tuple of integers corresponding to shape of convolution array 2.
    convolution_shape : tuple of ints, optional
        Size of the actually computed convolution. s1 + s2 - 1 by default.

    Returns
    -------
    BackendArray
        The array after applying the convolution mode.
    """
    # Remove padding to next fast Fourier length
    if convolution_shape is None:
        convolution_shape = [s1[i] + s2[i] - 1 for i in range(len(s1))]
    arr = arr[tuple(slice(x) for x in convolution_shape)]

    if convolution_mode not in ("full", "same", "valid"):
        raise ValueError("Supported convolution_mode are 'full', 'same' and 'valid'.")

    if convolution_mode == "full":
        subset = ...
    elif convolution_mode == "same":
        subset = center_slice(arr.shape, s1)
    elif convolution_mode == "valid":
        subset = center_slice(arr.shape, [x - y + 1 for x, y in zip(s1, s2)])
    return arr[subset]


def sliding_window_slices(
    shape: Tuple[int, ...], length: int, step: int
) -> Generator[Tuple[slice, ...], None, None]:
    """Yield overlapping hypercubic window slices tiling ``shape``.

    Parameters
    ----------
    shape : tuple of int
        Shape of the array to tile, of any dimensionality.
    length : int
        Edge length of the window along every axis.
    step : int
        Stride between consecutive windows along every axis.

    Yields
    ------
    tuple of slice
        One slice per axis selecting a single window. The last window along
        each axis is snapped to the array edge so the whole array is covered
        even when ``length`` does not divide the extent.
    """

    def starts(n):
        s = list(range(0, n - length + 1, step))
        if not s:
            raise ValueError(f"window length {length} larger than extent {n}")
        if s[-1] != n - length:
            s.append(n - length)
        return s

    for offsets in product(*(starts(n) for n in shape)):
        yield tuple(slice(o, o + length) for o in offsets)


def split_shape(
    shape: Tuple[int], splits: Dict, equal_shape: bool = True
) -> Tuple[slice]:
    """
    Splits ``shape`` into equally sized and potentially overlapping subsets.

    Parameters
    ----------
    shape : tuple of ints
        Shape to split.
    splits : dict
        Dictionary mapping axis number to number of splits.
    equal_shape : dict
        Whether the subsets should be of equal shape, True by default.

    Returns
    -------
    tuple
        Tuple of slice with requested split combinations.
    """
    ndim = len(shape)
    splits = {k: max(splits.get(k, 1), 1) for k in range(ndim)}
    ret_shape = np.divide(shape, tuple(splits[i] for i in range(ndim)))
    if equal_shape:
        ret_shape = np.ceil(ret_shape).astype(int)
    ret_shape = tuple(int(x) for x in ret_shape)

    slice_list = [
        tuple(
            (
                (slice((n_splits * length), (n_splits + 1) * length))
                if n_splits < splits.get(axis, 1) - 1
                else (
                    (slice(shape[axis] - length, shape[axis]))
                    if equal_shape
                    else (slice((n_splits * length), shape[axis]))
                )
            )
            for n_splits in range(splits.get(axis, 1))
        )
        for length, axis in zip(ret_shape, splits.keys())
    ]
    return tuple(product(*slice_list))


def _rigid_transform(
    coordinates: NDArray,
    rotation_matrix: NDArray,
    out: NDArray,
    translation: NDArray,
    coordinates_mask: NDArray = None,
    out_mask: NDArray = None,
    center: NDArray = None,
    **kwargs,
) -> None:
    """
    Apply a rigid transformation to given coordinates as

    rotation_matrix.T @ coordinates + translation

    Parameters
    ----------
    coordinates : NDArray
        An array representing the coordinates to be transformed (d,n).
    rotation_matrix : NDArray
        The rotation matrix to be applied (d,d).
    translation : NDArray
        The translation vector to be applied (d,).
    out : NDArray
        The output array to store the transformed coordinates (d,n).
    coordinates_mask : NDArray, optional
        An array representing the mask for the coordinates (d,t).
    out_mask : NDArray, optional
        The output array to store the transformed coordinates mask (d,t).
    center : NDArray, optional
        Coordinate center, defaults to the average along each axis.
    """
    if center is None:
        center = coordinates.mean(axis=1)

    coordinates = coordinates - center[:, None]
    out = np.matmul(rotation_matrix.T, coordinates, out=out)
    translation = np.add(translation, center)

    out = np.add(out, translation[:, None], out=out)
    if coordinates_mask is not None and out_mask is not None:
        np.matmul(rotation_matrix.T, coordinates_mask, out=out_mask)
        out_mask = np.add(out_mask, translation[:, None], out=out_mask)


def minimum_enclosing_box(coordinates: NDArray, **kwargs) -> Tuple[int, ...]:
    """
    Computes the minimal enclosing box around coordinates.

    Parameters
    ----------
    coordinates : NDArray
        Coordinates of shape (d,n) to compute the enclosing box of.
    margin : NDArray, optional
        Box margin, zero by default.

        .. deprecated:: 0.3.2

            Boxed are returned without margin.

    use_geometric_center : bool, optional
        Whether box accommodates the geometric or coordinate center, False by default.

        .. deprecated:: 0.3.2

            Boxes always accomodate the coordinate center

    Returns
    -------
    tuple of int
        Minimum enclosing box.
    """
    coordinates = np.asarray(coordinates).T
    coordinates = coordinates - coordinates.min(axis=0)
    coordinates = coordinates - coordinates.mean(axis=0)

    # Adding one avoids clipping during scipy.ndimage.affine_transform
    box_size = int(np.ceil(2 * np.linalg.norm(coordinates, axis=1).max()) + 1)
    return tuple(box_size for _ in range(coordinates.shape[1]))


def scramble_phases(arr: NDArray, seed: int = 42, **kwargs) -> NDArray:
    """
    Perform phase scrambling of ``arr``.

    Parameters
    ----------
    arr : NDArray
        Input data.
    seed : int, optional
        The seed for the phase scrambling, 42 by default.

    Returns
    -------
    NDArray
        Phase scrambled version of ``arr``.
    """
    amp = np.abs(np.fft.rfftn(arr))
    eps = np.finfo(amp.dtype).resolution

    rng = np.random.default_rng(seed)
    noise = np.fft.rfftn(rng.standard_normal(arr.shape, dtype=amp.dtype))
    np.divide(noise, np.maximum(np.abs(noise), eps), out=noise)

    noise = np.multiply(noise, amp, out=noise)
    ret = np.fft.irfftn(noise, s=arr.shape, axes=range(arr.ndim))
    if np.sign(ret.sum()) != np.sign(arr.sum()):
        ret *= -1
    return ret


def compute_extraction_box(
    centers: BackendArray, extraction_shape: Tuple[int], original_shape: Tuple[int]
):
    """Compute coordinates for extracting fixed-size regions around points.

    Parameters
    ----------
    centers : BackendArray
        Array of shape (n, d) containing n center coordinates in d dimensions.
    extraction_shape : tuple of int
        Desired shape of the extraction box.
    original_shape : tuple of int
        Shape of the original array from which extractions will be made.

    Returns
    -------
    obs_beg : BackendArray
        Starting coordinates for extraction, shape (n, d).
    obs_end : BackendArray
        Ending coordinates for extraction, shape (n, d).
    cand_beg : BackendArray
        Starting coordinates in output array, shape (n, d).
    cand_end : BackendArray
        Ending coordinates in output array, shape (n, d).
    keep : BackendArray
        Boolean mask of valid extraction boxes, shape (n,).
    """
    target_shape = be.to_backend_array(original_shape)
    extraction_shape = be.to_backend_array(extraction_shape)

    left_pad = be.astype(be.divide(extraction_shape, 2), int)
    right_pad = be.astype(be.add(left_pad, be.mod(extraction_shape, 2)), int)

    obs_beg = be.subtract(centers, left_pad)
    obs_end = be.add(centers, right_pad)

    obs_beg_clamp = be.maximum(obs_beg, 0)
    obs_end_clamp = be.minimum(obs_end, target_shape)

    clamp_change = be.sum(
        be.add(obs_beg != obs_beg_clamp, obs_end != obs_end_clamp), axis=1
    )

    cand_beg = left_pad - be.subtract(centers, obs_beg_clamp)
    cand_end = left_pad + be.subtract(obs_end_clamp, centers)

    stops = be.subtract(cand_end, extraction_shape)
    keep = be.sum(be.multiply(cand_beg == 0, stops == 0), axis=1) == centers.shape[1]
    keep = be.multiply(keep, clamp_change == 0)

    return obs_beg_clamp, obs_end_clamp, cand_beg, cand_end, keep


def create_mask(
    mask_type: str,
    soft_edge_width: float = 0,
    sigma_decay: float = None,
    method: str = "gaussian",
    **kwargs,
) -> NDArray:
    """
    Creates a mask of the specified type.

    Parameters
    ----------
    mask_type : str
        Type of the mask to be created. Can be one of:

            +-----------+---------------------------------------------------------+
            | box       | Box mask (see :py:meth:`box_mask`)                      |
            +-----------+---------------------------------------------------------+
            | tube      | Cylindrical mask (see :py:meth:`tube_mask`)             |
            +-----------+---------------------------------------------------------+
            | membrane  | Membrane mask (see :py:meth:`membrane_mask`)            |
            +-----------+---------------------------------------------------------+
            | ellipse   | Ellipsoidal mask (see :py:meth:`elliptical_mask`)       |
            +-----------+---------------------------------------------------------+
            | threshold | Density-based mask (see :py:meth:`threshold_mask`)      |
            +-----------+---------------------------------------------------------+
    soft_edge_width : float, optional
        Soft-edge width in voxels, 0 by default (hard edge).
    sigma_decay : float, optional
        Deprecated alias for *soft_edge_width*. If both are given,
        *soft_edge_width* takes precedence.
    method : str, optional
        Soft-edge method: ``"gaussian"`` (default) or ``"cosine"``.
    kwargs : dict
        Parameters passed to the individual mask creation functions.

    Returns
    -------
    NDArray
        The created mask.

    Raises
    ------
    ValueError
        If the mask_type is invalid.
    """
    import warnings
    from .mask import (
        elliptical_mask,
        box_mask,
        tube_mask,
        membrane_mask,
        threshold_mask,
    )

    if sigma_decay is not None:
        warnings.warn(
            "sigma_decay is deprecated, use soft_edge_width instead.",
            FutureWarning,
            stacklevel=2,
        )
        if soft_edge_width == 0:
            soft_edge_width = sigma_decay

    mapping = {
        "ellipse": elliptical_mask,
        "box": box_mask,
        "tube": tube_mask,
        "membrane": membrane_mask,
        "threshold": threshold_mask,
    }
    if mask_type not in mapping:
        raise ValueError(f"mask_type has to be one of {','.join(mapping.keys())}")

    return mapping[mask_type](
        **kwargs,
        soft_edge_width=soft_edge_width,
        method=method,
    )


def setup_filter(
    matching_data: MatchingData,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    pad_template_filter: bool = False,
    apply_target_filter: bool = False,
    **kwargs,
):
    from .filters import Compose

    backend_arr = type(be.zeros((1), dtype=be._float))
    template_filter = be.full(shape=(1,), fill_value=1, dtype=be._float)
    target_filter = be.full(shape=(1,), fill_value=1, dtype=be._float)
    if isinstance(matching_data.template_filter, backend_arr):
        template_filter = matching_data.template_filter

    if isinstance(matching_data.target_filter, backend_arr):
        target_filter = matching_data.target_filter

    filter_template = isinstance(matching_data.template_filter, Compose)
    filter_target = isinstance(matching_data.target_filter, Compose)

    # For now assume user-supplied template_filter is correctly padded
    if filter_target is None and target_filter is None:
        return template_filter

    # Extract spatial dimensions from 2-batch-dim fast_shape
    _, axes, _ = matching_data._batch_shape(fast_shape)
    real_shape = tuple(fast_shape[i] for i in axes)
    cmpl_shape = list(fast_ft_shape[i] for i in axes)

    real_tmpl_shape, cmpl_tmpl_shape = real_shape, cmpl_shape
    if not pad_template_filter:
        shape = matching_data._output_template_shape
        b = 1 if matching_data._has_batch else 0

        real_tmpl_shape = shape[b:]
        cmpl_tmpl_shape = list(real_tmpl_shape)
        cmpl_tmpl_shape[-1] = cmpl_tmpl_shape[-1] // 2 + 1

    # Broadcast over respective batch dimensions
    if matching_data._has_batch:
        cmpl_shape = [-1] + cmpl_shape
        cmpl_tmpl_shape = [-1] + cmpl_tmpl_shape

    target = matching_data.target
    tb = target.ndim - len(real_shape)
    target_axes = tuple(range(tb, target.ndim))

    filter_kwargs = kwargs | {
        "axes": target_axes,
        "return_real_fourier": True,
        "shape_is_real_fourier": False,
    }
    if filter_template:
        template_filter = matching_data.template_filter(
            shape=real_tmpl_shape, **filter_kwargs
        )["data"]
        template_filter = be.reshape(template_filter, cmpl_tmpl_shape)
        template_filter = be.to_backend_array(template_filter, be._float)
        template_filter = be.at(template_filter, ((0,) * template_filter.ndim), 0)

    if filter_target:
        target_filter = matching_data.target_filter(
            shape=real_shape, weight_type=None, **filter_kwargs
        )["data"]
        target_filter = be.reshape(target_filter, cmpl_shape)
        target_filter = be.to_backend_array(target_filter, be._float)
        target_filter = be.at(target_filter, ((0,) * target_filter.ndim), 0)

    if apply_target_filter and filter_target:
        # Applying the target filter is the only step that needs the target FFT.
        pad_shape = target.shape[:tb] + real_shape
        target_temp = be.topleft_pad(target, pad_shape)
        target_temp_ft = be.rfftn(
            be.astype(target_temp, be._float), s=real_shape, axes=target_axes
        )
        target_temp_ft = be.multiply(target_temp_ft, target_filter, out=target_temp_ft)
        target_temp = be.irfftn(
            target_temp_ft, s=target_temp.shape[tb:], axes=target_axes
        )
        matching_data._target = be.topleft_pad(target_temp, matching_data.target.shape)

    return template_filter, target_filter


def minimum_score_from_fp(std: float, n_correlations: int, n_fp: float) -> float:
    """Rickgauer et al. 2017 false-positive threshold."""
    from scipy.special import erfcinv

    return float(erfcinv(2 * n_fp / n_correlations) * np.sqrt(2) * std)


def write_pickle(data: object, filename: str) -> None:
    from .utils.serialization import write_pickle as _write_pickle

    warnings.warn(
        "Using write_pickle is deprecated and will raise an error "
        "in v0.3.5. Please use tme.utils.serialization.serialize instead.",
        DeprecationWarning,
    )
    return _write_pickle(data, filename)


def load_pickle(filename: str) -> object:
    from .utils.serialization import load_pickle as _load_pickle

    warnings.warn(
        "Using load_pickle is deprecated and will raise an error "
        "in v0.3.5. Please use tme.uils.serialization.deserialize instead.",
        DeprecationWarning,
    )
    return _load_pickle(filename)


def compute_parallelization_schedule(*args, **kwargs) -> Tuple[Dict, Tuple[int, int]]:
    from .memory import compute_schedule as _compute_schedule

    warnings.warn(
        "Using compute_parallelization_schedule is deprecated and will raise an error "
        "in v0.3.5. Please use tme.memory.compute_schedule instead.",
        DeprecationWarning,
    )
    return _compute_schedule(*args, **kwargs)
