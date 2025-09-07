"""
Utility functions for template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
import pickle
from shutil import move
from tempfile import mkstemp
from itertools import product
from gzip import open as gzip_open
from typing import Tuple, Dict, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .backends import backend as be
from .memory import estimate_memory_usage
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


def generate_tempfile_name(suffix: str = None) -> str:
    """
    Returns the path to a temporary file with given suffix. If defined. the
    environment variable TMPDIR is used as base.

    Parameters
    ----------
    suffix : str, optional
        File suffix. By default the file has no suffix.

    Returns
    -------
    str
        The generated filename
    """
    return mkstemp(suffix=suffix)[1]


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
        arr = np.array(arr)
        os.remove(memmap_filepath)
    return arr


def is_gzipped(filename: str) -> bool:
    """Check if a file is a gzip file by reading its magic number."""
    with open(filename, "rb") as f:
        return f.read(2) == b"\x1f\x8b"


def write_pickle(data: object, filename: str) -> None:
    """
    Serialize and write data to a file invalidating the input data.

    Parameters
    ----------
    data : iterable or object
        The data to be serialized.
    filename : str
        The name of the file where the serialized data will be written.

    See Also
    --------
    :py:meth:`load_pickle`
    """
    if type(data) not in (list, tuple):
        data = (data,)

    dirname = os.path.dirname(filename)
    with open(filename, "wb") as ofile, ThreadPoolExecutor() as executor:
        for i in range(len(data)):
            futures = []
            item = data[i]
            if isinstance(item, np.memmap):
                _, new_filename = mkstemp(suffix=".mm", dir=dirname)
                new_item = ("np.memmap", item.shape, item.dtype, new_filename)
                futures.append(executor.submit(move, item.filename, new_filename))
                item = new_item
            pickle.dump(item, ofile)
        for future in futures:
            future.result()


def load_pickle(filename: str) -> object:
    """
    Load and deserialize data written by :py:meth:`write_pickle`.

    Parameters
    ----------
    filename : str
        The name of the file to read and deserialize data from.

    Returns
    -------
    object or iterable
        The deserialized data.

    See Also
    --------
    :py:meth:`write_pickle`
    """

    def _load_pickle(file_handle):
        try:
            while True:
                yield pickle.load(file_handle)
        except EOFError:
            pass

    def _is_pickle_memmap(data):
        ret = False
        if isinstance(data[0], str):
            if data[0] == "np.memmap":
                ret = True
        return ret

    items = []
    func = open
    if is_gzipped(filename):
        func = gzip_open

    with func(filename, "rb") as ifile:
        for data in _load_pickle(ifile):
            if isinstance(data, tuple):
                if _is_pickle_memmap(data):
                    _, shape, dtype, filename = data
                    data = np.memmap(filename, shape=shape, dtype=dtype)
            items.append(data)
    return items[0] if len(items) == 1 else items


def compute_parallelization_schedule(
    shape1: NDArray,
    shape2: NDArray,
    max_cores: int,
    max_ram: int,
    matching_method: str,
    split_axes: Tuple[int] = None,
    backend: str = None,
    split_only_outer: bool = False,
    shape1_padding: NDArray = None,
    analyzer_method: str = None,
    max_splits: int = 256,
    float_nbytes: int = 4,
    complex_nbytes: int = 8,
    integer_nbytes: int = 4,
) -> Tuple[Dict, int, int]:
    """
    Computes a parallelization schedule for a given computation.

    This function estimates the amount of memory that would be used by a computation
    and breaks down the computation into smaller parts that can be executed in parallel
    without exceeding the specified limits on the number of cores and memory.

    Parameters
    ----------
    shape1 : NDArray
        The shape of the first input array.
    shape1_padding : NDArray, optional
        Padding for shape1, None by default.
    shape2 : NDArray
        The shape of the second input array.
    max_cores : int
        The maximum number of cores that can be used.
    max_ram : int
        The maximum amount of memory that can be used.
    matching_method : str
        The metric used for scoring the computations.
    split_axes : tuple
        Axes that can be used for splitting. By default all are considered.
    backend : str, optional
        Backend used for computations.
    split_only_outer : bool, optional
        Whether only outer splits sould be considered.
    analyzer_method : str
        The method used for score analysis.
    max_splits : int, optional
        The maximum number of parts that the computation can be split into,
        by default 256.
    float_nbytes : int
        Number of bytes of the used float, e.g. 4 for float32.
    complex_nbytes : int
        Number of bytes of the used complex, e.g. 8 for complex64.
    integer_nbytes : int
        Number of bytes of the used integer, e.g. 4 for int32.

    Notes
    -----
        This function assumes that no residual memory remains after each split,
        which not always holds true, e.g. when using
        :py:class:`tme.analyzer.MaxScoreOverRotations`.

    Returns
    -------
    dict
        The optimal splits for each axis of the first input tensor.
    int
        The number of outer jobs.
    int
        The number of inner jobs per outer job.
    """
    shape1 = tuple(int(x) for x in shape1)
    shape2 = tuple(int(x) for x in shape2)

    if shape1_padding is None:
        shape1_padding = np.zeros_like(shape1)
    core_assignments = []
    for i in range(1, int(max_cores**0.5) + 1):
        if max_cores % i == 0:
            core_assignments.append((i, max_cores // i))
            core_assignments.append((max_cores // i, i))

    if split_only_outer:
        core_assignments = [(1, max_cores)]

    possible_params, split_axis = [], np.argmax(shape1)

    split_axis_index = split_axis
    if split_axes is not None:
        split_axis, split_axis_index = split_axes[0], 0
    else:
        split_axes = tuple(i for i in range(len(shape1)))

    split_factor, n_splits = [1 for _ in range(len(shape1))], 0
    while n_splits <= max_splits:
        splits = {k: split_factor[k] for k in range(len(split_factor))}
        array_slices = split_shape(shape=shape1, splits=splits)
        array_widths = [
            tuple(x.stop - x.start for x in split) for split in array_slices
        ]
        n_splits = np.prod(list(splits.values()))

        for inner_cores, outer_cores in core_assignments:
            if outer_cores > n_splits:
                continue
            ram_usage = [
                estimate_memory_usage(
                    shape1=tuple(sum(x) for x in zip(shp, shape1_padding)),
                    shape2=shape2,
                    matching_method=matching_method,
                    analyzer_method=analyzer_method,
                    backend=backend,
                    ncores=inner_cores,
                    float_nbytes=float_nbytes,
                    complex_nbytes=complex_nbytes,
                    integer_nbytes=integer_nbytes,
                )
                for shp in array_widths
            ]
            max_usage = 0
            for i in range(0, len(ram_usage), outer_cores):
                usage = np.sum(ram_usage[i : (i + outer_cores)])
                if usage > max_usage:
                    max_usage = usage

            inits = n_splits // outer_cores
            if max_usage < max_ram:
                possible_params.append(
                    (*split_factor, outer_cores, inner_cores, n_splits, inits)
                )
        split_factor[split_axis] += 1

        split_axis_index += 1
        if split_axis_index == len(split_axes):
            split_axis_index = 0
        split_axis = split_axes[split_axis_index]

    possible_params = np.array(possible_params)
    if not len(possible_params):
        print(
            "No suitable assignment found. Consider increasing "
            "max_ram or decrease max_cores."
        )
        return None, None

    init = possible_params.shape[1] - 1
    possible_params = possible_params[
        np.lexsort((possible_params[:, init], possible_params[:, (init - 1)]))
    ]
    splits = {k: possible_params[0, k] for k in range(len(shape1))}
    core_assignment = (
        possible_params[0, len(shape1)],
        possible_params[0, (len(shape1) + 1)],
    )

    return splits, core_assignment


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
    arr = arr[tuple(slice(0, x) for x in convolution_shape)]

    if convolution_mode not in ("full", "same", "valid"):
        raise ValueError("Supported convolution_mode are 'full', 'same' and 'valid'.")

    if convolution_mode == "full":
        subset = ...
    elif convolution_mode == "same":
        subset = center_slice(arr.shape, s1)
    elif convolution_mode == "valid":
        subset = center_slice(arr.shape, [x - y + 1 for x, y in zip(s1, s2)])
    return arr[subset]


def compute_full_convolution_index(
    outer_shape: Tuple[int],
    inner_shape: Tuple[int],
    outer_split: Tuple[slice],
    inner_split: Tuple[slice],
) -> Tuple[slice]:
    """
    Computes the position of the convolution of pieces in the full convolution.

    Parameters
    ----------
    outer_shape : tuple
        Tuple of integers corresponding to the shape of the outer array.
    inner_shape : tuple
        Tuple of integers corresponding to the shape of the inner array.
    outer_split : tuple
        Tuple of slices used to split outer array (see :py:meth:`split_shape`).
    inner_split : tuple
        Tuple of slices used to split inner array (see :py:meth:`split_shape`).

    Returns
    -------
    tuple
        Tuple of slices corresponding to the position of the given convolution
        in the full convolution.
    """
    outer_shape = np.asarray(outer_shape)
    inner_shape = np.asarray(inner_shape)

    outer_width = np.array([outer.stop - outer.start for outer in outer_split])
    inner_width = np.array([inner.stop - inner.start for inner in inner_split])
    convolution_shape = outer_width + inner_width - 1

    end_inner = np.array([inner.stop for inner in inner_split]).astype(int)
    start_outer = np.array([outer.start for outer in outer_split]).astype(int)

    offsets = start_outer + inner_shape - end_inner

    score_slice = tuple(
        (slice(offset, offset + shape))
        for offset, shape in zip(offsets, convolution_shape)
    )

    return score_slice


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
    ret_shape = ret_shape.astype(int)

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

    splits = tuple(product(*slice_list))

    return splits


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


def scramble_phases(
    arr: NDArray, noise_proportion: float = 1.0, seed: int = 42, **kwargs
) -> NDArray:
    """
    Perform random phase scrambling of ``arr``.

    Parameters
    ----------
    arr : NDArray
        Input data.
    noise_proportion : float, optional
        Proportion of scrambled phases, 1.0 by default.
    seed : int, optional
        The seed for the random phase scrambling, 42 by default.

    Returns
    -------
    NDArray
        Phase scrambled version of ``arr``.
    """
    from .filters._utils import fftfreqn

    np.random.seed(seed)
    noise_proportion = max(min(noise_proportion, 1), 0)

    arr_fft = np.fft.fftn(arr)
    amp, ph = np.abs(arr_fft), np.angle(arr_fft)

    mask = (
        fftfreqn(
            arr_fft.shape, sampling_rate=1, compute_euclidean_norm=True, fftshift=False
        )
        <= 0.5
    )

    ph_noise = np.random.permutation(ph[mask])
    ph[mask] = ph[mask] * (1 - noise_proportion) + ph_noise * noise_proportion
    return np.real(np.fft.ifftn(amp * np.exp(1j * ph)))


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


def create_mask(mask_type: str, sigma_decay: float = 0, **kwargs) -> NDArray:
    """
    Creates a mask of the specified type.

    Parameters
    ----------
    mask_type : str
        Type of the mask to be created. Can be one of:

            +----------+---------------------------------------------------------+
            | box      | Box mask (see :py:meth:`box_mask`)                      |
            +----------+---------------------------------------------------------+
            | tube     | Cylindrical mask (see :py:meth:`tube_mask`)             |
            +----------+---------------------------------------------------------+
            | membrane | Cylindrical mask (see :py:meth:`membrane_mask`)         |
            +----------+---------------------------------------------------------+
            | ellipse  | Ellipsoidal mask (see :py:meth:`elliptical_mask`)       |
            +----------+---------------------------------------------------------+
    sigma_decay : float, optional
        Smoothing along mask edges using a Gaussian filter, 0 by default.
    kwargs : dict
        Parameters passed to the indivdual mask creation funcitons.

    Returns
    -------
    NDArray
        The created mask.

    Raises
    ------
    ValueError
        If the mask_type is invalid.
    """
    from .mask import elliptical_mask, box_mask, tube_mask, membrane_mask

    mapping = {
        "ellipse": elliptical_mask,
        "box": box_mask,
        "tube": tube_mask,
        "membrane": membrane_mask,
    }
    if mask_type not in mapping:
        raise ValueError(f"mask_type has to be one of {','.join(mapping.keys())}")

    mask = mapping[mask_type](**kwargs, sigma_decay=sigma_decay)
    return mask


def setup_filter(
    matching_data: MatchingData,
    fast_shape: Tuple[int],
    fast_ft_shape: Tuple[int],
    pad_template_filter: bool = False,
    apply_target_filter: bool = False,
):
    from .filters import Compose

    backend_arr = type(be.zeros((1), dtype=be._float_dtype))
    template_filter = be.full(shape=(1,), fill_value=1, dtype=be._float_dtype)
    target_filter = be.full(shape=(1,), fill_value=1, dtype=be._float_dtype)
    if isinstance(matching_data.template_filter, backend_arr):
        template_filter = matching_data.template_filter

    if isinstance(matching_data.target_filter, backend_arr):
        target_filter = matching_data.target_filter

    filter_template = isinstance(matching_data.template_filter, Compose)
    filter_target = isinstance(matching_data.target_filter, Compose)

    # For now assume user-supplied template_filter is correctly padded
    if filter_target is None and target_filter is None:
        return template_filter

    batch_mask = matching_data._batch_mask
    real_shape = matching_data._batch_shape(fast_shape, batch_mask, keepdims=False)
    cmpl_shape = matching_data._batch_shape(fast_ft_shape, batch_mask, keepdims=True)

    real_tmpl_shape, cmpl_tmpl_shape = real_shape, cmpl_shape
    if not pad_template_filter:
        shape = matching_data._output_template_shape

        real_tmpl_shape = matching_data._batch_shape(shape, batch_mask, keepdims=False)
        cmpl_tmpl_shape = matching_data._batch_shape(shape, batch_mask, keepdims=True)
        cmpl_tmpl_shape = list(cmpl_tmpl_shape)
        cmpl_tmpl_shape[-1] = cmpl_tmpl_shape[-1] // 2 + 1

    cmpl_shape = tuple(
        -1 if y else x for x, y in zip(cmpl_shape, matching_data._target_batch)
    )
    cmpl_tmpl_shape = list(
        -1 if y else x for x, y in zip(cmpl_tmpl_shape, matching_data._template_batch)
    )

    # We can have one flexible dimension and this makes projection matching easier
    if not any(matching_data._template_batch):
        cmpl_tmpl_shape[0] = -1

    # Avoid invalidating the meaning of some filters on padded batch dimensions
    target_shape = np.maximum(
        np.multiply(fast_shape, tuple(1 - x for x in matching_data._target_batch)),
        matching_data.target.shape,
    )
    target_shape = tuple(int(x) for x in target_shape)
    target_temp = be.topleft_pad(matching_data.target, target_shape)
    shape = matching_data._batch_shape(
        target_temp.shape, matching_data._target_batch, keepdims=False
    )
    axes = matching_data._batch_axis(matching_data._target_batch)
    target_temp_ft = be.rfftn(target_temp, s=shape, axes=axes)

    # Setup composable filters
    filter_kwargs = {
        "return_real_fourier": True,
        "shape_is_real_fourier": False,
        "data_rfft": target_temp_ft,
        "axes": matching_data._target_dim,
    }
    if filter_template:
        template_filter = matching_data.template_filter(
            shape=real_tmpl_shape, **filter_kwargs
        )["data"]
        template_filter = be.reshape(template_filter, cmpl_tmpl_shape)
        template_filter = be.astype(
            be.to_backend_array(template_filter), be._float_dtype
        )
        template_filter = be.at(template_filter, ((0,) * template_filter.ndim), 0)

    if filter_target:
        target_filter = matching_data.target_filter(
            shape=real_shape, weight_type=None, **filter_kwargs
        )["data"]
        target_filter = be.reshape(target_filter, cmpl_shape)
        target_filter = be.astype(be.to_backend_array(target_filter), be._float_dtype)
        target_filter = be.at(target_filter, ((0,) * target_filter.ndim), 0)

    if apply_target_filter and filter_target:
        target_temp_ft = be.multiply(target_temp_ft, target_filter, out=target_temp_ft)
        target_temp = be.irfftn(target_temp_ft, s=shape, axes=axes)
        matching_data._target = be.topleft_pad(target_temp, matching_data.target.shape)

    return template_filter, target_filter
