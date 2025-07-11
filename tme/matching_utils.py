"""
Utility functions for template matching.

Copyright (c) 2023 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import os
import pickle
from shutil import move
from joblib import Parallel
from tempfile import mkstemp
from itertools import product
from concurrent.futures import ThreadPoolExecutor
from typing import Tuple, Dict, Callable, Optional

import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter

from .backends import backend as be
from .memory import estimate_memory_usage
from .types import NDArray, BackendArray


def noop(*args, **kwargs):
    pass


def identity(arr, *args):
    return arr


def conditional_execute(
    func: Callable,
    execute_operation: bool,
    alt_func: Callable = noop,
) -> Callable:
    """
    Return the given function or a no-op function based on execute_operation.

    Parameters
    ----------
    func : Callable
        Callable.
    alt_func : Callable
        Callable to return if ``execute_operation`` is False, no-op by default.
    execute_operation : bool
        Whether to return ``func`` or a ``alt_func`` function.

    Returns
    -------
    Callable
        ``func`` if ``execute_operation`` else ``alt_func``.
    """

    return func if execute_operation else alt_func


def normalize_template(
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


def _normalize_template_overflow_safe(
    template: BackendArray, mask: BackendArray, n_observations: float, axis=None
) -> BackendArray:
    _template = be.astype(template, be._overflow_safe_dtype)
    _mask = be.astype(mask, be._overflow_safe_dtype)
    normalize_template(
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
    with open(filename, "rb") as ifile:
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


def _center_slice(current_shape: Tuple[int], new_shape: Tuple[int]) -> Tuple[slice]:
    """Extract the center slice of ``current_shape`` to retrieve ``new_shape``."""
    new_shape = tuple(int(x) for x in new_shape)
    current_shape = tuple(int(x) for x in current_shape)
    starts = tuple((x - y) // 2 for x, y in zip(current_shape, new_shape))
    stops = tuple(sum(stop) for stop in zip(starts, new_shape))
    box = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    return box


def centered(arr: BackendArray, new_shape: Tuple[int]) -> BackendArray:
    """
    Extract the centered portion of an array based on a new shape.

    Parameters
    ----------
    arr : BackendArray
        Input data.
    new_shape : tuple of ints
        Desired shape for the central portion.

    Returns
    -------
    BackendArray
        Central portion of the array with shape ``new_shape``.

    References
    ----------
    .. [1] https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_signaltools.py#L388
    """
    box = _center_slice(arr.shape, new_shape=new_shape)
    return arr[box]


def centered_mask(arr: BackendArray, new_shape: Tuple[int]) -> BackendArray:
    """
    Mask the centered portion of an array based on a new shape.

    Parameters
    ----------
    arr : BackendArray
        Input data.
    new_shape : tuple of ints
        Desired shape for the mask.

    Returns
    -------
    BackendArray
        Array with central portion unmasked and the rest set to 0.
    """
    box = _center_slice(arr.shape, new_shape=new_shape)
    mask = np.zeros_like(arr)
    mask[box] = 1
    arr *= mask
    return arr


def apply_convolution_mode(
    arr: BackendArray,
    convolution_mode: str,
    s1: Tuple[int],
    s2: Tuple[int],
    convolution_shape: Tuple[int] = None,
    mask_output: bool = False,
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
    mask_output : bool, optional
        Whether to mask values outside of convolution_mode rather than
        removing them. Defaults to False.

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

    func = centered_mask if mask_output else centered
    if convolution_mode == "full":
        return arr
    elif convolution_mode == "same":
        return func(arr, s1)
    elif convolution_mode == "valid":
        valid_shape = [s1[i] - s2[i] + 1 for i in range(arr.ndim)]
        return func(arr, valid_shape)


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


def rigid_transform(
    coordinates: NDArray,
    rotation_matrix: NDArray,
    out: NDArray,
    translation: NDArray,
    use_geometric_center: bool = False,
    coordinates_mask: NDArray = None,
    out_mask: NDArray = None,
    center: NDArray = None,
) -> None:
    """
    Apply a rigid transformation (rotation and translation) to given coordinates.

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
    use_geometric_center : bool, optional
        Whether to use geometric or coordinate center.
    """
    coordinate_dtype = coordinates.dtype
    center = coordinates.mean(axis=1) if center is None else center
    if not use_geometric_center:
        coordinates = coordinates - center[:, None]

    np.matmul(rotation_matrix, coordinates, out=out)
    if use_geometric_center:
        axis_max, axis_min = out.max(axis=1), out.min(axis=1)
        axis_difference = axis_max - axis_min
        translation = np.add(translation, center - axis_max + (axis_difference // 2))
    else:
        translation = np.add(translation, np.subtract(center, out.mean(axis=1)))

    out += translation[:, None]
    if coordinates_mask is not None and out_mask is not None:
        if not use_geometric_center:
            coordinates_mask = coordinates_mask - center[:, None]
        np.matmul(rotation_matrix, coordinates_mask, out=out_mask)
        out_mask += translation[:, None]

    if not use_geometric_center and coordinate_dtype != out.dtype:
        np.subtract(out.mean(axis=1), out.astype(int).mean(axis=1), out=translation)
        out += translation[:, None]


def minimum_enclosing_box(
    coordinates: NDArray, margin: NDArray = None, use_geometric_center: bool = False
) -> Tuple[int]:
    """
    Computes the minimal enclosing box around coordinates with margin.

    Parameters
    ----------
    coordinates : NDArray
        Coordinates of shape (d,n) to compute the enclosing box of.
    margin : NDArray, optional
        Box margin, zero by default.
    use_geometric_center : bool, optional
        Whether box accommodates the geometric or coordinate center, False by default.

    Returns
    -------
    tuple of ints
        Minimum enclosing box shape.
    """
    from .extensions import max_euclidean_distance

    point_cloud = np.asarray(coordinates)
    dim = point_cloud.shape[0]
    point_cloud = point_cloud - point_cloud.min(axis=1)[:, None]

    margin = np.zeros(dim) if margin is None else margin
    margin = np.asarray(margin).astype(int)

    norm_cloud = point_cloud - point_cloud.mean(axis=1)[:, None]
    # Adding one avoids clipping during scipy.ndimage.affine_transform
    shape = np.repeat(
        np.ceil(2 * np.linalg.norm(norm_cloud, axis=0).max()) + 1, dim
    ).astype(int)
    if use_geometric_center:
        hull = ConvexHull(point_cloud.T)
        distance, _ = max_euclidean_distance(point_cloud[:, hull.vertices].T)
        distance += np.linalg.norm(np.ones(dim))
        shape = np.repeat(np.rint(distance).astype(int), dim)

    return shape


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


def elliptical_mask(
    shape: Tuple[int],
    radius: Tuple[float],
    center: Optional[Tuple[float]] = None,
    orientation: Optional[NDArray] = None,
    sigma_decay: float = 0.0,
    cutoff_sigma: float = 3,
) -> NDArray:
    """
    Creates an ellipsoidal mask.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the mask to be created.
    radius : tuple of floats
        Radius of the mask.
    center : tuple of floats, optional
        Center of the mask, default to shape // 2.
    orientation : NDArray, optional.
        Orientation of the mask as rotation matrix with shape (d,d).

    Returns
    -------
    NDArray
        The created ellipsoidal mask.

    Raises
    ------
    ValueError
        If the length of center and radius is not one or the same as shape.

    Examples
    --------
    >>> from tme.matching_utils import elliptical_mask
    >>> mask = elliptical_mask(shape=(20,20), radius=(5,5), center=(10,10))
    """
    shape, radius = np.asarray(shape), np.asarray(radius)

    shape = shape.astype(int)
    if center is None:
        center = np.divide(shape, 2).astype(int)

    center = np.asarray(center, dtype=np.float32)
    radius = np.repeat(radius, shape.size // radius.size)
    center = np.repeat(center, shape.size // center.size)
    if radius.size != shape.size:
        raise ValueError("Length of radius has to be either one or match shape.")
    if center.size != shape.size:
        raise ValueError("Length of center has to be either one or match shape.")

    n = shape.size
    center = center.reshape((-1,) + (1,) * n)
    radius = radius.reshape((-1,) + (1,) * n)

    indices = np.indices(shape, dtype=np.float32) - center
    if orientation is not None:
        return_shape = indices.shape
        indices = indices.reshape(n, -1)
        rigid_transform(
            coordinates=indices,
            rotation_matrix=np.asarray(orientation),
            out=indices,
            translation=np.zeros(n),
            use_geometric_center=False,
        )
        indices = indices.reshape(*return_shape)

    dist = np.linalg.norm(indices / radius, axis=0)
    if sigma_decay > 0:
        sigma_decay = 2 * (sigma_decay / np.mean(radius)) ** 2
        mask = np.maximum(0, dist - 1)
        mask = np.exp(-(mask**2) / sigma_decay)
        mask *= mask > np.exp(-(cutoff_sigma**2) / 2)
    else:
        mask = (dist <= 1).astype(int)
    return mask


def tube_mask2(
    shape: Tuple[int],
    inner_radius: float,
    outer_radius: float,
    height: int,
    symmetry_axis: Optional[int] = 2,
    center: Optional[Tuple[float]] = None,
    orientation: Optional[NDArray] = None,
    epsilon: float = 0.5,
) -> NDArray:
    """
    Creates a tube mask.

    Parameters
    ----------
    shape : tuple
        Shape of the mask to be created.
    inner_radius : float
        Inner radius of the tube.
    outer_radius : float
        Outer radius of the tube.
    height : int
        Height of the tube.
    symmetry_axis : int, optional
        The axis of symmetry for the tube, defaults to 2.
    center : tuple of float, optional.
        Center of the mask, defaults to shape // 2.
    orientation : NDArray, optional.
        Orientation of the mask as rotation matrix with shape (d,d).
    epsilon : float, optional
        Tolerance to handle discretization errors, defaults to 0.5.

    Returns
    -------
    NDArray
        The created tube mask.

    Raises
    ------
    ValueError
        If ``inner_radius`` is larger than ``outer_radius``.
        If ``center`` and ``shape`` do not have the same length.
    """
    shape = np.asarray(shape, dtype=int)

    if center is None:
        center = np.divide(shape, 2).astype(int)

    center = np.asarray(center, dtype=np.float32)
    center = np.repeat(center, shape.size // center.size)
    if inner_radius > outer_radius:
        raise ValueError("inner_radius should be smaller than outer_radius.")
    if symmetry_axis > len(shape):
        raise ValueError(f"symmetry_axis can be not larger than {len(shape)}.")
    if center.size != shape.size:
        raise ValueError("Length of center has to be either one or match shape.")

    n = shape.size
    center = center.reshape((-1,) + (1,) * n)
    indices = np.indices(shape, dtype=np.float32) - center
    if orientation is not None:
        return_shape = indices.shape
        indices = indices.reshape(n, -1)
        rigid_transform(
            coordinates=indices,
            rotation_matrix=np.asarray(orientation),
            out=indices,
            translation=np.zeros(n),
            use_geometric_center=False,
        )
        indices = indices.reshape(*return_shape)

    mask = np.zeros(shape, dtype=bool)
    sq_dist = np.zeros(shape)
    for i in range(len(shape)):
        if i == symmetry_axis:
            continue
        sq_dist += indices[i] ** 2

    sym_coord = indices[symmetry_axis]
    half_height = height / 2
    height_mask = np.abs(sym_coord) <= half_height

    inner_mask = 1
    if inner_radius > epsilon:
        inner_mask = sq_dist >= ((inner_radius) ** 2 - epsilon)

    height_mask = np.abs(sym_coord) <= (half_height + epsilon)
    outer_mask = sq_dist <= ((outer_radius) ** 2 + epsilon)

    mask = height_mask & inner_mask & outer_mask
    return mask


def box_mask(
    shape: Tuple[int],
    center: Tuple[int],
    height: Tuple[int],
    sigma_decay: float = 0.0,
    cutoff_sigma: float = 0.0,
) -> np.ndarray:
    """
    Creates a box mask centered around the provided center point.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the output array.
    center : tuple of ints
        Center point coordinates of the box.
    height : tuple of ints
        Height (side length) of the box along each axis.

    Returns
    -------
    NDArray
        The created box mask.

    Raises
    ------
    ValueError
        If ``shape`` and ``center`` do not have the same length.
        If ``center`` and ``height`` do not have the same length.
    """
    if len(shape) != len(center) or len(center) != len(height):
        raise ValueError("The length of shape, center, and height must be consistent.")

    shape = tuple(int(x) for x in shape)
    center, height = np.array(center, dtype=int), np.array(height, dtype=int)

    half_heights = height // 2
    starts = np.maximum(center - half_heights, 0)
    stops = np.minimum(center + half_heights + np.mod(height, 2) + 1, shape)
    slice_indices = tuple(slice(*coord) for coord in zip(starts, stops))

    out = np.zeros(shape)
    out[slice_indices] = 1

    if sigma_decay > 0:
        mask_filter = gaussian_filter(out.astype(np.float32), sigma=sigma_decay)
        out = np.add(out, (1 - out) * mask_filter)
        out *= out > np.exp(-(cutoff_sigma**2) / 2)
    return out


def tube_mask(
    shape: Tuple[int],
    symmetry_axis: int,
    base_center: Tuple[int],
    inner_radius: float,
    outer_radius: float,
    height: int,
    sigma_decay: float = 0.0,
    **kwargs,
) -> NDArray:
    """
    Creates a tube mask.

    Parameters
    ----------
    shape : tuple
        Shape of the mask to be created.
    symmetry_axis : int
        The axis of symmetry for the tube.
    base_center : tuple
        Center of the tube.
    inner_radius : float
        Inner radius of the tube.
    outer_radius : float
        Outer radius of the tube.
    height : int
        Height of the tube.

    Returns
    -------
    NDArray
        The created tube mask.

    Raises
    ------
    ValueError
        If ``inner_radius`` is larger than ``outer_radius``.
        If ``height`` is larger than the symmetry axis.
        If ``base_center`` and ``shape`` do not have the same length.
    """
    if inner_radius > outer_radius:
        raise ValueError("inner_radius should be smaller than outer_radius.")

    if height > shape[symmetry_axis]:
        raise ValueError(f"Height can be no larger than {shape[symmetry_axis]}.")

    if symmetry_axis > len(shape):
        raise ValueError(f"symmetry_axis can be not larger than {len(shape)}.")

    if len(base_center) != len(shape):
        raise ValueError("shape and base_center need to have the same length.")

    shape = tuple(int(x) for x in shape)
    circle_shape = tuple(b for ix, b in enumerate(shape) if ix != symmetry_axis)
    circle_center = tuple(b for ix, b in enumerate(base_center) if ix != symmetry_axis)

    inner_circle = np.zeros(circle_shape)
    outer_circle = np.zeros_like(inner_circle)
    if inner_radius > 0:
        inner_circle = create_mask(
            mask_type="ellipse",
            shape=circle_shape,
            radius=inner_radius,
            center=circle_center,
            sigma_decay=sigma_decay,
        )
    if outer_radius > 0:
        outer_circle = create_mask(
            mask_type="ellipse",
            shape=circle_shape,
            radius=outer_radius,
            center=circle_center,
            sigma_decay=sigma_decay,
        )
    circle = outer_circle - inner_circle
    circle = np.expand_dims(circle, axis=symmetry_axis)

    center = base_center[symmetry_axis]
    start_idx = int(center - height // 2)
    stop_idx = int(center + height // 2 + height % 2)

    start_idx, stop_idx = max(start_idx, 0), min(stop_idx, shape[symmetry_axis])

    slice_indices = tuple(
        slice(None) if i != symmetry_axis else slice(start_idx, stop_idx)
        for i in range(len(shape))
    )
    tube = np.zeros(shape)
    tube[slice_indices] = circle

    return tube


def membrane_mask(
    shape: Tuple[int],
    radius: float,
    thickness: float,
    separation: float,
    symmetry_axis: int = 2,
    center: Optional[Tuple[float]] = None,
    sigma_decay: float = 0.5,
    cutoff_sigma: float = 3,
    **kwargs,
) -> NDArray:
    """
    Creates a membrane mask consisting of two parallel disks with Gaussian intensity profile.
    Uses efficient broadcasting approach: flat disk mask × height profile.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the mask to be created.
    radius : float
        Radius of the membrane disks.
    thickness : float
        Thickness of each disk in the membrane.
    separation : float
        Distance between the centers of the two disks.
    symmetry_axis : int, optional
        The axis perpendicular to the membrane disks, defaults to 2.
    center : tuple of floats, optional
        Center of the membrane (midpoint between the two disks), defaults to shape // 2.
    sigma_decay : float, optional
        Controls edge sharpness relative to radius, defaults to 0.5.
    cutoff_sigma : float, optional
        Cutoff for height profile in standard deviations, defaults to 3.

    Returns
    -------
    NDArray
        The created membrane mask with Gaussian intensity profile.

    Raises
    ------
    ValueError
        If ``thickness`` is negative.
        If ``separation`` is negative.
        If ``center`` and ``shape`` do not have the same length.
        If ``symmetry_axis`` is out of bounds.

    Examples
    --------
    >>> from tme.matching_utils import membrane_mask
    >>> mask = membrane_mask(shape=(50,50,50), radius=10, thickness=2, separation=15)
    """
    shape = np.asarray(shape, dtype=int)

    if center is None:
        center = np.divide(shape, 2).astype(float)

    center = np.asarray(center, dtype=np.float32)
    center = np.repeat(center, shape.size // center.size)

    if thickness < 0:
        raise ValueError("thickness must be non-negative.")
    if separation < 0:
        raise ValueError("separation must be non-negative.")
    if symmetry_axis >= len(shape):
        raise ValueError(f"symmetry_axis must be less than {len(shape)}.")
    if center.size != shape.size:
        raise ValueError("Length of center has to be either one or match shape.")

    disk_mask = elliptical_mask(
        shape=[x for i, x in enumerate(shape) if i != symmetry_axis],
        radius=radius,
        sigma_decay=sigma_decay,
        cutoff_sigma=cutoff_sigma,
    )

    axial_coord = np.arange(shape[symmetry_axis]) - center[symmetry_axis]
    height_profile = np.zeros((shape[symmetry_axis],), dtype=np.float32)
    for leaflet_pos in [-separation / 2, separation / 2]:
        leaflet_profile = np.exp(
            -((axial_coord - leaflet_pos) ** 2) / (2 * (thickness / 3) ** 2)
        )
        cutoff_threshold = np.exp(-(cutoff_sigma**2) / 2)
        leaflet_profile *= leaflet_profile > cutoff_threshold

        height_profile = np.maximum(height_profile, leaflet_profile)

    disk_mask = disk_mask.reshape(
        [x if i != symmetry_axis else 1 for i, x in enumerate(shape)]
    )
    height_profile = height_profile.reshape(
        [1 if i != symmetry_axis else x for i, x in enumerate(shape)]
    )

    return disk_mask * height_profile


def scramble_phases(
    arr: NDArray,
    noise_proportion: float = 1.0,
    seed: int = 42,
    normalize_power: bool = False,
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
    normalize_power : bool, optional
        Return value has same sum of squares as ``arr``.

    Returns
    -------
    NDArray
        Phase scrambled version of ``arr``.
    """
    from tme.filters._utils import fftfreqn

    np.random.seed(seed)
    noise_proportion = max(min(noise_proportion, 1), 0)

    arr_fft = np.fft.fftn(arr)
    amp, ph = np.abs(arr_fft), np.angle(arr_fft)

    # Scrambling up to nyquist gives more uniform noise distribution
    mask = np.fft.ifftshift(
        fftfreqn(arr_fft.shape, sampling_rate=1, compute_euclidean_norm=True) <= 0.5
    )

    ph_noise = np.random.permutation(ph[mask])
    ph[mask] = ph[mask] * (1 - noise_proportion) + ph_noise * noise_proportion
    ret = np.real(np.fft.ifftn(amp * np.exp(1j * ph)))

    if normalize_power:
        np.divide(ret - ret.min(), ret.max() - ret.min(), out=ret)
        np.multiply(ret, np.subtract(arr.max(), arr.min()), out=ret)
        np.add(ret, arr.min(), out=ret)
        scaling = np.divide(np.abs(arr).sum(), np.abs(ret).sum())
        np.multiply(ret, scaling, out=ret)

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


class TqdmParallel(Parallel):
    """
    A minimal Parallel implementation using tqdm for progress reporting.

    Parameters:
    -----------
    tqdm_args : dict, optional
        Dictionary of arguments passed to tqdm.tqdm
    *args, **kwargs:
        Arguments to pass to joblib.Parallel
    """

    def __init__(self, tqdm_args: Dict = {}, *args, **kwargs):
        from tqdm import tqdm

        super().__init__(*args, **kwargs)
        self.pbar = tqdm(**tqdm_args)

    def __call__(self, iterable, *args, **kwargs):
        self.n_tasks = len(iterable) if hasattr(iterable, "__len__") else None
        return super().__call__(iterable, *args, **kwargs)

    def print_progress(self):
        if self.n_tasks is None:
            return super().print_progress()

        if self.n_tasks != self.pbar.total:
            self.pbar.total = self.n_tasks
            self.pbar.refresh()

        self.pbar.n = self.n_completed_tasks
        self.pbar.refresh()

        if self.n_completed_tasks >= self.n_tasks:
            self.pbar.close()
