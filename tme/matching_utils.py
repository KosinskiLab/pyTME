""" Utility functions for template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import traceback
import pickle
from shutil import move
from tempfile import mkstemp
from itertools import product
from typing import Tuple, Dict, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from .extensions import max_euclidean_distance
from .matching_memory import estimate_ram_usage
from .helpers import quaternion_to_rotation_matrix, load_quaternions_by_angle


def handle_traceback(last_type, last_value, last_traceback):
    """
    Handle sys.exc_info().

    Parameters
    ----------
    last_type : type
        The type of the last exception.
    last_value :
        The value of the last exception.
    last_traceback : traceback
        The traceback object encapsulating the call stack at the point
        where the exception originally occurred.

    Raises
    ------
    Exception
        Re-raises the last exception.
    """
    if last_type is None:
        return None
    traceback.print_tb(last_traceback)
    raise Exception(last_value)
    # raise last_type(last_value)


def generate_tempfile_name(suffix=None):
    """
    Returns the path to a potential temporary file location. If the environment
    variable TME_TMPDIR is defined, the temporary file will be created there.
    Otherwise the default tmp directory will be used.

    Parameters
    ----------
    suffix : str, optional
        File suffix. By default the file has no suffix.

    Returns
    -------
    str
        The generated filename
    """
    tmp_dir = os.environ.get("TMPDIR", None)
    _, filename = mkstemp(suffix=suffix, dir=tmp_dir)
    return filename


def array_to_memmap(arr: NDArray, filename: str = None) -> str:
    """
    Converts a numpy array to a np.memmap.

    Parameters
    ----------
    arr : np.ndarray
        The numpy array to be converted.
    filename : str, optional
        Desired filename for the memmap. If not provided, a temporary
        file will be created.

    Notes
    -----
        If the environment variable TME_TMPDIR is defined, the temporary
        file will be created there. Otherwise the default tmp directory
        will be used.

    Returns
    -------
    str
        The filename where the memmap was written to.
    """
    if filename is None:
        filename = generate_tempfile_name()

    shape, dtype = arr.shape, arr.dtype
    arr_memmap = np.memmap(filename, mode="w+", dtype=dtype, shape=shape)

    arr_memmap[:] = arr[:]
    arr_memmap.flush()

    return filename


def memmap_to_array(arr: NDArray) -> NDArray:
    """
    Converts a np.memmap into an numpy array.

    Parameters
    ----------
    arr : np.memmap
        The numpy array to be converted.

    Returns
    -------
    np.ndarray
        The converted array.
    """
    if type(arr) == np.memmap:
        memmap_filepath = arr.filename
        arr = np.array(arr)
        os.remove(memmap_filepath)
    return arr


def close_memmap(arr: np.ndarray) -> None:
    """
    Remove the file associated with a numpy memmap array.

    Parameters
    ----------
    arr : np.ndarray
        The numpy array which might be a memmap.
    """
    try:
        os.remove(arr.filename)
        # arr._mmap.close()
    except Exception:
        pass


def write_pickle(data: object, filename: str) -> None:
    """
    Serialize and write data to a file invalidating the input data in
    the process. This function  uses type-specific serialization for
    certain objects, such as np.memmap, for optimized storage. Other
    objects are serialized using standard pickle.

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
        The shape of the first input tensor.
    shape1_padding : NDArray, optional
        Padding for shape1 used for each split. None by defauly
    shape2 : NDArray
        The shape of the second input tensor.
    max_cores : int
        The maximum number of cores that can be used.
    max_ram : int
        The maximum amount of memory that can be used.
    matching_method : str
        The metric used for scoring the computations.
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
    shape1, shape2 = np.array(shape1), np.array(shape2)
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
    split_factor, n_splits = [1 for _ in range(len(shape1))], 0
    while n_splits <= max_splits:
        splits = {k: split_factor[k] for k in range(len(split_factor))}
        array_slices = split_numpy_array_slices(shape=shape1, splits=splits)
        array_widths = [
            tuple(x.stop - x.start for x in split) for split in array_slices
        ]
        n_splits = np.prod(list(splits.values()))

        for inner_cores, outer_cores in core_assignments:
            if outer_cores > n_splits:
                continue
            ram_usage = [
                estimate_ram_usage(
                    shape1=np.add(shp, shape1_padding),
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
        split_axis += 1
        if split_axis == shape1.size:
            split_axis = 0

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
    splits = {k: possible_params[0, k] for k in range(shape1.size)}
    core_assignment = (
        possible_params[0, shape1.size],
        possible_params[0, (shape1.size + 1)],
    )

    return splits, core_assignment


def centered(arr: NDArray, newshape: Tuple[int]) -> NDArray:
    """
    Extract the centered portion of an array based on a new shape.

    Parameters
    ----------
    arr : NDArray
        Input array.
    newshape : tuple
        Desired shape for the central portion.

    Returns
    -------
    NDArray
        Central portion of the array with shape `newshape`.

    References
    ----------
    .. [1] https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_signaltools.py#L388
    """
    new_shape = np.asarray(newshape)
    current_shape = np.array(arr.shape)
    starts = (current_shape - new_shape) // 2
    stops = starts + newshape
    box = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    return arr[box]


def centered_mask(arr: NDArray, newshape: Tuple[int]) -> NDArray:
    """
    Mask the centered portion of an array based on a new shape.

    Parameters
    ----------
    arr : NDArray
        Input array.
    newshape : tuple
        Desired shape for the mask.

    Returns
    -------
    NDArray
        Array with central portion unmasked and the rest set to 0.
    """
    new_shape = np.asarray(newshape)
    current_shape = np.array(arr.shape)
    starts = (current_shape - new_shape) // 2
    stops = starts + newshape
    box = tuple(slice(start, stop) for start, stop in zip(starts, stops))
    mask = np.zeros_like(arr)
    mask[box] = 1
    arr *= mask
    return arr


def apply_convolution_mode(
    arr: NDArray,
    convolution_mode: str,
    s1: Tuple[int],
    s2: Tuple[int],
    mask_output: bool = False,
) -> NDArray:
    """
    Applies convolution_mode to arr.

    Parameters
    ----------
    arr : NDArray
        Numpy array containing convolution result of arrays with shape s1 and s2.
    convolution_mode : str
        Analogous to mode in ``scipy.signal.convolve``:

        +---------+----------------------------------------------------------+
        | 'full'  | returns full template matching result of the inputs.     |
        +---------+----------------------------------------------------------+
        | 'valid' | returns elements that do not rely on zero-padding..      |
        +---------+----------------------------------------------------------+
        | 'same'  | output is the same size as s1.                           |
        +---------+----------------------------------------------------------+
    s1 : tuple
        Tuple of integers corresponding to shape of convolution array 1.
    s2 : tuple
        Tuple of integers corresponding to shape of convolution array 2.
    mask_output : bool, optional
        Whether to mask values outside of convolution_mode rather than
        removing them. Defaults to False.

    Returns
    -------
    NDArray
        The numpy array after applying the convolution mode.

    References
    ----------
    .. [1] https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_signaltools.py#L519
    """
    # This removes padding to next fast fourier length
    arr = arr[tuple(slice(s1[i] + s2[i] - 1) for i in range(len(s1)))]

    if convolution_mode not in ("full", "same", "valid"):
        raise ValueError("Supported convolution_mode are 'full', 'same' and 'valid'.")

    func = centered_mask if mask_output else centered
    if convolution_mode == "full":
        return arr
    elif convolution_mode == "same":
        return func(arr, s1)
    elif convolution_mode == "valid":
        valid_shape = [s1[i] - s2[i] + s2[i] % 2 for i in range(arr.ndim)]
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
        Tuple of slices used to split outer array
        (see :py:meth:`split_numpy_array_slices`).
    inner_split : tuple
        Tuple of slices used to split inner array
        (see :py:meth:`split_numpy_array_slices`).

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


def split_numpy_array_slices(
    shape: NDArray, splits: Dict, margin: NDArray = None
) -> Tuple[slice]:
    """
    Returns a tuple of slices to subset a numpy array into pieces along multiple axes.

    Parameters
    ----------
    shape : NDArray
        Shape of the array to split.
    splits : dict
        A dictionary where the keys are the axis numbers and the values
        are the number of splits along that axis.
    margin : NDArray, optional
        Padding on the left hand side of the array.

    Returns
    -------
    tuple
        A tuple of slices, where each slice corresponds to a split along an axis.
    """
    ndim = len(shape)
    if margin is None:
        margin = np.zeros(ndim, dtype=int)
    splits = {k: max(splits.get(k, 0), 1) for k in range(ndim)}
    new_shape = np.divide(shape, [splits.get(i, 1) for i in range(ndim)]).astype(int)

    slice_list = [
        tuple(
            (slice(max((n_splits * length) - margin[axis], 0), (n_splits + 1) * length))
            if n_splits < splits.get(axis, 1) - 1
            else (slice(max((n_splits * length) - margin[axis], 0), shape[axis]))
            for n_splits in range(splits.get(axis, 1))
        )
        for length, axis in zip(new_shape, splits.keys())
    ]

    splits = tuple(product(*slice_list))

    return splits


def get_rotation_matrices(
    angular_sampling: float, dim: int = 3, use_optimized_set: bool = True
) -> NDArray:
    """
    Returns rotation matrices in format k x dim x dim, where k is determined
    by ``angular_sampling``.

    Parameters
    ----------
    angular_sampling : float
        The angle in degrees used for the generation of rotation matrices.
    dim : int, optional
        Dimension of the rotation matrices.
    use_optimized_set : bool, optional
        Whether to use pre-computed rotational sets with more optimal sampling.
        Currently only available when dim=3.

    Notes
    -----
        For the case of dim = 3 optimized rotational sets are used, otherwise
        QR-decomposition.

    Returns
    -------
    NDArray
        Array of shape (k, dim, dim) containing k rotation matrices.
    """
    if dim == 3 and use_optimized_set:
        quaternions, *_ = load_quaternions_by_angle(angular_sampling)
        ret = quaternion_to_rotation_matrix(quaternions)
    else:
        num_rotations = dim * (dim - 1) // 2
        k = int((360 / angular_sampling) ** num_rotations)
        As = np.random.randn(k, dim, dim)
        ret, _ = np.linalg.qr(As)
        dets = np.linalg.det(ret)
        neg_dets = dets < 0
        ret[neg_dets, :, -1] *= -1
    return ret


def minimum_enclosing_box(
    coordinates: NDArray,
    margin: NDArray = None,
    use_geometric_center: bool = False,
) -> Tuple[int]:
    """
    Computes the minimal enclosing box around coordinates with margin.

    Parameters
    ----------
    coordinates : NDArray
        Coordinates of which the enclosing box should be computed. The shape
        of this array should be [d, n] with d dimensions and n coordinates.
    margin : NDArray, optional
        Box margin. Defaults to None.
    use_geometric_center : bool, optional
        Whether the box should accommodate the geometric or the coordinate
        center. Defaults to False.

    Returns
    -------
    tuple
        Integers corresponding to the minimum enclosing box shape.
    """
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


def crop_input(
    target: "Density",
    template: "Density",
    target_mask: "Density" = None,
    template_mask: "Density" = None,
    map_cutoff: float = 0,
    template_cutoff: float = 0,
) -> Tuple[int]:
    """
    Crop target and template maps for efficient fitting. Input densities
    are cropped in place.

    Parameters
    ----------
    target : Density
        Target to be fitted on.
    template : Density
        Template to fit onto the target.
    target_mask : Density, optional
        Path to mask of target. Will be croppped like target.
    template_mask : Density, optional
        Path to mask of template. Will be cropped like template.
    map_cutoff : float, optional
        Cutoff value for trimming the target Density. Default is 0.
    map_cutoff : float, optional
        Cutoff value for trimming the template Density. Default is 0.

    Returns
    -------
    Tuple[int]
        Tuple containing reference fit index
    """
    convolution_shape_init = np.add(target.shape, template.shape) - 1
    # If target and template are aligned, fitting should return this index
    reference_fit = np.subtract(template.shape, 1)

    target_box = tuple(slice(0, x) for x in target.shape)
    if map_cutoff is not None:
        target_box = target.trim_box(cutoff=map_cutoff)

    target_mask_box = target_box
    if target_mask is not None and map_cutoff is not None:
        target_mask_box = target_mask.trim_box(cutoff=map_cutoff)
    target_box = tuple(
        slice(min(arr.start, mask.start), max(arr.stop, mask.stop))
        for arr, mask in zip(target_box, target_mask_box)
    )

    template_box = tuple(slice(0, x) for x in template.shape)
    if template_cutoff is not None:
        template_box = template.trim_box(cutoff=template_cutoff)

    template_mask_box = template_box
    if template_mask is not None and template_cutoff is not None:
        template_mask_box = template_mask.trim_box(cutoff=template_cutoff)
    template_box = tuple(
        slice(min(arr.start, mask.start), max(arr.stop, mask.stop))
        for arr, mask in zip(template_box, template_mask_box)
    )

    cut_right = np.array(
        [shape - x.stop for shape, x in zip(template.shape, template_box)]
    )
    cut_left = np.array([x.start for x in target_box])

    origin_difference = np.divide(target.origin - template.origin, target.sampling_rate)
    origin_difference = origin_difference.astype(int)

    target.adjust_box(target_box)
    template.adjust_box(template_box)

    if target_mask is not None:
        target_mask.adjust_box(target_box)
    if template_mask is not None:
        template_mask.adjust_box(template_box)

    reference_fit -= cut_right + cut_left + origin_difference

    convolution_shape = np.array(target.shape)
    convolution_shape += np.array(template.shape) - 1

    print(f"Cropped volume of target is: {target.shape}")
    print(f"Cropped volume of template is: {template.shape}")
    saving = 1 - (np.prod(convolution_shape)) / np.prod(convolution_shape_init)
    saving *= 100

    print(
        "Cropping changed array size from "
        f"{round(4*np.prod(convolution_shape_init)/1e6, 3)} MB "
        f"to {round(4*np.prod(convolution_shape)/1e6, 3)} MB "
        f"({'-' if saving > 0 else ''}{abs(round(saving, 2))}%)"
    )
    return reference_fit


def euler_to_rotationmatrix(angles: Tuple[float]) -> NDArray:
    """
    Convert Euler angles to a rotation matrix.

    Parameters
    ----------
    angles : tuple
        A tuple representing the Euler angles in degrees.

    Returns
    -------
    NDArray
        The generated rotation matrix.
    """
    n_angles = len(angles)
    angle_convention = "zyx"[:n_angles]
    if n_angles == 1:
        angles = (angles, 0, 0)
    rotation_matrix = (
        Rotation.from_euler(angle_convention, angles, degrees=True)
        .as_matrix()
        .astype(np.float32)
    )
    return rotation_matrix


def euler_from_rotationmatrix(rotation_matrix: NDArray) -> Tuple:
    """
    Convert a rotation matrix to euler angles.

    Parameters
    ----------
    rotation_matrix : NDArray
        A 2 x 2 or 3 x 3 rotation matrix in z y x form.

    Returns
    -------
    Tuple
        The generate euler angles in degrees
    """
    if rotation_matrix.shape[0] == 2:
        temp_matrix = np.eye(3)
        temp_matrix[:2, :2] = rotation_matrix
        rotation_matrix = temp_matrix
    euler_angles = (
        Rotation.from_matrix(rotation_matrix)
        .as_euler("zyx", degrees=True)
        .astype(np.float32)
    )
    return euler_angles


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
        An array representing the coordinates to be transformed [d x N].
    rotation_matrix : NDArray
        The rotation matrix to be applied [d x d].
    translation : NDArray
        The translation vector to be applied [d].
    out : NDArray
        The output array to store the transformed coordinates.
    coordinates_mask : NDArray, optional
        An array representing the mask for the coordinates [d x T].
    out_mask : NDArray, optional
        The output array to store the transformed coordinates mask.
    use_geometric_center : bool, optional
        Whether to use geometric or coordinate center.

    Returns
    -------
    None
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


def _format_string(string: str) -> str:
    """
    Formats a string by adding quotation marks if it contains white spaces.

    Parameters
    ----------
    string : str
        Input string to be formatted.

    Returns
    -------
    str
        Formatted string with added quotation marks if needed.
    """
    if " " in string:
        return f"'{string}'"
    # Occurs e.g. for C1' atoms. The trailing whitespace is necessary.
    if string.count("'") == 1:
        return f'"{string}"'
    return string


def _format_mmcif_colunns(subdict: Dict) -> Dict:
    """
    Formats the columns of a mmcif dictionary.

    Parameters
    ----------
    subdict : dict
        Input dictionary where each key corresponds to a column and the
        values are iterables containing the column values.

    Returns
    -------
    dict
        Formatted dictionary with the columns of the mmcif file.
    """
    subdict = {k: [_format_string(s) for s in v] for k, v in subdict.items()}
    key_length = {
        key: len(max(value, key=lambda x: len(x), default=""))
        for key, value in subdict.items()
    }
    padded_subdict = {
        key: [s.ljust(key_length[key] + 1) for s in values]
        for key, values in subdict.items()
    }
    return padded_subdict


def create_mask(
    mask_type: str, sigma_decay: float = 0, mask_cutoff: float = 0.135, **kwargs
) -> NDArray:
    """
    Creates a mask of the specified type.

    Parameters
    ----------
    mask_type : str
        Type of the mask to be created. Can be "ellipse", "box", or "tube".
    sigma_decay : float, optional
        Standard deviation of an optionally applied Gaussian filter.
    mask_cutoff : float, optional
        Values below mask_cutoff will be set to zero. By default, exp(-2).
    kwargs : dict
        Additional parameters required by the mask creating functions.

    Returns
    -------
    NDArray
        The created mask.

    Raises
    ------
    ValueError
        If the mask_type is invalid.

    See Also
    --------
    :py:meth:`elliptical_mask`
    :py:meth:`box_mask`
    :py:meth:`tube_mask`
    """
    mapping = {"ellipse": elliptical_mask, "box": box_mask, "tube": tube_mask}
    if mask_type not in mapping:
        raise ValueError(f"mask_type has to be one of {','.join(mapping.keys())}")

    mask = mapping[mask_type](**kwargs)
    if sigma_decay > 0:
        mask = gaussian_filter(mask.astype(np.float32), sigma=sigma_decay)

    mask[mask < mask_cutoff] = 0

    return mask


def elliptical_mask(
    shape: Tuple[int], radius: Tuple[float], center: Tuple[int]
) -> NDArray:
    """
    Creates an ellipsoidal mask.

    Parameters
    ----------
    shape : tuple of ints
        Shape of the mask to be created.
    radius : tuple of floats
        Radius of the ellipse.
    center : tuple of ints
        Center of the ellipse.

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
    >>> mask = elliptical_mask(shape = (20,20), radius = (5,5), center = (10,10))
    """
    center, shape, radius = np.asarray(center), np.asarray(shape), np.asarray(radius)

    radius = np.repeat(radius, shape.size // radius.size)
    center = np.repeat(center, shape.size // center.size)

    if radius.size != shape.size:
        raise ValueError("Length of radius has to be either one or match shape.")
    if center.size != shape.size:
        raise ValueError("Length of center has to be either one or match shape.")

    n = shape.size
    center = center.reshape((-1,) + (1,) * n)
    radius = radius.reshape((-1,) + (1,) * n)

    mask = np.linalg.norm((np.indices(shape) - center) / radius, axis=0)
    mask = (mask <= 1).astype(int)

    return mask


def box_mask(shape: Tuple[int], center: Tuple[int], height: Tuple[int]) -> np.ndarray:
    """
    Creates a box mask centered around the provided center point.

    Parameters
    ----------
    shape : Tuple[int]
        Shape of the output array.
    center : Tuple[int]
        Center point coordinates of the box.
    height : Tuple[int]
        Height (side length) of the box along each axis.

    Returns
    -------
    NDArray
        The created box mask.
    """
    if len(shape) != len(center) or len(center) != len(height):
        raise ValueError("The length of shape, center, and height must be consistent.")

    # Calculate min and max coordinates for the box using the center and half-heights
    center, height = np.array(center, dtype=int), np.array(height, dtype=int)

    half_heights = height // 2
    starts = np.maximum(center - half_heights, 0)
    stops = np.minimum(center + half_heights + np.mod(height, 2) + 1, shape)
    slice_indices = tuple(slice(*coord) for coord in zip(starts, stops))

    out = np.zeros(shape)
    out[slice_indices] = 1
    return out


def tube_mask(
    shape: Tuple[int],
    symmetry_axis: int,
    base_center: Tuple[int],
    inner_radius: float,
    outer_radius: float,
    height: int,
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
        Center of the base circle of the tube.
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
        If the inner radius is larger than the outer radius. Or height is larger
        than the symmetry axis shape.
    """
    if inner_radius > outer_radius:
        raise ValueError("inner_radius should be smaller than outer_radius.")

    if height > shape[symmetry_axis]:
        raise ValueError(f"Height can be no larger than {shape[symmetry_axis]}.")

    if symmetry_axis > len(shape):
        raise ValueError(f"symmetry_axis can be not larger than {len(shape)}.")

    circle_shape = tuple(b for ix, b in enumerate(shape) if ix != symmetry_axis)
    base_center = tuple(b for ix, b in enumerate(base_center) if ix != symmetry_axis)

    inner_circle = np.zeros(circle_shape)
    outer_circle = np.zeros_like(inner_circle)
    if inner_radius > 0:
        inner_circle = create_mask(
            mask_type="ellipse",
            shape=circle_shape,
            radius=inner_radius,
            center=base_center,
        )
    if outer_radius > 0:
        outer_circle = create_mask(
            mask_type="ellipse",
            shape=circle_shape,
            radius=outer_radius,
            center=base_center,
        )
    circle = outer_circle - inner_circle
    circle = np.expand_dims(circle, axis=symmetry_axis)

    center = shape[symmetry_axis] // 2
    start_idx = center - height // 2
    stop_idx = center + height // 2 + height % 2

    slice_indices = tuple(
        slice(None) if i != symmetry_axis else slice(start_idx, stop_idx)
        for i in range(len(shape))
    )
    tube = np.zeros(shape)
    tube[slice_indices] = np.repeat(circle, height, axis=symmetry_axis)

    return tube


def scramble_phases(
    arr: NDArray, noise_proportion: float = 0.5, seed: int = 42
) -> NDArray:
    """
    Applies random phase scrambling to a given array.

    This function takes an input array, applies a Fourier transform, then scrambles the
    phase with a given proportion of noise, and finally applies an
    inverse Fourier transform to the scrambled data. The phase scrambling
    is controlled by a random seed.

    Parameters
    ----------
    arr : NDArray
        The input array to be scrambled.
    noise_proportion : float, optional
        The proportion of noise in the phase scrambling, by default 0.5.
    seed : int, optional
        The seed for the random phase scrambling, by default 42.

    Returns
    -------
    NDArray
        The array with scrambled phases.

    Raises
    ------
    ValueError
        If noise_proportion is not within [0, 1].
    """
    if noise_proportion < 0 or noise_proportion > 1:
        raise ValueError("noise_proportion has to be within [0, 1].")

    arr_fft = np.fft.fftn(arr)

    amp = np.abs(arr_fft)
    ph = np.angle(arr_fft)

    np.random.seed(seed)
    ph_noise = np.random.permutation(ph)
    ph_new = ph * (1 - noise_proportion) + ph_noise * noise_proportion
    ret = np.real(np.fft.ifftn(amp * np.exp(1j * ph_new)))
    return ret


def conditional_execute(func: Callable, execute_operation: bool = True) -> Callable:
    """
    Return the given function or a no-operation function based on execute_operation.

    Parameters
    ----------
    func : callable
        The function to be executed if execute_operation is True.
    execute_operation : bool, optional
        A flag that determines whether to return `func` or a no-operation function.
        Default is True.

    Returns
    -------
    callable
        Either the given function `func` or a no-operation function.

    Examples
    --------
    >>> def greet(name):
    ...     return f"Hello, {name}!"
    ...
    >>> operation = conditional_execute(greet, False)
    >>> operation("Alice")
    >>> operation = conditional_execute(greet, True)
    >>> operation("Alice")
    'Hello, Alice!'
    """

    def noop(*args, **kwargs):
        """No operation function."""
        pass

    return func if execute_operation else noop
