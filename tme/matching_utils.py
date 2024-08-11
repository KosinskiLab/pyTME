""" Utility functions for template matching.

    Copyright (c) 2023 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""
import os
import yaml
import pickle
from shutil import move
from tempfile import mkstemp
from itertools import product
from typing import Tuple, Dict, Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from scipy.spatial.transform import Rotation

from .backends import backend as be
from .memory import estimate_ram_usage
from .types import NDArray, BackendArray
from .extensions import max_euclidean_distance


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
    template: BackendArray, mask: BackendArray, n_observations: float
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

    Returns
    -------
    BackendArray
        Standardized input data.

    References
    ----------
    .. [1]  Hrabe T. et al, J. Struct. Biol. 178, 177 (2012).
    """
    masked_mean = be.sum(be.multiply(template, mask)) / n_observations
    masked_std = be.sum(be.multiply(be.square(template), mask))
    masked_std = be.subtract(masked_std / n_observations, be.square(masked_mean))
    masked_std = be.sqrt(be.maximum(masked_std, 0))

    template = be.subtract(template, masked_mean, out=template)
    template = be.divide(template, masked_std, out=template)
    return be.multiply(template, mask, out=template)


def _normalize_template_overflow_safe(
    template: BackendArray, mask: BackendArray, n_observations: float
) -> BackendArray:
    _template = be.astype(template, be._overflow_safe_dtype)
    _mask = be.astype(mask, be._overflow_safe_dtype)
    normalize_template(template=_template, mask=_mask, n_observations=n_observations)
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
    tmp_dir = os.environ.get("TMPDIR", None)
    _, filename = mkstemp(suffix=suffix, dir=tmp_dir)
    return filename


def array_to_memmap(arr: NDArray, filename: str = None) -> str:
    """
    Converts a obj:`numpy.ndarray` to a obj:`numpy.memmap`.

    Parameters
    ----------
    arr : obj:`numpy.ndarray`
        Input data.
    filename : str, optional
        Path to new memmap, :py:meth:`generate_tempfile_name` is used by default.

    Returns
    -------
    str
        Path to the memmap.
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
    splits = {k: possible_params[0, k] for k in range(shape1.size)}
    core_assignment = (
        possible_params[0, shape1.size],
        possible_params[0, (shape1.size + 1)],
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
            (slice((n_splits * length), (n_splits + 1) * length))
            if n_splits < splits.get(axis, 1) - 1
            else (slice(shape[axis] - length, shape[axis]))
            if equal_shape
            else (slice((n_splits * length), shape[axis]))
            for n_splits in range(splits.get(axis, 1))
        )
        for length, axis in zip(ret_shape, splits.keys())
    ]

    splits = tuple(product(*slice_list))

    return splits


def get_rotation_matrices(
    angular_sampling: float, dim: int = 3, use_optimized_set: bool = True
) -> NDArray:
    """
    Returns rotation matrices with desired ``angular_sampling`` rate.

    Parameters
    ----------
    angular_sampling : float
        The desired angular sampling in degrees.
    dim : int, optional
        Dimension of the rotation matrices.
    use_optimized_set : bool, optional
        Use optimized rotational sets, True by default and available for dim=3.

    Notes
    -----
        For dim = 3 optimized sets are used, otherwise QR-decomposition.

    Returns
    -------
    NDArray
        Array of shape (n, d, d) containing n rotation matrices.
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


def get_rotations_around_vector(
    cone_angle: float,
    cone_sampling: float,
    axis_angle: float = 360.0,
    axis_sampling: float = None,
    vector: Tuple[float] = (1, 0, 0),
    n_symmetry: int = 1,
    convention: str = None,
) -> NDArray:
    """
    Generate rotations describing the possible placements of a vector in a cone.

    Parameters
    ----------
    cone_angle : float
        The half-angle of the cone in degrees.
    cone_sampling : float
        Angular increment used for sampling points on the cone in degrees.
    axis_angle : float, optional
        The total angle of rotation around the vector axis in degrees (default is 360.0).
    axis_sampling : float, optional
        Angular increment used for sampling points around the vector axis in degrees.
        If None, it takes the value of `cone_sampling`.
    vector : Tuple[float], optional
        Cartesian coordinates in zyx convention.
    n_symmetry : int, optional
        Number of symmetry axis around the vector axis.
    convention : str, optional
        Convention for angles. By default returns rotation matrices.

    Returns
    -------
    NDArray
        An array of rotation angles represented as Euler angles (phi, theta, psi) in degrees.
        The shape of the array is (n, 3), where `n` is the total number of rotation angles.
        Each row represents a set of rotation angles.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere

    """
    if axis_sampling is None:
        axis_sampling = cone_sampling

    # Heuristic to estimate necessary number of points on sphere
    theta = np.linspace(0, cone_angle, round(cone_angle / cone_sampling) + 1)
    number_of_points = np.ceil(
        360 * np.divide(np.sin(np.radians(theta)), cone_sampling),
    )
    number_of_points = int(np.sum(number_of_points + 1) + 2)

    # Golden Spiral
    indices = np.arange(0, number_of_points, dtype=float) + 0.5
    radius = cone_angle * np.sqrt(indices / number_of_points)
    theta = np.pi * (1 + np.sqrt(5)) * indices

    angles_vector = Rotation.from_euler(
        angles=rotation_aligning_vectors([1, 0, 0], vector, convention="zyx"),
        seq="zyx",
        degrees=True,
    )

    # phi, theta, psi
    axis_angle /= n_symmetry
    phi_steps = np.maximum(np.round(axis_angle / axis_sampling), 1).astype(int)
    phi = np.linspace(0, axis_angle, phi_steps + 1)[:-1]
    np.add(phi, angles_vector.as_euler("zyx", degrees=True)[0], out=phi)
    angles = np.stack(
        [radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(radius)], axis=1
    )
    angles = np.repeat(angles, phi_steps, axis=0)
    angles[:, 2] = np.tile(phi, radius.size)

    angles = Rotation.from_euler(angles=angles, seq="zyx", degrees=True)
    angles = angles_vector * angles

    if convention is None:
        rotation_angles = angles.as_matrix()
    else:
        rotation_angles = angles.as_euler(seq=convention, degrees=True)

    return rotation_angles


def load_quaternions_by_angle(
    angular_sampling: float,
) -> Tuple[NDArray, NDArray, float]:
    """
    Get orientations and weights proportional to the given angular_sampling.

    Parameters
    ----------
    angular_sampling : float
        Requested angular sampling.

    Returns
    -------
    Tuple[NDArray, NDArray, float]
        Quaternion representations of orientations, weights associated with each
        quaternion and closest angular sampling to the requested sampling.
    """
    # Metadata contains (N orientations, rotational sampling, coverage as values)
    with open(
        os.path.join(os.path.dirname(__file__), "data", "metadata.yaml"), "r"
    ) as infile:
        metadata = yaml.full_load(infile)

    set_diffs = {
        setname: abs(angular_sampling - set_angle)
        for setname, (_, set_angle, _) in metadata.items()
    }
    fname = min(set_diffs, key=set_diffs.get)

    infile = os.path.join(os.path.dirname(__file__), "data", fname)
    quat_weights = np.load(infile)

    quat = quat_weights[:, :4]
    weights = quat_weights[:, -1]
    angle = metadata[fname][0]

    return quat, weights, angle


def quaternion_to_rotation_matrix(quaternions: NDArray) -> NDArray:
    """
    Convert quaternions to rotation matrices.

    Parameters
    ----------
    quaternions : NDArray
        Quaternion data of shape (n, 4).

    Returns
    -------
    NDArray
        Rotation matrices corresponding to the given quaternions.
    """
    q0 = quaternions[:, 0]
    q1 = quaternions[:, 1]
    q2 = quaternions[:, 2]
    q3 = quaternions[:, 3]

    s = np.linalg.norm(quaternions, axis=1) * 2
    rotmat = np.zeros((quaternions.shape[0], 3, 3), dtype=np.float64)

    rotmat[:, 0, 0] = 1.0 - s * ((q2 * q2) + (q3 * q3))
    rotmat[:, 0, 1] = s * ((q1 * q2) - (q0 * q3))
    rotmat[:, 0, 2] = s * ((q1 * q3) + (q0 * q2))

    rotmat[:, 1, 0] = s * ((q2 * q1) + (q0 * q3))
    rotmat[:, 1, 1] = 1.0 - s * ((q3 * q3) + (q1 * q1))
    rotmat[:, 1, 2] = s * ((q2 * q3) - (q0 * q1))

    rotmat[:, 2, 0] = s * ((q3 * q1) - (q0 * q2))
    rotmat[:, 2, 1] = s * ((q3 * q2) + (q0 * q1))
    rotmat[:, 2, 2] = 1.0 - s * ((q1 * q1) + (q2 * q2))

    np.around(rotmat, decimals=8, out=rotmat)

    return rotmat


def euler_to_rotationmatrix(angles: Tuple[float], convention: str = "zyx") -> NDArray:
    """
    Convert Euler angles to a rotation matrix.

    Parameters
    ----------
    angles : tuple
        A tuple representing the Euler angles in degrees.
    convention : str, optional
        Euler angle convention.

    Returns
    -------
    NDArray
        The generated rotation matrix.
    """
    n_angles = len(angles)
    angle_convention = convention[:n_angles]
    if n_angles == 1:
        angles = (angles, 0, 0)
    rotation_matrix = Rotation.from_euler(angle_convention, angles, degrees=True)
    return rotation_matrix.as_matrix().astype(np.float32)


def euler_from_rotationmatrix(
    rotation_matrix: NDArray, convention: str = "zyx"
) -> Tuple:
    """
    Convert a rotation matrix to euler angles.

    Parameters
    ----------
    rotation_matrix : NDArray
        A 2 x 2 or 3 x 3 rotation matrix in zyx form.
    convention : str, optional
        Euler angle convention, zyx by default.

    Returns
    -------
    Tuple
        The generate euler angles in degrees
    """
    if rotation_matrix.shape[0] == 2:
        temp_matrix = np.eye(3)
        temp_matrix[:2, :2] = rotation_matrix
        rotation_matrix = temp_matrix
    rotation = Rotation.from_matrix(rotation_matrix)
    return rotation.as_euler(convention, degrees=True).astype(np.float32)


def rotation_aligning_vectors(
    initial_vector: NDArray, target_vector: NDArray = [1, 0, 0], convention: str = None
):
    """
    Compute the rotation matrix or Euler angles required to align an initial vector with a target vector.

    Parameters
    ----------
    initial_vector : NDArray
        The initial vector to be rotated.
    target_vector : NDArray, optional
        The target vector to align the initial vector with. Default is [1, 0, 0].
    convention : str, optional
        The generate euler angles in degrees. If None returns a rotation matrix instead.

    Returns
    -------
    rotation_matrix_or_angles : NDArray or tuple
        Rotation matrix if convention is None else tuple of euler angles.
    """
    initial_vector = np.asarray(initial_vector, dtype=np.float32)
    target_vector = np.asarray(target_vector, dtype=np.float32)
    initial_vector /= np.linalg.norm(initial_vector)
    target_vector /= np.linalg.norm(target_vector)

    rotation_matrix = np.eye(len(initial_vector))
    if not np.allclose(initial_vector, target_vector):
        rotation_axis = np.cross(initial_vector, target_vector)
        rotation_angle = np.arccos(np.dot(initial_vector, target_vector))
        k = rotation_axis / np.linalg.norm(rotation_axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        rotation_matrix = np.eye(3)
        rotation_matrix += np.sin(rotation_angle) * K
        rotation_matrix += (1 - np.cos(rotation_angle)) * np.dot(K, K)

    if convention is None:
        return rotation_matrix

    angles = euler_from_rotationmatrix(rotation_matrix, convention=convention)
    return angles


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

            +---------+----------------------------------------------------------+
            | box     | Box mask (see :py:meth:`box_mask`)                       |
            +---------+----------------------------------------------------------+
            | tube    | Cylindrical mask (see :py:meth:`tube_mask`)              |
            +---------+----------------------------------------------------------+
            | ellipse | Ellipsoidal mask (see :py:meth:`elliptical_mask`)        |
            +---------+----------------------------------------------------------+
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
    mapping = {"ellipse": elliptical_mask, "box": box_mask, "tube": tube_mask}
    if mask_type not in mapping:
        raise ValueError(f"mask_type has to be one of {','.join(mapping.keys())}")

    mask = mapping[mask_type](**kwargs)
    if sigma_decay > 0:
        mask_filter = gaussian_filter(mask.astype(np.float32), sigma=sigma_decay)
        mask = np.add(mask, (1 - mask) * mask_filter)
        mask[mask < np.exp(-np.square(sigma_decay))] = 0

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
    >>> from tme.matching_utils import elliptical_mask
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
        )
    if outer_radius > 0:
        outer_circle = create_mask(
            mask_type="ellipse",
            shape=circle_shape,
            radius=outer_radius,
            center=circle_center,
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


def scramble_phases(
    arr: NDArray,
    noise_proportion: float = 0.5,
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
        Proportion of scrambled phases, 0.5 by default.
    seed : int, optional
        The seed for the random phase scrambling, 42 by default.
    normalize_power : bool, optional
        Return value has same sum of squares as ``arr``.

    Returns
    -------
    NDArray
        Phase scrambled version of ``arr``.
    """
    np.random.seed(seed)
    noise_proportion = max(min(noise_proportion, 1), 0)

    arr_fft = np.fft.fftn(arr)
    amp, ph = np.abs(arr_fft), np.angle(arr_fft)

    ph_noise = np.random.permutation(ph)
    ph_new = ph * (1 - noise_proportion) + ph_noise * noise_proportion
    ret = np.real(np.fft.ifftn(amp * np.exp(1j * ph_new)))

    if normalize_power:
        np.divide(ret - ret.min(), ret.max() - ret.min(), out=ret)
        np.multiply(ret, np.subtract(arr.max(), arr.min()), out=ret)
        np.add(ret, arr.min(), out=ret)
        scaling = np.divide(np.abs(arr).sum(), np.abs(ret).sum())
        np.multiply(ret, scaling, out=ret)

    return ret
