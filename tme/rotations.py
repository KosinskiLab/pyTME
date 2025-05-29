""" Implements various means of generating rotation matrices.

    Copyright (c) 2023-2025 European Molecular Biology Laboratory

    Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import yaml
import warnings
from typing import Tuple
from os.path import join, dirname

import numpy as np
from scipy.spatial.transform import Rotation

from .types import NDArray

__all__ = [
    "get_cone_rotations",
    "align_vectors",
    "euler_to_rotationmatrix",
    "euler_from_rotationmatrix",
    "get_rotation_matrices",
    "align_to_axis",
]


def _sample_cone(
    angle: float, sampling: float, axis: Tuple[float] = (1, 0, 0)
) -> NDArray:
    """
    Sample points on a cone surface around cone_axis using golden spiral distribution.

    Parameters
    ----------
    angle : float
        The half-angle of the cone in degrees.
    sampling : float
        Angular increment used for sampling points in degrees.
    axis : tuple of floats
        Vector to align the cone with.

    Returns
    -------
    NDArray
        Array of points around axis with shape n,3.

    References
    ----------
    .. [1] https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
    """
    theta = np.linspace(0, angle, round(angle / sampling) + 1)
    number_of_points = np.ceil(
        360 * np.divide(np.sin(np.radians(theta)), sampling),
    )
    number_of_points = int(np.sum(number_of_points + 1) + 2)

    indices = np.arange(0, number_of_points, dtype=float) + 0.5
    radius = np.radians(angle * np.sqrt(indices / number_of_points))
    theta = np.pi * (1 + np.sqrt(5)) * indices

    points = np.stack(
        [
            np.cos(radius),
            np.cos(theta) * np.sin(radius),
            np.sin(theta) * np.sin(radius),
        ],
        axis=1,
    )

    rotation = Rotation.from_euler(
        angles=align_vectors((1, 0, 0), axis, seq="zyz"),
        seq="zyz",
        degrees=True,
    )
    return rotation.apply(points)


def get_cone_rotations(
    cone_angle: float,
    cone_sampling: float,
    axis: Tuple[float] = (1, 0, 0),
    axis_angle: float = 360.0,
    axis_sampling: float = None,
    reference: Tuple[float] = (1, 0, 0),
    n_symmetry: int = 1,
    seq: str = None,
) -> NDArray:
    """
    Generate rotations describing the possible placements of a vector in a cone.

    Parameters
    ----------
    cone_angle : float
        The half-angle of the cone in degrees.
    cone_sampling : float
        Angular increment used for sampling points on the cone in degrees.
    axis : Tuple[float], optional
        Base-vector of the cone.
    axis_angle : float, optional
        The total angle of rotation around the cone axis in degrees (default is 360.0).
    axis_sampling : float, optional
        Angular increment used for sampling points around the cone axis in degrees.
        If None, it takes the value of cone_sampling.
    reference : Tuple[float], optional
        Returned rotations will map this point onto the cone. In practice, this is
        the principal axis of the template.
    n_symmetry : int, optional
        Number of symmetry axis around the vector axis.
    seq : str, optional
        Convention for angles. By default returns rotation matrices.

    Returns
    -------
    NDArray
        An arary of rotations represented as stack of rotation matrices when convention
        None (n, 3, 3) or an array of Euler angles (n, 3) for available conventions.
    """
    if axis_sampling is None:
        axis_sampling = cone_sampling

    points = _sample_cone(angle=cone_angle, sampling=cone_sampling, axis=axis)
    reference = np.asarray(reference).astype(np.float32)
    reference /= np.linalg.norm(reference)

    axis_angle /= n_symmetry
    phi_steps = np.maximum(np.round(axis_angle / axis_sampling), 1).astype(int)
    phi = np.linspace(0, axis_angle, phi_steps + 1)[:-1]

    axis_rotation = Rotation.from_rotvec(axis * np.radians(phi)[:, None])
    all_rotations = [
        axis_rotation * Rotation.from_matrix(align_vectors(reference, x))
        for x in points
    ]

    rotations = Rotation.concatenate(all_rotations)
    if seq is None:
        return rotations.as_matrix()

    return rotations.as_euler(seq=seq, degrees=True)


def align_vectors(base: NDArray, target: NDArray = (0, 0, 1), seq: str = None):
    """
    Compute the rotation matrix or Euler angles required to align an initial vector with a target vector.

    Parameters
    ----------
    base : NDArray
        The basis vector.
    target : NDArray, optional
        The vector to map base to, defaults to (0,0,1).
    seq : str, optional
        Euler angle convention, None returns a rotation matrix instead.

    Returns
    -------
    NDArray
        Rotation matrix if seq is None otherwise Euler angles in desired convention
    """
    base = np.asarray(base, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)

    rotation, error = Rotation.align_vectors(target, base)
    if seq is None:
        return rotation.as_matrix()
    return rotation.as_euler(seq=seq, degrees=True)


def euler_to_rotationmatrix(angles: Tuple[float], seq: str = "zyz") -> NDArray:
    """
    Convert Euler angles to a rotation matrix.

    Parameters
    ----------
    angles : tuple
        A tuple representing the Euler angles in degrees.
    seq : str, optional
        Euler angle convention.

    Returns
    -------
    NDArray
        The generated rotation matrix.
    """
    n_angles = len(angles)
    angle_convention = seq[:n_angles]
    if n_angles == 1:
        angles = (angles, 0, 0)
    rotation_matrix = Rotation.from_euler(
        seq=angle_convention, angles=angles, degrees=True
    )
    return rotation_matrix.as_matrix().astype(np.float32)


def euler_from_rotationmatrix(rotation_matrix: NDArray, seq: str = "zyz") -> Tuple:
    """
    Convert a rotation matrix to euler angles.

    Parameters
    ----------
    rotation_matrix : NDArray
        A 2 x 2 or 3 x 3 rotation matrix.
    seq : str, optional
        Euler angle convention, zyz by default.

    Returns
    -------
    Tuple
        The generate euler angles in degrees
    """
    if rotation_matrix.shape[0] == 2:
        temp_matrix = np.eye(3)
        temp_matrix[:2, :2] = rotation_matrix
        rotation_matrix = temp_matrix

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rotation = Rotation.from_matrix(rotation_matrix)
        angles = rotation.as_euler(seq=seq, degrees=True).astype(np.float32)
    return angles


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
        quaternions, *_ = _load_quaternions_by_angle(angular_sampling)
        ret = Rotation.from_quat(quaternions).as_matrix()
    else:
        num_rotations = dim * (dim - 1) // 2
        k = int((360 / angular_sampling) ** num_rotations)
        As = np.random.randn(k, dim, dim)
        ret, _ = np.linalg.qr(As)
        dets = np.linalg.det(ret)
        neg_dets = dets < 0
        ret[neg_dets, :, -1] *= -1
        ret[0] = np.eye(dim, dtype=ret.dtype)
    return ret


def _load_quaternions_by_angle(
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
        Quaternions, associated weights and realized angular sampling rate.
    """
    # Metadata contains (N orientations, rotational sampling, coverage as values)
    with open(join(dirname(__file__), "data", "metadata.yaml"), "r") as infile:
        metadata = yaml.full_load(infile)

    set_diffs = {
        setname: abs(angular_sampling - set_angle)
        for setname, (_, set_angle, _) in metadata.items()
    }
    fname = min(set_diffs, key=set_diffs.get)

    infile = join(dirname(__file__), "data", fname)
    quat_weights = np.load(infile)

    quat = quat_weights[:, :4]
    weights = quat_weights[:, -1]
    angle = metadata[fname][0]

    return quat, weights, angle


def align_to_axis(
    coordinates: NDArray,
    weights: NDArray = None,
    axis: int = 2,
    flip: bool = False,
    eigenvector_index: int = 0,
) -> NDArray:
    """
    Calculate a rotation matrix that aligns the principal axis of a point cloud
    with a specified coordinate axis.

    Parameters
    ----------
    coordinates : NDArray
        Array of 3D coordinates with shape (n, 3) representing the point cloud.
    weights : NDArray
        Coordinate weighting factors with shape (n,).
    axis : int, optional
        The target axis to align with, defaults to 2 (z-axis).
    flip : bool, optional
        Whether to align with the negative direction of the axis, default is False.
    eigenvector_index : int, optional
        Index of eigenvector to select, sorted by descending eigenvalues.
        0 = largest eigenvalue (most variance), 1 = second largest, etc.
        Default is 0 (primary principal component).

    Returns
    -------
    NDArray
        3x3 rotation matrix that aligns the principal component of the
        coordinates with the specified axis.
    """
    axis = int(axis)
    coordinates = np.asarray(coordinates)
    alignment_axis = np.array(
        [0 if i != axis else 1 for i in range(coordinates.shape[1])]
    )
    if flip:
        alignment_axis *= -1

    ndim = coordinates.shape[1]
    if eigenvector_index >= ndim:
        raise ValueError(f"eigenvector_index has to be less than {ndim}.")

    avg = np.average(coordinates, axis=0, weights=weights)
    coordinates = coordinates - avg
    cov_matrix = np.cov(coordinates.T, aweights=weights)

    # Eigenvalues are already sorted in ascending order
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    eigenvector = eigenvectors[:, -(eigenvector_index + 1)]
    return align_vectors(eigenvector, alignment_axis)
