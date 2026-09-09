"""
Implements various means of generating rotation matrices.

Copyright (c) 2023-2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import yaml
import warnings
from typing import Tuple, Optional
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
    angle: float, sampling: float, axis: Tuple[float] = (0, 0, 1)
) -> NDArray:
    """
    Sample points uniformly on a spherical cap.

    Parameters
    ----------
    angle : float
        Half-angle of the cone in degrees.
    sampling : float
        Target angular spacing between points in degrees.
    axis : tuple of floats
        Cone axis direction.

    Returns
    -------
    NDArray
        Array of points around axis with shape n,3.
    """
    angle = np.radians(min(max(angle, 0), 180))

    # Surface area of unit spherical cap A = 2π(1 - cos(angle))
    # Each point covers approximately sampling squre in steradians
    cap_area = 2 * np.pi * (1 - np.cos(angle))
    area_per_point = (np.radians(sampling)) ** 2
    n_samples = max(1, int(np.ceil(cap_area / area_per_point)))

    # Same construction as Mosaic
    indices = np.arange(0, n_samples, dtype=float) + 0.5
    phi = np.arccos(1 - (1 - np.cos(angle)) * indices / n_samples)
    theta = np.pi * (1 + 5**0.5) * indices

    points = np.column_stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]
    )
    rotation = Rotation.from_matrix(align_vectors((0, 0, 1), axis))
    return rotation.apply(points)


def get_cone_rotations(
    cone_angle: float,
    cone_sampling: float,
    axis_angle: Optional[float] = None,
    axis_sampling: Optional[float] = None,
    reference: Tuple[float] = (0, 0, 1),
    n_symmetry: Optional[int] = None,
    **kwargs,
) -> NDArray:
    """
    Generate rotations describing the possible placements of a vector in a cone.

    Parameters
    ----------
    cone_angle : float
        Half-angle of the cone in degrees. Defines the maximum angular deviation
        from the reference direction. Must be in range (0, 180].
    cone_sampling : float
        Angular spacing between sample points on the cone surface in degrees.
    axis_angle : float, optional
        Total rotation angle around the reference direction in degrees. Defaults
        to 360.0 for complete in-plane rotation.
    axis_sampling : float, optional
        Angular spacing for in-plane rotations along the reference in degrees.
        If None, uses the value of cone_sampling.
    reference : Tuple[float], optional
        The central direction of the cone as a 3D vector (x, y, z). Rotations
        will map this direction onto the cone surface. Defaults to z unit vector.
    n_symmetry : int, optional
        Symmetry order of the object around the reference direction.
        The axis_angle is divided by this value. For example, use n_symmetry=2
        for C2 symmetry. Default is 1 (no symmetry).

    Returns
    -------
    NDArray
        Array of rotation matrices with shape (n, 3, 3).

    Examples
    --------
    Sample orientations within 30° of the z-axis with full in-plane rotation:

    >>> rotations = get_cone_rotations(
    ...     cone_angle=30.0,
    ...     cone_sampling=10.0,
    ...     reference=(0, 0, 1)
    ... )

    Limited search with 2-fold symmetry around x-axis:

    >>> rotations = get_cone_rotations(
    ...     cone_angle=45.0,
    ...     cone_sampling=15.0,
    ...     axis_angle=180.0,
    ...     reference=(1, 0, 0),
    ...     n_symmetry=2
    ... )

    Notes
    -----
    The total number of rotations is approximately:
        N ≈ (2π(1 - cos(cone_angle)) / cone_sampling²) × (axis_angle / axis_sampling)
    """
    axis_angle = 360.0 if axis_angle is None else axis_angle
    axis_sampling = cone_sampling if axis_sampling is None else axis_sampling

    reference = np.asarray(reference).astype(np.float32)
    reference /= np.linalg.norm(reference)

    if n_symmetry is not None:
        axis_angle /= n_symmetry

    phi_steps = np.maximum(np.round(axis_angle / axis_sampling), 1).astype(int)
    phi = np.linspace(-axis_angle / 2, axis_angle / 2, phi_steps, endpoint=False)
    axis_rotation = Rotation.from_rotvec(reference * np.radians(phi)[:, None])

    if cone_angle <= 0:
        return axis_rotation.as_matrix()

    points = _sample_cone(angle=cone_angle, sampling=cone_sampling, axis=reference)
    rotations = Rotation.concatenate(
        [
            axis_rotation * Rotation.from_matrix(align_vectors(reference, x))
            for x in points
        ]
    )
    return rotations.as_matrix()


def align_vectors(base: NDArray, target: NDArray = (0, 0, 1)) -> NDArray:
    """
    Compute the rotation matrix or Euler angles required to align an initial
    vector with a target vector. As align_vectors(base, target) @ base = target

    Parameters
    ----------
    base : NDArray
        The basis vector. Can be shape (d,) or (n, d).
    target : NDArray, optional
        The vector to map base to. Can be shape (d,) or (n, d). Default is (0,0,1).

    Returns
    -------
    NDArray
        Rotation matrix mapping base to target.
    """
    base = np.atleast_2d(base)
    target = np.atleast_2d(target)

    nb, nt = base.shape[0], target.shape[0]
    if not (nt == nb or nt == 1 or nb == 1):
        raise ValueError(
            f"Incompatible shapes: base {base.shape}, target {target.shape}. "
            "Provide either a single target or one per base."
        )

    base = base / np.linalg.norm(base, axis=1, keepdims=True)
    target = target / np.linalg.norm(target, axis=1, keepdims=True)

    # Rotation.from_quat expects scalar-last. Support for scalar first via
    # scalar_first flag was not added until scipy v.14.0
    quat = _align_vectors_to_quat(base, target)
    rotation = Rotation.from_quat(quat[:, (1, 2, 3, 0)])

    rotation = rotation.as_matrix().astype(np.float32)
    if base.shape[0] == 1:
        return np.squeeze(rotation)
    return rotation


def _align_vectors_to_quat(vec1: np.ndarray, vec2: np.ndarray) -> np.ndarray:
    """
    Compute quaternions for shortest rotation aligning vec1 to vec2.

    Parameters
    ----------
    vec1, vec2 : np.ndarray
        Normalized vectors, shape (N, 3).

    Returns
    -------
    np.ndarray
        Quaternions [w, x, y, z], shape (N, 4).
    """
    axis = np.cross(vec1, vec2)
    cos_angle = np.sum(vec1 * vec2, axis=1)

    aligned = cos_angle > (1 - 1e-8)
    opposite = cos_angle < (-1 + 1e-8)
    normal = ~(aligned | opposite)

    quaternions = np.empty((vec1.shape[0], 4))
    quaternions[aligned] = [1, 0, 0, 0]

    # Half angle
    if np.any(normal):
        w = np.sqrt((1 + cos_angle[normal]) / 2)
        xyz = axis[normal] / (2 * w[:, np.newaxis])
        quaternions[normal, 0] = w
        quaternions[normal, 1:] = xyz

    # Opposite
    if np.any(opposite):
        for i in np.where(opposite)[0]:
            v = vec1[i]
            perp = np.cross(v, [1, 0, 0]) if abs(v[0]) < 0.9 else np.cross(v, [0, 1, 0])
            perp = perp / np.linalg.norm(perp)
            quaternions[i] = [0, perp[0], perp[1], perp[2]]

    return quaternions


def euler_to_rotationmatrix(angles: Tuple[float], seq: str = "ZYZ") -> NDArray:
    """
    Convert Euler angles to a rotation matrix.

    Parameters
    ----------
    angles : tuple
        Euler angles in degrees.
    seq : str, optional
        Euler angle convention, defaults to ZYZ.

    Returns
    -------
    NDArray
        Corresponding rotation matrix.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        rotation = Rotation.from_euler(seq=seq, angles=angles, degrees=True)
        return rotation.as_matrix().astype(np.float32)


def euler_from_rotationmatrix(rotation_matrix: NDArray, seq: str = "ZYZ") -> NDArray:
    """
    Convert a rotation matrix to Euler angles.

    Parameters
    ----------
    rotation_matrix : NDArray
        Rotation matrix (d,d).
    seq : str, optional
        Euler angle convention, default to intrinsic ZYZ.

    Returns
    -------
    NDArray
        Corresponding Euler angles in degrees.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return Rotation.from_matrix(rotation_matrix).as_euler(seq=seq, degrees=True)


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
        return Rotation.from_quat(quaternions).as_matrix()

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
        Quaternions (x,y,z,w), associated weights and realized angular sampling.
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

    # Quat weights are scalar first (w,x,y,z), but scipy expects (x,y,z,w)
    quat = quat_weights[:, (1, 2, 3, 0)]
    weights = quat_weights[:, -1]
    return quat, weights, metadata[fname][0]


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


def get_symmetry_matrices(
    symmetry_type: str, axis: Tuple[float] = (0, 0, 1)
) -> NDArray:
    """
    Get rotation matrices describing point group symmetries.

    Parameters
    ----------
    symmetry_type : str
        Type of symmetry, supported are 'Cn' and 'Dn'.
    axis : Tuple[float], optional
        Symmetry axis as (x, y, z) vector, defaults to (0, 0, 1).

    Returns
    -------
    NDArray
        Array of rotation matrices with shape (n, 3, 3).

    """
    axis = np.array(axis, dtype=np.float32)
    axis = axis / np.linalg.norm(axis)

    try:
        n = int(symmetry_type[1:])
    except IndexError:
        n = 1

    matrices = []
    symmetry = symmetry_type.upper()[0]
    if symmetry == "C":

        for i in range(n):
            angle = 2 * np.pi * i / n
            R = Rotation.from_rotvec(angle * axis)
            matrices.append(R.as_matrix().astype(np.float32))

    elif symmetry == "D":
        # First add the Cn rotations around main axis
        matrices.extend(get_symmetry_matrices(f"C{n}", axis=axis))

        # Then add n 180° rotations around perpendicular axes
        _, _, vh = np.linalg.svd(axis.reshape(1, -1))

        perp = vh[-1].astype(np.float32)
        perp = perp / np.linalg.norm(perp)
        for i in range(n):
            angle = np.pi * i / n
            R = Rotation.from_rotvec(angle * axis)

            R_180 = Rotation.from_rotvec(np.pi * R.apply(perp))
            matrices.append(R_180.as_matrix().astype(np.float32))
    else:
        raise ValueError(f"Unsupported symmetry type: {symmetry_type}")
    return np.array(matrices)


def _canonical_quaternion(quats: NDArray) -> NDArray:
    """Flip quaternion sign so the largest-magnitude component is positive."""
    idx = np.argmax(np.abs(quats), axis=-1)
    signs = np.sign(np.take_along_axis(quats, idx[..., None], axis=-1))
    signs[signs == 0] = 1
    return quats * signs


def reduce_rotations_by_symmetry(
    rotations: NDArray, symmetry: str = "C1", axis: Tuple[float] = (0, 0, 1)
) -> NDArray:
    """
    Reduce rotation matrices to a point-group fundamental domain.

    Each rotation is kept only when it is the canonical (lexicographically
    largest, sign-canonicalized quaternion) member of its own symmetry orbit
    ``{S_k . R}``. The orbit uses left multiplicatin to align with the rotation
    pulling convention, so orientations that produce the identical
    rotatde density differ by ``R ~ S_k . R``.

    Parameters
    ----------
    rotations : NDArray
        Rotation matrices with shape (n, 3, 3).
    symmetry : str, optional
        Point-group symmetry as 'C<n>' or 'D<n>', 'C1' by default.
    axis : Tuple[float], optional
        Symmetry axis as (x, y, z), defaults to (0, 0, 1).

    Returns
    -------
    NDArray
        Rotation matrices with shape (m, 3, 3), m <= n.
    """
    sym_ops = get_symmetry_matrices(symmetry, axis=axis)
    if sym_ops.shape[0] <= 1:
        return rotations

    keep = []
    for i in range(rotations.shape[0]):
        orbit = _canonical_quaternion(
            Rotation.from_matrix(sym_ops @ rotations[i]).as_quat()
        )
        order = np.lexsort(orbit.T[::-1])
        # get_symmetry_matrices lists the identity first, so orbit element 0 is
        # rotations[i] itself. Keep it when it holds the canonical value.
        if np.allclose(orbit[0], orbit[order[-1]], atol=1e-6):
            keep.append(i)
    return rotations[keep]
