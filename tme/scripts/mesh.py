#!python3
"""Generate triangular surface meshes from segmentations.

Copyright (c) 2026 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

import argparse
from pathlib import Path
from typing import Optional

import numpy as np

from tme import Density
from tme.utils import cli
from tme.utils.logging import get_logger, setup_logging

logger = get_logger("mesh")


def _require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise ImportError(
            "open3d is required for pytme utils mesh. "
            "Install with: pip install 'pytme[mesh]'"
        ) from exc
    return o3d


def _axis_array(value, ndim):
    arr = np.asarray(value, dtype=np.float64).ravel()
    if arr.size == 1:
        arr = np.full(ndim, float(arr.item()))
    return arr


def marching_cubes(
    density: Density,
    threshold: float = 0.5,
    smoothing_iterations: int = 10,
    simplify: float = 10.0,
):
    """Marching-cubes isosurface from a dense segmentation.

    Parameters
    ----------
    density : Density
        Input segmentation. Treated as a scalar field thresholded at ``threshold``.
    threshold : float
        Isosurface level.
    smoothing_iterations : int
        Taubin smoothing iterations applied to the output mesh.
    simplify : float
        Triangle-count reduction factor. The raw isosurface is decimated to
        ``n_triangles / simplify`` triangles by quadric edge collapse, which
        minimizes surface distortion. A marching-cubes mesh at voxel resolution
        is heavily oversampled, so the default thins it substantially while
        preserving shape. ``0`` disables decimation.

    Returns
    -------
    open3d.geometry.TriangleMesh
        Surface mesh in the same physical coordinates as the input density.
    """
    o3d = _require_open3d()
    from skimage import measure

    ndim = density.data.ndim
    spacing = _axis_array(density.sampling_rate, ndim)
    origin = _axis_array(density.origin, ndim)

    padded = np.pad(density.data, 1, mode="constant", constant_values=0)
    vertices, faces, _, _ = measure.marching_cubes(
        padded, level=threshold, spacing=tuple(spacing)
    )
    vertices = vertices + origin - spacing

    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices),
        o3d.utility.Vector3iVector(faces.astype(np.int32)),
    )

    if simplify > 1:
        target = max(4, round(len(faces) / simplify))
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target)

    if smoothing_iterations > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iterations)

    mesh.compute_vertex_normals()
    return mesh


def _medial_clustering(
    points: np.ndarray, cutoff: float, rng: np.random.Generator
) -> np.ndarray:
    """Greedy centre-of-mass clustering at ``cutoff``."""
    from scipy.spatial import KDTree

    tree = KDTree(points)
    unassigned = np.ones(len(points), dtype=bool)
    clusters = []
    while unassigned.any():
        candidates = np.flatnonzero(unassigned)
        seed = int(candidates[rng.integers(len(candidates))])
        idx = np.asarray(tree.query_ball_point(points[seed], cutoff), dtype=np.int64)
        idx = idx[unassigned[idx]]
        clusters.append(points[idx].mean(axis=0))
        unassigned[idx] = False
    return np.asarray(clusters)


def poisson_mesh(
    density: Density,
    threshold: float = 0.5,
    thickness: float = 1.0,
    depth: int = 9,
    k_neighbors: int = 15,
    deldist: Optional[float] = None,
    rng: np.random.Generator = None,
):
    """Poisson surface reconstruction fit to the medial surface.

    Parameters
    ----------
    density : Density
        Input segmentation.
    threshold : float
        Binarization threshold.
    thickness : float
        Shell thickness used as for clustering in spatial units.
    depth : int
        Octree depth for Poisson reconstruction. Higher is finer and slower.
    k_neighbors : int
        Neighbourhood size for normal estimation and consistent orientation.
    deldist : float, optional
        Drop mesh vertices further than this distance from the medial point
        cloud, removing the triangles Poisson extrapolates.Defaults to
        ``1.5 * thickness``; 0 deactivates this feature.
    rng : np.random.Generator, optional
        Random generator. Defaults to a fresh ``default_rng``.

    Returns
    -------
    open3d.geometry.TriangleMesh
        Surface mesh in the same physical coordinates as the input density.
    """
    o3d = _require_open3d()

    ndim = density.data.ndim
    spacing = _axis_array(density.sampling_rate, ndim)
    origin = _axis_array(density.origin, ndim)

    mask = density.data > threshold
    if not np.any(mask):
        raise ValueError(
            f"No foreground voxels at threshold {threshold}. "
            "Check the input segmentation or lower --threshold."
        )

    points = np.column_stack(np.nonzero(mask)).astype(np.float64)
    points = points * spacing + origin

    if rng is None:
        rng = np.random.default_rng()
    medial = _medial_clustering(points, cutoff=thickness, rng=rng)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(medial)

    k = min(k_neighbors, max(len(medial) - 1, 1))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_consistent_tangent_plane(k=k)

    mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=0, scale=1.2, linear_fit=False
    )

    if deldist is None:
        deldist = 1.2 * thickness
    if deldist > 0:
        from scipy.spatial import KDTree

        distances, _ = KDTree(medial).query(np.asarray(mesh.vertices))
        mesh.remove_vertices_by_mask(distances > deldist)

    mesh.compute_vertex_normals()
    return mesh


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="pytme utils mesh",
        description=(
            "Create a triangular surface mesh from a binary segmentation. "
            "The two paths differ in their normal field. Marching cubes "
            "(default) traces the iso-surface directly, so for a thin-shell "
            "segmentation the mesh wraps both faces and normals point in and "
            "out of the shell. Pass --thickness to fit a Poisson mesh to the "
            "shell's medial surface instead, which yields a single coherently "
            "oriented normal field suitable for orientation-constrained "
            "matching once the global direction is verified."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-i",
        "--segmentation",
        type=cli.existing_file,
        required=True,
        help="Segmentation density file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help=(
            "Output mesh path. Written as Wavefront .obj; the extension is "
            "added if missing."
        ),
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Isosurface level for binarizing the segmentation (default: 0.5).",
    )
    parser.add_argument(
        "--thickness",
        type=cli.check_positive,
        default=None,
        help=(
            "Shell thickness in the same units as the sampling rate "
            "(typically Angstrom). Should be at least the membrane "
            "thickness, e.g. ~50 . Triggers Poisson reconstruction "
            "fit to the medial surface, with thickness used as the "
            "cutoff that collapses opposing shell to their midpoints."
        ),
    )
    parser.add_argument(
        "--smoothing",
        type=int,
        default=10,
        help=(
            "Taubin smoothing iterations applied to marching-cubes output. "
            "Ignored when --thickness is given (default: 10)."
        ),
    )
    parser.add_argument(
        "--simplify",
        type=float,
        default=8.0,
        help=(
            "Triangle-count reduction factor for marching-cubes output, "
            "applied by quadric decimation. The raw isosurface is heavily "
            "oversampled at voxel resolution; the default thins it ~10x while "
            "preserving shape. Use 0 to disable. Ignored when --thickness is "
            "given (default: 10.0)."
        ),
    )
    parser.add_argument(
        "--depth",
        type=int,
        default=9,
        help=(
            "Poisson octree depth. Higher is finer but slower. "
            "Only used with --thickness (default: 9)."
        ),
    )
    parser.add_argument(
        "--deldist",
        type=float,
        default=None,
        help=(
            "Drop Poisson mesh vertices further than this distance (in "
            "sampling-rate units) from the input data, removing triangles "
            "extrapolated to close the surface. Defaults to 1.2x --thickness; "
            "use 0 to keep the full surface. Only used with --thickness."
        ),
    )
    return parser


def main():
    setup_logging()
    parser = _build_parser()
    args = parser.parse_args()

    o3d = _require_open3d()
    density = Density.from_file(args.segmentation)

    if args.thickness is None:
        logger.info("Marching cubes at threshold %.3f", args.threshold)
        mesh = marching_cubes(
            density,
            threshold=args.threshold,
            smoothing_iterations=args.smoothing,
            simplify=args.simplify,
        )
    else:
        logger.info(
            "Poisson reconstruction at thickness %.3f (depth %d)",
            args.thickness,
            args.depth,
        )
        mesh = poisson_mesh(
            density,
            threshold=args.threshold,
            thickness=args.thickness,
            depth=args.depth,
            deldist=args.deldist,
        )

    output = Path(args.output)
    if output.suffix.lower() != ".obj":
        output = output.with_suffix(".obj")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not o3d.io.write_triangle_mesh(str(output), mesh):
        raise OSError(f"open3d failed to write mesh to {output}")

    logger.info(
        "Wrote %d vertices and %d faces to %s",
        len(mesh.vertices),
        len(mesh.triangles),
        output,
    )


if __name__ == "__main__":
    main()
