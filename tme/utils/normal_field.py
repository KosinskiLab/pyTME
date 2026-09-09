"""
Compute the normal vector field imposed by a triangular mesh
using the signed distance function on voxel grids.

Copyright (c) 2025 European Molecular Biology Laboratory

Author: Valentin Maurer <valentin.maurer@embl-hamburg.de>
"""

from typing import Tuple, Optional, Literal

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial import KDTree

from ..types import NDArray
from ..backends import backend as be
from ..matching_utils import split_shape

__all__ = ["compute_normal_field"]


def _get_query_indices_band(
    mesh,
    shape,
    offset,
    voxel_size,
    height,
    max_length=512,
):
    import open3d as o3d

    grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh(mesh, voxel_size)
    origin = np.round(np.asarray(grid.origin) / voxel_size).astype(int)

    surf_idx = np.array([v.grid_index for v in grid.get_voxels()])
    np.add(surf_idx, origin, out=surf_idx)
    np.subtract(surf_idx, offset, out=surf_idx)
    surf_idx = be.to_backend_array(surf_idx, be._int)

    query_indices = []
    splits = {i: max(1, int(np.ceil(shape[i] / max_length))) for i in range(3)}
    for slices in split_shape(tuple(shape), splits):
        starts = np.array([s.start for s in slices])
        stops = np.array([s.stop for s in slices])
        pad_starts = tuple(int(x) for x in np.maximum(0, starts - height))
        pad_stops = tuple(int(x) for x in np.minimum(shape, stops + height))

        valid = (
            be.sum(
                be.multiply(
                    surf_idx >= be.to_backend_array(pad_starts),
                    surf_idx < be.to_backend_array(pad_stops),
                ),
                axis=1,
            )
            == surf_idx.shape[1]
        )

        if be.sum(valid) == 0:
            continue

        valid_idx = be.subtract(surf_idx[valid], be.to_backend_array(pad_starts))
        vol_shape = tuple(int(y - x) for x, y in zip(pad_starts, pad_stops))

        vol = be.zeros(vol_shape, dtype=bool)
        vol = be.at(vol, tuple(valid_idx.T), True)
        dist = be.distance_transform_edt(~vol)

        inner_starts = tuple(int(x) for x in np.subtract(starts, pad_starts))
        inner_stops = tuple(int(x) for x in np.subtract(stops, pad_starts))
        inner_slices = tuple(slice(inner_starts[i], inner_stops[i]) for i in range(3))

        mask = dist[inner_slices] <= height
        if be.sum(mask) > 0:
            indices = be.stack(be.where(mask)).T
            indices = be.add(indices, be.to_backend_array(starts))
            query_indices.append(be.to_numpy_array(indices).astype(np.int32))
        vol, dist = None, None

    if not query_indices:
        return np.empty((0, 3), dtype=np.int32)
    return np.concatenate(query_indices, axis=0, dtype=np.int32)


def _build_scene(mesh):
    import open3d as o3d

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _compute_distance_volume(
    mesh,
    shape,
    offset,
    voxel_size,
    height=None,
    trust_vertex_normals=False,
    scene=None,
):
    import open3d as o3d

    if height is not None:
        query_indices = _get_query_indices_band(
            mesh,
            shape,
            offset,
            height=height + 4,
            voxel_size=voxel_size,
        )
    else:
        idx = np.meshgrid(*[np.arange(x) for x in shape], indexing="ij")
        query_indices = np.stack([x.ravel() for x in idx], axis=1, dtype=np.int32)

    if len(query_indices) == 0:
        return np.full(shape, 1e8, dtype=np.float32)

    voxel_centers = (query_indices + offset + 0.5) * voxel_size

    if scene is None:
        scene = _build_scene(mesh)
    query_tensor = o3d.core.Tensor(voxel_centers.astype(np.float32))

    if not trust_vertex_normals:
        distances = scene.compute_signed_distance(query_tensor).numpy()
    else:
        triangles = np.asarray(mesh.triangles)
        vertex_normals = np.asarray(mesh.vertex_normals)

        result = scene.compute_closest_points(query_tensor)
        closest_points = result["points"].numpy()
        tri_idx = result["primitive_ids"].numpy()
        uvs = result["primitive_uvs"].numpy()

        v0, v1, v2 = triangles[tri_idx].T
        w = 1.0 - uvs[:, 0] - uvs[:, 1]
        ref_normals = (
            w[:, None] * vertex_normals[v0]
            + uvs[:, 0:1] * vertex_normals[v1]
            + uvs[:, 1:2] * vertex_normals[v2]
        )

        direction = voxel_centers - closest_points
        distances = np.linalg.norm(direction, axis=1).astype(np.float32)
        distances *= np.where(np.einsum("ij,ij->i", direction, ref_normals) < 0, -1, 1)

    # edt = np.zeros(shape, dtype=np.float32)
    edt = np.full(shape, 5 * np.abs(distances).max(), dtype=np.float32)
    edt[tuple(query_indices.T)] = distances

    # if height is not None:
    #     computed = np.zeros(shape, dtype=bool)
    #     computed[tuple(query_indices.T)] = True

    #     dist_from_band, nearest_idx = be.distance_transform_edt(
    #         ~computed, return_distances=True, return_indices=True
    #     )
    #     fill = ~computed & (dist_from_band <= 5 * height)
    #     nearest_vals = edt[tuple(nearest_idx[:, fill])]
    #     edt[fill] = (
    #         nearest_vals
    #         + np.sign(nearest_vals) * dist_from_band[fill] * voxel_size
    #     )

    return edt


def _replace_medial_axis_normals(
    mesh, voxel_indices, normals, norms, sdf_values, voxel_size, scene=None
):
    """Replace unreliable SDF gradient normals near the medial axis with the
    barycentric interpolation of the closest-triangle vertex normals,
    sign-adjusted by the SDF value to match the active sign convention."""
    import open3d as o3d

    unreliable = norms.ravel() < 0.3 * voxel_size
    if not np.any(unreliable):
        return normals

    centers = (voxel_indices[unreliable] + 0.5) * voxel_size
    if scene is None:
        scene = _build_scene(mesh)

    result = scene.compute_closest_points(o3d.core.Tensor(centers.astype(np.float32)))
    triangle_ids = result["primitive_ids"].numpy()
    uvs = result["primitive_uvs"].numpy()

    triangles = np.asarray(mesh.triangles)
    vertex_normals = np.asarray(mesh.vertex_normals)
    vn = vertex_normals[triangles[triangle_ids]]

    w0 = 1.0 - uvs[:, 0] - uvs[:, 1]
    direction = (
        w0[:, None] * vn[:, 0] + uvs[:, 0, None] * vn[:, 1] + uvs[:, 1, None] * vn[:, 2]
    )
    direction /= np.linalg.norm(direction, axis=1, keepdims=True) + 1e-10
    direction *= np.where(sdf_values[unreliable] >= 0, 1, -1)[:, None]
    normals[unreliable] = direction
    return normals


def _filter_boundary_artifacts(
    mesh,
    valid_indices,
    voxel_size,
    normal_cutoff,
    _bary_eps=1e-2,
    level=0.0,
    scene=None,
):
    """Reject voxels whose closest mesh point lies on a boundary/edge vertex
    and whose surface distance differs from level by more than normal cutoff voxels"""
    import open3d as o3d

    edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    if len(edges) == 0:
        return np.ones(len(valid_indices), dtype=bool)

    triangles = np.asarray(mesh.triangles)

    boundary_verts = np.zeros(len(mesh.vertices), dtype=bool)
    boundary_verts[np.unique(edges)] = True

    sorted_boundary = np.sort(edges, axis=1)
    max_vid = max(triangles.max(), sorted_boundary.max()) + 1
    boundary_keys = (
        sorted_boundary[:, 0].astype(np.int64) * max_vid + sorted_boundary[:, 1]
    )
    edge_pairs = np.sort(triangles[:, [[0, 1], [0, 2], [1, 2]]], axis=2)
    edge_keys = edge_pairs[..., 0].astype(np.int64) * max_vid + edge_pairs[..., 1]
    edge_mask = np.isin(edge_keys, boundary_keys)

    if scene is None:
        scene = _build_scene(mesh)

    voxel_centers = (valid_indices + 0.5) * voxel_size
    result = scene.compute_closest_points(
        o3d.core.Tensor(voxel_centers.astype(np.float32))
    )
    closest_points = result["points"].numpy()
    tri_idx = result["primitive_ids"].numpy()
    uvs = result["primitive_uvs"].numpy()

    u, v = uvs[:, 0], uvs[:, 1]
    w = 1.0 - u - v

    bary_opposite = np.stack([v, u, w], axis=1)
    on_edge = np.any((bary_opposite < _bary_eps) & edge_mask[tri_idx], axis=1)

    bary = np.stack([w, u, v], axis=1)
    dominant = np.argmax(bary, axis=1)
    on_vertex = (
        bary[np.arange(len(dominant)), dominant] > 1.0 - _bary_eps
    ) & boundary_verts[triangles[tri_idx, dominant]]

    on_boundary = on_edge | on_vertex
    dist = np.linalg.norm(voxel_centers - closest_points, axis=1)
    return ~(on_boundary & (np.abs(dist - level) > normal_cutoff * voxel_size))


def _erode_mesh_boundary(mesh, min_distance):
    """Remove boundary triangles until all open edges have receded by
    at least *min_distance* (real-space units) from the original boundary."""
    import open3d as o3d

    edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
    if len(edges) == 0:
        return mesh

    tree = KDTree(np.asarray(mesh.vertices)[np.unique(edges)])

    while True:
        edges = np.asarray(mesh.get_non_manifold_edges(allow_boundary_edges=False))
        if len(edges) == 0:
            break

        boundary_ids = np.unique(edges)
        dists, _ = tree.query(np.asarray(mesh.vertices)[boundary_ids])
        erode_ids = set(boundary_ids[dists < min_distance].tolist())
        if not erode_ids:
            break

        triangles = np.asarray(mesh.triangles)
        erode_arr = np.fromiter(erode_ids, dtype=triangles.dtype, count=len(erode_ids))
        keep = ~np.isin(triangles, erode_arr).any(axis=1)
        if not keep.any():
            break

        mesh.triangles = o3d.utility.Vector3iVector(triangles[keep])
        mesh.remove_unreferenced_vertices()

    return mesh


def compute_normal_field(
    mesh,
    voxel_size: float,
    height: int,
    shape: Optional[Tuple[int, int, int]] = None,
    offset: Optional[Tuple[int, int, int]] = None,
    normal_offset: Optional[float] = None,
    mode: Literal[
        "outward", "inward", "natural", "outside_only", "inside_only"
    ] = "outward",
    trust_vertex_normals: bool = True,
    normal_cutoff: Optional[float] = 0,
    boundary_cutoff: Optional[float] = None,
) -> Tuple[NDArray, NDArray]:
    """
    Compute normal vector field from a triangle mesh using SDF gradients.

    Parameters
    ----------
    mesh : open3d.geometry.TriangleMesh or str
        Input mesh in real-space coordinates, or path to a mesh file.
    voxel_size : float
        Voxel size in real-space units.
    height : int
        Maximum distance from mesh surface in voxel units.
    shape : tuple of int, optional
        Region of interest shape in voxels. Derived from mesh bounds if None.
    offset : tuple of int, optional
        Region of interest offset in voxels. Derived from mesh bounds if None.
    normal_offset : tuple of float, optional
        Translate all vertices offset times their corresponding normal vector.
    mode : {"outward", "inward", "natural", "outside_only", "inside_only"}
        Controls which voxels to include and normal orientation.
    trust_vertex_normals : bool
        Use vertex normals to determine SDF sign convention.
    normal_cutoff : float, optional
        Reject voxels on boundary edges/vertices farther than this many
        voxels from the surface. None disables filtering.
    boundary_cutoff : float, optional
        Erode open mesh boundaries by this many voxels before computing
        the normal field. None disables erosion.

    Returns
    -------
    voxel_indices : ndarray, shape (n, 3)
        Voxel indices where normals are defined.
    normals : ndarray, shape (n, 3)
        Unit normal vectors at each voxel.
    """
    try:
        import open3d as o3d
    except ImportError:
        raise ImportError(
            "open3d is required for mesh-based orientation constraints. "
            "Install it with: pip install pytme['mesh']"
        ) from None

    if isinstance(mesh, str):
        mesh = o3d.io.read_triangle_mesh(mesh)

    mesh = o3d.geometry.TriangleMesh(mesh)
    if not trust_vertex_normals or not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    _offset = 0.0 if normal_offset is None else normal_offset
    shift = _offset * voxel_size
    band_pad = int(np.ceil(abs(_offset)))

    if offset is None:
        mesh_min = np.asarray(mesh.get_axis_aligned_bounding_box().min_bound)
        offset = np.floor(mesh_min / voxel_size).astype(int) - height - band_pad

    # o3d's mesh.translate function, appears to be broken on some systems
    offset = np.asarray(offset)
    mesh.vertices = o3d.utility.Vector3dVector(
        np.asarray(mesh.vertices) - np.multiply(offset, voxel_size)
    )

    if shape is None:
        mesh_max = np.asarray(mesh.get_axis_aligned_bounding_box().max_bound)
        shape = np.ceil(mesh_max / voxel_size).astype(int) + height + band_pad + 8

    if boundary_cutoff is not None:
        mesh = _erode_mesh_boundary(mesh, boundary_cutoff * voxel_size)

    scene = _build_scene(mesh)

    shape = np.asarray(shape)
    sdf_volume = _compute_distance_volume(
        mesh,
        shape,
        np.zeros(3, dtype=int),
        voxel_size,
        height=height + band_pad,
        trust_vertex_normals=trust_vertex_normals,
        scene=scene,
    )
    threshold = height * voxel_size

    valid = np.abs(sdf_volume - shift) <= threshold
    if mode == "outside_only":
        valid &= sdf_volume > shift
    elif mode == "inside_only":
        valid &= sdf_volume < shift

    if normal_cutoff is not None:
        valid_indices = np.argwhere(valid)
        keep = _filter_boundary_artifacts(
            mesh,
            valid_indices,
            voxel_size,
            normal_cutoff,
            level=abs(shift),
            scene=scene,
        )
        valid[tuple(valid_indices[~keep].T)] = False

    normals = np.stack(
        [
            gaussian_filter(
                sdf_volume, sigma=1, order=[int(i == ax) for i in range(3)]
            )[valid]
            for ax in range(3)
        ],
        axis=1,
    )
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= norms + 1e-10

    voxel_indices = np.argwhere(valid)
    sdf_valid = sdf_volume[valid]
    normals = _replace_medial_axis_normals(
        mesh, voxel_indices, normals, norms, sdf_valid, voxel_size, scene=scene
    )
    if mode == "natural":
        normals[sdf_valid < shift] *= -1
    elif mode in ("inward", "inside_only"):
        normals *= -1

    return (voxel_indices + offset).astype(np.int32), normals
