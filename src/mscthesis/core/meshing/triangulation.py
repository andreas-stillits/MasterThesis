from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import open3d as o3d
from skimage import measure

from ...log import log_call


def _clean_mesh(mesh: o3d.geometry.TriangleMesh) -> o3d.geometry.TriangleMesh:
    """
    Clean the mesh by removing degenerate triangles, duplicated vertices,
    and non-manifold edges.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input mesh to be cleaned.

    Returns:
        o3d.geometry.TriangleMesh: The cleaned mesh.
    """
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_non_manifold_edges()
    mesh.compute_vertex_normals()
    return mesh


def _metadata(
    mesh: o3d.geometry.TriangleMesh,
    pre_area: float,
    post_area: float,
    pre_volume: float,
    post_volume: float,
    shrinkage_tolerance: float,
    success: bool,
) -> dict[str, Any]:
    """
    Create metadata dictionary for the triangulated mesh.

    Args:
        mesh (o3d.geometry.TriangleMesh): The triangulated mesh.
        pre_area (float): Surface area before processing.
        post_area (float): Surface area after processing.
        pre_volume (float): Volume before processing.
        post_volume (float): Volume after processing.
        shrinkage_tolerance (float): Tolerance for acceptable shrinkage.
        success (bool): Whether the triangulation was successful (manifold and watertight).

    Returns:
        dict[str, Any]: Metadata dictionary.
    """
    num_vertices = len(np.asarray(mesh.vertices))
    num_elements = len(np.asarray(mesh.triangles))

    area_shrinkage = abs(pre_area - post_area) / pre_area
    volume_shrinkage = abs(pre_volume - post_volume) / pre_volume
    shrinkage_acceptable = (area_shrinkage <= shrinkage_tolerance) and (
        volume_shrinkage <= shrinkage_tolerance
    )

    return {
        "num_vertices": num_vertices,
        "num_elements": num_elements,
        "pre_area": pre_area,
        "post_area": post_area,
        "area_shrinkage": area_shrinkage,
        "pre_volume": pre_volume,
        "post_volume": post_volume,
        "volume_shrinkage": volume_shrinkage,
        "shrinkage_acceptable": shrinkage_acceptable,
        "success": success,
    }


@log_call()
def triangulate_voxels(
    voxels: np.ndarray[tuple[int, int, int],],
    smoothing_iterations: int = 15,
    decimation_target: int = 10_000,
    shrinkage_tolerance: float = 0.10,
    spacing: Iterable[float] = (1.0, 1.0, 1.0),
) -> tuple[o3d.geometry.TriangleMesh, dict[str, Any]]:
    """
    Triangulate a voxel model using the marching cubes algorithm.

    Args:
        voxels (np.ndarray): 3D numpy array of shape (X, Y, Z) with binary values,
            where 1 indicates presence of tissue and 0 indicates airspace.
        smoothing_iterations (int): Number of iterations for Taubin smoothing.
        decimation_target (int): Target number of triangles after mesh decimation.
        shrinkage_tolerance (float): Maximum acceptable shrinkage ratio for area and volume.
        spacing (Iterable[float]): Voxel spacing in each dimension (dx, dy, dz).

    Returns:
        o3d.geometry.TriangleMesh: The triangulated mesh.
        dict: Metadata including number of vertices and faces.
    """
    # apply marching cubes algorithm
    verts, faces, normals, values = measure.marching_cubes(
        voxels, spacing=spacing, level=0.5
    )
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    mesh = _clean_mesh(mesh)

    # get surface area and volume
    pre_area = mesh.get_surface_area()
    pre_volume = mesh.get_volume()

    # apply smoothing
    mesh = mesh.filter_smooth_taubin(number_of_iterations=smoothing_iterations)
    mesh = _clean_mesh(mesh)

    # apply mesh decimation
    current_triangle_count = len(np.asarray(mesh.triangles))
    target_triangle_count = decimation_target
    if target_triangle_count < current_triangle_count:
        mesh = mesh.simplify_quadric_decimation(
            target_number_of_triangles=target_triangle_count
        )
        mesh = _clean_mesh(mesh)

    # get post-processing area and volume
    post_area = mesh.get_surface_area()
    post_volume = mesh.get_volume()

    # check that mesh is manifold and water tight
    is_edge_manifold = mesh.is_edge_manifold()
    is_vertex_manifold = mesh.is_vertex_manifold()
    is_watertight = mesh.is_watertight()

    success = is_edge_manifold and is_vertex_manifold and is_watertight

    metadata = _metadata(
        mesh,
        pre_area,
        post_area,
        pre_volume,
        post_volume,
        shrinkage_tolerance,
        success,
    )

    return mesh, metadata
