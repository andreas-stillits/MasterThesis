from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, cast

import gmsh
import numpy as np

from ...utilities.log import log_call

# set namespace
kernel = gmsh.model.occ
field = gmsh.model.mesh.field

# monkey patch silent initialization
_original_initialize = gmsh.initialize


def _silent_initialize(*args, **kwargs) -> None:
    """Initialize GMSH without printing to stdout"""
    _original_initialize(*args, **kwargs)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)


def _metadata(
    plug_aspect: float,
    mesophyll_area: float,
) -> dict[str, Any]:
    """Collect relevant metadata from the meshing process to be stored with the command execution record"""
    plug_area = np.pi * plug_aspect**2
    mesophyll_area_fraction = mesophyll_area / plug_area

    return {
        "plug_aspect": plug_aspect,
        "plug_area": plug_area,
        "mesophyll_area": mesophyll_area,
        "mesophyll_area_fraction": mesophyll_area_fraction,
    }


def _iterative_affine_transformation(
    entity: list[tuple[int, int]],
    transformation: Callable,
    error: Callable,
    max_iterations: int = 5,
    tolerance: float = 1e-6,
    target_size: float = 1.0,
) -> int:
    """
    Iteratively apply an affine transformation to an entity until the error is below the tolerance
    Args:
        entity (list[tuple[int, int]]): [(dim, tag)]
        transformation (Callable): function that takes center, size, target_size and returns a 4x4 affine transformation matrix
        error (Callable): function that takes center, size, target_size and returns the error
        max_iterations (int): maximum number of iterations
        tolerance (float): error tolerance
        target_size (float): desired size after transformation
    Returns:
        int: number of iterations performed
    """
    count = 0
    for _ in range(max_iterations):
        center, size = _get_bbox(entity)
        current_error = abs(error(center, size, target_size))
        if current_error < tolerance:
            break
        transform = transformation(center, size, target_size)
        kernel.affineTransform(entity, transform)
        kernel.synchronize()
        count += 1
    return count


def _get_bbox(entity: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    """
    Get bounding box of a given entity
    Args:
        entity (list[tuple[int, int]]): [(dim, tag)]
    Returns:
        tuple[np.ndarray, np.ndarray]: center (3,) and size (3,) of the bounding box
    """
    xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(*entity[0])
    bbox_min = np.array([xmin, ymin, zmin])
    bbox_max = np.array([xmax, ymax, zmax])
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = bbox_max - bbox_min
    return bbox_center, bbox_size


@log_call()
def build_gmsh_model(
    entities: list[tuple[int, int]],
    boundary_margin_fraction: float,
    substomatal_cavity_margin_fraction: float,
) -> list[tuple[int, int]]:
    """
    Build the gmsh model from imported entities.
    Args:
        entities (list[tuple[int, int]]): List of (dim, tag) tuples
        boundary_margin_fraction (float): Margin fraction for minimal distance to plug boundary
        substomatal_cavity_margin_fraction (float): Margin fraction for minimal distance to the stomatal surface
    Returns:
        list[tuple[int, int]]: List of (dim, tag) tuples representing the airspace entity
    """
    # ====== Identify appropriate cylinder plug dimensions ======

    # shift to center at origin
    center, size = _get_bbox(entities)
    kernel.translate(entities, -center[0], -center[1], -center[2])
    kernel.synchronize()

    # perform 2D meshing and extract the point furthest away from origin in xy-plane
    gmsh.model.mesh.generate(2)
    node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
    node_coords = np.array(node_coords).reshape(-1, 3)
    distances = np.linalg.norm(node_coords[:, :2], axis=1)
    max_distance = np.max(distances)

    # calculate cylinder geometry
    center, size = _get_bbox(entities)

    bottom_z = center[2] - size[2] * (
        0.5 + substomatal_cavity_margin_fraction
    )  # z-coordinate of the bottom cylinder surface

    height = size[2] * (
        1 + substomatal_cavity_margin_fraction + boundary_margin_fraction
    )

    # determine the appropriate dimensions for the cylinder plug
    bottom_surface = (center[0], center[1], bottom_z)
    axis = (0, 0, height)
    radius = (1 + boundary_margin_fraction) * max_distance

    # create the cylinder plug
    cylinder = [(3, kernel.addCylinder(*bottom_surface, *axis, radius))]
    kernel.synchronize()

    # perform boolean cut to create airspace
    airspace, _ = kernel.cut(cylinder, entities, removeObject=True, removeTool=True)
    kernel.synchronize()

    # Retain only the largest volume as airspace
    volumes = gmsh.model.getEntities(dim=3)
    largest_volume = 0
    largest_volume_tag = None
    # identify largest volume
    for dim, tag in volumes:
        mass = kernel.getMass(dim, tag)
        if mass > largest_volume:
            largest_volume = mass
            largest_volume_tag = tag
    # remove all other volumes
    for dim, tag in volumes:
        if tag != largest_volume_tag:
            kernel.remove(
                [(dim, tag)]
            )  # recurvsive=True will remove all lower dimensional entities shared at the boundary

    kernel.synchronize()
    largest_volume_tag = cast(int, largest_volume_tag)
    airspace = [(3, largest_volume_tag)]

    # Iteratively apply affine transformation to airspace to center bottom surface at origin and scale height to 1
    def _transformation(
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        target_size: float,
    ) -> list[float]:
        """Generate affine transformation matrix to scale and translate entity"""
        scale = target_size / size[2]
        return [
            scale,
            0,
            0,
            -center[0] * scale,
            0,
            scale,
            0,
            -center[1] * scale,
            0,
            0,
            scale,
            -(center[2] - size[2] / 2) * scale,
            0,
            0,
            0,
            1,
        ]

    def _error(
        center: tuple[float, float, float],
        size: tuple[float, float, float],
        target_size: float,
    ) -> float:
        """Calculate relative error in height"""
        return (size[2] - target_size) / target_size

    _ = _iterative_affine_transformation(
        airspace,
        _transformation,
        _error,
        max_iterations=5,
        tolerance=1e-6,
        target_size=1.0,
    )

    return airspace


@log_call()
def assign_physical_groups(
    airspace: list[tuple[int, int]],
    tolerance: float,
) -> tuple[dict[str, Any], float, float]:
    """
    Assign physical groups: airspace volume, top surface, bottom surface, curved surface, mesophyll surfaces
    Args:
        airspace (list[tuple[int, int]]): List of (dim, tag) tuples representing the airspace entity
        tolerance (float): Tolerance for relative difference from expected area when identifying curved face
    Returns:
        tuple[dict[str, list[int] | int], float, float]: Tuple containing dictionary with tags for physical groups, the plug aspect ratio, and the mesophyll surface area
    """
    # determine curved face tag
    # OBS: this approach of identification by area only works if the curved area 2 pi r is unique up to tolerace
    # However, top and bottom surfaces will always be distinctly caught by the COM z-coordinate check below
    center, size = _get_bbox(airspace)
    # calculate target curved area from cylinder dimensions (elliptical cross-section due to possible slight asymmetry in transform)
    a = size[0] / 2
    b = size[1] / 2
    curved_area_target = np.pi * (
        3 * (a + b) - np.sqrt((3 * a + b) * (a + 3 * b))
    )  # approximation of ellipse circumference to account for slight transform assymetry

    curved_area_found = []
    curved_area_tag = None
    top_area_tag = None
    bottom_area_tag = None

    def _iscurved(tag: int) -> bool:
        area = kernel.getMass(2, tag)
        trigger = abs(area / curved_area_target - 1) <= tolerance
        if trigger:
            curved_area_found.append(area)
        return trigger

    # airspace
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in airspace], 1, name="airspace")

    # ====== surfaces ======
    # get all surfaces
    surfaces = gmsh.model.getEntities(dim=2)

    mesophyll_surface_tags = []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[2], 1.0):
            # top surface
            gmsh.model.addPhysicalGroup(2, [tag], 2, name="top_surface")
            top_area_tag = tag
        elif np.isclose(com[2], 0.0):
            # bottom surface
            gmsh.model.addPhysicalGroup(2, [tag], 3, name="bottom_surface")
            bottom_area_tag = tag
        elif _iscurved(tag):
            # curved surface of cylinder
            gmsh.model.addPhysicalGroup(2, [tag], 4, name="curved_surface")
            curved_area_tag = tag
        else:
            # other surfaces
            mesophyll_surface_tags.append(tag)
    gmsh.model.addPhysicalGroup(2, mesophyll_surface_tags, 5, name="mesophyll_surfaces")

    assert (
        len(curved_area_found) == 1
    ), f"Error identifying curved face of cylinder. Found {len(curved_area_found)} curved faces with relative errors from target: {[area/curved_area_target - 1 for area in curved_area_found]}"

    assert (
        top_area_tag is not None and bottom_area_tag is not None
    ), "Error identifying top or bottom surface of cylinder"

    tags = {
        "mesophyll_surface_tags": mesophyll_surface_tags,
        "curved_area_tag": curved_area_tag,
        "top_area_tag": top_area_tag,
        "bottom_area_tag": bottom_area_tag,
    }

    plug_aspect = float((a + b) / 2)
    mesophyll_area = 0
    for tag in mesophyll_surface_tags:
        mesophyll_area += kernel.getMass(2, tag)

    return tags, plug_aspect, mesophyll_area


@log_call()
def configure_meshfield(
    tags: dict[str, Any],
    plug_aspect: float,
    global_resolution_factor: float,
    min_stomatal_feature: float,
    max_stomatal_feature: float,
    min_cellular_feature: float,
    max_stomatal_dist_factor: float,
    min_cellular_dist_factor: float,
    max_cellular_dist_factor: float,
    min_boundary_dist_factor: float,
    max_boundary_dist_factor: float,
    min_points_boundary: int,
    max_points_boundary: int,
) -> None:
    """
    Configure the mesh size field in gmsh.
    Args:
        tags (dict[str, list[int] | int]): Dictionary containing tags for physical groups
        plug_aspect (float): Aspect ratio of the plug (radius/height)
        global_resolution_factor (float): Factor to adjust global resolution
        min_stomatal_feature (float): Minimum feature size for stomata
        max_stomatal_feature (float): Maximum feature size for stomata
        min_cellular_feature (float): Minimum feature size for mesophyll cells
        min_stomatal_dist_factor (float): Factor to adjust minimum distance for stomatal mesh
        max_stomatal_dist_factor (float): Factor to adjust maximum distance for stomatal mesh
        min_cellular_dist_factor (float): Factor to adjust minimum distance for cellular mesh
        max_cellular_dist_factor (float): Factor to adjust maximum distance for cellular mesh
        min_boundary_dist_factor (float): Factor to adjust minimum distance for boundary mesh
        max_boundary_dist_factor (float): Factor to adjust maximum distance for boundary mesh
        min_points_boundary (int): Minimum number of points on boundary
        max_points_boundary (int): Maximum number of points on boundary
    """
    # Calculate resolution and distance parameters based on the provided factors and the size of the plug
    min_stomatal_res = min_stomatal_feature * global_resolution_factor
    min_cellular_res = min_cellular_feature * global_resolution_factor
    min_boundary_res = (
        2 * np.pi * plug_aspect / max_points_boundary * global_resolution_factor
    )
    max_global_res = (
        2 * np.pi * plug_aspect / min_points_boundary * global_resolution_factor
    )
    #
    max_stomatal_dist = max_stomatal_feature * max_stomatal_dist_factor
    min_cellular_dist = min_cellular_feature * min_cellular_dist_factor
    max_cellular_dist = min_cellular_feature * max_cellular_dist_factor
    min_boundary_dist = (
        min_boundary_res * min_boundary_dist_factor / global_resolution_factor
    )
    max_boundary_dist = (
        min_boundary_res * max_boundary_dist_factor / global_resolution_factor
    )
    # add a distance field away from a point source at 0,0,0
    point_tag = kernel.addPoint(0, 0, 0)
    point_distance = field.add("Distance")
    field.setNumbers(point_distance, "PointsList", [point_tag])
    point_threshold = field.add("Threshold")
    field.setNumber(point_threshold, "InField", point_distance)
    field.setNumber(point_threshold, "LcMin", min_stomatal_res)
    field.setNumber(point_threshold, "LcMax", max_global_res)
    field.setNumber(point_threshold, "DistMin", max_stomatal_feature)
    field.setNumber(point_threshold, "DistMax", max_stomatal_dist)

    # # control distance to bottom inlet surface
    # inlet_distance = field.add("Distance")
    # field.setNumbers(inlet_distance, "FacesList", [tags["bottom_area_tag"]])
    # inlet_threshold = field.add("Threshold")
    # field.setNumber(inlet_threshold, "InField", inlet_distance)
    # field.setNumber(inlet_threshold, "LcMin", min_stomatal_res)
    # field.setNumber(inlet_threshold, "LcMax", max_global_res)
    # field.setNumber(inlet_threshold, "DistMin", min_stomatal_dist)
    # field.setNumber(inlet_threshold, "DistMax", max_stomatal_dist)
    #
    # control distance to mesophyll cell surfaces
    mesophyll_distance = field.add("Distance")
    field.setNumbers(mesophyll_distance, "FacesList", tags["mesophyll_surface_tags"])
    mesophyll_threshold = field.add("Threshold")
    field.setNumber(mesophyll_threshold, "InField", mesophyll_distance)
    field.setNumber(mesophyll_threshold, "LcMin", min_cellular_res)
    field.setNumber(mesophyll_threshold, "LcMax", max_global_res)
    field.setNumber(mesophyll_threshold, "DistMin", min_cellular_dist)
    field.setNumber(mesophyll_threshold, "DistMax", max_cellular_dist)
    #
    # control distance to plug boundary
    boundary_distance = field.add("Distance")
    field.setNumbers(
        boundary_distance,
        "FacesList",
        [tags["curved_area_tag"], tags["top_area_tag"], tags["bottom_area_tag"]],
    )
    boundary_threshold = field.add("Threshold")
    field.setNumber(boundary_threshold, "InField", boundary_distance)
    field.setNumber(boundary_threshold, "LcMin", min_boundary_res)
    field.setNumber(boundary_threshold, "LcMax", max_global_res)
    field.setNumber(boundary_threshold, "DistMin", min_boundary_dist)
    field.setNumber(boundary_threshold, "DistMax", max_boundary_dist)
    #
    minimum_field = field.add("Min")
    field.setNumbers(
        minimum_field,
        "FieldsList",
        [point_threshold, mesophyll_threshold, boundary_threshold],
    )
    field.setAsBackgroundMesh(minimum_field)
    kernel.synchronize()
    return


@log_call()
def run_gmsh_session(
    brep_file: str | Path,
    output_mesh_file: str | Path,
    global_resolution_factor: float,
    min_stomatal_feature: float,
    max_stomatal_feature: float,
    min_cellular_feature: float,
    max_stomatal_dist_factor: float,
    min_cellular_dist_factor: float,
    max_cellular_dist_factor: float,
    min_boundary_dist_factor: float,
    max_boundary_dist_factor: float,
    min_points_boundary: int,
    max_points_boundary: int,
    boundary_margin_fraction: float,
    substomatal_cavity_margin_fraction: float,
    tolerance: float,
) -> dict[str, Any]:
    """
    Run the gmsh meshing session.
    Args:
        brep_file (str | Path): Path to the input BREP file.
        output_mesh_file (str | Path): Path to the output mesh file.
        global_resolution_factor (float): Factor to adjust global resolution.
        cell_resolution_factor (float): Factor to adjust cell resolution.
        minimum_stomatal_aspect (float): Minimum aspect ratio of the stomatal cavity to be resolved.
        minimum_distance_factor (float): Factor to adjust minimum distance for mesh refinement.
        maximum_distance_factor (float): Factor to adjust maximum distance for mesh refinement.
        boundary_margin_fraction (float): Margin fraction for minimal distance to plug boundary.
        substomatal_cavity_margin_fraction (float): Margin fraction for minimal distance to the stomatal surface.
        tolerance (float): Tolerance for identifying curved face.
    Returns:
        dict[str, Any]: Metadata dictionary containing relevant information about the meshing process and the resulting mesh.
    """
    _silent_initialize()
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    gmsh.model.add("Leaf Plug Model")
    entities = kernel.importShapes(str(brep_file))
    kernel.synchronize()

    airspace = build_gmsh_model(
        entities,
        boundary_margin_fraction,
        substomatal_cavity_margin_fraction,
    )

    tags, plug_aspect, mesophyll_area = assign_physical_groups(airspace, tolerance)

    configure_meshfield(
        tags,
        plug_aspect,
        global_resolution_factor,
        min_stomatal_feature,
        max_stomatal_feature,
        min_cellular_feature,
        max_stomatal_dist_factor,
        min_cellular_dist_factor,
        max_cellular_dist_factor,
        min_boundary_dist_factor,
        max_boundary_dist_factor,
        min_points_boundary,
        max_points_boundary,
    )

    gmsh.model.mesh.generate(3)
    gmsh.write(str(output_mesh_file))

    return _metadata(plug_aspect, mesophyll_area)


@log_call()
def build_cylinder_model(
    output_mesh_file: str | Path,
    plug_aspect: float,
    global_resolution_factor: float,
    min_stomatal_feature: float,
    max_stomatal_feature: float,
    max_stomatal_dist_factor: float,
    min_boundary_dist_factor: float,
    max_boundary_dist_factor: float,
    min_points_boundary: int,
    max_points_boundary: int,
    tolerance: float,
) -> dict[str, Any]:
    """
    Build a simple cylinder model in gmsh with the given aspect ratio.
    Args:
        output_mesh_file (str | Path): Path to the output mesh file
        plug_aspect (float): Aspect ratio of the cylinder (radius/height)
        global_resolution_factor (float): Factor for global mesh resolution
        min_stomatal_feature (float): Minimum feature size for stomata
        min_stomatal_dist_factor (float): Factor for minimum stomatal distance
        max_stomatal_dist_factor (float): Factor for maximum stomatal distance
        min_boundary_dist_factor (float): Factor for minimum boundary distance
        max_boundary_dist_factor (float): Factor for maximum boundary distance
        min_points_boundary (int): Minimum number of points on boundary
        max_points_boundary (int): Maximum number of points on boundary
        tolerance (float): Tolerance for surface identification
    Returns:
        dict[str, Any]: metadata
    """
    _silent_initialize()
    gmsh.model.add("Ideal cylinder Model")
    kernel.synchronize()
    # construct cylinder
    height = 1.0
    radius = plug_aspect * height
    bottom_surface = (0, 0, 0)
    axis = (0, 0, height)
    cylinder = [(3, kernel.addCylinder(*bottom_surface, *axis, radius))]
    kernel.synchronize()

    # assign physical groups
    # determine curved face tag
    # OBS: this approach of identification by area only works if the curved area 2 pi r is unique up to tolerace
    # However, top and bottom surfaces will always be distinctly caught by the COM z-coordinate check below
    # calculate target curved area from cylinder dimensions
    curved_area_target = 2 * np.pi * radius

    curved_area_found = []
    curved_area_tag = None
    top_area_tag = None
    bottom_area_tag = None

    def _iscurved(tag: int) -> bool:
        area = kernel.getMass(2, tag)
        trigger = abs(area / curved_area_target - 1) <= tolerance
        if trigger:
            curved_area_found.append(area)
        return trigger

    # airspace
    gmsh.model.addPhysicalGroup(3, [tag for dim, tag in cylinder], 1, name="airspace")

    # ====== surfaces ======
    # get all surfaces
    surfaces = gmsh.model.getEntities(dim=2)

    mesophyll_surface_tags = []
    for dim, tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(dim, tag)
        if np.isclose(com[2], 1.0):
            # top surface
            gmsh.model.addPhysicalGroup(2, [tag], 2, name="top_surface")
            top_area_tag = tag
        elif np.isclose(com[2], 0.0):
            # bottom surface
            gmsh.model.addPhysicalGroup(2, [tag], 3, name="bottom_surface")
            bottom_area_tag = tag
        elif _iscurved(tag):
            # curved surface of cylinder
            gmsh.model.addPhysicalGroup(2, [tag], 4, name="curved_surface")
            curved_area_tag = tag
        else:
            # other surfaces
            mesophyll_surface_tags.append(tag)

    assert (
        len(mesophyll_surface_tags) == 0
    ), f"Error identifying surfaces. Found unexpected additional surfaces with tags: {mesophyll_surface_tags}"

    assert (
        len(curved_area_found) == 1
    ), f"Error identifying curved face of cylinder. Found {len(curved_area_found)} curved faces with relative errors from target: {[area/curved_area_target - 1 for area in curved_area_found]}"

    assert (
        top_area_tag is not None and bottom_area_tag is not None
    ), "Error identifying top or bottom surface of cylinder"

    tags = {
        "mesophyll_surface_tags": mesophyll_surface_tags,
        "curved_area_tag": curved_area_tag,
        "top_area_tag": top_area_tag,
        "bottom_area_tag": bottom_area_tag,
    }
    kernel.synchronize()

    # mesh and write to file
    # Calculate resolution and distance parameters based on the provided factors and the size of the plug
    min_stomatal_res = min_stomatal_feature * global_resolution_factor
    min_boundary_res = (
        2 * np.pi * plug_aspect / max_points_boundary * global_resolution_factor
    )
    max_global_res = (
        2 * np.pi * plug_aspect / min_points_boundary * global_resolution_factor
    )
    #
    max_stomatal_dist = max_stomatal_feature * max_stomatal_dist_factor
    min_boundary_dist = (
        min_boundary_res * min_boundary_dist_factor / global_resolution_factor
    )
    max_boundary_dist = (
        min_boundary_res * max_boundary_dist_factor / global_resolution_factor
    )

    # control distance to bottom inlet surface
    inlet_distance = field.add("Distance")
    field.setNumbers(inlet_distance, "FacesList", [tags["bottom_area_tag"]])
    inlet_threshold = field.add("Threshold")
    field.setNumber(inlet_threshold, "InField", inlet_distance)
    field.setNumber(inlet_threshold, "LcMin", min_stomatal_res)
    field.setNumber(inlet_threshold, "LcMax", max_global_res)
    field.setNumber(inlet_threshold, "DistMin", max_stomatal_feature)
    field.setNumber(inlet_threshold, "DistMax", max_stomatal_dist)
    #
    # control distance to plug boundary
    boundary_distance = field.add("Distance")
    field.setNumbers(boundary_distance, "FacesList", [tags["curved_area_tag"]])
    boundary_threshold = field.add("Threshold")
    field.setNumber(boundary_threshold, "InField", boundary_distance)
    field.setNumber(boundary_threshold, "LcMin", min_boundary_res)
    field.setNumber(boundary_threshold, "LcMax", max_global_res)
    field.setNumber(boundary_threshold, "DistMin", min_boundary_dist)
    field.setNumber(boundary_threshold, "DistMax", max_boundary_dist)
    #
    minimum_field = field.add("Min")
    field.setNumbers(
        minimum_field,
        "FieldsList",
        [inlet_threshold, boundary_threshold],
    )
    field.setAsBackgroundMesh(minimum_field)
    kernel.synchronize()

    gmsh.model.mesh.generate(3)
    gmsh.write(str(output_mesh_file))

    return {}
