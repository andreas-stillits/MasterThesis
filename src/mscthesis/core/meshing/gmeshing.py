from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import gmsh
import numpy as np

from mscthesis.config import ProjectConfig

from ...log import log_call

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


def _scale_and_center(
    entity: list[tuple[int, int]],
    max_iterations: int = 5,
    tolerance: float = 1e-6,
    target_size: float = 1.0,
) -> int:
    """
    Iteratively apply an affine transformation to an entity until the error is below the tolerance
    Args:
        entity (list[tuple[int, int]]): [(dim, tag)]
        max_iterations (int): maximum number of iterations
        tolerance (float): error tolerance
        target_size (float): desired size after transformation
    Returns:
        int: number of iterations performed
    """

    def _transformation(
        center: np.ndarray,
        size: np.ndarray,
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

    count = 0
    for _ in range(max_iterations):
        center, size = _get_bbox(entity)
        current_error = abs(size[2] - target_size)
        if current_error < tolerance:
            break
        transform = _transformation(center, size, target_size)
        kernel.affineTransform(entity, transform)
        kernel.synchronize()
        count += 1
    return count


@dataclass
class Tags:
    AIRSPACE: int = 1
    TOP: int = 2
    BOTTOM: int = 3
    CURVED: int = 4
    INLET: int = 5
    INLET_BOUNDARY: int = 6
    MESOPHYLL: int = 7


def _summary():
    print("GMSH STATE:")
    print("VOLUMES:")
    for vol in kernel.getEntities(dim=3):
        print(vol, f"Mass: {kernel.getMass(*vol):.3f}")
    print("SURFACES:")
    for surf in kernel.getEntities(dim=2):
        print(surf, f"Mass: {kernel.getMass(*surf):.3f}")
    print("CURVES:")
    for curve in kernel.getEntities(dim=1):
        print(curve, f"Length: {kernel.getMass(*curve):.3f}")
    print("END", "\n")


@log_call()
def build_pipe_model(
    plug_aspect: float,
) -> int:
    # INITIALIZATION
    _silent_initialize()
    gmsh.option.setNumber("Geometry.OCCBoundsUseStl", 1)
    gmsh.model.add("Leaf Plug Model")

    # MAKE CYLINDER
    bottom_surface = (0, 0, 0)
    axis = (0, 0, 1)
    airspace_tag = kernel.addCylinder(*bottom_surface, *axis, plug_aspect)
    kernel.synchronize()
    return airspace_tag


@log_call()
def build_sample_model(
    cadmodel_path: str | Path,
) -> int:
    raise NotImplementedError


@log_call()
def mesh_model(
    airspace_tag: int,
    plug_aspect: float,
    stomatal_aspect: float,
    scale_factor: float,
    global_max_num: int,
    edge_min_num: int,
    edge_dist_min: float,
    edge_dist_max: float,
    cell_min: float,
    cell_dist_min: float,
    cell_dist_max: float,
    inlet_min: float,
    inlet_dist_min: float,
    inlet_dist_max: float,
    atol: float,
) -> None:
    # BASELINE
    assert stomatal_aspect < plug_aspect, (
        "Stomatal aspect ratio must be smaller than plug aspect ratio"
    )
    volumes = kernel.getEntities(dim=3)
    assert len(volumes) == 1, "Expected exactly one volume in the model"
    initial_surface_count = len(kernel.getEntities(dim=2))

    # MAKE INLET
    inlet_tag = kernel.addDisk(0, 0, 0, stomatal_aspect, stomatal_aspect)
    kernel.synchronize()

    # IDENTIFY BOTTOM SURFACE TAG
    surfaces = kernel.getEntities(dim=2)
    bottom_tag: int = -1
    for dim, tag in surfaces:
        com = kernel.getCenterOfMass(dim, tag)
        if np.isclose(com[2], 0):
            bottom_tag = tag
            break
    assert bottom_tag != -1, "Failed to identify bottom surface tag"

    # FRAGMENT SO MESH CONFORMS TO INLET INTERFACE
    out_dimtags, _ = kernel.fragment([(3, airspace_tag)], [(2, inlet_tag)])
    kernel.synchronize()

    # ASSERT ONLY (initial_surface_count + 1) SURFACES AND 1 VOLUME
    assert len(kernel.getEntities(dim=2)) == initial_surface_count + 1, (
        f"Expected {initial_surface_count + 1} surfaces after fragmentation"
    )
    volume = kernel.getEntities(dim=3)
    assert len(volume) == 1, "Expected exactly one volume after fragmentation"

    # _summary()

    # IDENTIFY SURFACES BY THEIR AREA (MASS)
    tags: dict[int, list[int] | int] = {}
    tags[Tags.AIRSPACE] = volume[0][1]
    plug_target = np.pi * plug_aspect**2
    inlet_target = np.pi * stomatal_aspect**2
    ring_target = plug_target - inlet_target
    curved_target = 2 * np.pi * plug_aspect * 1.0

    # print(
    #     f"plug_target: {plug_target:.3f}, inlet_target: {inlet_target:.3f}, ring_target: {ring_target:.3f}, curved_target: {curved_target:.3f}"
    # )
    mesophyll_tags: list[int] = []

    def _isclose(x: float, y: float, atol: float = atol) -> bool:
        return abs(x - y) < atol * y

    for dim, tag in kernel.getEntities(dim=2):
        com = kernel.getCenterOfMass(dim, tag)
        mass = kernel.getMass(dim, tag)
        if np.isclose(com[2], 1.0) and _isclose(mass, plug_target):
            tags[Tags.TOP] = tag
        elif np.isclose(com[2], 0.0) and _isclose(mass, inlet_target):
            tags[Tags.INLET] = tag
        elif np.isclose(com[2], 0.0) and _isclose(mass, ring_target):
            tags[Tags.BOTTOM] = tag
        elif _isclose(mass, curved_target):
            tags[Tags.CURVED] = tag
        else:
            mesophyll_tags.append(tag)
    if len(mesophyll_tags) > 0:
        tags[Tags.MESOPHYLL] = mesophyll_tags

    # print(json.dumps(tags, indent=4), "\n")
    # _summary()

    # IDENTIFY INLET INTERFACE
    target_circumference = 2 * np.pi * stomatal_aspect
    inner_boundary = gmsh.model.getBoundary(
        [(2, tags[Tags.INLET])], oriented=False, recursive=False
    )
    outer_boundary = gmsh.model.getBoundary(
        [(2, tags[Tags.BOTTOM])], oriented=False, recursive=False
    )
    inner_curves = {tag for dim, tag in inner_boundary if dim == 1}
    outer_curves = {tag for dim, tag in outer_boundary if dim == 1}

    interface_curves = list(inner_curves.intersection(outer_curves))
    assert len(interface_curves) == 1, (
        "Expected exactly one curve at the inlet interface"
    )
    assert _isclose(kernel.getMass(1, interface_curves[0]), target_circumference), (
        "Expected inner boundary to be target circle"
    )
    tags[Tags.INLET_BOUNDARY] = interface_curves[0]

    # ASSIGN PHYSICAL GROUPS
    gmsh.model.addPhysicalGroup(3, [tags[Tags.AIRSPACE]], name="Airspace")
    gmsh.model.addPhysicalGroup(2, [tags[Tags.TOP]], name="Top")
    gmsh.model.addPhysicalGroup(2, [tags[Tags.BOTTOM]], name="Bottom")
    gmsh.model.addPhysicalGroup(2, [tags[Tags.CURVED]], name="Curved")
    gmsh.model.addPhysicalGroup(2, [tags[Tags.INLET]], name="Inlet")
    gmsh.model.addPhysicalGroup(1, [tags[Tags.INLET_BOUNDARY]], name="InletBoundary")
    if Tags.MESOPHYLL in tags:
        gmsh.model.addPhysicalGroup(2, [tags[Tags.MESOPHYLL]], name="MesophyllCells")

    # MESHING
    global_max_ = 2 * np.pi * plug_aspect / global_max_num * scale_factor
    edge_min_ = 2 * np.pi * plug_aspect / edge_min_num * scale_factor
    cell_min_ = cell_min * scale_factor
    inlet_min_ = inlet_min * scale_factor

    field_list = []
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)

    # add distance field away from the edge:
    edge_distance = field.add("Distance")
    field.setNumbers(
        edge_distance,
        "FacesList",
        [tags[Tags.CURVED], tags[Tags.TOP], tags[Tags.BOTTOM], tags[Tags.INLET]],
    )
    edge_threshold = field.add("Threshold")
    field.setNumber(edge_threshold, "InField", edge_distance)
    field.setNumber(edge_threshold, "LcMin", edge_min_)
    field.setNumber(edge_threshold, "LcMax", global_max_)
    field.setNumber(edge_threshold, "DistMin", edge_dist_min)
    field.setNumber(edge_threshold, "DistMax", edge_dist_max)
    field_list.append(edge_threshold)

    # add distance field away from the inlet boundary:
    inlet_boundary_distance = field.add("Distance")
    field.setNumbers(
        inlet_boundary_distance,
        "CurvesList",
        [tags[Tags.INLET_BOUNDARY]],
    )
    field.setNumber(inlet_boundary_distance, "Sampling", 100)
    inlet_boundary_threshold = field.add("Threshold")
    field.setNumber(inlet_boundary_threshold, "InField", inlet_boundary_distance)
    field.setNumber(inlet_boundary_threshold, "LcMin", inlet_min_)
    field.setNumber(inlet_boundary_threshold, "LcMax", global_max_)
    field.setNumber(inlet_boundary_threshold, "DistMin", inlet_dist_min)
    field.setNumber(inlet_boundary_threshold, "DistMax", inlet_dist_max)
    field_list.append(inlet_boundary_threshold)

    # control distance to mesophyll cell surfaces
    if Tags.MESOPHYLL in tags:
        mesophyll_distance = field.add("Distance")
        field.setNumbers(mesophyll_distance, "FacesList", tags[Tags.MESOPHYLL])
        mesophyll_threshold = field.add("Threshold")
        field.setNumber(mesophyll_threshold, "InField", mesophyll_distance)
        field.setNumber(mesophyll_threshold, "LcMin", cell_min_)
        field.setNumber(mesophyll_threshold, "LcMax", global_max_)
        field.setNumber(mesophyll_threshold, "DistMin", cell_dist_min)
        field.setNumber(mesophyll_threshold, "DistMax", cell_dist_max)
        field_list.append(mesophyll_threshold)

    # combine fields by taking the minimum at each point
    minimum_field = field.add("Min")
    field.setNumbers(
        minimum_field,
        "FieldsList",
        field_list,
    )
    field.setAsBackgroundMesh(minimum_field)
    kernel.synchronize()

    # FINALIZE
    gmsh.model.mesh.generate(3)

    gmsh.write("tmp/test.msh")
    gmsh.fltk.run()

    gmsh.finalize()


def main() -> int:
    print("testing...")

    config = ProjectConfig().meshing
    config_dict = config.model_dump()
    args = {}
    for key, value in config_dict.items():
        if key not in ["boundary_margin", "substomatal_margin"]:
            args[key] = value

    plug_aspect = 0.25
    airspace_tag = build_pipe_model(plug_aspect)
    mesh_model(airspace_tag, plug_aspect, **args)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
