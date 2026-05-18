from __future__ import annotations

from typing import Any

import numpy as np
from stillib_random import from_seed

from ...log import log_call
from ..geo import compute_geodesics
from .utils import (
    get_sample_seed,
    initialize_meshgrid,
)


@log_call()
def _metadata(
    plug_aspect: float,
    centers: np.ndarray,
    radii: np.ndarray,
    voxels: np.ndarray,
) -> dict[str, Any]:
    """
    Generate metadata for a voxel model non-overlapping spheres.

    Args:
        random_seed (int): Seed used for random number generation.
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        centers (np.ndarray): Array of cell centers of shape (N, 3).
        radii (np.ndarray): Array of cell radii of shape (N,).
        voxels (np.ndarray): The generated voxel model of shape (X, Y, Z).

    Returns:
        dict[str, Any]: Metadata dictionary.
    """
    metadata: dict[str, Any] = {}
    metadata["plug_aspect"] = plug_aspect
    metadata["num_cells_placed"] = len(centers)
    metadata["min_radius"] = float(np.min(radii)) if len(radii) > 0 else 0.0
    metadata["mean_radius"] = float(np.mean(radii)) if len(radii) > 0 else 0.0
    metadata["max_radius"] = float(np.max(radii)) if len(radii) > 0 else 0.0
    metadata["mean_cell_volume"] = (
        float(np.mean([(4 / 3) * np.pi * r**3 for r in radii]))  # type: ignore
        if len(radii) > 0
        else 0.0
    )
    metadata["mean_porosity"] = 1.0 - float(np.sum(voxels) / (np.pi / 4 * voxels.size))
    metadata["std_porosity"] = float(
        np.std(1.0 - np.sum(voxels, axis=(0, 1)) / (voxels.shape[0] * voxels.shape[1]))
    )
    metadata["mesophyll_area_fraction"] = float(
        len(centers) * 4 * np.sum(radii**2) / plug_aspect**2
    )
    metadata["type"] = "metaballs"
    return metadata


def _distance_field(
    X: np.ndarray,
    Y: np.ndarray,
    Z: np.ndarray,
    center: np.ndarray,
    radius: float,
    factor: float,
    power: float,
) -> np.ndarray:
    dist = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
    field = radius / (dist + 1e-8)  # add small epsilon to avoid division by zero
    field = field**power
    mask = dist > factor * radius
    field[mask] = 0.0
    return field


@log_call()
def generate_voxels_from_seed(
    random_seed: int,
    resolution: int,
    plug_aspect: float,
    separation: float,
    max_attempts: int,
    num_cells: int,
    radius_min: float,
    radius_max: float,
    factor: float,
    power: float,
    threshold: float,
) -> tuple[np.ndarray[tuple[int, int, int]], dict[str, Any]]:
    """
    Generate uniform swiss cheese voxel model for a given random seed.

    Args:
        random_seed (int): Seed for random number generation.
        resolution (int): Number of voxels along each axis.
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        separation (float): Minimum separation distance between cells and boundaries.
        max_attempts (int): Maximum number of attempts to place each cell.
        num_cells (int): Number of cells (spheres) to place in the model.
        radius_min (float): Minimum radius of the cells (spheres).
        radius_max (float): Maximum radius of the cells (spheres).
        factor (float): Factor for scaling the metaball field.
        power (float): Power for raising the metaball field.
        threshold (float): Threshold for binarizing the voxel grid.
    Returns:
        np.ndarray: 3D numpy array of shape (planar_resolution, planar_resolution, resolution).
        dict[str, Any]: Metadata dictionary.
    """

    rng = from_seed(random_seed).generator()

    field, (X, Y, Z) = initialize_meshgrid(plug_aspect, resolution, dtype="float32")

    # initialize cell lists and determine placement boundaries
    centers = np.zeros((num_cells, 3))
    radii = np.zeros((num_cells,))
    max_r = plug_aspect - radius_min - separation
    min_z = radius_min + separation
    max_z = 1 - radius_min - separation

    if max_r <= 0:
        raise ValueError(
            f"Cell size {radius_min} and separation {separation} too large for given plug aspect {plug_aspect}."
        )

    # placement of cells
    for i in range(num_cells):
        attempts = 0
        while attempts < max_attempts:
            # draw random cell center
            attempts += 1
            center = np.array(
                [
                    rng.uniform(-max_r, max_r),
                    rng.uniform(-max_r, max_r),
                    rng.uniform(min_z, max_z),
                ]
            )

            # draw random cell radius
            radius = rng.uniform(radius_min, radius_max)

            # enforce cyllindrical boundary
            if np.linalg.norm(center[:2]) + radius - radius_min > max_r:
                attempts += 1
                continue
            if (
                center[2] + radius - radius_min > max_z
                or center[2] - radius + radius_min < min_z
            ):
                attempts += 1
                continue

            centers[i] = center
            radii[i] = radius
            field += _distance_field(X, Y, Z, center, radius, factor, power)
            break

        else:  # executed only if while loop is not stopped by break - then we dont attempt to place any further spheres
            break  # break out of the for loop

    # threshold the voxels to binary values
    voxels = (field > threshold).astype(np.uint8)

    # remove cells where radius is still zero (not placed)
    centers = centers[radii > 0]
    radii = radii[radii > 0]

    metadata = _metadata(
        plug_aspect,
        centers,
        radii,
        voxels,
    )

    return voxels, metadata


@log_call()
def generate_voxels_from_sample_id(
    sample_id: str,
    base_seed: int,
    resolution: int,
    plug_aspect: float,
    separation: float,
    max_attempts: int,
    num_cells: int,
    radius_min: float,
    radius_max: float,
    factor: float,
    power: float,
    threshold: float,
) -> tuple[np.ndarray[tuple[int, int, int]], dict[str, Any]]:
    """
    Generate uniform swiss cheese voxel models for a given sample ID.

    Args:
        sample_id (str): Unique identifier for the sample. Mappable to int.
        base_seed (int): Base seed for random number generation.
        resolution (int): Number of voxels along each axis.
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        separation (float): Minimum separation distance between cells and boundaries.
        max_attempts (int): Maximum attempts to place each cell without overlap.
        num_cells (int): Number of cells (spheres) to place in the model.
        radius_min (float): Minimum radius of the cells.
        radius_max (float): Maximum radius of the cells.
        factor (float): Factor for scaling the metaball field.
        power (float): Power for raising the metaball field.
        threshold (float): Threshold for binarizing the voxel grid.
    Returns:
        np.ndarray: 3D numpy array of shape (planar_resolution, planar_resolution, resolution)
        with uint8 values, where 1 indicates presence of tissue (cell)
        and 0 indicates airspace.
        dict[str, Any]: Metadata dictionary.
    """

    # calulate and fix sample seed
    random_seed = get_sample_seed(base_seed, sample_id)

    voxels, metadata = generate_voxels_from_seed(
        random_seed,
        resolution,
        plug_aspect,
        separation,
        max_attempts,
        num_cells,
        radius_min,
        radius_max,
        factor,
        power,
        threshold,
    )

    return voxels, metadata


# ================================================================================
#                              SEARCH FUNCTIONALITY
# ================================================================================


def generate_voxels_from_rng(
    rng: np.random.Generator,
    resolution: int,
    plug_aspect: float,
    separation: float,
    max_attempts: int,
    num_cells: int,
    radius_min: float,
    radius_max: float,
    factor: float,
    power: float,
    threshold: float,
) -> tuple[np.ndarray[tuple[int, int, int]], dict[str, Any]]:
    """
    Generate uniform swiss cheese voxel model for a given random seed.

    Args:
        rng (np.random.Generator): Random number generator instance.
        resolution (int): Number of voxels along each axis.
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        separation (float): Minimum separation distance between cells and boundaries.
        max_attempts (int): Maximum number of attempts to place each cell.
        num_cells (int): Number of cells (spheres) to place in the model.
        radius_min (float): Minimum radius of the cells (spheres).
        radius_max (float): Maximum radius of the cells (spheres).
        factor (float): Factor for scaling the metaball field.
        power (float): Power for raising the metaball field.
        threshold (float): Threshold for binarizing the voxel grid.
    Returns:
        np.ndarray: 3D numpy array of shape (planar_resolution, planar_resolution, resolution).
        dict[str, Any]: Metadata dictionary.
    """

    field, (X, Y, Z) = initialize_meshgrid(plug_aspect, resolution, dtype="float32")

    # initialize cell lists and determine placement boundaries
    centers = np.zeros((num_cells, 3))
    radii = np.zeros((num_cells,))
    max_r = plug_aspect - radius_min - separation
    min_z = radius_min + separation
    max_z = 1 - radius_min - separation

    if max_r <= 0:
        raise ValueError(
            f"Cell size {radius_min} and separation {separation} too large for given plug aspect {plug_aspect}."
        )

    # placement of cells
    for i in range(num_cells):
        attempts = 0
        while attempts < max_attempts:
            # draw random cell center
            attempts += 1
            center = np.array(
                [
                    rng.uniform(-max_r, max_r),
                    rng.uniform(-max_r, max_r),
                    rng.uniform(min_z, max_z),
                ]
            )

            # draw random cell radius
            radius = rng.uniform(radius_min, radius_max)

            # enforce cyllindrical boundary
            if np.linalg.norm(center[:2]) + radius - radius_min > max_r:
                attempts += 1
                continue
            if (
                center[2] + radius - radius_min > max_z
                or center[2] - radius + radius_min < min_z
            ):
                attempts += 1
                continue

            # check for overlaps
            if np.all(
                np.linalg.norm(centers[:i] - center, axis=1)
                > (radii[:i] + radius + separation)
            ):
                centers[i] = center
                radii[i] = radius
                field += _distance_field(X, Y, Z, center, radius, factor, power)
                break

        else:  # executed only if while loop is not stopped by break - then we dont attempt to place any further spheres
            break  # break out of the for loop

    # threshold the voxels to binary values
    voxels = (field > threshold).astype(np.uint8)

    # remove cells where radius is still zero (not placed)
    centers = centers[radii > 0]
    radii = radii[radii > 0]

    # mask unconnected airspaces as solid

    geo, _, _ = compute_geodesics(voxels)

    padded_geo = np.full(voxels.size, np.inf)
    padded_geo[(voxels.ravel() == 0)] = geo
    mask = np.isinf(padded_geo).reshape(voxels.shape)
    voxels[mask] = 1  # mark as solid

    geo, _, _ = compute_geodesics(voxels)
    assert not np.any(np.isinf(geo)), "Unable to remove unconnected airspaces"

    porosity = 1.0 - float(np.sum(voxels) / (np.pi / 4 * voxels.size))
    assert porosity > 0.20, "Porosity below 20%, likely degenerate geometry"

    metadata = _metadata(
        plug_aspect,
        centers,
        radii,
        voxels,
    )

    return voxels, metadata
