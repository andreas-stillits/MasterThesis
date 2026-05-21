from __future__ import annotations

from typing import Any

import numpy as np
from stillib_random import from_seed

from ...config import ProjectConfig
from ...log import log_call
from ..visualization import visualize_voxels
from .utils import initialize_meshgrid


@log_call()
def _metadata(
    plug_aspect: float,
    num_cells_placed: int,
    voxels: np.ndarray,
) -> dict[str, Any]:
    """
    Generate metadata for a voxel model non-overlapping spheres.

    Args:
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        num_cells_placed (int): Number of cells successfully placed in the model.
        voxels (np.ndarray): The generated voxel model of shape (X, Y, Z).

    Returns:
        dict[str, Any]: Metadata dictionary.
    """
    metadata: dict[str, Any] = {}
    metadata["plug_aspect"] = plug_aspect
    metadata["num_cells_placed"] = num_cells_placed
    metadata["mean_porosity"] = 1.0 - float(np.sum(voxels) / (np.pi / 4 * voxels.size))
    metadata["std_porosity"] = float(
        np.std(1.0 - np.sum(voxels, axis=(0, 1)) / (voxels.shape[0] * voxels.shape[1]))
    )
    metadata["type"] = "contrast"
    return metadata


@log_call()
def generate_voxels_from_rng(
    rng: np.random.Generator,
    resolution: int,
    plug_aspect: float,
    separation: float,
    max_attempts: int,
    num_cells: int,
    radius_min: float,
    radius_max: float,
    division: float,
) -> tuple[np.ndarray[tuple[int, int, int]], dict[str, Any]]:

    voxels, (X, Y, Z) = initialize_meshgrid(plug_aspect, resolution)

    rod_height = 1 - division - 2 * separation
    assert rod_height > 2 * radius_min, (
        "Cells to large for the prescribed division and separation. Not enough space for palisade cells."
    )
    rod_z_center = division + separation + rod_height / 2

    # draw mixed radii spheres for the spongy compartment
    centers_spongy, radii_spongy = np.zeros((num_cells, 3)), np.zeros(num_cells)
    max_r = plug_aspect - radius_min - separation
    min_z = radius_min + separation
    max_z = division - radius_min - separation
    assert max_z > min_z, (
        "Cell size to large for given division and plug aspect. Not enough space for spongy cells."
    )
    assert max_r > 0, (
        "Cell size and separation too large for given plug aspect. No radial space for cells."
    )

    # DRAW SPONGY CELLS
    for i in range(num_cells):
        attempts = 0
        while attempts < max_attempts:
            center = np.array(
                [
                    rng.uniform(-max_r, max_r),
                    rng.uniform(-max_r, max_r),
                    rng.uniform(min_z, max_z),
                ]
            )
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
                np.linalg.norm(centers_spongy[:i] - center, axis=1)
                > (radii_spongy[:i] + radius + separation)
            ):
                centers_spongy[i] = center
                radii_spongy[i] = radius
                break
            attempts += 1
        else:
            break

        # update voxels for spongy compartment

        distance = np.sqrt(
            (X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2
        )
        voxels |= (distance <= radius).astype(np.uint8)

    radii_spongy = radii_spongy[radii_spongy > 0]
    num_cells_placed = len(radii_spongy)

    # DRAW PALISADE CELLS
    centers_palisade = np.zeros((num_cells, 3))
    radii_palisade = np.zeros(num_cells)
    heights_palisade = np.zeros(num_cells)

    for i in range(num_cells):
        attempts = 0
        while attempts < max_attempts:
            center = np.array(
                [rng.uniform(-max_r, max_r), rng.uniform(-max_r, max_r), rod_z_center]
            )

            radius = rng.uniform(radius_min, radius_max)

            # enforce cyllindrical boundary
            if np.linalg.norm(center[:2]) + radius - radius_min > max_r:
                attempts += 1
                continue

            # check for overlaps
            if np.all(
                np.linalg.norm(centers_palisade - center, axis=1)
                > (radii_palisade + radius + separation)
            ):
                centers_palisade[i] = center
                radii_palisade[i] = radius
                heights_palisade[i] = rod_height - 2 * radius
                break

        else:
            break

        # update voxels for palisade compartment
        mask = (
            (
                ((X - center[0]) ** 2 + (Y - center[1]) ** 2 <= radius**2)
                & (center[2] - rod_height / 2 + radius <= Z)
                & (center[2] + rod_height / 2 - radius >= Z)
            )
            | (
                (X - center[0]) ** 2
                + (Y - center[1]) ** 2
                + (Z - center[2] - rod_height / 2 + radius) ** 2
                < radius**2
            )
            | (
                (X - center[0]) ** 2
                + (Y - center[1]) ** 2
                + (Z - center[2] + rod_height / 2 - radius) ** 2
                < radius**2
            )
        )
        voxels |= mask.astype(np.uint8)

    radii_palisade = radii_palisade[radii_palisade > 0]
    num_cells_placed += len(radii_palisade)

    metadata = _metadata(
        plug_aspect,
        num_cells_placed,
        voxels,
    )

    return voxels, metadata


def main() -> int:

    config = ProjectConfig()

    rng = from_seed(config.synthesis.base_seed).generator()

    voxels, metadata = generate_voxels_from_rng(
        rng,
        config.synthesis.resolution,
        config.synthesis.plug_aspect,
        config.synthesis.separation,
        config.synthesis.max_attempts,
        config.synthesis.contrast.num_cells,
        config.synthesis.contrast.radius_min,
        config.synthesis.contrast.radius_max,
        config.synthesis.contrast.division,
    )

    porosity = 1 - np.sum(voxels) / voxels.size / (np.pi / 4)
    print(f"Porosity: {porosity:.3f}")
    visualize_voxels(voxels)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
