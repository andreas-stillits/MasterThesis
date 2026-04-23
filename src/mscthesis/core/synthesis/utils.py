from __future__ import annotations

import numpy as np

from ...log import log_call


@log_call()
def get_sample_seed(base_seed: int, sample_id: str) -> int:
    """
    Get a unique deterministic seed for a given sample ID based on a base seed.

    Args:
        base_seed (int): The base seed for random number generation.
        sample_id (str): The unique identifier for the sample.

    Returns:
        int: A unique seed for the sample.
    """
    return base_seed + 17 * 31 * 53 * int(sample_id)


@log_call()
def initialize_meshgrid(
    plug_aspect: float,
    resolution: int,
) -> tuple[np.ndarray[tuple[int, int, int]], tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Initialize a 3D meshgrid for voxel generation.

    Args:
        plug_aspect (float): Ratio of plug radius to plug thickness/height.
        resolution (int): Number of voxels along the z axis (axial resolution).

    Returns:
        A tuple containing the empty voxel grid (np.ndarray[tuple[int, int, int]])
        and the meshgrid arrays (X, Y, Z) (tuple[np.ndarray, np.ndarray, np.ndarray]).
    """
    planar_resolution = int(2 * plug_aspect * resolution)
    x = np.linspace(-plug_aspect, plug_aspect, planar_resolution)
    y = np.linspace(-plug_aspect, plug_aspect, planar_resolution)
    z = np.linspace(0, 1, resolution)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # initialize empty voxels
    voxels = np.zeros(
        (planar_resolution, planar_resolution, resolution), dtype=np.uint8
    )

    return voxels, (X, Y, Z)
