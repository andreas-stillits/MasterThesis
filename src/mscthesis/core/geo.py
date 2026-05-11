from __future__ import annotations

from typing import Any

import numpy as np
import scipy
from stillib_random import from_seed

from ..log import log_call


def _derive_points(shape: tuple[int, int, int]) -> np.ndarray:
    nx, ny, nz = shape
    x = np.linspace(-nx / nz / 2, nx / nz / 2, nx)
    y = np.linspace(-ny / nz / 2, ny / nz / 2, ny)
    z = np.linspace(0, 1, nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    return points


def _map_point(point: np.ndarray, valid_points: np.ndarray) -> int:
    idx = np.argmin(np.linalg.norm(valid_points - point, axis=1))
    return int(idx)


@log_call()
def compute_geodesics(
    voxels: np.ndarray,
    start_point: tuple[float, float, float] = (0.0, 0.0, 0.0),
    material_id: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

    nx, ny, nz = voxels.shape
    points = _derive_points((nx, ny, nz))

    # transform voxel grid to work as a mask for the array of points
    airspace_mask = ~(voxels.ravel() == material_id)

    # map indices to graph node indices
    grid_to_node = -np.ones(points.shape[0], dtype=int)
    valid_indices = np.where(airspace_mask)[0]
    grid_to_node[valid_indices] = np.arange(
        valid_indices.size
    )  # label nodes consecutively and obstructions with -1

    valid_points = points[airspace_mask]
    n_nodes = valid_points.shape[0]

    # BUILD 26-NEIGHBOR CONNECTIVITY

    rows = []
    cols = []
    weights = []

    def flat_index(i, j, k):
        return i * (ny * nz) + j * nz + k

    dx = 1.0 / nz
    dy = 1.0 / nz
    dz = 1.0 / nz

    neighbor_offsets = [
        (1, 0, 0, dx),
        (1, -1, 0, dx * np.sqrt(2)),
        (1, 1, 0, dx * np.sqrt(2)),
        (1, 0, -1, dx * np.sqrt(2)),
        (1, 0, 1, dx * np.sqrt(2)),
        (1, 1, 1, dx * np.sqrt(3)),
        (1, 1, -1, dx * np.sqrt(3)),
        (1, -1, 1, dx * np.sqrt(3)),
        (1, -1, -1, dx * np.sqrt(3)),
        (-1, 0, 0, dx),
        (-1, -1, 0, dx * np.sqrt(2)),
        (-1, 1, 0, dx * np.sqrt(2)),
        (-1, 0, -1, dx * np.sqrt(2)),
        (-1, 0, 1, dx * np.sqrt(2)),
        (-1, 1, 1, dx * np.sqrt(3)),
        (-1, 1, -1, dx * np.sqrt(3)),
        (-1, -1, 1, dx * np.sqrt(3)),
        (-1, -1, -1, dx * np.sqrt(3)),
        (0, 1, 0, dy),
        (0, 1, -1, dy * np.sqrt(2)),
        (0, 1, 1, dy * np.sqrt(2)),
        (0, -1, 0, dy),
        (0, -1, -1, dy * np.sqrt(2)),
        (0, -1, 1, dy * np.sqrt(2)),
        (0, 0, 1, dz),
        (0, 0, -1, dz),
    ]

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                idx0 = flat_index(i, j, k)
                node0 = grid_to_node[idx0]
                if node0 < 0:
                    continue  # skip obstructions
                for di, dj, dk, weight in neighbor_offsets:
                    ii, jj, kk = i + di, j + dj, k + dk
                    if not (0 <= ii < nx and 0 <= jj < ny and 0 <= kk < nz):
                        continue  # out of bounds
                    idx1 = flat_index(ii, jj, kk)
                    node1 = grid_to_node[idx1]
                    if node1 < 0:
                        continue  # skip obstructions

                    rows.append(node0)
                    cols.append(node1)
                    weights.append(weight)

    A = scipy.sparse.coo_matrix(
        (weights, (rows, cols)), shape=(n_nodes, n_nodes)
    ).tocsr()

    starting_point = np.array(start_point)
    start_node = _map_point(starting_point, valid_points)

    # Compute shortest path
    geodesic_from_start = scipy.sparse.csgraph.shortest_path(
        A,
        directed=False,
        indices=start_node,
        method="D",
    )
    euclid_from_start = np.linalg.norm(valid_points - valid_points[start_node], axis=1)
    return geodesic_from_start, euclid_from_start, valid_points


@log_call()
def compute_for_targets(
    target_points: np.ndarray,
    geodesic_from_start: np.ndarray,
    euclid_from_start: np.ndarray,
    valid_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    tortuosities = np.zeros(target_points.shape[0])
    lateral_lengthening = np.zeros(target_points.shape[0])
    for i, point in enumerate(target_points):
        idx = _map_point(point, valid_points)
        geo_dist = geodesic_from_start[idx]
        euclid_dist = euclid_from_start[idx]
        tortuosity = geo_dist / euclid_dist if euclid_dist > 0 else np.nan
        tortuosities[i] = tortuosity
        lateral_lengthening[i] = point[2] / euclid_dist if euclid_dist > 0 else np.nan
    tortuosities = tortuosities[~np.isnan(tortuosities)]
    lateral_lengthening = lateral_lengthening[~np.isnan(lateral_lengthening)]
    return tortuosities, lateral_lengthening


@log_call()
def sample_zplane(
    target: float,
    valid_points: np.ndarray,
    n_samples: int | None = None,
) -> np.ndarray:
    target_mask = np.isclose(valid_points[:, 2], target)
    target_points = valid_points[target_mask]
    if n_samples is not None and target_points.shape[0] > n_samples:
        root = from_seed(123456)
        rng = root.generator()
        indices = rng.choice(target_points.shape[0], size=n_samples, replace=False)
        target_points = target_points[indices]

    return target_points


@log_call()
def sample_surfaces(
    voxels: np.ndarray,
    n_samples: int | None = None,
    material_id: int = 1,
) -> np.ndarray:
    nx, ny, nz = voxels.shape
    all_points = _derive_points((nx, ny, nz))

    # construct a list of points that are adjacent to the material_id surface
    surface_mask = np.zeros_like(voxels, dtype=bool)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                if voxels[i, j, k] == material_id:
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            for dk in [-1, 0, 1]:
                                ii, jj, kk = i + di, j + dj, k + dk
                                if (
                                    0 <= ii < nx
                                    and 0 <= jj < ny
                                    and 0 <= kk < nz
                                    and voxels[ii, jj, kk] != material_id
                                ):
                                    surface_mask[ii, jj, kk] = True
    surface_points = all_points[surface_mask.ravel()]
    if n_samples is not None and surface_points.shape[0] > n_samples:
        root = from_seed(123456)
        rng = root.generator()
        indices = rng.choice(surface_points.shape[0], size=n_samples, replace=False)
        surface_points = surface_points[indices]

    return surface_points


@log_call()
def geometry(voxels: np.ndarray, n_samples: int | None = None) -> dict[str, Any]:
    results: dict[str, Any] = {}

    geo, euc, valid_points = compute_geodesics(voxels)
    tor, lat = compute_for_targets(
        sample_surfaces(voxels, n_samples=n_samples), geo, euc, valid_points
    )
    summary = {
        "tortuosity": np.mean(tor),
        "tortuosity_factor": np.mean(tor**2),
        "lateral_lengthening": np.mean(lat),
    }
    results["surfaces"] = summary
    #
    tor, lat = compute_for_targets(
        sample_zplane(1.0, valid_points, n_samples=n_samples), geo, euc, valid_points
    )
    summary = {
        "tortuosity": np.mean(tor),
        "tortuosity_factor": np.mean(tor**2),
        "lateral_lengthening": np.mean(lat),
    }
    results["top"] = summary
    return results
