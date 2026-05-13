import numpy as np
from scipy import ndimage
from skimage import measure

from mscthesis.config import ProjectConfig
from mscthesis.core.io import load_voxels
from mscthesis.paths import ProjectPaths

paths = ProjectPaths(ProjectConfig().behavior.storage_root)

n = 100

x = np.linspace(-0.5, 0.5, n)
y = np.linspace(-0.5, 0.5, n)
z = np.linspace(0, 1, n)
X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

voxels = np.zeros((n, n, n), dtype=bool)

# assign all points inside a sphere to False
center = np.array([0, 0, 0.5])
radius = 0.3
distances = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2 + (Z - center[2]) ** 2)
voxels[distances < radius] = True


def mesh_surface_area_from_binary(volume, spacing=(1.0, 1.0, 1.0), z_axis=2):
    """
    Estimate surface area of a binary 3D solid using a signed distance field
    and marching cubes.

    volume: 3D boolean array, True = inside object
    spacing: voxel spacing in array axis order, e.g. (dz, dy, dx)
    """
    volume = volume.astype(bool)

    # Signed distance: negative inside, positive outside
    outside_dist = ndimage.distance_transform_edt(~volume, sampling=spacing)
    inside_dist = ndimage.distance_transform_edt(volume, sampling=spacing)
    sdf = outside_dist - inside_dist

    verts, faces, normals, values = measure.marching_cubes(
        sdf, level=0.0, spacing=spacing
    )

    tris = verts[faces]
    a = tris[:, 0, :]
    b = tris[:, 1, :]
    c = tris[:, 2, :]

    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)

    z_centroids = tris[:, :, z_axis].mean(axis=1)

    return np.sum(areas * z_centroids) / np.sum(areas), np.sum(areas)


voxels = load_voxels(paths.selected_sample("00000").synthesis.voxels.require())

factor, area = mesh_surface_area_from_binary(voxels, spacing=(1 / n, 1 / n, 1 / n))
print(f"Estimated factor: {factor:.4f}")
