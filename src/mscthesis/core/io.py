from __future__ import annotations

from pathlib import Path

import adios4dolfinx as a4x
import gmsh
import numpy as np
import open3d as o3d
import pandas as pd
from dolfinx import fem
from dolfinx.io import gmshio
from mpi4py import MPI

from ..core.solvers import MeshContext
from ..log import log_call


@log_call()
def load_voxels(file_path: str | Path) -> np.ndarray:
    """
    Load a 3D voxel grid from a .npy file.

    Args:
        file_path (str | Path): Path to the .npy file containing the voxel grid.

    Returns:
        np.ndarray: The loaded 3D voxel grid.
    """
    voxels = np.load(file_path)
    return voxels


@log_call()
def save_voxels(file_path: str | Path, voxels: np.ndarray) -> None:
    """
    Save a voxel model to a binary .npy file.

    Args:
        file_path (str | Path): The output file path for the .npy file.
        voxels (np.ndarray): 3D numpy array representing the voxel model.
    """
    np.save(file_path, voxels)
    return


@log_call()
def load_surface_mesh(file_path: str | Path) -> o3d.geometry.TriangleMesh:
    """
    Load a surface mesh from a file.

    Args:
        file_path (str | Path): Path to the mesh file.
    Returns:
        o3d.geometry.TriangleMesh: The loaded surface mesh.
    """
    mesh = o3d.io.read_triangle_mesh(file_path)
    if mesh.is_empty():
        raise OSError(f"Failed to read mesh from {file_path}")
    return mesh


@log_call()
def save_surface_mesh(file_path: str | Path, mesh: o3d.geometry.TriangleMesh) -> None:
    """
    Save a surface mesh to a file.

    Args:
        file_path (str | Path): The output filename for the mesh file.
        mesh (o3d.geometry.TriangleMesh): The surface mesh to save.
    """
    written = o3d.io.write_triangle_mesh(file_path, mesh)
    if not written:
        raise OSError(f"Failed to write mesh to {file_path}")
    return


# monkey patch for silent GMSH
_original_initialize = gmsh.initialize


def _quiet_initialize(*args, **kwargs):
    """Initialize GMSH without printing to stdout."""
    _original_initialize(*args, **kwargs)
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.option.setNumber("General.Verbosity", 0)


@log_call()
def load_volumetric_mesh(file_path: str | Path) -> MeshContext:
    """
    Load a volumetric mesh from a Gmsh file.

    Args:
        file_path (str | Path): Path to the Gmsh .msh file.
    Returns:
        MeshContext: A MeshContext object containing the mesh and associated tags.
    """
    # override the gmsh.initialize() call in gmshio to suppress output
    gmsh.initialize = _quiet_initialize
    mesh, cell_tags, facet_tags = gmshio.read_from_msh(
        file_path, MPI.COMM_SELF, 0, gdim=3
    )

    mesh_ctx = MeshContext(
        mesh,
        cell_tags,
        facet_tags,
    )

    return mesh_ctx


@log_call()
def save_fem_solution(
    file_path: str | Path,
    solution: fem.Function,
    mesh_ctx: MeshContext,
) -> None:
    """
    Save FEniCSx solution as a .bp file
    Args:
        file_path (str | Path): The output filename for the .bp file.
        solution (fem.Function): The FEniCSx solution to save.
        mesh_ctx (MeshContext): The mesh context containing the mesh and tags.
    """
    file_path = Path(file_path)
    a4x.write_mesh(file_path, mesh_ctx.mesh)
    a4x.write_meshtags(
        file_path, mesh_ctx.mesh, mesh_ctx.cell_tags, meshtag_name="cell_tags"
    )
    a4x.write_meshtags(
        file_path, mesh_ctx.mesh, mesh_ctx.facet_tags, meshtag_name="facet_tags"
    )
    a4x.write_function(file_path, solution, name="solution")
    return


@log_call()
def load_fem_solution(
    file_path: str | Path, order: int = 2
) -> tuple[fem.Function, MeshContext]:
    """
    Load a FEniCSx solution from a .bp file
    Args:
        file_path (str | Path): The path to the .bp file containing the solution.
        order (int): The polynomial order of the function space.
    Returns:
        tuple[fem.Function, MeshContext]: The loaded solution and mesh context.
    """
    file_path = Path(file_path)
    mesh = a4x.read_mesh(file_path, MPI.COMM_SELF)
    cell_tags = a4x.read_meshtags(file_path, mesh, meshtag_name="cell_tags")
    facet_tags = a4x.read_meshtags(file_path, mesh, meshtag_name="facet_tags")
    functionspace = fem.functionspace(mesh, ("Lagrange", order))
    solution = fem.Function(functionspace)
    a4x.read_function(file_path, solution, name="solution")  # type: ignore[reportArgumentType]
    mesh_ctx = MeshContext(mesh, cell_tags, facet_tags)
    return solution, mesh_ctx  # type: ignore[reportArgumentType]


@log_call()
def save_dataframe(file_path: str | Path, dataframe: pd.DataFrame) -> None:
    """
    Save a pandas DataFrame to a .csv file.

    Args:
        file_path (str | Path): The output filename for the .csv file.
        dataframe (pd.DataFrame): The DataFrame to save.
    """
    dataframe.to_csv(file_path, decimal=",", sep="\t")
    return


@log_call()
def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    """
    Load a pandas DataFrame from a .csv file.

    Args:
        file_path (str | Path): The path to the .csv file containing the DataFrame.
    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    dataframe = pd.read_csv(file_path, decimal=",", sep="\t")
    return dataframe
