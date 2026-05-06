from __future__ import annotations

import argparse
from pathlib import Path

from ....config import ProjectConfig
from ....core.io import load_fem_solution, load_surface_mesh, load_voxels
from ....core.visualization import (
    visualize_fem_solution,
    visualize_surface_mesh,
    visualize_volumetric_mesh,
    visualize_voxels,
)
from ....ids import validate_sample_id
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the print-config command."""
    paths = ProjectPaths(config.behavior.storage_root)

    if ":" in args.key:
        parts: list[str] = args.key.split(":")
        if len(parts) == 2:
            sample_id, name = parts
            specifier = None
        elif len(parts) == 3:
            sample_id, name, specifier = parts
            specifier = int(specifier) if specifier.isdigit() else None
        else:
            print(f"Invalid key format: {args.key}")
            return
        #
        sample_id = validate_sample_id(sample_id, config.behavior.sample_id_digits)
        if name == "voxels":
            path = paths.selected_sample(sample_id).synthesis.voxels.require()
            voxels = load_voxels(path)
            visualize_voxels(voxels)
        elif name == "surface":
            path = paths.selected_sample(sample_id).triangulation.mesh.require()
            mesh = load_surface_mesh(path)
            visualize_surface_mesh(mesh)
        elif name == "mesh" and specifier is not None:
            path = paths.selected_sample(sample_id).meshing(specifier).mesh.require()
            visualize_volumetric_mesh(path)
        elif name == "diffusion" and specifier is not None:
            path = (
                paths.selected_sample(sample_id)
                .solutions(specifier)
                .diffusion.solution.require()
            )
            solution, mesh_ctx = load_fem_solution(path, order=2)
            visualize_fem_solution(solution, mesh_ctx.mesh)
        elif name == "photoactive" and specifier is not None:
            path = (
                paths.selected_sample(sample_id)
                .solutions(specifier)
                .photoactive.solution.require()
            )
            solution, mesh_ctx = load_fem_solution(path, order=2)
            visualize_fem_solution(solution, mesh_ctx.mesh)
        else:
            raise ValueError(f"Unsupported object name: {name}")
    else:
        try:
            path = Path(args.key)
        except Exception as e:
            print(f"Failed to convert key: {args.key} to a valid file path")
            raise e
        #
        if not path.exists():
            print(f"File {path} does not exist.")
            return
        if not path.is_file():
            print(f"Path {path} is not a file.")
            return
        #
        if path.suffix == ".npy":
            voxels = load_voxels(path)
            visualize_voxels(voxels)
        elif path.suffix == ".stl":
            mesh = load_surface_mesh(path)
            visualize_surface_mesh(mesh)
        elif path.suffix == ".msh":
            visualize_volumetric_mesh(path)
        elif path.suffix == ".bp":
            solution, mesh_ctx = load_fem_solution(path, order=2)
            visualize_fem_solution(solution, mesh_ctx.mesh)
        else:
            print(f"Unsupported file type: {path.suffix}")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize",
        help="Visualize a specified project object.",
    )
    parser.add_argument(
        "key",
        type=str,
        help="A string key for the object to visualize. Supported specifiers are <sample_id>:<object_name>:<specifier>, or <path>",
    )

    parser.set_defaults(cmd=_cmd)
