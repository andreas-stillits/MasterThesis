from __future__ import annotations

import argparse
from pathlib import Path

from ....config import ProjectConfig
from ....core.io import load_surface_mesh, load_voxels
from ....core.visualization import visualize_surface_mesh, visualize_voxels
from ....ids import validate_sample_id
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the print-config command."""
    paths = ProjectPaths(config.behavior.storage_root)

    if ":" in args.specifier:
        sample_id, name = args.specifier.split(":")
        sample_id = validate_sample_id(sample_id, config.behavior.sample_id_digits)
        if name == "voxels":
            path = paths.sample(sample_id).synthesis.voxels.require()
            voxels = load_voxels(path)
            visualize_voxels(voxels)
        elif name == "surface":
            path = paths.sample(sample_id).triangulation.mesh.require()
            mesh = load_surface_mesh(path)
            visualize_surface_mesh(mesh)
        else:
            raise ValueError(f"Unsupported object name: {name}")
    else:
        try:
            path = Path(args.specifier)
        except Exception as e:
            print(f"Failed to convert specifier: {args.specifier} to a valid file path")
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
        else:
            print(f"Unsupported file type: {path.suffix}")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "visualize",
        help="Visualize a specified project object.",
    )
    parser.add_argument(
        "specifier",
        type=str,
        help="A string specifier for the object to visualize. Supported specifiers: <sample_id>:<object_name>, or <path>",
    )

    parser.set_defaults(cmd=_cmd)
