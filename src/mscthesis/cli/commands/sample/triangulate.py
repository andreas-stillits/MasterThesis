from __future__ import annotations

import argparse
import os
import subprocess
from typing import cast

from ....config import ProjectConfig, save_config
from ....core.io import load_voxels, save_surface_mesh
from ....core.meshing.triangulation import triangulate_voxels
from ....core.visualization import visualize_surface_mesh
from ....ids import validate_sample_id
from ....manifest import dump_manifest, fetch_from_manifest
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the triangulate command."""
    cmdconfig = config.triangulation
    sample_id: str = validate_sample_id(
        args.sample_id, required_digits=config.behavior.sample_id_digits
    )
    # prepare paths
    paths = ProjectPaths(config.behavior.storage_root).sample(sample_id)
    process_paths = paths.triangulation
    process_paths.root.ensure()

    voxels_path = paths.synthesis.voxels.require()
    voxels = load_voxels(voxels_path)

    num_cells_placed = cast(
        int, fetch_from_manifest(paths.synthesis.manifest.require(), "num_cells_placed")
    )

    decimation_target = cmdconfig.elements_per_cell * num_cells_placed

    # do work
    mesh, metadata = triangulate_voxels(
        voxels,
        cmdconfig.smoothing_iterations,
        decimation_target,
        cmdconfig.shrinkage_tolerance,
        cmdconfig.spacing,
    )

    # save data
    save_surface_mesh(
        process_paths.mesh.path,
        mesh,
    )

    if metadata["success"]:
        # save paths as environment variables for FreeCAD to read
        env = os.environ.copy()
        env["INPUT_STL"] = os.path.abspath(process_paths.mesh.path)
        env["OUTPUT_BREP"] = os.path.abspath(process_paths.cadmodel.path)

        # Export to BREP using an external tool (e.g., FreeCAD command line)
        # Fail loudly if this process fails
        try:
            process = subprocess.run(
                [cmdconfig.freecad_cmd, cmdconfig.freecad_script_path],
                env=env,
            )
            if process.returncode != 0:
                raise RuntimeError(
                    f"FreeCAD command failed with return code {process.returncode}"
                )
        except Exception as exc:
            raise RuntimeError("Failed to export BREP using FreeCAD") from exc

        metadata["brep_exported"] = True
    # silently continue even if surface mesh was not water tight and manifold
    else:
        metadata["brep_exported"] = False

    # save config
    save_config(process_paths.config.path, config, "triangulation")

    # save manifest
    dump_manifest(
        process_paths.manifest.path,
        command_name="triangulate",
        sample_id=sample_id,
        inputs={"voxels": voxels_path},
        outputs={
            "mesh": process_paths.mesh.path,
            "cadmodel": process_paths.cadmodel.path,
        },
        metadata=metadata,
        tool_version=config.meta.project_version,
    )

    if args.show:
        visualize_surface_mesh(mesh)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "triangulate",
        help="Generate a surface mesh from voxel data.",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="Unique identifier for the generated sample",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Visualize the output",
    )
    parser.set_defaults(cmd=_cmd)
