from __future__ import annotations

import argparse

from mscthesis.core.visualization import visualize_volumetric_mesh

from ....config import ProjectConfig, save_config
from ....core.meshing.gmeshing import build_sample_model, mesh_model
from ....ids import validate_sample_id
from ....manifest import dump_manifest
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the mesh command."""
    cmdconfig = config.meshing
    sample_id: str = validate_sample_id(
        args.sample_id, required_digits=config.behavior.sample_id_digits
    )
    # prepare paths
    paths = ProjectPaths(config.behavior.storage_root).sample(sample_id)

    process_paths = paths.meshing
    process_paths.root.ensure()

    airspace_tag, plug_aspect = build_sample_model(
        paths.triangulation.cadmodel.require(),
        cmdconfig.boundary_margin,
        cmdconfig.substomatal_margin,
        cmdconfig.atol,
    )

    mesh_model(
        process_paths.mesh.ensure(),
        airspace_tag,
        plug_aspect,
        **cmdconfig.mesh_field.model_dump(),
    )

    # save config
    save_config(process_paths.config.path, config, "meshing")

    # save manifest
    dump_manifest(
        process_paths.manifest.path,
        command_name="mesh",
        sample_id=sample_id,
        inputs={"cadmodel": paths.triangulation.cadmodel.path},
        outputs={"mesh": process_paths.mesh.path},
        metadata={
            "plug_aspect": plug_aspect,
            "stomatal_aspect": cmdconfig.mesh_field.stomatal_aspect,
        },
        tool_version=config.meta.project_version,
    )

    if args.show:
        visualize_volumetric_mesh(process_paths.mesh.path)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "mesh",
        help="Generate a mesh for the given sample.",
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
