from __future__ import annotations

import argparse

from ....config import ProjectConfig, save_config
from ....core.io import save_voxels
from ....core.synthesis.metaballs import generate_voxels_from_sample_id
from ....core.visualization import visualize_voxels
from ....ids import validate_sample_id
from ....manifest import dump_manifest
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the synthesize-mixed command."""
    cmdconfig = config.synthesis
    sample_id: str = validate_sample_id(
        args.sample_id, required_digits=config.behavior.sample_id_digits
    )
    # prepare paths
    paths = ProjectPaths(config.behavior.storage_root).sample(sample_id)
    process_paths = paths.synthesis
    process_paths.root.ensure()

    # do work
    voxels, metadata = generate_voxels_from_sample_id(
        sample_id,
        cmdconfig.base_seed,
        cmdconfig.resolution,
        cmdconfig.plug_aspect,
        cmdconfig.separation,
        cmdconfig.max_attempts,
        cmdconfig.metaballs.num_cells,
        cmdconfig.metaballs.radius_min,
        cmdconfig.metaballs.radius_max,
        cmdconfig.metaballs.threshold,
    )

    # save data
    save_voxels(process_paths.voxels.path, voxels)

    # save config
    save_config(process_paths.config.path, config, "synthesis")

    # save manifest
    dump_manifest(
        process_paths.manifest.path,
        command_name="synthesize-metaballs",
        sample_id=sample_id,
        inputs={},
        outputs={"voxels": process_paths.voxels.path},
        metadata=metadata,
        tool_version=config.meta.project_version,
    )

    if args.show:
        visualize_voxels(voxels)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "synthesize-metaballs",
        help="Generate a metaballs voxel model.",
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
