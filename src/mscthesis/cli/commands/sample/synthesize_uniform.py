from __future__ import annotations

import argparse

from ....config import ProjectConfig, save_config
from ....core.io import save_voxels
from ....core.synthesis.uniform import generate_voxels_from_sample_id
from ....ids import validate_sample_id
from ....manifest import dump_manifest
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the synthesize-uniform command."""
    cmdconfig = config.synthesis
    sample_id: str = validate_sample_id(
        args.sample_id, required_digits=config.behavior.sample_id_digits
    )
    # prepare paths
    paths = ProjectPaths(config.behavior.storage_root)
    process_paths = paths.sample(sample_id).synthesis
    process_paths.root.ensure()
    output_path = process_paths.voxels.path

    # do work
    voxels, metadata = generate_voxels_from_sample_id(
        sample_id,
        cmdconfig.base_seed,
        cmdconfig.resolution,
        cmdconfig.plug_aspect,
        cmdconfig.uniform.num_cells,
        cmdconfig.uniform.radius,
        cmdconfig.separation,
        cmdconfig.max_attempts,
    )

    # save data
    save_voxels(output_path, voxels)

    # save config
    save_config(process_paths.config.path, config, "synthesis")

    # save manifest
    dump_manifest(
        process_paths.manifest.path,
        command_name="synthesize-uniform",
        sample_id=sample_id,
        inputs={},
        outputs={"voxels": output_path},
        metadata=metadata,
        tool_version=config.meta.project_version,
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "synthesize-uniform",
        help="Generate a uniform swiss cheese voxel model.",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="Unique identifier for the generated sample",
    )
    parser.set_defaults(cmd=_cmd)
