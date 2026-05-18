from __future__ import annotations

import argparse

from mscthesis.core.io import load_voxels
from mscthesis.core.visualization import visualize_voxels

from ....config import ProjectConfig
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:

    paths = ProjectPaths(config.behavior.storage_root)
    selected = [p for p in paths.selected.path.iterdir() if p.is_dir()]
    selected_ids = [s.name for s in selected if s.name.isdigit()]

    deletion_ids = []

    for sample_id in selected_ids:
        voxel_path = paths.selected_sample(sample_id).synthesis.voxels.require()
        voxels = load_voxels(voxel_path)
        visualize_voxels(voxels)
        delete = input("Delete this sample? (y/n): ")
        if delete.lower() == "y":
            deletion_ids.append(sample_id)
    if deletion_ids:
        print("The following samples will be deleted:")
        for sample_id in deletion_ids:
            print(sample_id)
        confirm = input("Are you sure you want to delete these samples? (y/n): ")
        if confirm.lower() == "y":
            for sample_id in deletion_ids:
                root = paths.selected_sample(sample_id).root.path
                for item in root.iterdir():
                    if item.is_file():
                        item.unlink()
                root.rmdir()
                #
                root = paths.candidate_sample(sample_id).root.path
                for item in root.iterdir():
                    if item.is_file():
                        item.unlink()
                root.rmdir()
            print("Selected samples have been deleted.")
        else:
            print("Aborting deletion.")
    else:
        print("No samples selected for deletion.")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "skim-selected",
        help="Skim selected samples and visualize them.",
    )
    parser.set_defaults(cmd=_cmd)
    return
