from __future__ import annotations

import argparse
from pathlib import Path

from ....config import ProjectConfig
from ....ids import validate_sample_id
from ....paths import ProjectPaths


def delete_sample(root: Path) -> None:
    for item in root.iterdir():
        if item.is_dir():
            delete_sample(item)
        else:
            item.unlink()
    root.rmdir()
    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    # load a list of selected samples
    selected = [p for p in paths.selected.path.iterdir() if p.is_dir()]
    selected_ids = [s.name for s in selected if s.name.isdigit()]
    deletion_ids = []
    with open(args.filename, encoding="utf-8") as f:
        for line in f:
            sample_id = line.strip()
            try:
                sample_id = validate_sample_id(
                    sample_id, config.behavior.sample_id_digits
                )
            except ValueError:
                print(f"Invalid sample ID: {sample_id}")
                continue
            if sample_id in selected_ids:
                deletion_ids.append(sample_id)
    if deletion_ids:
        print("The following samples will be deleted:")
        for sample_id in deletion_ids:
            print(sample_id)
        confirm = input("Are you sure you want to delete these samples? (y/n): ")
        if confirm.lower() == "y":
            for sample_id in deletion_ids:
                delete_sample(paths.selected_sample(sample_id).root.path)
                delete_sample(paths.candidate_sample(sample_id).root.path)
            print("Selected samples have been deleted.")
        else:
            print("Aborting deletion.")
    else:
        print("No samples selected for deletion.")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "delete-selected",
        help="Delete selected samples.",
    )
    parser.add_argument(
        "filename",
        type=str,
        help="Path to the file containing the list of sample IDs to delete.",
    )
    parser.set_defaults(cmd=_cmd)
    return
