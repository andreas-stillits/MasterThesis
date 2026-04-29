from __future__ import annotations

import argparse

from ....config import ProjectConfig
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the wim command."""
    paths = ProjectPaths(config.behavior.storage_root).samples.path
    max_sample_id = 0
    for path in paths.iterdir():
        if path.is_dir() and path.name.isdigit():
            sample_id = int(path.name)
            if sample_id > max_sample_id:
                max_sample_id = sample_id
    print(f"Maximum sample ID: {max_sample_id:0{config.behavior.sample_id_digits}d}")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "wim",
        help="Determine the maximum sample ID present in the storage root and print it to the console.",
    )
    parser.set_defaults(cmd=_cmd)
