from __future__ import annotations

import argparse
from pathlib import Path

from ....config import ProjectConfig, save_config


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the print-config command."""
    target_path: Path = args.out / config.meta.config_name
    print(target_path)
    if target_path.exists() and not args.force:
        response = input(
            f"File {target_path} already exists. Do you want to overwrite it? [y/n]: "
        )
        if response.lower() != "y":
            print("Aborting.")
            return
    else:
        print(f"Saving config to {target_path}...")
    save_config(target_path, config, "behavior", "solver")
    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "print-config",
        help="Print the current project configuration to a file.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path.cwd(),
        help="Path to save the initialized config.json file. If the file already exists, you must confirm overwriting with --force",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Overwrite the output file if it already exists without asking for confirmation.",
    )
    parser.set_defaults(cmd=_cmd)
