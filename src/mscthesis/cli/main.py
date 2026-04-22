from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..config import ProjectConfig, build_project_config
from ..paths import ProjectPaths

try:
    import argcomplete
except ImportError:
    argcomplete = None


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="msc", description="Master Thesis Command Line Interface"
    )
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a partial project config.json file for reproduction purposes.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error output."
    )
    # subparsers = parser.add_subparsers(dest="command", title="Commands", required=True)
    # #
    # # umbrella command for sample commands
    # sample_parser = subparsers.add_parser(
    #     "sample", help="Commands acting on sample geometries"
    # )
    # sample_subparsers = sample_parser.add_subparsers(
    #     dest="sample_command", title="Sample Commands", required=True
    # )
    # # the envoked command is stored under args.sample_command
    # #
    # # umbrella command for ideal pipe commands
    # pipe_parser = subparsers.add_parser(
    #     "pipe", help="Commands acting on ideal pipe geometries"
    # )
    # pipe_subparsers = pipe_parser.add_subparsers(
    #     dest="pipe_command", title="Pipe Commands", required=True
    # )
    # # the envoked command is stored under args.pipe_command
    return parser


def main(argv: list[str] | None = None) -> int:

    parser = _build_parser()

    # Enable tab completion if argcomplete is available. User must run: 'eval "$(register-python-argcomplete mscthesis)"'
    # in bash per session or add to .bashrc
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)
    config: ProjectConfig = build_project_config(args.config)
    paths: ProjectPaths = ProjectPaths(config.behavior.storage_root)
    paths.sample("00012").synthesis.ensure()  # example of how to use paths
    paths.pipes.meshes.ensure()  # example of how to use paths

    # if hasattr(args, "cmd"):
    #     start_time = time.perf_counter()

    #     args.cmd(config, args)

    #     duration = time.perf_counter() - start_time

    #     if not args.quiet:
    #         print(f"Command '{args.cmd.__name__}' completed in {duration:.3f} seconds.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
