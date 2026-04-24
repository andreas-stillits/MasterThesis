from __future__ import annotations

import argparse
import time
from pathlib import Path

from ..config import ProjectConfig, build_project_config
from ..log import setup_logging
from .commands.sample import (
    mesh,
    solve_active,
    solve_diffusion,
    synthesize_uniform,
    triangulate,
)
from .commands.utils import print_config, visualize

try:
    import argcomplete
except ImportError:
    argcomplete = None


def _wire_global_flags(parser: argparse.ArgumentParser) -> None:
    """Add global flags to the parser."""
    parser.add_argument(
        "-c",
        "--config",
        type=Path,
        help="Path to a partial project config.json file for reproduction purposes.",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress non-error output."
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable logging to file. Logs will only be printed to console.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level. Default is INFO.",
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="msc", description="Master Thesis Command Line Interface"
    )
    _wire_global_flags(parser)
    subparsers = parser.add_subparsers(dest="command", title="Commands", required=True)
    #
    # =================================================================================
    # umbrella command for sample commands
    # =================================================================================
    #
    sample_parser = subparsers.add_parser(
        "sample", help="Commands acting on sample geometries"
    )
    sample_subparsers = sample_parser.add_subparsers(
        dest="sample_command",
        title="Sample Commands",
        required=True,
        # the envoked command is stored under args.sample_command
    )
    synthesize_uniform.add_parser(sample_subparsers)
    triangulate.add_parser(sample_subparsers)
    mesh.add_parser(sample_subparsers)
    solve_active.add_parser(sample_subparsers)
    solve_diffusion.add_parser(sample_subparsers)
    #
    # ================================================================================
    # umbrella command for ideal pipe commands
    # ================================================================================
    #
    pipe_parser = subparsers.add_parser(
        "pipe", help="Commands acting on ideal pipe geometries"
    )
    pipe_subparsers = pipe_parser.add_subparsers(
        dest="pipe_command",
        title="Pipe Commands",
        required=True,
        # the envoked command is stored under args.pipe_command
    )
    #
    # ================================================================================
    # utility commands
    # ================================================================================
    #
    utils_parser = subparsers.add_parser(
        "utils", help="Utility commands for project maintenance and debugging"
    )
    utils_subparsers = utils_parser.add_subparsers(
        dest="utils_command",
        title="Utility Commands",
        required=True,
        # the envoked command is stored under args.utils_command
    )
    print_config.add_parser(utils_subparsers)
    visualize.add_parser(utils_subparsers)

    return parser


def main(argv: list[str] | None = None) -> int:

    parser = _build_parser()

    # Enable tab completion if argcomplete is available. User must run: 'eval "$(register-python-argcomplete mscthesis)"'
    # in bash per session or add to .bashrc
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)
    config: ProjectConfig = build_project_config(args.config)
    log_path: Path = config.behavior.storage_root / config.meta.log_name
    setup_logging(log_path, args.log_level, args.quiet, args.no_log)

    if hasattr(args, "cmd"):
        start_time = time.perf_counter()

        args.cmd(config, args)

        duration = time.perf_counter() - start_time

        if not args.quiet:
            cmdname = args.command
            if args.command in ["sample", "pipe", "utils"]:
                cmdname = f"{args.command} {getattr(args, f'{args.command}_command')}"
            print(f"Command: '{cmdname}' completed in {duration:.2f} seconds.")

        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
