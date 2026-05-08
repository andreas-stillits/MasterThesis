from __future__ import annotations

import argparse
import json
from typing import Any

from stillib_parallelism import collect, print_progress

from ....config import ProjectConfig
from ....core.geo import geometry
from ....core.io import load_voxels
from ....paths import ProjectPaths

_STATE: dict[str, Any] = {}


def initializer(config: ProjectConfig, force: bool) -> None:
    _STATE["config"] = config
    _STATE["force"] = force
    return


def analyze_geometry(sample_id: str) -> None:
    config: ProjectConfig = _STATE["config"]
    force: bool = _STATE["force"]
    paths = ProjectPaths(config.behavior.storage_root).candidate_sample(sample_id)
    if not paths.synthesis.geometry.exists() or force:
        voxel_path = paths.synthesis.voxels.require()
        voxels = load_voxels(voxel_path)
        geo: dict[str, Any] = geometry(voxels, n_samples=1000)
        paths.synthesis.geometry.path.write_text(
            json.dumps(geo, indent=4, default=str), encoding="utf-8"
        )

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    # load a list of candidate samples
    candidates = [p for p in paths.candidates.path.iterdir() if p.is_dir()]
    candidate_ids = [c.name for c in candidates if c.name.isdigit()]

    report = collect(
        candidate_ids,
        analyze_geometry,
        max_workers=config.max_workers,
        initializer=initializer,
        initargs=(config, args.force),
        progress_callback=print_progress,
        error_policy="raise",
        ordering="completion",
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "analyze-geometry", help="Analyze the geometry of a sample."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-calculation even if results already exist.",
    )
    parser.set_defaults(cmd=_cmd)
    return
