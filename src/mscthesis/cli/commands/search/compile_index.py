from __future__ import annotations

import argparse
import subprocess
from typing import Any

import pandas as pd

from ....config import ProjectConfig
from ....core.io import save_dataframe
from ....manifest import fetch_from_manifest
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    paths.candidates.ensure()
    paths.selected.ensure()

    # load a list of candidates
    candidates = [p for p in paths.candidates.path.iterdir() if p.is_dir()]

    # load a list of selected samples
    selected = [p for p in paths.selected.path.iterdir() if p.is_dir()]
    selected_ids = [s.name for s in selected if s.name.isdigit()]

    # compile index
    index: list[dict[str, Any]] = []
    for candidate in candidates:
        sample_id = candidate.name
        plug_aspect, porosity, type = fetch_from_manifest(
            paths.candidate_sample(sample_id).synthesis.manifest.require(),
            "plug_aspect",
            "mean_porosity",
            "type",
        )
        in_selected = sample_id in selected_ids
        index.append(
            {
                "sample_id": sample_id,
                "plug_aspect": plug_aspect,
                "porosity": porosity,
                "type": type,
                "selected": in_selected,
            }
        )
    df = pd.DataFrame(index)
    df.reset_index(drop=True, inplace=True)
    save_dataframe(paths.index.path, df)

    if args.show:
        cmd = ["msc", "search", "show-index"]
        if args.selected_only:
            cmd.append("--selected-only")
        subprocess.run(cmd, check=True)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compile-index", help="Compile candidate configurations for search"
    )
    parser.set_defaults(cmd=_cmd)
    parser.add_argument(
        "-o",
        "--selected-only",
        action="store_true",
        help="Show only selected configurations in the index state",
    )
    parser.add_argument(
        "-s", "--show", action="store_true", help="Show the index state"
    )
    return
