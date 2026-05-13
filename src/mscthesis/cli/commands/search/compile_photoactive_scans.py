from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from ....config import ProjectConfig
from ....core.io import load_dataframe, save_dataframe
from ....core.photoactive import derive_summary
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    paths.selected.ensure()

    selected = [p for p in paths.selected.path.iterdir() if p.is_dir()]
    selected_ids = [s.name for s in selected if s.name.isdigit()]

    index = pd.DataFrame()
    for sample_id in selected_ids:
        sample_paths = paths.selected_sample(sample_id)
        for specifier in config.search.stomatal_aspect_set:
            scanning_paths = sample_paths.scanning(specifier)

            if scanning_paths.scan.exists():
                content = load_dataframe(scanning_paths.scan.path)
                index = pd.concat([index, content], ignore_index=True)
    index.reset_index(drop=True, inplace=True)
    index.sort_values(
        by=["sample_id", "specifier", "absorption", "transport"], inplace=True
    )

    save_dataframe(paths.photoactive_index.path, index)

    summary = derive_summary(index)
    save_dataframe(paths.photoactive_summary.path, summary)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compile-photoactive-scans",
        help="Compile photoactive scans into a single dataframe.",
    )
    parser.set_defaults(cmd=_cmd)
