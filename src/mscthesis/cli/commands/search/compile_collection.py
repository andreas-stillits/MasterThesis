from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd

from ....config import ProjectConfig
from ....core.io import load_dataframe, save_dataframe
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    paths.selected.ensure()

    diff = load_dataframe(paths.diffusion_summary.require())
    photo = load_dataframe(paths.photoactive_summary.require())

    sample_ids = photo["sample_id"].unique()
    specifiers = photo["specifier"].unique()

    collection = photo.copy()
    collection["diffusive_resistance_m"] = np.nan
    collection["diffusive_pipe_resistance_m"] = np.nan
    collection["diffusive_inlet_resistance"] = np.nan

    # collection has multiple rows for each sample_id and specifier pair
    # I want to fill the diffusive columns for each sample_id and specifier pair with the corresponding values from diff
    for sample_id in sample_ids:
        for specifier in specifiers:
            mask_photo = (collection["sample_id"] == sample_id) & (
                collection["specifier"] == specifier
            )
            mask_diff = (diff["sample_id"] == sample_id) & (
                diff["specifier"] == specifier
            )

            if not mask_diff.any():
                continue

            collection.loc[mask_photo, "diffusive_resistance_m"] = diff.loc[
                mask_diff, "resistance_m"
            ].values[0]
            collection.loc[mask_photo, "diffusive_pipe_resistance_m"] = diff.loc[
                mask_diff, "pipe_resistance_m"
            ].values[0]
            collection.loc[mask_photo, "diffusive_inlet_resistance"] = diff.loc[
                mask_diff, "inlet_resistance"
            ].values[0]

    collection.dropna(inplace=True)

    save_dataframe(paths.collection.path, collection)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compile-collection",
        help="Compile collection of results into a single dataframe.",
    )
    parser.set_defaults(cmd=_cmd)
