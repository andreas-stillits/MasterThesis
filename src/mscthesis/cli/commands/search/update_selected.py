from __future__ import annotations

import argparse
import shutil
import subprocess

import numpy as np

from ....config import ProjectConfig
from ....core.io import load_dataframe
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    paths.candidates.ensure()
    paths.selected.ensure()

    # load index and update selected column based on presence in selected directory
    index = load_dataframe(paths.index.require())
    plug_aspect_set = config.search.plug_aspect_set.values()
    gridsize = config.search.selected.porosity_gridsize
    porosity_set = np.arange(0.20, 1.0, gridsize)

    for kind in index["type"].unique():
        index_ = index[index["type"] == kind]
        for plug_aspect in plug_aspect_set:
            df = index_[index_["plug_aspect"] == plug_aspect]
            for porosity in porosity_set:
                df_ = df[
                    (df["porosity"] > porosity) & (df["porosity"] < porosity + gridsize)
                ]
                if df_.empty:
                    continue
                sample_ids = df_["sample_id"].tolist()
                # if some already in selected, skip
                if any(
                    (paths.selected / sample_id).exists() for sample_id in sample_ids
                ):
                    continue
                # else, pick one at random by drawing a random integer index and copy it to selected
                selected_sample_id = sample_ids[np.random.randint(len(sample_ids))]
                source = paths.candidate_sample(selected_sample_id).root.require()
                target = paths.selected_sample(selected_sample_id).root.path
                if source.is_dir():
                    shutil.copytree(source, target)

    # update index
    subprocess.run(["msc", "search", "compile-index"], check=True)

    # show index state if requested
    if args.show:
        cmd = ["msc", "search", "show-index", "--selected-only"]
        subprocess.run(cmd, check=True)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "update-selected", help="Update index of selected configurations for search"
    )
    parser.set_defaults(cmd=_cmd)
    parser.add_argument(
        "-s", "--show", action="store_true", help="Show the index state"
    )
    return
