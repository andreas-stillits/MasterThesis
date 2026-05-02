from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from stillib_plotting import figure, set_axis_labels, use_style

from ....config import ProjectConfig
from ....core.io import save_dataframe
from ....manifest import fetch_from_manifest
from ....paths import ProjectPaths

COLORS = {
    "uniform": "#1f77b4",
    "mixed": "#ff7f0e",
    "metaballs": "#2ca02c",
}


def _plot_index_state(df: pd.DataFrame) -> None:
    use_style()

    fig, ax = figure(size="single")
    for type, group in df.groupby("type"):
        ax.scatter(
            group["plug_aspect"],
            group["porosity"],
            label=type,
            color=COLORS.get(str(type)),
            edgecolor="black",
        )
    set_axis_labels(ax, r"Plug Aspect Ratio $R$", r"Mean Porosity $\theta$")
    ax.set_xlim(0.05, 0.45)
    ax.set_ylim(0.0, 1.05)
    ax.legend(title="Type", loc="center left", bbox_to_anchor=(1.0, 1.0))
    plt.show()
    return


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
        _plot_index_state(df)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compile-candidates", help="Compile candidate configurations for search"
    )
    parser.set_defaults(cmd=_cmd)
    parser.add_argument(
        "-s", "--show", action="store_true", help="Show the index state"
    )
    return
