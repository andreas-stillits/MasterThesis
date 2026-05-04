from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
from stillib_plotting import figure, set_axis_labels, use_style

from ....config import ProjectConfig
from ....core.io import load_dataframe
from ....paths import ProjectPaths

COLORS = {
    "uniform": "#1f77b4",
    "mixed": "#ff7f0e",
    "metaballs": "#2ca02c",
}


def plot_index_state(df: pd.DataFrame, selected_only: bool = False) -> None:
    use_style()

    fig, ax = figure(size="single")
    if selected_only:
        df = df[df["selected"]]
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
    # only call legend if the was something to plot
    if not df.empty:
        ax.legend(title="Type", loc="center left", bbox_to_anchor=(1.0, 1.0))
    plt.show()
    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)

    df = load_dataframe(paths.index.require())
    plot_index_state(df, selected_only=args.selected_only)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("show-index", help="Show the index state")
    parser.set_defaults(cmd=_cmd)
    parser.add_argument(
        "-o",
        "--selected-only",
        action="store_true",
        help="Show only selected configurations in the index state",
    )
    return
