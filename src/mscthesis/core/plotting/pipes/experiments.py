from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stillib_plotting import figure, save, use_style


def plot_experiments(df: pd.DataFrame, output_dir: str | Path | None = None) -> None:
    use_style()
    #
    fig1, ax1 = figure(size="single")
    # plot lines, resistance_mean as a function of stomatal_aspect for each plug_aspect
    for plug_aspect, group in df.groupby("plug_aspect"):
        ax1.errorbar(
            group["stomatal_aspect"],
            group["resistance_mean"],
            yerr=group["resistance_std"],
            marker="x",
            capsize=2,
            label=f"plug_aspect={plug_aspect:.2f}",
        )
    ax1.set_xlabel("stomatal aspect")
    ax1.set_ylabel("resistance mean")
    #
    if output_dir is not None:
        output_dir = Path(output_dir)

        save(fig1, output_dir / "by_stomatal.pdf")
    #
    fig2, ax2 = figure(size="single")
    # plot lines, resistance_mean as a function of stomatal_aspect for each plug_aspect
    for stomatal_aspect, group in df.groupby("stomatal_aspect"):
        ax2.errorbar(
            group["plug_aspect"],
            group["resistance_mean"],
            yerr=group["resistance_std"],
            marker="x",
            capsize=2,
            label=f"stomatal_aspect={stomatal_aspect:.2f}",
        )
    ax2.set_xlabel("plug aspect")
    ax2.set_ylabel("resistance mean")
    #
    if output_dir is not None:
        output_dir = Path(output_dir)

        save(fig2, output_dir / "by_plug.pdf")

    plt.show()

    return
