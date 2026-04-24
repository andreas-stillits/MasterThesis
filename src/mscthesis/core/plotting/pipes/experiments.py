from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from stillib_plotting import figure, save, use_style


def plot_experiments(df: pd.DataFrame, output_path: str | Path | None = None) -> None:
    use_style()
    fig, ax = figure(size="single")
    # plot lines, resistance_mean as a function of stomatal_aspect for each plug_aspect
    for plug_aspect, group in df.groupby("plug_aspect"):
        ax.errorbar(
            group["stomatal_aspect"],
            group["resistance_mean"],
            yerr=group["resistance_std"],
            marker="x",
            capsize=2,
            label=f"plug_aspect={plug_aspect:.2f}",
        )
    ax.set_xlabel("stomatal aspect")
    ax.set_ylabel("resistance mean")
    if output_path is not None:
        save(fig, output_path)

    plt.show()

    return
