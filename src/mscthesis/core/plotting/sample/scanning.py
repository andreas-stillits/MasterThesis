from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stillib_plotting import figure, label_panel, save, set_axis_labels, use_style


def _npy(df: pd.DataFrame, column: str) -> np.ndarray:
    return np.abs(df[column].values)


def plot_scanning_results(
    df: pd.DataFrame,
    output_dir: str | Path | None = None,
    show=True,
) -> None:
    use_style()

    transport = _npy(df, "transport")
    absorption = _npy(df, "absorption")
    chii = _npy(df, "substomatal_mean")
    chim = _npy(df, "mesophyll_mean")
    variation = np.sqrt(_npy(df, "mesophyll_var")) / chim
    a_n = _npy(df, "mesophyll_flux_sol") / _npy(df, "plug_area")
    resistance = (chii - chim) / a_n

    def _set_labels(ax: plt.Axes) -> None:  # type: ignore
        ax.set_xscale("log")
        ax.set_yscale("log")
        set_axis_labels(ax, r"Absorption $\phi$", r"Transport $\gamma$")
        return

    # Resistance plot
    fig1, ax1 = figure(size="single")
    # sc = ax1.scatter(
    #     absorption, transport, c=resistance, cmap="inferno", marker="o", s=100
    # )
    sc = ax1.scatter(
        absorption, transport, c=resistance, cmap="inferno", marker="o", s=100
    )
    plt.colorbar(sc, ax=ax1, label=r"Resistance $\rho$")
    _set_labels(ax1)

    # Substomatal plot
    fig2, ax2 = figure(size="single")
    sc = ax2.scatter(absorption, transport, c=chii, cmap="inferno", marker="o", s=100)
    plt.colorbar(sc, ax=ax2, label=r"Substomatal $\chi_i$")
    _set_labels(ax2)

    # Mesophyll plot
    fig3, ax3 = figure(size="single")
    sc = ax3.scatter(absorption, transport, c=chim, cmap="inferno", marker="o", s=100)
    plt.colorbar(sc, ax=ax3, label=r"Mesophyll $\chi_m$")
    _set_labels(ax3)

    # Coefficient of variation plot
    fig4, ax4 = figure(size="single")
    sc = ax4.scatter(
        absorption, transport, c=variation, cmap="inferno", marker="o", s=100
    )
    plt.colorbar(sc, ax=ax4, label=r"Coefficient of Variation $\delta_m$")
    _set_labels(ax4)

    if output_dir is not None:
        out = Path(output_dir)
        save(fig1, out / "resistance.pdf")
        save(fig2, out / "substomatal.pdf")
        save(fig3, out / "mesophyll.pdf")
        save(fig4, out / "variation.pdf")

    if show:
        plt.show()
    return
