from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stillib_plotting import (
    despine,
    figure,
    gridlines,
    label_panel,
    panel_grid,
    save,
    set_axis_labels,
    use_style,
)

colors = {
    "substomatal_mean": "#053389",
    "mesophyll_mean": "#087313",
    "top_mean": "#0483d8",
    "airspace_mean": "#1e7e1e",
    "curved_flux_grad": "#903b7a",
    "top_flux_grad": "#8e0606",
    "mesophyll_flux_sol": "#317134",
    "stomatal_flux_grad": "#DB5A0A",
    "bottom_flux_grad": "#af1a1a",
    "total_flux_grad": "#000000",
    "resistance": "#2F8CC6",
}

_LINESTYLES = ["--", "-"]


def _npy(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.abs(df[key].to_numpy())


def _rel_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.abs((x - y) / x)
    rel_error[np.isclose(rel_error, 0.0, atol=1e-9)] = np.nan
    return rel_error


def _split_by_order(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = (df[df["order"] == 1], df[df["order"] == 2])
    for df in dfs:
        df.sort_values("scale_factor", inplace=True)
    return dfs


def plot_validation(
    df: pd.DataFrame,
    output_path: str | Path,
    show: bool = True,
) -> None:
    """
    Plot validation results and save as .pdf
    """
    use_style()
    df1, df2 = _split_by_order(df)

    fig, axs = panel_grid(
        nrows=1,
        ncols=2,
        size="double",
        gridspec_kw={"wspace": 0.50},
    )

    h = _npy(df2, "scale_factor")
    true_flux = _npy(df2, "top_flux_grad")[np.argmin(h)]

    # plot boundary values relative to target values
    def _fill_in(ax: plt.Axes, df: pd.DataFrame, linestyle: str) -> None:
        keys = ["curved_flux_grad", "total_flux_grad"]
        labels = [r"$\Sigma_R$", r"$\partial \Omega$"]
        for key, label in zip(keys, labels):
            ax.plot(
                h,
                _npy(df, key) / true_flux,
                color=colors[key],
                label=label if linestyle == "-" else "",
                linestyle=linestyle,
            )

    for df, ls in zip([df1, df2], _LINESTYLES):
        _fill_in(axs[0], df, ls)
    label_panel(axs[0], "(a)")
    set_axis_labels(
        axs[0],
        xlabel="Scale factor h []",
        ylabel=r"Integral boundary flux relative to $A_N$ []",
    )
    axs[0].set_xscale("log")
    axs[0].legend(loc="center left", bbox_to_anchor=(1.0, 1.0))

    # plot convergence of flux and solution to CG2 finest resolution values
    keys = [
        "top_flux_grad",
        "substomatal_mean",
        "top_mean",
        "airspace_mean",
        "resistance",
    ]
    targets = {key: _npy(df2, key)[np.argmin(h)] for key in keys}
    labels = [
        r"$Q_{ad}$",
        r"$\chi_{s}$",
        r"$\chi_{ad}$",
        r"$\chi_{ias}$",
        r"Resistance $\rho$",
    ]

    for df, ls in zip([df1, df2], _LINESTYLES):
        for key, label in zip(keys, labels):
            signal = _rel_error(targets[key], _npy(df, key))
            if np.all(np.isnan(signal)):
                continue
            axs[1].plot(
                h,
                signal,
                color=colors[key],
                label=label if ls == "-" else "",
                linestyle=ls,
            )
    label_panel(axs[1], "(b)")
    set_axis_labels(
        axs[1],
        xlabel="Scale factor h []",
        ylabel=r"Relative difference to CG2($h_{min}$) []",
    )
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].legend(loc="center left", bbox_to_anchor=(1.0, 1.0))

    #
    save(fig, output_path)
    #
    if show:
        plt.show()
    plt.close("all")
    return
