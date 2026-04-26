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
    "curved_flux_grad": "#903b7a",
    "top_flux_grad": "#8e0606",
    "mesophyll_flux_sol": "#317134",
    "stomatal_flux_grad": "#DB5A0A",
    "bottom_flux_grad": "#af1a1a",
    "total_flux_grad": "#000000",
}


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


def _std_layout(
    ax: plt.Axes,
    xlabel: str | None = None,
    ylabel: str | None = None,
    title: str | None = None,
    xlog: bool = True,
    ylog: bool = False,
    legend: bool = True,
    loc: str = "center left",
    ymin: float | None = None,
    ymax: float | None = None,
) -> None:
    set_axis_labels(ax, xlabel=xlabel, ylabel=ylabel)
    if title is not None:
        ax.set_title(title)
    gridlines(ax)
    if xlog:
        ax.set_xscale("log")
    if ylog:
        ax.set_yscale("log")
    if legend:
        ax.legend(loc=loc, bbox_to_anchor=(1.0, 1.0))
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(1.0, None)
    return


_LINESTYLES = ["--", "-"]


def create_qoi_figure(df1: pd.DataFrame, df2: pd.DataFrame, ax: plt.Axes) -> None:
    keys = ["substomatal_mean", "top_mean", "stomatal_flux_grad", "top_flux_grad"]
    labels = [r"$\chi_{s}$", r"$\chi_{ad}$", r"$Q_{s}$", r"$Q_{ad}$"]
    #
    ax.set_xscale("log")
    ax_ = ax.twinx()
    h = _npy(df1, "scale_factor")
    for ls, df in zip(_LINESTYLES, [df1, df2], strict=True):
        for key, label in zip(keys, labels, strict=True):
            _ax = ax_ if "flux" in key else ax
            _ax.plot(
                h,
                _npy(df, key),
                linestyle=ls,
                label=label if ls == "-" else "",
                color=colors[key],
            )
    ax.set_xlabel("Scale factor h []")
    ax.set_ylabel(r"Concentrations $\chi$ []", color=colors["substomatal_mean"])
    ax.tick_params(
        axis="y",
        colors=colors["substomatal_mean"],
        labelcolor=colors["substomatal_mean"],
    )
    ax.set_ylim(0.0, None)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 1.0))
    # change yaxis color of ax to match the flux colors
    ax_.set_ylabel(r"Flow rates $Q$ []", color=colors["top_flux_grad"])
    ax_.tick_params(
        axis="y", colors=colors["top_flux_grad"], labelcolor=colors["top_flux_grad"]
    )
    ax_.set_ylim(0.0, None)
    ax_.legend(loc="center left", bbox_to_anchor=(1.0, 0.0))
    return


def create_bc_adherence_figure(
    df1: pd.DataFrame, df2: pd.DataFrame, ax: plt.Axes
) -> None:
    h = _npy(df1, "scale_factor")
    for ls, df in zip(_LINESTYLES, [df1, df2], strict=True):
        # Neumann
        keys = ["curved_flux_grad", "bottom_flux_grad"]
        labels = [r"$\Sigma_R$ Neumann", r"$\Sigma_{ab}$ \ $\Sigma_s$ Neumann"]
        for key, label in zip(keys, labels, strict=True):
            ax.plot(
                h,
                _npy(df, key),
                linestyle=ls,
                label=label if ls == "-" else "",
                color=colors[key],
            )
        # Dirichlet
        data = [
            _rel_error(_npy(df, "par_chii"), _npy(df, "substomatal_mean")),
            _rel_error(_npy(df, "par_chit"), _npy(df, "top_mean")),
        ]
        keys = ["substomatal_mean", "top_mean"]
        labels = [r"$\Sigma_{s}$ Dirichlet", r"$\Sigma_{ad}$ Dirichlet"]
        for signal, key, label in zip(data, keys, labels, strict=True):
            ax.plot(
                h,
                signal,
                linestyle=ls,
                label=label if ls == "-" else "",
                color=colors[key],
            )
    #
    despine(ax)
    set_axis_labels(ax, xlabel="Scale factor h []", ylabel="Error []")
    ax.set_ylim(0.0, None)
    ax.set_xscale("log")
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 1.0))
    return


def create_convergence_figure(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    ax: plt.Axes,
) -> None:
    h = _npy(df1, "scale_factor")
    #
    keys = [
        "stomatal_flux_grad",
        "top_flux_grad",
        "curved_flux_grad",
        "total_flux_grad",
    ]
    labels = [r"$Q_{s}$", r"$Q_{ad}$", r"$Q_{R}$", r"$Q_{total}$"]
    for ls, df in zip(_LINESTYLES, [df1, df2], strict=True):
        for key, label in zip(keys, labels, strict=True):
            truth = _npy(df2, key)[np.argmin(h)]
            signal = _rel_error(truth, _npy(df, key))
            ax.plot(
                h,
                signal,
                linestyle=ls,
                label=label if ls == "-" else "",
                color=colors[key],
            )
    ax.hlines(
        1e-2, np.min(h), np.max(h), color="grey", linestyle="-.", label="1% threshold"
    )
    set_axis_labels(
        ax,
        xlabel="Scale factor h []",
        ylabel=r"Relative difference to CG2($h_{min}$) []",
    )
    despine(ax)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # ax.set_ylim(1e-6, None)
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 1.0))

    return


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
        ncols=3,
        size=(10.0, 2.0),
        gridspec_kw={"wspace": 1.0},
    )
    create_qoi_figure(df1, df2, axs[0])
    label_panel(axs[0], "(a)")
    #
    create_bc_adherence_figure(df1, df2, axs[1])
    label_panel(axs[1], "(b)")
    #
    create_convergence_figure(df1, df2, axs[2])
    label_panel(axs[2], "(c)")
    #
    save(fig, output_path)
    #
    if show:
        plt.show()
    plt.close("all")
    return
