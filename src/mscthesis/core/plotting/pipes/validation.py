from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stillib_plotting import (
    figure,
    gridlines,
    label_panel,
    panel_grid,
    save,
    set_axis_labels,
    use_style,
)

colors = {
    "substomatal_mean": "#052D76",
    "mesophyll_mean": "#087313",
    "top_mean": "#63148d",
    "curved_flux_grad": "#903b7a",
    "top_flux_grad": "#9467bd",
    "mesophyll_flux_sol": "#317134",
    "stomatal_flux_grad": "#283375",
}


def _npy(df: pd.DataFrame, key: str) -> np.ndarray:
    return np.abs(df[key].to_numpy())


def _rel_error(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_error = np.abs((x - y) / x)
    rel_error[np.isclose(rel_error, 0.0)] = np.nan
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


def create_qoi_figure(df: pd.DataFrame, ax0: plt.Axes, ax1: plt.Axes) -> None:
    df1, df2 = _split_by_order(df)
    x = _npy(df1, "scale_factor")
    #
    keys_conc = ["substomatal_mean", "mesophyll_mean", "top_mean"]
    labels_conc = [r"$C_{st}$", r"$C_{m}$", r"$C_{top}$"]

    keys_flux = ["mesophyll_flux_sol", "top_flux_grad"]
    labels_flux = [r"$A_{m}$", r"$A_{top}$"]

    for df, linestyle in zip([df1, df2], ["-", "--"], strict=True):
        labels_conc = ["", "", ""] if linestyle == "--" else labels_conc
        labels_flux = ["", ""] if linestyle == "--" else labels_flux
        for key, label in zip(keys_conc, labels_conc, strict=True):
            ax0.plot(
                x, _npy(df, key), linestyle=linestyle, label=label, color=colors[key]
            )
        for key, label in zip(keys_flux, labels_flux, strict=True):
            ax1.plot(
                x, _npy(df, key), linestyle=linestyle, label=label, color=colors[key]
            )

    _std_layout(
        ax0, ylabel="Mean concentrations", title="Conc. QoI", ymin=0.0, ymax=1.05
    )
    _std_layout(ax1, xlabel="Resolution factor", ylabel="Flow rates", title="Flux QoI")
    return


def create_bc_adherence_figure(df: pd.DataFrame, ax0: plt.Axes, ax1: plt.Axes) -> None:
    df1, df2 = _split_by_order(df)
    x = _npy(df1, "scale_factor")

    #
    for df, linestyle in zip([df1, df2], ["-", "--"], strict=True):
        flux_m_direct = _npy(df, "mesophyll_flux_grad")
        flux_m_equiv = _npy(df, "mesophyll_flux_sol")
        flux_truth = flux_m_equiv[np.argmin(x)]
        flux_curved = _npy(df, "curved_flux_grad") / flux_truth
        fluc_top = _npy(df, "top_flux_grad") / flux_truth
        ax0.plot(
            x,
            flux_curved,
            linestyle=linestyle,
            label="Curved BC" if linestyle == "-" else "",
            color=colors["curved_flux_grad"],
        )
        ax0.plot(
            x,
            fluc_top,
            linestyle=linestyle,
            label="Top BC" if linestyle == "-" else "",
            color=colors["top_flux_grad"],
        )
        #
        ax1.plot(
            x,
            _rel_error(flux_m_equiv, flux_m_direct),
            linestyle=linestyle,
            label="mesophyll" if linestyle == "-" else "",
            color=colors["mesophyll_flux_sol"],
        )

    _std_layout(ax0, ylabel="Relative flux magnitude", title="Neumann BC adherence")
    _std_layout(
        ax1,
        xlabel="Resolution factor",
        ylabel="Relative difference",
        title="Robin BC adherence",
        ylog=False,
    )
    return


def create_convergence_figure(
    df: pd.DataFrame,
    ax: plt.Axes,
    tolerance: float = 1e-2,
    ignore_threshold: float = 1e-1,
) -> None:
    df1, df2 = _split_by_order(df)
    x = _npy(df1, "scale_factor")
    #
    keys_conc = ["substomatal_mean", "mesophyll_mean", "top_mean"]
    labels_conc = [r"$C_{st}$", r"$C_{m}$", r"$C_{top}$"]
    keys_flux = ["mesophyll_flux_sol", "top_flux_grad"]
    labels_flux = [r"$A_{m}$", r"$A_{top}$"]
    ax.hlines(
        tolerance,
        np.min(x),
        np.max(x),
        colors="#af1a1a",
        linestyles="-.",
        linewidth=2.0,
        label=f"{100*tolerance:.0f}% tolerance",
    )
    true_flux = _npy(df2, "mesophyll_flux_sol")[np.argmin(x)]
    for df, linestyle in zip([df1, df2], ["-", "--"], strict=True):
        labels_conc = ["", "", ""] if linestyle == "--" else labels_conc
        labels_flux = ["", ""] if linestyle == "--" else labels_flux
        #
        for key, label in zip(keys_conc, labels_conc, strict=True):
            q1s = _npy(df, key)
            q2 = _npy(df2, key)[np.argmin(x)]
            ax.plot(
                x,
                _rel_error(q2, q1s),
                label=label,
                linestyle=linestyle,
                color=colors[key],
            )

        for key, label in zip(keys_flux, labels_flux, strict=True):
            q1s = _npy(df, key)
            q2 = true_flux if key != "top_flux_grad" else _npy(df2, key)[np.argmin(x)]
            if (
                key == "top_flux_grad"
                and q1s[np.argmin(x)] / true_flux < ignore_threshold
            ):
                continue

            ax.plot(
                x,
                _rel_error(q2, q1s),
                label=label,
                linestyle=linestyle,
                color=colors[key],
            )

    _std_layout(
        ax,
        xlabel="Resolution factor",
        ylabel="Relative difference to CG2",
        title="Convergence to CG2 solution",
        ylog=True,
        ymin=1e-4,
    )

    return


def plot_validation(
    df: pd.DataFrame,
    output_path: str | Path,
    tolerance: float = 1e-2,
    ignore_threshold: float = 1e-1,
    show: bool = True,
) -> None:
    """
    Plot validation results and save as .pdf
    """
    use_style()
    fig = plt.figure(figsize=(12.0, 4.2))
    spec = fig.add_gridspec(nrows=2, ncols=3, hspace=0.5, wspace=0.8)
    ax10 = fig.add_subplot(spec[0, 0])
    ax11 = fig.add_subplot(spec[1, 0], sharex=ax10)
    ax10.tick_params(axis="x", which="both", labelbottom=False)
    ax20 = fig.add_subplot(spec[0, 1])
    ax21 = fig.add_subplot(spec[1, 1], sharex=ax20)
    ax20.tick_params(axis="x", which="both", labelbottom=False)
    ax3 = fig.add_subplot(spec[:, 2])
    #
    create_qoi_figure(df, ax10, ax11)
    label_panel(ax10, "(a)")
    label_panel(ax11, "(b)")
    #
    create_bc_adherence_figure(df, ax20, ax21)
    label_panel(ax20, "(c)")
    label_panel(ax21, "(d)")
    #
    create_convergence_figure(
        df, ax3, tolerance=tolerance, ignore_threshold=ignore_threshold
    )
    label_panel(ax3, "(e)")
    save(fig, output_path)
    #
    if show:
        plt.show()
    plt.close("all")
    return
