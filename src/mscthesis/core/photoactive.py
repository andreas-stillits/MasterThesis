from __future__ import annotations

import numpy as np
import pandas as pd

from ..log import log_call


@log_call()
def derive_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.copy()

    # calculate the observed resistance
    summary["assimilation_rate"] = np.abs(
        summary["mesophyll_flux_sol"] / summary["plug_area"]
    )
    summary["resistance_m"] = (
        summary["substomatal_mean"] - summary["mesophyll_mean"]
    ) / summary["assimilation_rate"]

    # calculate the coefficient of variation over the mesophyll surfaces and airspace
    summary["mesophyll_variation"] = (
        np.sqrt(summary["mesophyll_var"]) / summary["mesophyll_mean"]
    )
    summary["airspace_variation"] = (
        np.sqrt(summary["airspace_var"]) / summary["airspace_mean"]
    )

    # define the pipe resistance for each sample id as the resistance where specifier == 0
    sample_ids = summary["sample_id"].unique()
    for sample_id in sample_ids:
        specifier_0 = summary[
            (summary["sample_id"] == sample_id) & (summary["specifier"] == 0)
        ]
        if not specifier_0.empty:
            pipe_resistance_m = specifier_0["resistance_m"].values[0]
            summary.loc[summary["sample_id"] == sample_id, "pipe_resistance_m"] = (
                pipe_resistance_m
            )

    # filter the dataframe for clarity
    summary = summary[
        [
            "sample_id",
            "specifier",
            "absorption",
            "transport",
            "compensation",
            "substomatal_mean",
            "mesophyll_mean",
            "mesophyll_variation",
            "airspace_mean",
            "airspace_variation",
            "assimilation_rate",
            "assimilation_substomatal",
            "assimilation_mesophyll_mean",
            "resistance_m",
            "pipe_resistance_m",
            "stomatal_area_fraction",
            "mesophyll_area_fraction",
            "plug_area",
        ]
    ]

    return summary
