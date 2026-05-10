from __future__ import annotations

import numpy as np
import pandas as pd

from ..log import log_call


@log_call()
def derive_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = df.copy()

    # for all specifier == 0, set stomatal_aspect to plug_aspect
    summary.loc[summary["specifier"] == 0, "stomatal_aspect"] = summary.loc[
        summary["specifier"] == 0, "plug_aspect"
    ]

    # calculate the observed top and mean surface resistance
    summary["assimilation_rate"] = np.abs(
        summary["top_flux_grad"] / summary["plug_area"]
    )
    summary["resistance_t"] = (
        summary["substomatal_mean"] - summary["top_mean"]
    ) / summary["assimilation_rate"]
    summary["resistance_m"] = (
        summary["substomatal_mean"] - summary["mesophyll_mean"]
    ) / summary["assimilation_rate"]

    # define the pipe resistance for each sample id as the resistance where specifier == 0
    sample_ids = summary["sample_id"].unique()
    for sample_id in sample_ids:
        specifier_0 = summary[
            (summary["sample_id"] == sample_id) & (summary["specifier"] == 0)
        ]
        if not specifier_0.empty:
            pipe_resistance_t = specifier_0["resistance_t"].values[0]
            pipe_resistance_m = specifier_0["resistance_m"].values[0]
            summary.loc[summary["sample_id"] == sample_id, "pipe_resistance_t"] = (
                pipe_resistance_t
            )
            summary.loc[summary["sample_id"] == sample_id, "pipe_resistance_m"] = (
                pipe_resistance_m
            )

    # define the calculated inlet resistance
    summary["inlet_resistance"] = (
        0.75
        * summary["plug_aspect"]
        * (summary["plug_aspect"] / summary["stomatal_aspect"] - 1)
    )

    # define the total calculated resistances
    summary["calculated_resistance_t"] = (
        summary["pipe_resistance_t"] + summary["inlet_resistance"]
    )
    summary["calculated_resistance_m"] = (
        summary["pipe_resistance_m"] + summary["inlet_resistance"]
    )

    # calculate the standard formulas for ias resistance
    summary["standard_tomas"] = (
        summary["surface_tortuosity_factor"] / summary["porosity"] / 2
    )
    summary["standard_earles"] = (
        summary["surface_tortuosity_factor"]
        * summary["surface_lateral"]
        / summary["porosity"]
        / 2
    )

    # filter the dataframe for clarity
    summary = summary[
        [
            "sample_id",
            "specifier",
            "synthesis_type",
            "plug_aspect",
            "stomatal_aspect",
            "porosity",
            "surface_tortuosity_factor",
            "surface_lateral",
            "top_tortuosity_factor",
            "top_lateral",
            "assimilation_rate",
            "resistance_t",
            "resistance_m",
            "pipe_resistance_t",
            "pipe_resistance_m",
            "inlet_resistance",
            "calculated_resistance_t",
            "calculated_resistance_m",
            "standard_tomas",
            "standard_earles",
            "stomatal_area_fraction",
            "mesophyll_area_fraction",
            "plug_area",
        ]
    ]

    return summary
