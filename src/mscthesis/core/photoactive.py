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
    summary["resistance_active"] = (
        summary["substomatal_mean"] - summary["mesophyll_mean"]
    ) / summary["assimilation_rate"]

    # calculate the coefficient of variation over the mesophyll surfaces and airspace
    summary["mesophyll_variation"] = (
        np.sqrt(summary["mesophyll_var"]) / summary["mesophyll_mean"]
    )

    summary["consistency"] = summary["mesophyll_var"] / (
        (1 - summary["mesophyll_mean"])
        * (summary["mesophyll_mean"] - summary["compensation"])
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
            "assimilation_rate",
            "resistance_active",
            "consistency",
            "mesophyll_area_fraction",
            "porosity",
        ]
    ]

    return summary
