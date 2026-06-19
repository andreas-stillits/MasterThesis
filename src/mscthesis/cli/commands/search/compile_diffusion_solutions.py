from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from ....config import ProjectConfig
from ....core.io import save_dataframe
from ....manifest import fetch_from_manifest
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    paths.selected.ensure()

    # load a list of selected samples
    selected = [p for p in paths.selected.path.iterdir() if p.is_dir()]
    selected_ids = [s.name for s in selected if s.name.isdigit()]

    # compile index
    index: list[dict[str, Any]] = []
    for sample_id in selected_ids:
        sample_paths = paths.selected_sample(sample_id)
        synthesis_type, target_plug_aspect = fetch_from_manifest(
            sample_paths.synthesis.manifest.require(), "type", "plug_aspect"
        )
        # get geometry information
        geo: dict[str, Any] = json.loads(
            sample_paths.synthesis.geometry.require().read_text(encoding="utf-8")
        )

        tortuosity: float = geo["surfaces"]["tortuosity_factor"]
        lateral: float = geo["surfaces"]["lateral_lengthening"]

        for specifier in config.search.stomatal_aspect_set:

            content: dict[str, Any] = {
                "type": synthesis_type,
                "sample_id": sample_id,
                "specifier": specifier,
            }

            # extract from diffusion, neumann, and dirichlet
            diffusion_paths = sample_paths.diffusion(specifier)

            if diffusion_paths.manifest.exists():
                (
                    plug_area,
                    stomatal_area,
                    porosity,
                    surface_centroid,
                    substomatal_mean,
                    top_mean,
                    mesophyll_mean,
                    top_flux_grad,
                ) = fetch_from_manifest(
                    diffusion_paths.manifest.require(),
                    "plug_area",
                    "stomatal_area",
                    "porosity",
                    "surface_centroid",
                    "substomatal_mean",
                    "top_mean",
                    "mesophyll_mean",
                    "top_flux_grad",
                )
                content["plug_aspect"] = np.sqrt(plug_area / np.pi)
                content["stomatal_aspect"] = np.sqrt(stomatal_area / np.pi)
                content["r_empty_calc"] = 1.0 + (
                    np.pi
                    / 4
                    * content["plug_aspect"]
                    * (content["plug_aspect"] / content["stomatal_aspect"] - 1)
                    if content["stomatal_aspect"] < 0.95 * content["plug_aspect"]
                    else 0.0
                )
                content["tomas"] = tortuosity / porosity / 2
                content["earles"] = tortuosity * lateral / porosity / 2
                content["surface_centroid"] = surface_centroid
                content["r_porous_top"] = np.abs(
                    (substomatal_mean - top_mean) / (top_flux_grad / plug_area)
                )
                content["r_porous_mean"] = np.abs(
                    (substomatal_mean - mesophyll_mean) / (top_flux_grad / plug_area)
                )
            else:
                raise FileNotFoundError(
                    f"Diffusion manifest not found for sample {sample_id} specifier {specifier}"
                )

            dirichlet_paths = sample_paths.dirichlet(specifier)

            if dirichlet_paths.manifest.exists():
                substomatal_mean, mesophyll_mean, mesophyll_flux_grad = (
                    fetch_from_manifest(
                        dirichlet_paths.manifest.require(),
                        "substomatal_mean",
                        "mesophyll_mean",
                        "mesophyll_flux_grad",
                    )
                )
                content["r_dirichlet"] = np.abs(
                    (substomatal_mean - mesophyll_mean)
                    / (mesophyll_flux_grad / plug_area)
                )
            else:
                raise FileNotFoundError(
                    f"Dirichlet manifest not found for sample {sample_id} specifier {specifier}"
                )

            neumann_paths = sample_paths.neumann(specifier)

            if neumann_paths.manifest.exists():
                substomatal_mean, mesophyll_mean, mesophyll_flux_grad = (
                    fetch_from_manifest(
                        neumann_paths.manifest.require(),
                        "substomatal_mean",
                        "mesophyll_mean",
                        "mesophyll_flux_grad",
                    )
                )
                content["r_neumann"] = np.abs(
                    (substomatal_mean - mesophyll_mean)
                    / (mesophyll_flux_grad / plug_area)
                )
            else:
                raise FileNotFoundError(
                    f"Neumann manifest not found for sample {sample_id} specifier {specifier}"
                )

            empty_paths = sample_paths.empty(specifier)

            if empty_paths.manifest.exists():
                substomatal_mean, top_mean, top_flux_grad = fetch_from_manifest(
                    empty_paths.manifest.require(),
                    "substomatal_mean",
                    "top_mean",
                    "top_flux_grad",
                )
                content["r_empty"] = np.abs(
                    (substomatal_mean - top_mean) / (top_flux_grad / plug_area)
                )
            else:
                raise FileNotFoundError(
                    f"Empty manifest not found for sample {sample_id} specifier {specifier}"
                )

            index.append(content)

    df = pd.DataFrame(index)
    df.reset_index(drop=True, inplace=True)

    sample_ids = df["sample_id"].unique()
    for sample_id in sample_ids:
        specifier_0 = df[(df["sample_id"] == sample_id) & (df["specifier"] == 0)]
        if not specifier_0.empty:
            r_porous_top = specifier_0["r_porous_top"].values[0]
            r_porous_mean = specifier_0["r_porous_mean"].values[0]
            df.loc[df["sample_id"] == sample_id, "r_porous_top_0"] = r_porous_top
            df.loc[df["sample_id"] == sample_id, "r_porous_mean_0"] = r_porous_mean

    df.reset_index(drop=True, inplace=True)
    save_dataframe(paths.diffusion_index.path, df)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compile-diffusion-solutions",
        help="Compile selected diffusion solutions results from search",
    )
    parser.set_defaults(cmd=_cmd)
    return
