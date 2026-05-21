from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd

from ....config import ProjectConfig
from ....core.diffusion import derive_summary
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

        for specifier in config.search.stomatal_aspect_set:
            diffusion_paths = sample_paths.diffusion(specifier)

            if diffusion_paths.manifest.exists():
                content: dict[str, Any] = {
                    "sample_id": sample_id,
                    "synthesis_type": synthesis_type,
                    "target_plug_aspect": target_plug_aspect,
                    "specifier": specifier,
                    "solver": "diffusion",
                    "surface_tortuosity_factor": geo["surfaces"]["tortuosity_factor"],
                    "surface_lateral": geo["surfaces"]["lateral_lengthening"],
                    "surface_post_average": geo["surfaces"]["post_average"],
                    "top_tortuosity_factor": geo["top"]["tortuosity_factor"],
                    "top_lateral": geo["top"]["lateral_lengthening"],
                    "top_post_average": geo["top"]["post_average"],
                }
                plug_aspect, stomatal_aspect = fetch_from_manifest(
                    sample_paths.meshing(specifier).manifest.require(),
                    "plug_aspect",
                    "stomatal_aspect",
                )
                content["plug_aspect"] = plug_aspect
                content["stomatal_aspect"] = stomatal_aspect

                with open(diffusion_paths.manifest.path) as file:
                    manifest = json.load(file)
                    meta = manifest.get("meta", {})
                    del meta["parameters"]
                    content.update(meta)
                index.append(content)
            #
            else:
                continue

    df = pd.DataFrame(index)
    df.reset_index(drop=True, inplace=True)
    save_dataframe(paths.diffusion_index.path, df)

    #
    summary = derive_summary(df)
    save_dataframe(paths.diffusion_summary.path, summary)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "compile-diffusion-solutions",
        help="Compile selected diffusion solutions results from search",
    )
    parser.set_defaults(cmd=_cmd)
    return
