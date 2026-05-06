from __future__ import annotations

import argparse
from typing import Any

from stillib_parallelism import collect, print_progress

from mscthesis.core.io import load_dataframe

from ....config import ProjectConfig, save_config
from ....core.meshing.gmeshing import build_sample_model, mesh_model, mesh_porous_model
from ....manifest import dump_manifest
from ....paths import ProjectPaths

_STATE: dict[str, Any] = {}


def initializer(config: ProjectConfig, force: bool) -> None:
    _STATE["config"] = config
    _STATE["paths"] = ProjectPaths(config.behavior.storage_root)
    _STATE["force"] = force
    return


def execute_meshing(sample_id: str) -> None:
    config: ProjectConfig = _STATE["config"]
    paths: ProjectPaths = _STATE["paths"]
    force: bool = _STATE["force"]

    sample_paths = paths.selected_sample(sample_id)

    for specifier, stomatal_aspect in config.search.stomatal_aspect_set.items():
        meshing_paths = sample_paths.meshing(specifier)
        meshing_paths.root.ensure()

        if not meshing_paths.mesh.exists() or force:
            airspace_tag, plug_aspect_model = build_sample_model(
                sample_paths.triangulation.cadmodel.require(),
                config.meshing.boundary_margin,
                config.meshing.substomatal_margin,
                config.meshing.atol,
            )

            config_dict = config.model_dump()

            if specifier != 0:
                config_dict["meshing"]["mesh_field"]["stomatal_aspect"] = (
                    stomatal_aspect
                )

                mesh_model(
                    meshing_paths.mesh.path,
                    airspace_tag,
                    plug_aspect_model,
                    **config_dict["meshing"]["mesh_field"],
                )
            else:
                config_dict["meshing"]["mesh_field"]["stomatal_aspect"] = (
                    plug_aspect_model
                )
                subdict = config_dict["meshing"]["mesh_field"]
                # delete the stomatal aspect key since it is not used in this variant
                del subdict["stomatal_aspect"]

                mesh_porous_model(
                    meshing_paths.mesh.path,
                    airspace_tag,
                    plug_aspect_model,
                    **subdict,
                )

            # save config and manifest
            save_config(
                meshing_paths.config.path,
                ProjectConfig.model_validate(config_dict),
                "meshing",
            )
            dump_manifest(
                meshing_paths.manifest.path,
                command_name="mesh-selected",
                sample_id=sample_id,
                inputs={"cadmodel": sample_paths.triangulation.cadmodel.path},
                outputs={"mesh": meshing_paths.mesh.path},
                metadata={
                    "plug_aspect": plug_aspect_model,
                    "stomatal_aspect": stomatal_aspect,
                },
                tool_version=config.meta.project_version,
            )
        else:
            continue

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    index = load_dataframe(paths.index.require())
    selected_ids = index[index["selected"]]["sample_id"].tolist()

    report = collect(
        selected_ids,
        execute_meshing,
        max_workers=config.max_workers,
        initializer=initializer,
        initargs=(config, args.force),
        progress_callback=print_progress,
        ordering="completion",
        error_policy="collect",
    )

    if not report.ok:
        print(f"Meshing completed with {len(report.failures)} errors:")
        for failure in report.failures:
            print(
                f"- Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}",
                "\n",
            )
        print("saving list of failures to file: '<storage_root>/meshing_failures.txt'")
        failures_path = paths.base / "meshing_failures.txt"
        with open(failures_path, "w") as f:
            for failure in report.failures:
                f.write(
                    f"Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}\n"
                )
    else:
        print("Meshing completed successfully for all selected samples.")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("mesh-selected", help="Mesh the selected samples.")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force meshing even if the .msh already exists.",
    )
    parser.set_defaults(cmd=_cmd)
    return
