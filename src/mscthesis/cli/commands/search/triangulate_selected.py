from __future__ import annotations

import argparse
import os
import subprocess
from typing import Any

from stillib_parallelism import collect, print_progress

from ....config import ProjectConfig, save_config
from ....core.io import load_dataframe, load_voxels, save_surface_mesh
from ....core.meshing.triangulation import triangulate_voxels
from ....manifest import dump_manifest, fetch_from_manifest
from ....paths import ProjectPaths

_STATE: dict[str, Any] = {}


def initializer(config: ProjectConfig, force: bool) -> None:
    _STATE["config"] = config
    _STATE["paths"] = ProjectPaths(config.behavior.storage_root)
    _STATE["force"] = force

    # construct set of parameters for retry logic
    initial_target: int = config.triangulation.elements_per_cell
    increment: int = config.search.selected.retries["triangulation"][
        "elements_per_cell_increment"
    ]
    max_attempts: int = config.search.selected.retries["triangulation"]["max_attempts"]
    params = [initial_target + i * increment for i in range(max_attempts)]
    _STATE["elements_per_cell_set"] = params

    return


def execute_triangulation(sample_id: str) -> None:
    config: ProjectConfig = _STATE["config"]
    paths: ProjectPaths = _STATE["paths"]
    force: bool = _STATE["force"]
    elements_per_cell_set: list[int] = _STATE["elements_per_cell_set"]

    attempt: int = 0
    sample_config = config.model_dump()

    # check if .brep already exists and skip if not forced
    sample_paths = paths.selected_sample(sample_id)
    sample_paths.triangulation.root.ensure()
    brep_path = sample_paths.triangulation.cadmodel

    #
    if not brep_path.exists() or force:
        voxels = load_voxels(sample_paths.synthesis.voxels.require())
        num_cells_placed = fetch_from_manifest(
            sample_paths.synthesis.manifest.require(), "num_cells_placed"
        )
        assert isinstance(num_cells_placed, int)

        # retry loop with different parameters
        for idx, elements_per_cell in enumerate(elements_per_cell_set):
            attempt = idx
            decimation_target = elements_per_cell * num_cells_placed
            try:
                mesh, metadata = triangulate_voxels(
                    voxels,
                    config.triangulation.smoothing_iterations,
                    decimation_target,
                    config.triangulation.shrinkage_tolerance,
                    config.triangulation.spacing,
                )

                if not metadata["success"]:
                    raise RuntimeError(
                        "Triangulation failed for .STL generation with current parameters."
                    )
                # if successful, export to BREP, save mesh, config, and manifest and break out of retry loop
                save_surface_mesh(sample_paths.triangulation.mesh.path, mesh)

                env = os.environ.copy()
                env["INPUT_STL"] = os.path.abspath(sample_paths.triangulation.mesh.path)
                env["OUTPUT_BREP"] = os.path.abspath(
                    sample_paths.triangulation.cadmodel.path
                )
                try:
                    process = subprocess.run(
                        [
                            config.triangulation.freecad_cmd,
                            config.triangulation.freecad_script_path,
                        ],
                        env=env,
                    )
                    if process.returncode != 0:
                        raise RuntimeError(
                            f"FreeCAD command failed with return code {process.returncode}"
                        )
                except Exception as exc:
                    raise RuntimeError("Failed to export BREP using FreeCAD") from exc

                metadata["brep_exported"] = True

                break

            except Exception:
                continue
        else:
            raise RuntimeError("Triangulation failed for all parameter sets.")

        # we reach this point if brep export was successful
        # save config
        sample_config["triangulation"]["elements_per_cell"] = elements_per_cell_set[
            attempt
        ]
        save_config(
            sample_paths.triangulation.config.path,
            ProjectConfig.model_validate(sample_config),
            "triangulation",
        )
        # save manifest
        metadata["attempts"] = attempt + 1
        dump_manifest(
            sample_paths.triangulation.manifest.path,
            command_name="triangulate-selected",
            sample_id=sample_id,
            inputs={"voxels": sample_paths.synthesis.voxels.require()},
            outputs={
                "mesh": sample_paths.triangulation.mesh.path,
                "cad_model": sample_paths.triangulation.cadmodel.path,
            },
            metadata=metadata,
            tool_version=config.meta.project_version,
        )

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    index = load_dataframe(paths.index.require())
    selected_ids = index[index["selected"]]["sample_id"].tolist()

    report = collect(
        selected_ids,
        execute_triangulation,
        max_workers=config.max_workers,
        initializer=initializer,
        initargs=(config, args.force),
        progress_callback=print_progress,
        ordering="completion",
        error_policy="collect",
    )

    if not report.ok:
        print(f"Triangulation completed with {len(report.failures)} errors:")
        for failure in report.failures:
            print(
                f"- Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}",
                "\n",
            )
        print(
            "saving list of failures to file: '<storage_root>/triangulation_failures.txt'"
        )
        failures_path = paths.failures.ensure() / "triangulation_failures.txt"
        with open(failures_path, "w") as f:
            for failure in report.failures:
                f.write(
                    f"Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}\n"
                )
    else:
        print("Triangulation completed successfully for all selected samples.")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "triangulate-selected", help="Triangulate the selected samples."
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force triangulation even if the .brep already exists.",
    )
    parser.set_defaults(cmd=_cmd)
    return
