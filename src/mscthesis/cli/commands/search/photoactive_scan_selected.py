from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import pandas as pd
from stillib_parallelism import collect, print_progress

from mscthesis.core.plotting.sample.scanning import plot_scanning_results

from ....config import ProjectConfig, save_config
from ....core.io import (
    load_dataframe,
    load_volumetric_mesh,
    save_dataframe,
    save_fem_solution,
)
from ....core.solvers import (
    DiffusionSolver,
    MeshContext,
    PhotoactiveSolver,
    SolverContext,
)
from ....manifest import dump_manifest
from ....paths import ProjectPaths

_STATE: dict[str, Any] = {}


def initializer(config: ProjectConfig, force: bool) -> None:
    _STATE["config"] = config
    _STATE["paths"] = ProjectPaths(config.behavior.storage_root)
    _STATE["force"] = force
    return


def _get_chii(
    transport: float,
    absorption: float,
    compensation: float,
    preliminary_resistance: float,
) -> float:
    return 1 - (1 - compensation) / (
        1 + transport / preliminary_resistance + transport / absorption
    )


def _get_logspace(start: float, stop: float, num: int) -> np.ndarray:
    return np.logspace(np.log10(start), np.log10(stop), num)


def execute_scanning(sample_id: str) -> None:
    config: ProjectConfig = _STATE["config"]
    paths: ProjectPaths = _STATE["paths"]
    force: bool = _STATE["force"]

    sample_paths = paths.selected_sample(sample_id)

    # get preliminary resistance from diffusion_summary.csv
    diffusion_summary = load_dataframe(paths.diffusion_summary.require())
    sample_summary = diffusion_summary[diffusion_summary["sample_id"] == sample_id]
    if sample_summary.empty:
        raise ValueError(
            f"No diffusion summary found for sample_id: {sample_id} in diffusion_summary.csv"
        )
    preliminary_resistance = float(sample_summary.iloc[0]["resistance_m"])

    for specifier in config.search.stomatal_aspect_set:
        scanning_paths = sample_paths.scanning(specifier)
        scanning_paths.root.ensure()

        if not scanning_paths.scan.exists() or force:
            # load the mesh
            mesh_ctx: MeshContext = load_volumetric_mesh(
                sample_paths.meshing(specifier).mesh.require()
            )
            solver = PhotoactiveSolver(
                SolverContext(**config.solver_ctx.model_dump()),
                mesh_ctx,
            )

            results: list[dict[str, Any]] = []
            compensation = config.scanning.compensation

            for transport, absorption in (
                (transport, absorption)
                for transport in _get_logspace(
                    config.scanning.transport_min,
                    config.scanning.transport_max,
                    config.scanning.transport_num,
                )
                for absorption in _get_logspace(
                    config.scanning.absorption_min,
                    config.scanning.absorption_max,
                    config.scanning.absorption_num,
                )
            ):
                chii = _get_chii(
                    transport,
                    absorption,
                    compensation,
                    preliminary_resistance,
                )
                solution, analysis = solver.solve_for(chii, absorption, compensation)
                content = {
                    "sample_id": sample_id,
                    "specifier": specifier,
                    "transport": transport,
                    "absorption": absorption,
                    "compensation": compensation,
                    "chii": chii,
                    **analysis,
                }
                results.append(content)

            # save results to csv
            df = pd.DataFrame(results)
            save_dataframe(scanning_paths.scan.path, df)

            # save config
            save_config(scanning_paths.config.path, config, "scanning", "solver_ctx")

            # save manifest
            dump_manifest(
                scanning_paths.manifest.path,
                command_name="photoactive-scan-selected",
                sample_id=sample_id,
                inputs={
                    "mesh": sample_paths.meshing(specifier).mesh.path,
                    "diffusion_summary": paths.diffusion_summary.path,
                },
                outputs={"scan": scanning_paths.scan.path},
                metadata={},
                tool_version=config.meta.project_version,
            )

            del solver

        else:
            continue

        # plot results
        dataframe = load_dataframe(scanning_paths.scan.require())
        plot_scanning_results(dataframe, scanning_paths.plots.ensure(), show=False)

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    index = load_dataframe(paths.index.require())
    selected_ids = index[index["selected"]]["sample_id"].tolist()

    report = collect(
        selected_ids,
        execute_scanning,
        max_workers=config.max_workers,
        initializer=initializer,
        initargs=(config, args.force),
        progress_callback=print_progress,
        ordering="completion",
        error_policy="collect",
    )

    if not report.ok:
        print(f"Scanning completed with {len(report.failures)} errors:")
        for failure in report.failures:
            print(
                f"- Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}",
                "\n",
            )
        print(
            "saving list of failures to file: '<storage_root>/failures/scanning_failures.txt'"
        )
        failures_path = paths.failures.ensure() / "scanning_failures.txt"
        with open(failures_path, "w") as f:
            for failure in report.failures:
                f.write(
                    f"Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}\n"
                )
    else:
        print("Scanning completed successfully for all selected samples.")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "photoactive-scan-selected",
        help="Solve the diffusion problem for the selected samples.",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force solving even if the solutions already exists.",
    )
    parser.set_defaults(cmd=_cmd)
    return
