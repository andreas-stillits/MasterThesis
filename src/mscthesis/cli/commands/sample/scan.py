from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from stillib_parallelism import collect, print_progress

from ....config import ProjectConfig, save_config
from ....core.io import load_dataframe, load_volumetric_mesh, save_dataframe
from ....core.plotting.sample.scanning import plot_scanning_results
from ....core.solvers import MeshContext, PhotoactiveSolver, SolverContext
from ....ids import validate_sample_id
from ....manifest import dump_manifest
from ....paths import ProjectPaths


@dataclass
class Task:
    index: int
    chii: float
    absorption: float
    compensation: float


def _get_logspace(start: float, stop: float, num: int) -> np.ndarray:
    return np.logspace(np.log10(start), np.log10(stop), num)


def _get_chii(
    transport: float,
    absorption: float,
    compensation: float,
    preliminary_resistance: float,
) -> float:
    return 1 - (1 - compensation) / (
        1 + transport / preliminary_resistance + transport / absorption
    )


def make_tasks(
    transport_min: float,
    transport_max: float,
    transport_num: int,
    absorption_min: float,
    absorption_max: float,
    absorption_num: int,
    compensation: float,
    preliminary_resistance: float,
) -> list[Task]:
    transport_values = _get_logspace(transport_min, transport_max, transport_num)
    absorption_values = _get_logspace(absorption_min, absorption_max, absorption_num)
    tasks: list[Task] = []

    for idx, (transport, absorption) in enumerate(
        (transport, absorption)
        for transport in transport_values
        for absorption in absorption_values
    ):
        chii = _get_chii(transport, absorption, compensation, preliminary_resistance)
        tasks.append(Task(idx, chii, absorption, compensation))
    return tasks


_STATE: dict[str, Any] = {}


def intializer(config: ProjectConfig, sample_id: str) -> None:
    global _STATE
    paths = ProjectPaths(config.behavior.storage_root)
    solver_ctx = SolverContext(**config.solver_ctx.model_dump())
    mesh_ctx: MeshContext = load_volumetric_mesh(
        paths.sample(sample_id).meshing().mesh.require()
    )
    solver = PhotoactiveSolver(solver_ctx, mesh_ctx)
    _STATE["solver"] = solver
    _STATE["sample_id"] = sample_id
    return


def execute_task(task: Task) -> dict[str, Any]:
    solver: PhotoactiveSolver = _STATE["solver"]
    sample_id: str = _STATE["sample_id"]
    #
    solution, analysis = solver.solve_for(
        task.chii,
        task.absorption,
        task.compensation,
    )
    return {
        "index": task.index,
        "sample_id": sample_id,
        "chii": task.chii,
        "absorption": task.absorption,
        "compensation": task.compensation,
        **analysis,
    }


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the scan command."""
    cmdconfig = config.scanning
    sample_id: str = validate_sample_id(
        args.sample_id, required_digits=config.behavior.sample_id_digits
    )
    # prepare paths
    paths = ProjectPaths(config.behavior.storage_root).sample(sample_id)

    process_paths = paths.scanning()
    process_paths.root.ensure()

    if not args.skip:
        tasks = make_tasks(**cmdconfig.model_dump())

        report = collect(
            tasks,
            execute_task,
            max_workers=config.max_workers,
            initializer=intializer,
            initargs=(config, sample_id),
            progress_callback=print_progress,
            error_policy="raise",
        )

        aggregate = [item.result for item in report.completed]
        dataframe = pd.DataFrame(aggregate)
        save_dataframe(process_paths.scan.path, dataframe)

        # save config
        save_config(process_paths.config.path, config, "scanning", "solver_ctx")

        # save manifest
        dump_manifest(
            process_paths.manifest.path,
            command_name="scan",
            sample_id=sample_id,
            inputs={"mesh": paths.meshing().mesh.path},
            outputs={"scan": process_paths.scan.path},
            metadata={
                "num_simulations": len(aggregate),
            },
            tool_version=config.meta.project_version,
        )

    # plot results
    dataframe = load_dataframe(process_paths.scan.path)
    plot_scanning_results(
        dataframe,
        output_dir=process_paths.plots.ensure(),
        show=args.show,
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "scan",
        help="Scan the photoactive problem with the specified parameters",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="Unique identifier for the generated sample",
    )
    parser.add_argument(
        "--skip",
        action="store_true",
        help="Skip the scan if the output file already exists. Default is False.",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Show the generated plot after saving. Default is False.",
    )
    parser.set_defaults(cmd=_cmd)
    return
