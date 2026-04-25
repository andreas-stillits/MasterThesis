from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from stillib_parallelism import collect, print_progress

from ....config import ProjectConfig, save_config
from ....core.io import load_dataframe, load_volumetric_mesh, save_dataframe
from ....core.meshing.gmeshing import build_pipe_model, mesh_model
from ....core.plotting.pipes.validation import plot_validation
from ....core.solvers import DiffusionSolver, MeshContext, SolverContext
from ....paths import ProjectPaths, ValidationPaths


@dataclass
class Task:
    index: int
    plug_aspect: float
    stomatal_aspect: float
    scale_factor: float


def make_tasks(
    scale_min: float,
    scale_max: float,
    scale_num: int,
    plug_aspect: float,
    stomatal_aspect: float,
) -> list[Task]:
    tasks: list[Task] = []
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), scale_num)
    for idx, scale in enumerate(scales):
        tasks.append(Task(idx, plug_aspect, stomatal_aspect, scale))
    return tasks


_STATE: dict[str, Any] = {}


def initialize_worker(force: bool, tag: str) -> None:
    global _STATE
    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root).pipes.validation(tag)
    _STATE = {
        "config": config,
        "paths": paths,
        "force": force,
        "tag": tag,
    }


def execute_task(task: Task) -> list[dict[str, Any]] | None:
    global _STATE
    config: ProjectConfig = _STATE["config"]
    paths: ValidationPaths = _STATE["paths"]
    force: bool = _STATE["force"]

    # get path for this task
    mesh = paths.mesh(task.scale_factor).file

    # check if mesh already exists
    if not mesh.exists() or force:
        # build and mesh the model
        mesh_field_settings = config.meshing.mesh_field.model_dump()
        mesh_field_settings["stomatal_aspect"] = task.stomatal_aspect
        mesh_field_settings["scale_factor"] = task.scale_factor
        # perform meshing
        mesh_model(
            mesh.path,
            *build_pipe_model(task.plug_aspect),
            **mesh_field_settings,
        )

    mesh_ctx: MeshContext = load_volumetric_mesh(mesh.require())

    # solve diffusion if not already done or if force is True
    results = paths.results
    if not results.exists() or force:

        qois: list[dict[str, Any]] = []

        for order in [1, 2]:
            solver_ctx = config.solver_ctx.model_dump()
            solver_ctx["order"] = order
            solver = DiffusionSolver(
                SolverContext(**solver_ctx),
                mesh_ctx,
            )
            solution, analysis = solver.solve_for(*config.pipes.parameter_sets[0])
            chii = analysis["substomatal_mean"]
            chit = analysis["top_mean"]
            flux = analysis["top_flux_grad"] / analysis["plug_area"]
            resistance = np.abs((chii - chit) / flux)
            qois.append(
                {
                    "scale_factor": task.scale_factor,
                    "order": order,
                    "resistance": resistance,
                    **analysis,
                }
            )

        return qois

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the pipes validate command."""

    paths = ProjectPaths(config.behavior.storage_root).pipes.validation(args.tag)
    paths.root.ensure()

    tasks = make_tasks(**config.pipes.validation.model_dump())

    report = collect(
        tasks,
        execute_task,
        initializer=initialize_worker,
        initargs=(args.force, args.tag),
        progress_callback=print_progress,
        error_policy="raise",
    )
    aggregate = []
    for item in report.completed:
        if item.result is not None:
            aggregate.extend(item.result)

    if aggregate:
        dataframe = pd.DataFrame(aggregate)
        dataframe.sort_values(["scale_factor", "order"]).reset_index(
            drop=True, inplace=True
        )
        save_dataframe(paths.results.path, dataframe)

    save_config(
        paths.config.path,
        config,
        "pipes",
        "meshing",
        "solver_ctx",
    )

    dataframe = load_dataframe(paths.results.require())
    plot_validation(dataframe, paths.plot.path, show=args.show)
    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Execute validation of pipe analysis meshing settings by convergence of QoIs",
    )
    parser.add_argument(
        "tag",
        type=str,
        help="A unique tag to identify this validation run. Used for organizing output files.",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Visualize the output",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force re-calculation of all steps, even if outputs already exist",
    )
    parser.set_defaults(cmd=_cmd)
