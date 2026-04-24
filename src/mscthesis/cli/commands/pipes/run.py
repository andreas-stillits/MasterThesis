from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from stillib_parallelism import collect, print_progress

from ....config import ProjectConfig, save_config
from ....core.io import load_dataframe, load_volumetric_mesh, save_dataframe
from ....core.meshing.gmeshing import build_pipe_model, mesh_model
from ....core.plotting.pipes.experiments import plot_experiments
from ....core.solvers import DiffusionSolver, MeshContext, SolverContext
from ....paths import PipePaths, ProjectPaths


@dataclass
class Task:
    index: int
    plug_aspect: float
    stomatal_aspect: float

    def describe(self) -> str:
        return f"Task {self.index}: plug_aspect={self.plug_aspect:.2f}, \
             stomatal_aspect={self.stomatal_aspect:.2f}"


def make_tasks(
    plug_aspect_min: float,
    plug_aspect_max: float,
    plug_aspect_delta: float,
    stomatal_aspect_min: float,
    stomatal_aspect_delta: float,
) -> list[Task]:
    tasks: list[Task] = []
    plug_aspects = np.arange(
        plug_aspect_min, plug_aspect_max + plug_aspect_delta, plug_aspect_delta
    )
    stomatal_aspects = np.arange(
        stomatal_aspect_min,
        plug_aspect_max + stomatal_aspect_delta,
        stomatal_aspect_delta,
    )
    idx = 0
    for plug_aspect in plug_aspects:
        mask = stomatal_aspects <= plug_aspect
        if not np.any(mask):
            continue
        for stomatal_aspect in stomatal_aspects[mask]:
            if np.isclose(stomatal_aspect, plug_aspect):
                stomatal_aspect = plug_aspect - 1e-3
            tasks.append(Task(idx, plug_aspect, stomatal_aspect))
            idx += 1
    return tasks


_STATE: dict[str, Any] = {}


def initialize_worker(force: bool) -> None:
    global _STATE
    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root).pipes
    _STATE = {
        "config": config,
        "paths": paths,
        "force": force,
    }


def execute_task(task: Task) -> dict[str, int | float]:
    global _STATE
    config: ProjectConfig = _STATE["config"]
    paths: PipePaths = _STATE["paths"]
    force: bool = _STATE["force"]

    # get path for this task
    mesh = paths.experiments.mesh(task.index).file

    # mesh if not already done or if force is True
    if not mesh.exists() or force:
        mesh_field_settings = config.meshing.mesh_field.model_dump()
        mesh_field_settings["stomatal_aspect"] = task.stomatal_aspect
        # perform the meshing
        mesh_model(
            mesh.path,
            *build_pipe_model(task.plug_aspect),
            **mesh_field_settings,
        )

    mesh_ctx: MeshContext = load_volumetric_mesh(mesh.require())

    # solve diffusion if not already done or if force is True
    results_path = paths.experiments.results
    if not results_path.exists() or force:
        solver = DiffusionSolver(
            SolverContext(**config.solver_ctx.model_dump()),
            mesh_ctx,
        )

        resistances = []
        for parameters in config.pipes.parameter_sets:
            solution, analysis = solver.solve_for(*parameters)
            chii = analysis["substomatal_mean"]
            chit = analysis["top_mean"]
            flux = analysis["top_flux_grad"] / analysis["plug_area"]
            resistances.append(np.abs((chii - chit) / flux))

        solution, analysis = solver.solve_for(*config.solve_diffusion.parameters)
        return {
            "index": task.index,
            "plug_aspect": task.plug_aspect,
            "stomatal_aspect": task.stomatal_aspect,
            "resistance_mean": float(np.mean(resistances)),
            "resistance_std": float(np.std(resistances)),
        }

    return {}


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the pipe run command."""

    paths = ProjectPaths(config.behavior.storage_root).pipes
    paths.experiments.root.ensure()
    paths.experiments.meshes.ensure()

    tasks = make_tasks(**config.pipes.make.model_dump())

    report = collect(
        tasks,
        execute_task,
        initializer=initialize_worker,
        initargs=(args.force,),
        progress_callback=print_progress,
        error_policy="raise",
    )

    aggregate = [item.result for item in report.completed if item.result if item.result]

    if aggregate:
        dataframe = pd.DataFrame([item.result for item in report.completed])
        dataframe.sort_values(["plug_aspect", "stomatal_aspect"]).reset_index(
            drop=True, inplace=True
        )
        save_dataframe(paths.experiments.results.path, dataframe)

    save_config(paths.experiments.config.path, config, "pipes", "meshing", "solver_ctx")

    if args.show:
        dataframe = load_dataframe(paths.experiments.results.require())
        plot_experiments(dataframe, paths.experiments.plots.ensure() / "summary.pdf")
    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "run",
        help="Execute ideal pipe analysis pipeline: meshing -> diffusion solve -> postprocessing",
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
