from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from stillib_parallelism import collect, print_progress

from ....config import ProjectConfig, save_config
from ....core.io import load_dataframe, load_volumetric_mesh, save_dataframe
from ....core.meshing.gmeshing import build_sample_model, mesh_model
from ....core.plotting.search.validation import plot_validation
from ....core.solvers import MeshContext, PhotoactiveSolver, SolverContext
from ....ids import validate_sample_id
from ....paths import ProjectPaths, ValidationPaths


@dataclass
class Task:
    index: int
    stomatal_aspect: float
    scale_factor: float


def make_tasks(
    scale_min: float,
    scale_max: float,
    scale_num: int,
    stomatal_aspect: float,
) -> list[Task]:
    tasks: list[Task] = []
    scales = np.logspace(np.log10(scale_min), np.log10(scale_max), scale_num)
    for idx, scale in enumerate(scales):
        tasks.append(Task(idx, stomatal_aspect, scale))
    return tasks


_STATE: dict[str, Any] = {}


def initialize_worker(force: bool, sample_id: str) -> None:
    global _STATE
    config = ProjectConfig()
    paths = ProjectPaths(config.behavior.storage_root).validation(sample_id)
    _STATE = {
        "config": config,
        "paths": paths,
        "force": force,
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
            *build_sample_model(
                paths.triangulation.cadmodel.require(),
                config.meshing.boundary_margin,
                config.meshing.substomatal_margin,
                config.meshing.atol,
            ),
            **mesh_field_settings,
        )
    # solve photoactive if not already done or if force is True
    if not paths.results.exists() or force:
        qois: list[dict[str, Any]] = []

        mesh_ctx: MeshContext = load_volumetric_mesh(mesh.require())

        for order in [1, 2]:
            solver_ctx = config.solver_ctx.model_dump()
            solver_ctx["order"] = order
            solver = PhotoactiveSolver(
                SolverContext(**solver_ctx),
                mesh_ctx,
            )
            parameters = config.search.selected.validation.parameter_set
            solution, analysis = solver.solve_for(
                *parameters,
            )
            chii = analysis["substomatal_mean"]
            chim = analysis["top_mean"]
            flux = analysis["mesophyll_flux_sol"] / analysis["plug_area"]
            resistance = np.abs((chii - chim) / flux)
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
    paths = ProjectPaths(config.behavior.storage_root)
    sample_id = validate_sample_id(
        args.selected_sample_id, config.behavior.sample_id_digits
    )
    sample_paths = paths.selected_sample(sample_id)
    assert (
        sample_paths.root.exists()
    ), f"Selected sample with ID '{sample_id}' does not exist at path: {sample_paths.root}"
    assert (
        sample_paths.synthesis.voxels.exists()
    ), f"Selected sample with ID '{sample_id}' does not have a voxels file at path: {sample_paths.synthesis.voxels}"
    assert (
        sample_paths.triangulation.cadmodel.exists()
    ), f"Selected sample with ID '{sample_id}' does not have a triangulation CAD model file at path: {sample_paths.triangulation.cadmodel}"

    validation_paths = paths.validation(sample_id)
    validation_paths.root.ensure()

    # copy the contents of the synthesis/ directory to a counterpart in validation/
    shutil.copytree(
        sample_paths.synthesis.root.require(),
        validation_paths.synthesis.root.ensure(),
        dirs_exist_ok=True,
    )
    shutil.copytree(
        sample_paths.triangulation.root.require(),
        validation_paths.triangulation.root.ensure(),
        dirs_exist_ok=True,
    )

    # delete existing files if force is True
    if args.force:
        for path in validation_paths.meshes.path.glob("*.msh"):
            # delete mesh files
            path.unlink()
    validation_config = config.search.selected.validation.model_dump()
    del validation_config["parameter_set"]

    tasks = make_tasks(**validation_config)

    report = collect(
        tasks,
        execute_task,
        max_workers=config.max_workers,
        initializer=initialize_worker,
        initargs=(args.force, sample_id),
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
        save_dataframe(validation_paths.results.path, dataframe)

    save_config(
        validation_paths.config.path,
        config,
        "search",
        "meshing",
        "solver_ctx",
    )

    dataframe = load_dataframe(validation_paths.results.require())
    plot_validation(dataframe, validation_paths.plot.path, show=args.show)
    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "validate",
        help="Validate the generated meshes by running a set of predefined tests and visualizations.",
    )
    parser.add_argument(
        "selected_sample_id",
        type=str,
        help="The sample ID of the selected sample to validate.",
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
