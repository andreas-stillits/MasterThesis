from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from typing import Any

import numpy as np

from ....config import ProjectConfig, save_config
from ....core.io import load_dataframe, load_volumetric_mesh, save_dataframe
from ....core.meshing.gmeshing import build_pipe_model, mesh_model
from ....core.plotting.pipes.validation import plot_validation
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
        "sample_id": sample_id,
    }


def execute_task(task: Task) -> list[dict[str, Any]] | None:

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

    # now run validation test as for pipes on the copied .brep file

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
