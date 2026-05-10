from __future__ import annotations

import argparse
from typing import Any

from stillib_parallelism import collect, print_progress

from mscthesis.core.io import load_dataframe

from ....config import ProjectConfig, save_config
from ....core.io import load_volumetric_mesh, save_fem_solution
from ....core.solvers import (
    DiffusionSolver,
    MeshContext,
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


def execute_solving(sample_id: str) -> None:
    config: ProjectConfig = _STATE["config"]
    paths: ProjectPaths = _STATE["paths"]
    force: bool = _STATE["force"]

    sample_paths = paths.selected_sample(sample_id)

    for specifier in config.search.stomatal_aspect_set:
        diffusion_paths = sample_paths.diffusion(specifier)
        diffusion_paths.root.ensure()

        if not diffusion_paths.solution.exists() or force:
            # load mesh
            mesh_ctx: MeshContext = load_volumetric_mesh(
                sample_paths.meshing(specifier).mesh.require()
            )

            solver = DiffusionSolver(
                SolverContext(**config.solver_ctx.model_dump()),
                mesh_ctx,
            )
            solution, analysis = solver.solve_for(*config.solve_diffusion.parameters)

            save_fem_solution(diffusion_paths.solution.path, solution, mesh_ctx)
            save_config(
                diffusion_paths.config.path,
                config,
                "solve_diffusion",
                "solver_ctx",
            )
            dump_manifest(
                diffusion_paths.manifest.path,
                command_name="solve-diffusion",
                sample_id=sample_id,
                inputs={"mesh": sample_paths.meshing(specifier).mesh.path},
                outputs={"solution": diffusion_paths.solution.path},
                metadata={
                    "parameters": config.solve_diffusion.parameters,
                    **analysis,
                },
                tool_version=config.meta.project_version,
            )
            del solver

        else:
            continue

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)
    index = load_dataframe(paths.index.require())
    selected_ids = index[index["selected"]]["sample_id"].tolist()

    report = collect(
        selected_ids,
        execute_solving,
        max_workers=config.max_workers,
        initializer=initializer,
        initargs=(config, args.force),
        progress_callback=print_progress,
        ordering="completion",
        error_policy="collect",
    )

    if not report.ok:
        print(f"Solving completed with {len(report.failures)} errors:")
        for failure in report.failures:
            print(
                f"- Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}",
                "\n",
            )
        print("saving list of failures to file: '<storage_root>/solving_failures.txt'")
        failures_path = paths.failures.ensure() / "solving_failures.txt"
        with open(failures_path, "w") as f:
            for failure in report.failures:
                f.write(
                    f"Sample ID: {failure.task}, Error: {failure.exc_type}: {failure.exc_message}\n"
                )
    else:
        print("Solving completed successfully for all selected samples.")

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "diffusion-solve-selected",
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
