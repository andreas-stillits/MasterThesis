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


def execute_solving(sample_id: str) -> None:
    config: ProjectConfig = _STATE["config"]
    paths: ProjectPaths = _STATE["paths"]
    force: bool = _STATE["force"]

    sample_paths = paths.selected_sample(sample_id)

    for specifier in config.search.stomatal_aspect_set:
        solutions_paths = sample_paths.solutions(specifier)
        solutions_paths.root.ensure()

        if (
            not solutions_paths.diffusion.solution.exists()
            or not solutions_paths.photoactive.solution.exists()
            or force
        ):
            # load mesh
            mesh_ctx: MeshContext = load_volumetric_mesh(
                sample_paths.meshing(specifier).mesh.require()
            )

            if not solutions_paths.diffusion.solution.exists() or force:
                solver = DiffusionSolver(
                    SolverContext(**config.solver_ctx.model_dump()),
                    mesh_ctx,
                )
                solution, analysis = solver.solve_for(
                    *config.solve_diffusion.parameters
                )
                chii = analysis["substomatal_mean"]
                chit = analysis["top_mean"]
                a_n = analysis["top_flux_grad"] / analysis["plug_area"]
                resistance = abs((chii - chit) / a_n)

                save_fem_solution(
                    solutions_paths.diffusion.solution.ensure(), solution, mesh_ctx
                )
                save_config(
                    solutions_paths.diffusion.config.path,
                    config,
                    "solve_diffusion",
                    "solver_ctx",
                )
                dump_manifest(
                    solutions_paths.diffusion.manifest.path,
                    command_name="solve-diffusion",
                    sample_id=sample_id,
                    inputs={"mesh": sample_paths.meshing(specifier).mesh.path},
                    outputs={"solution": solutions_paths.diffusion.solution.path},
                    metadata={
                        "parameters": config.solve_diffusion.parameters,
                        "resistance": resistance,
                        **analysis,
                    },
                    tool_version=config.meta.project_version,
                )
                del solver

            if not solutions_paths.photoactive.solution.exists() or force:
                solver = PhotoactiveSolver(
                    SolverContext(**config.solver_ctx.model_dump()),
                    mesh_ctx,
                )
                solution, analysis = solver.solve_for(*config.solve_active.parameters)
                chii = analysis["substomatal_mean"]
                chim = analysis["mesophyll_mean"]
                a_n = analysis["mesophyll_flux_sol"] / analysis["plug_area"]
                resistance = abs((chii - chim) / a_n)

                save_fem_solution(
                    solutions_paths.photoactive.solution.ensure(), solution, mesh_ctx
                )
                save_config(
                    solutions_paths.photoactive.config.path,
                    config,
                    "solve_active",
                    "solver_ctx",
                )
                dump_manifest(
                    solutions_paths.photoactive.manifest.path,
                    command_name="solve-photoactive",
                    sample_id=sample_id,
                    inputs={"mesh": sample_paths.meshing(specifier).mesh.path},
                    outputs={"solution": solutions_paths.photoactive.solution.path},
                    metadata={
                        "parameters": config.solve_active.parameters,
                        "resistance": resistance,
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
    parser = subparsers.add_parser("solve-selected", help="Solve the selected samples.")
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force solving even if the solutions already exists.",
    )
    parser.set_defaults(cmd=_cmd)
    return
