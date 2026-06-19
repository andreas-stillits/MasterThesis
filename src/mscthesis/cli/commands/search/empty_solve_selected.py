from __future__ import annotations

import argparse
from typing import Any

from stillib_parallelism import collect, print_progress

from mscthesis.core.io import load_dataframe
from mscthesis.core.meshing.gmeshing import (
    build_pipe_model,
    mesh_model,
    mesh_porous_model,
)

from ....config import ProjectConfig, save_config
from ....core.io import load_volumetric_mesh, save_fem_solution
from ....core.solvers import (
    DiffusionSolver,
    MeshContext,
    SolverContext,
)
from ....manifest import dump_manifest, fetch_from_manifest
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

        source_paths = sample_paths.meshing(specifier)
        stomatal_aspect, plug_aspect = fetch_from_manifest(
            source_paths.manifest.require(),
            "stomatal_aspect",
            "plug_aspect",
        )
        mesh_field_settings = config.meshing.mesh_field.model_dump()
        mesh_field_settings["stomatal_aspect"] = stomatal_aspect
        mesh_path = paths.empty.ensure() / f"mesh_{sample_id}_{specifier}.msh"

        if specifier != 0:
            mesh_model(
                mesh_path,
                *build_pipe_model(plug_aspect),
                **mesh_field_settings,
            )
        else:
            del mesh_field_settings["stomatal_aspect"]
            mesh_porous_model(
                mesh_path,
                *build_pipe_model(plug_aspect),
                **mesh_field_settings,
            )

        mesh_ctx: MeshContext = load_volumetric_mesh(mesh_path)
        solver = DiffusionSolver(
            SolverContext(**config.solver_ctx.model_dump()),
            mesh_ctx,
        )
        solution, analysis = solver.solve_for(*config.solve_diffusion.parameters)
        empty_paths = sample_paths.empty(specifier)
        empty_paths.root.ensure()

        save_config(
            empty_paths.config.path,
            config,
            "solve_diffusion",
            "solver_ctx",
        )
        dump_manifest(
            empty_paths.manifest.path,
            command_name="empty-solve-selected",
            sample_id=sample_id,
            inputs={},
            outputs={},
            metadata={
                "parameters": config.solve_diffusion.parameters,
                **analysis,
            },
            tool_version=config.meta.project_version,
        )
        del solver

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
        "empty-solve-selected",
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
