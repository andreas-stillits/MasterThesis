from __future__ import annotations

import argparse

from ....config import ProjectConfig, save_config
from ....core.io import load_volumetric_mesh, save_fem_solution
from ....core.solvers import (
    DiffusionSolver,
    MeshContext,
    SolverContext,
)
from ....core.visualization import visualize_fem_solution
from ....ids import validate_sample_id
from ....manifest import dump_manifest
from ....paths import ProjectPaths


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the solve-diffusion command."""
    cmdconfig = config.solve_diffusion
    sample_id: str = validate_sample_id(
        args.sample_id, required_digits=config.behavior.sample_id_digits
    )
    # prepare paths
    paths = ProjectPaths(config.behavior.storage_root).sample(sample_id)

    process_paths = paths.solutions.diffusion
    process_paths.root.ensure()

    mesh_ctx: MeshContext = load_volumetric_mesh(paths.meshing.mesh.require())

    solver = DiffusionSolver(
        SolverContext(**config.solver_ctx.model_dump()),
        mesh_ctx,
    )

    solution, analysis = solver.solve_for(*cmdconfig.parameters)

    # save config
    save_config(process_paths.config.path, config, "solve_diffusion", "solver_ctx")

    # save manifest
    dump_manifest(
        process_paths.manifest.path,
        command_name="solve-diffusion",
        sample_id=sample_id,
        inputs={"mesh": paths.meshing.mesh.path},
        outputs={"solution": process_paths.solution.path},
        metadata={
            "parameters": cmdconfig.parameters,
            **analysis,
        },
        tool_version=config.meta.project_version,
    )

    if not args.no_save:
        save_fem_solution(process_paths.solution.path, solution, mesh_ctx)

    if args.show:
        visualize_fem_solution(solution, mesh_ctx.mesh)

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "solve-diffusion",
        help="Solve the diffusion problem with the specified parameters",
    )
    parser.add_argument(
        "sample_id",
        type=str,
        help="Unique identifier for the generated sample",
    )
    parser.add_argument(
        "-s",
        "--show",
        action="store_true",
        help="Visualize the output",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the output to disk",
    )
    parser.set_defaults(cmd=_cmd)
