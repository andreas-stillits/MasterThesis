from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from stillib_parallelism import collect, print_progress

from ....config import ProjectConfig, SynthesisConfig, save_config
from ....core.io import save_voxels
from ....core.synthesis.uniform import (
    generate_voxels_from_sample_id as uniform_from_sample_id,
)
from ....ids import validate_sample_id
from ....manifest import dump_manifest
from ....paths import ProjectPaths


@dataclass
class Task:
    index: int
    sample_id: str
    synthesis_type: str
    parameters: dict[str, Any]


def make_tasks(
    start_sample_id: str,
    num_samples: int,
    synthesis_type: str,
    config: ProjectConfig,
) -> list[Task]:
    tasks: list[Task] = []
    num_cells = np.linspace(
        config.synthesis.num_cells_min,
        config.synthesis.num_cells_max,
        num_samples,
        dtype=int,
    )
    synthesis_dict = config.synthesis.model_dump()
    for idx in range(num_samples):
        sample_id = int(start_sample_id) + idx
        sample_id = str(sample_id).zfill(config.behavior.sample_id_digits)
        sample_id = validate_sample_id(
            sample_id, required_digits=config.behavior.sample_id_digits
        )
        blacklist = ["uniform", "mixed", "metaballs", "num_cells_min", "num_cells_max"]
        parameters = {
            key: value for key, value in synthesis_dict.items() if key not in blacklist
        }
        parameters.update(synthesis_dict[synthesis_type])
        parameters["num_cells"] = num_cells[idx]
        tasks.append(
            Task(
                idx,
                sample_id,
                synthesis_type,
                parameters,
            )
        )
    return tasks


_STATE: dict[str, Any] = {}


def initialize_worker(config: ProjectConfig) -> None:
    global _STATE
    paths = ProjectPaths(config.behavior.storage_root)
    _STATE = {
        "paths": paths,
    }


def execute_task(task: Task) -> None:
    global _STATE
    paths: ProjectPaths = _STATE["paths"]
    sample_paths = paths.sample(task.sample_id)
    process_paths = sample_paths.synthesis
    process_paths.root.ensure()

    if task.synthesis_type == "uniform":
        voxels, metadata = uniform_from_sample_id(
            task.sample_id,
            **task.parameters,
        )

        # save data
        save_voxels(process_paths.voxels.path, voxels)

        # save config
        # process_paths.config.path.write_text(
        #     json.dumps(task.config.model_dump(), indent=4, default=str),
        #     encoding="utf-8",
        # )

        # save manifest
        dump_manifest(
            process_paths.manifest.path,
            command_name=f"synthesize-{task.synthesis_type}",
            sample_id=task.sample_id,
            inputs={},
            outputs={"voxels": process_paths.voxels.path},
            metadata=metadata,
            tool_version=ProjectConfig().meta.project_version,
        )

    elif task.synthesis_type == "mixed" or task.synthesis_type == "metaballs":
        pass
    else:
        raise ValueError(f"Invalid synthesis type: {task.synthesis_type}")

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    """Command function for the batch synthesize command."""

    start_sample_id: str = validate_sample_id(
        args.start_sample_id, required_digits=config.behavior.sample_id_digits
    )

    tasks = make_tasks(
        start_sample_id,
        args.num_samples,
        args.type,
        config,
    )

    _ = collect(
        tasks,
        execute_task,
        max_workers=8,
        initializer=initialize_worker,
        initargs=(config,),
        progress_callback=print_progress,
        error_policy="raise",
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "synthesize",
        help="Generate a batch of synthetic voxel models",
    )
    parser.add_argument(
        "type",
        type=str,
        choices=["uniform", "mixed", "metaballs"],
        help="Type of synthesis to perform",
    )
    parser.add_argument(
        "start_sample_id",
        type=str,
        help="Unique identifier for the first generated sample",
    )
    parser.add_argument(
        "num_samples",
        type=int,
        help="Number of samples to generate",
    )
    parser.set_defaults(cmd=_cmd)
