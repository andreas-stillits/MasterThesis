from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from stillib_parallelism import collect, print_progress
from stillib_random import RNGStream, from_entropy
from stillib_random.multiprocessing import TaskStream, assign_streams

from ....config import ProjectConfig, save_config
from ....core.io import save_voxels
from ....core.synthesis.mixed import generate_voxels_from_rng
from ....ids import asstr
from ....manifest import dump_manifest
from ....paths import ProjectPaths


@dataclass
class Candidate:
    sample_id: str
    plug_aspect: float
    num_cells: int
    radius_min: float
    radius_max: float


def generate_candidates(config: ProjectConfig, start_id: int) -> list[Candidate]:
    candidates: list[Candidate] = []
    sample_id = start_id

    for num_cells in config.search.candidates.num_cells_set:
        for radius_center in config.search.candidates.radius_center_set:
            for plug_aspect in config.search.plug_aspect_set.values():
                radius_width = config.search.candidates.radius_width
                radius_min = max(0.01, radius_center - radius_width)
                radius_max = radius_center + radius_width
                sample_id_str = asstr(sample_id, config.behavior.sample_id_digits)
                candidate = Candidate(
                    sample_id_str,
                    plug_aspect,
                    num_cells,
                    radius_min,
                    radius_max,
                )
                candidates.append(candidate)
                sample_id += 1

    return candidates


_STATE: dict[str, Any] = {}


def initializer(config: ProjectConfig) -> None:
    _STATE["config"] = config
    return


def create_candidate(taskstream: TaskStream) -> None:
    candidate: Candidate = taskstream.task
    stream = RNGStream.from_manifest(taskstream.manifest)
    cursor = stream.cursor()
    rng = cursor.generator()
    #
    config: ProjectConfig = _STATE["config"]
    paths = ProjectPaths(config.behavior.storage_root).candidate_sample(
        candidate.sample_id
    )
    snapshot = cursor.snapshot()

    #
    voxels, manifest = generate_voxels_from_rng(
        rng,
        config.synthesis.resolution,
        candidate.plug_aspect,
        config.synthesis.separation,
        config.synthesis.max_attempts,
        candidate.num_cells,
        candidate.radius_min,
        candidate.radius_max,
    )

    # io
    paths.synthesis.root.ensure()
    cursor.save_snapshot(paths.synthesis.snapshot.path, snapshot)
    save_voxels(paths.synthesis.voxels.path, voxels)

    # config
    candidate_config = config.model_dump()
    candidate_config["synthesis"]["plug_aspect"] = candidate.plug_aspect
    candidate_config["synthesis"]["mixed"]["num_cells"] = candidate.num_cells
    candidate_config["synthesis"]["mixed"]["radius_min"] = candidate.radius_min
    candidate_config["synthesis"]["mixed"]["radius_max"] = candidate.radius_max

    save_config(
        paths.synthesis.config.path,
        ProjectConfig.model_validate(candidate_config),
        "synthesis",
    )

    # manifest
    dump_manifest(
        paths.synthesis.manifest.path,
        command_name="gen-candidates-mixed",
        sample_id=candidate.sample_id,
        inputs={},
        outputs={"voxels": paths.synthesis.voxels.path},
        metadata=manifest,
        tool_version=config.meta.project_version,
    )

    return


def _cmd(config: ProjectConfig, args: argparse.Namespace) -> None:
    paths = ProjectPaths(config.behavior.storage_root)

    # get a sample ID to start at as max in candidates/ or 0 if no candidates exist yet
    existing_candidates = [
        p.name for p in paths.candidates.ensure().iterdir() if p.is_dir()
    ]
    next_id = 0
    if existing_candidates:
        next_id = 1 + max([int(c) for c in existing_candidates if c.isdigit()])
    #
    new_candidates: list[Candidate] = generate_candidates(config, next_id)
    root_stream: RNGStream = from_entropy()
    taskstreams: list[TaskStream] = assign_streams(
        new_candidates, root_stream, prefix="candidate"
    )
    #
    collect(
        taskstreams,
        create_candidate,
        max_workers=config.max_workers,
        progress_callback=print_progress,
        initializer=initializer,
        initargs=(config,),
        error_policy="raise",
    )

    return


def add_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser(
        "gen-candidates-mixed", help="Generate candidate configurations for search"
    )
    parser.set_defaults(cmd=_cmd)
    return
