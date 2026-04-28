from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel
from pydantic.config import ConfigDict


class MetaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = "mscthesis"
    project_version: str = "1.0.0"
    config_name: str = "config.json"
    log_name: str = "run.log"


class BehaviorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_root: Path = Path.home() / "coding" / "MasterThesis" / ".treasury"
    sample_id_digits: int = 5


class SynthesizeUniformConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_cells: int = 100
    radius: float = 0.08


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uniform: SynthesizeUniformConfig = SynthesizeUniformConfig()
    base_seed: int = 123456
    resolution: int = 100
    plug_aspect: float = 0.25
    separation: float = 0.01
    max_attempts: int = 10_000


class TriangulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smoothing_iterations: int = 5
    elements_per_cell: int = 100
    shrinkage_tolerance: float = 0.10
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    freecad_cmd: str = "freecadcmd-daily"
    freecad_script_path: str = (
        "/home/andreasstillits/coding/master/src/mscthesis/core/meshing/breping.py"
    )


class MeshFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stomatal_aspect: float = 0.05
    scale_factor: float = 1.0
    global_max_num: int = 25
    edge_min_num: int = 50
    edge_dist_min: float = 0.025
    edge_dist_max: float = 0.050
    cell_min: float = 0.01
    cell_dist_min: float = 0.01
    cell_dist_max: float = 0.05
    inlet_min: float = 0.01
    inlet_dist_min: float = 0.10
    inlet_dist_max: float = 0.50
    atol: float = 0.005


class MeshingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh_field: MeshFieldConfig = MeshFieldConfig()
    boundary_margin: float = 0.05
    substomatal_margin: float = 0.05
    atol: float = 0.005


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    petsc_options: dict[str, str | float] = {
        "ksp_type": "cg",
        "ksp_rtol": 1e-8,
        "pc_type": "jacobi",
    }
    quadrature_degree: int = 4
    order: int = 2


class PhotoactiveSolveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: tuple[float, ...] = (0.7, 1.0, 0.1)


class DiffusionSolveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: tuple[float, ...] = (0.7, 0.2)


class PipesMakeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plug_aspect_min: float = 0.10
    plug_aspect_max: float = 0.50
    plug_aspect_delta: float = 0.025
    stomatal_aspect_min: float = 0.025
    stomatal_aspect_delta: float = 0.025


class PipesValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_min: float = 1.0
    scale_max: float = 4.0
    scale_num: int = 32
    plug_aspect: float = 0.50
    stomatal_aspect: float = 0.10


class PipesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    make: PipesMakeConfig = PipesMakeConfig()
    validation: PipesValidationConfig = PipesValidationConfig()
    parameter_sets: tuple[tuple[float, float], ...] = (
        (0.80, 0.20),
        (0.80, 0.40),
        (0.80, 0.60),
        (0.60, 0.20),
        (0.60, 0.40),
    )


class ProjectConfig(BaseModel):
    meta: MetaConfig = MetaConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    solver_ctx: SolverConfig = SolverConfig()
    synthesis: SynthesisConfig = SynthesisConfig()
    triangulation: TriangulationConfig = TriangulationConfig()
    meshing: MeshingConfig = MeshingConfig()
    solve_active: PhotoactiveSolveConfig = PhotoactiveSolveConfig()
    solve_diffusion: DiffusionSolveConfig = DiffusionSolveConfig()
    pipes: PipesConfig = PipesConfig()

    def dump_json(self) -> str:
        return json.dumps(self.model_dump(), indent=4, default=str)


# ================================================================================
#                                       HELPERS
# ================================================================================


def filter_config(config: ProjectConfig, *keys: str) -> dict[str, Any]:
    """Filter a ProjectConfig to only include specified keys."""

    config_dict = config.model_dump()
    filtered_dict = {key: config_dict[key] for key in keys if key in config_dict}
    return filtered_dict


def save_config(path: str | Path, config: ProjectConfig, *keys: str) -> None:
    """Dump a ProjectConfig to a JSON file."""
    path = Path(path)
    path.write_text(
        json.dumps(filter_config(config, *keys), indent=4, default=str),
        encoding="utf-8",
    )


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists() and not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a nested dictionary with another dictionary."""
    result = dict(base)

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def build_project_config(
    path: Path | None, overrides: dict[str, Any] | None = None
) -> ProjectConfig:
    config = ProjectConfig()
    config_dict = config.model_dump()

    # update with supplied partial config file if provided
    if path is not None:
        config_dict = deep_update(config_dict, load_config(path))

    # update with overrides if present
    if overrides is not None:
        config_dict = deep_update(config_dict, overrides)

    return ProjectConfig.model_validate(config_dict)
