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


class SynthesizeMixedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_cells: int = 200
    radius_min: float = 0.04
    radius_max: float = 0.10


class SynthesizeMetaBallsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_cells: int = 200
    radius_min: float = 0.04
    radius_max: float = 0.06
    threshold: float = 4.2


class SynthesisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uniform: SynthesizeUniformConfig = SynthesizeUniformConfig()
    mixed: SynthesizeMixedConfig = SynthesizeMixedConfig()
    metaballs: SynthesizeMetaBallsConfig = SynthesizeMetaBallsConfig()
    base_seed: int = 123456
    resolution: int = 100
    plug_aspect: float = 0.25
    separation: float = 0.005
    max_attempts: int = 10_000
    num_cells_min: int = 25
    num_cells_max: int = 200


class TriangulationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    smoothing_iterations: int = 5
    elements_per_cell: int = 150
    shrinkage_tolerance: float = 0.10
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    freecad_cmd: str = "freecadcmd-daily"
    freecad_script_path: str = (
        "/home/andreasstillits/coding/master/src/mscthesis/core/meshing/breping.py"
    )


class MeshFieldConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stomatal_aspect: float = 0.02
    scale_factor: float = 1.5
    global_max_num: int = 25
    edge_min_num: int = 50
    edge_dist_min: float = 0.025
    edge_dist_max: float = 0.050
    cell_min: float = 0.01
    cell_dist_min: float = 0.02
    cell_dist_max: float = 0.05
    inlet_min: float = 0.005
    inlet_dist_min: float = 0.05
    inlet_dist_max: float = 0.25
    atol: float = 0.005


class MeshingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mesh_field: MeshFieldConfig = MeshFieldConfig()
    boundary_margin: float = 0.02
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

    parameters: tuple[float, ...] = (0.8, 0.1, 0.1)


class DiffusionSolveConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameters: tuple[float, ...] = (0.8, 0.4)


class ScanningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    transport_min: float = 0.01
    transport_max: float = 100.0
    transport_num: int = 11
    absorption_min: float = 0.01
    absorption_max: float = 100.0
    absorption_num: int = 11
    compensation: float = 0.1


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
    plug_aspect: float = 0.10
    stomatal_aspect: float = 0.05


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


class CandidateConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    num_cells_set: list[int] = [25, 50, 100, 150, 200]
    radius_center_set: list[float] = [0.04, 0.06, 0.08, 0.10]
    radius_width: float = 0.02


class SelectedValidationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scale_min: float = 1.0
    scale_max: float = 4.0
    scale_num: int = 16
    stomatal_aspect: float = 0.02
    parameter_set: tuple[float, float, float] = (0.9, 10.0, 0.1)


class SelectedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    validation: SelectedValidationConfig = SelectedValidationConfig()
    retries: dict[str, Any] = {
        "triangulation": {
            "elements_per_cell_increment": 25,
            "max_attempts": 4,
        }
    }
    porosity_gridsize: float = 0.10
    porous_inlet_specifier: int = 0


class SearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plug_aspect_set: dict[int, float] = {
        0: 0.10,
        1: 0.15,
        2: 0.20,
        3: 0.25,
        4: 0.30,
        5: 0.35,
        6: 0.40,
    }
    stomatal_aspect_set: dict[int, float] = {
        # reserve 0 for the variant were stomatal area ~ plug area
        0: 0.000,
        1: 0.020,
        2: 0.030,
        3: 0.040,
        4: 0.050,
    }
    candidates: CandidateConfig = CandidateConfig()
    selected: SelectedConfig = SelectedConfig()


class ProjectConfig(BaseModel):
    meta: MetaConfig = MetaConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    solver_ctx: SolverConfig = SolverConfig()
    synthesis: SynthesisConfig = SynthesisConfig()
    triangulation: TriangulationConfig = TriangulationConfig()
    meshing: MeshingConfig = MeshingConfig()
    solve_active: PhotoactiveSolveConfig = PhotoactiveSolveConfig()
    solve_diffusion: DiffusionSolveConfig = DiffusionSolveConfig()
    scanning: ScanningConfig = ScanningConfig()
    pipes: PipesConfig = PipesConfig()
    search: SearchConfig = SearchConfig()
    max_workers: int = 8

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
