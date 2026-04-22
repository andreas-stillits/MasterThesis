from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Literal

from pydantic import BaseModel
from pydantic.config import ConfigDict


class MetaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = "mscthesis"
    project_version: str = "1.0.0"
    config_name: str = "config.json"


class BehaviorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    storage_root: Path = Path.home() / "coding" / "MasterThesis" / ".treasury"
    sample_id_digits: int = 5


class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ksp_type: Literal["preonly", "cg"] = "cg"
    ksp_rtol: float = 1e-8
    pc_type: Literal["lu", "jacobi"] = "jacobi"
    quadrature_degree: int = 4
    order: int = 2


class ProjectConfig(BaseModel):

    meta: MetaConfig = MetaConfig()
    behavior: BehaviorConfig = BehaviorConfig()
    solver: SolverConfig = SolverConfig()

    def dump_json(self) -> str:
        return json.dumps(self.model_dump(), indent=4, default=str)


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
