from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import ufl
from dolfinx import default_scalar_type, fem
from dolfinx.fem.petsc import LinearProblem
from dolfinx.mesh import (
    Mesh,
    MeshTags,
    compute_midpoints,
    locate_entities_boundary,
    meshtags,
)


@dataclass(frozen=True)
class Tags:
    AIRSPACE: int = 1
    TOP: int = 2
    BOTTOM: int = 3
    CURVED: int = 4
    MESOPHYLL: int = 5
    INLET: int = 6


@dataclass(frozen=True)
class MeshContext:
    mesh: Mesh
    cell_tags: MeshTags
    facet_tags: MeshTags


@dataclass(frozen=True)
class SolverContext:
    ksp_type: str = "cg"
    ksp_rtol: float = 1e-8
    pc_type: str = "jacobi"
    quadrature_degree: int = 4
    order: int = 2
