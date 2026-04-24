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
)

from .meshing.gmeshing import Tags


@dataclass
class MeshContext:
    mesh: Mesh
    cell_tags: MeshTags
    facet_tags: MeshTags

    def __post_init__(self) -> None:
        self.gdim = self.mesh.geometry.dim
        self.tdim = self.mesh.topology.dim
        self.fdim = self.mesh.topology.dim - 1


@dataclass(frozen=True)
class SolverContext:
    petsc_options: dict[str, str | float] = {
        "ksp_type": "cg",
        "ksp_rtol": 1e-8,
        "pc_type": "jacobi",
    }
    quadrature_degree: int = 4
    order: int = 2


def _integral(expression: Any, measure: ufl.Measure) -> float:
    return float(
        fem.assemble_scalar(
            fem.form(expression * measure)  # type: ignore[reportArgumentType]
        )
    )


class BaseSolver:
    def __init__(self, solver_ctx: SolverContext, mesh_ctx: MeshContext) -> None:
        self.solver_ctx = solver_ctx
        self.mesh_ctx = mesh_ctx
        self.tags = Tags()
        self.functionspace = fem.functionspace(
            self.mesh_ctx.mesh, ("CG", self.solver_ctx.order)
        )
        inlet_facets = self.mesh_ctx.facet_tags.find(self.tags.INLET)
        self.inlet_dofs = fem.locate_dofs_topological(
            self.functionspace, self.mesh_ctx.fdim, inlet_facets
        )
        top_facets = self.mesh_ctx.facet_tags.find(self.tags.TOP)
        self.top_dofs = fem.locate_dofs_topological(
            self.functionspace, self.mesh_ctx.fdim, top_facets
        )
        self.dx = ufl.Measure(
            "dx",
            domain=self.mesh_ctx.mesh,
            subdomain_data=self.mesh_ctx.cell_tags,
            metadata={"quadrature_degree": self.solver_ctx.quadrature_degree},
        )
        self.ds = ufl.Measure(
            "ds",
            domain=self.mesh_ctx.mesh,
            subdomain_data=self.mesh_ctx.facet_tags,
            metadata={"quadrature_degree": self.solver_ctx.quadrature_degree},
        )
        # placeholder variables
        self.compensation = fem.Constant(self.mesh_ctx.mesh, default_scalar_type(0.0))
        self.surface_coeff = fem.Constant(self.mesh_ctx.mesh, default_scalar_type(0.0))
        self.chii = fem.Constant(self.mesh_ctx.mesh, default_scalar_type(0.0))
        self.chit = fem.Constant(self.mesh_ctx.mesh, default_scalar_type(0.0))
        # Dimensions and area fractions
        self.airspace_volume = _integral(1.0, self.dx(self.tags.AIRSPACE))
        self.plug_area = _integral(1.0, self.ds(self.tags.TOP))
        self.plug_aspect = np.sqrt(self.plug_area / np.pi)
        self.curved_area = _integral(1.0, self.ds(self.tags.CURVED))
        self.mesophyll_area = _integral(1.0, self.ds(self.tags.MESOPHYLL))
        self.mesophyll_area_fraction = self.mesophyll_area / self.plug_area
        self.stomatal_area = _integral(1.0, self.ds(self.tags.INLET))
        self.stomatal_aspect = np.sqrt(self.stomatal_area / np.pi)
        self.stomatal_area_fraction = self.stomatal_area / self.plug_area
        self.porosity = self.airspace_volume / self.plug_area * 1.0
        return

    def compute_gradient(self, solution: fem.Function) -> fem.Function:
        grad_order = max(self.solver_ctx.order - 1, 0)  # Ensure order is at least 0
        gradientspace = fem.functionspace(
            self.mesh_ctx.mesh, ("DG", grad_order, (self.mesh_ctx.gdim,))
        )
        d_chi = ufl.TrialFunction(gradientspace)
        v = ufl.TestFunction(gradientspace)
        a_proj = ufl.inner(d_chi, v) * self.dx
        L_proj = ufl.inner(ufl.grad(solution), v) * self.dx
        projection_problem = LinearProblem(
            a_proj,
            L_proj,
            bcs=[],
            petsc_options=self.solver_ctx.petsc_options,
        )
        return projection_problem.solve()

    def analyze(self, solution: fem.Function, gradient: fem.Function) -> dict[str, Any]:
        normal = ufl.FacetNormal(self.mesh_ctx.mesh)
        stomatal_flux_grad = _integral(
            ufl.dot(gradient, normal), self.ds(self.tags.INLET)
        )
        bottom_flux_grad = _integral(
            ufl.dot(gradient, normal), self.ds(self.tags.BOTTOM)
        )
        mesophyll_flux_grad = _integral(
            ufl.dot(gradient, normal), self.ds(self.tags.MESOPHYLL)
        )
        mesophyll_flux_sol = _integral(
            self.surface_coeff * (solution - self.compensation),  # type: ignore
            self.ds(self.tags.MESOPHYLL),
        )
        curved_flux_grad = _integral(
            ufl.dot(gradient, normal), self.ds(self.tags.CURVED)
        )
        top_flux_grad = _integral(ufl.dot(gradient, normal), self.ds(self.tags.TOP))
        total_flux_grad = _integral(ufl.dot(gradient, normal), self.ds)
        #
        substomatal_mean = (
            _integral(solution, self.ds(self.tags.INLET)) / self.stomatal_area
        )
        top_mean = _integral(solution, self.ds(self.tags.TOP)) / self.plug_area
        mesophyll_mean = 0.0
        mesophyll_var = 0.0
        if self.mesophyll_area > 0.0:
            mesophyll_mean = (
                _integral(solution, self.ds(self.tags.MESOPHYLL)) / self.mesophyll_area
            )
            mesophyll_var = (
                _integral(
                    (solution - mesophyll_mean) ** 2,  # type: ignore
                    self.ds(self.tags.MESOPHYLL),  # type: ignore
                )
                / self.mesophyll_area
            )
        airspace_mean = (
            _integral(solution, self.dx(self.tags.AIRSPACE)) / self.airspace_volume
        )
        airspace_var = (
            _integral((solution - airspace_mean) ** 2, self.dx(self.tags.AIRSPACE))  # type: ignore
            / self.airspace_volume
        )
        #
        tol = 1e-6
        transport = (
            np.abs(mesophyll_flux_sol / (1.0 - substomatal_mean) / self.plug_area)
            if substomatal_mean < 1.0 - tol
            else None
        )

        return {
            "stomatal_flux_grad": stomatal_flux_grad,
            "bottom_flux_grad": bottom_flux_grad,
            "mesophyll_flux_grad": mesophyll_flux_grad,
            "mesophyll_flux_sol": mesophyll_flux_sol,
            "curved_flux_grad": curved_flux_grad,
            "top_flux_grad": top_flux_grad,
            "total_flux_grad": total_flux_grad,
            "sub_stomatal_mean": substomatal_mean,
            "top_mean": top_mean,
            "mesophyll_mean": mesophyll_mean,
            "mesophyll_var": mesophyll_var,
            "airspace_mean": airspace_mean,
            "airspace_var": airspace_var,
            "transport": transport,
            "plug_area": self.plug_area,
            "curved_area": self.curved_area,
            "mesophyll_area": self.mesophyll_area,
            "stomatal_area": self.stomatal_area,
            "stomatal_area_fraction": self.stomatal_area_fraction,
            "mesophyll_area_fraction": self.mesophyll_area_fraction,
            "porosity": self.porosity,
        }

    def solve_for(self, *args, **kwargs) -> tuple[fem.Function, dict[str, Any]]:
        raise NotImplementedError("Subclasses must implement the solve_for method.")


# =======================================================================
#                         PHOTOACTIVE SOLVER
# =======================================================================


class PhotoactiveSolver(BaseSolver):
    def __init__(self, solver_ctx: SolverContext, mesh_ctx: MeshContext) -> None:
        super().__init__(solver_ctx, mesh_ctx)
        #
        chi = ufl.TrialFunction(self.functionspace)
        v = ufl.TestFunction(self.functionspace)

        a = ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(
            self.tags.AIRSPACE
        ) + self.surface_coeff * chi * v * self.ds(self.tags.MESOPHYLL)  # type: ignore
        L = self.surface_coeff * self.compensation * v * self.ds(self.tags.MESOPHYLL)  # type: ignore

        bc_inlet = fem.dirichletbc(self.chii, self.inlet_dofs, self.functionspace)

        self.problem = LinearProblem(
            a,
            L,
            bcs=[bc_inlet],
            petsc_options=self.solver_ctx.petsc_options,
        )
        return

    def solve_for(
        self, chii: float, absorption: float, compensation: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.surface_coeff.value = default_scalar_type(
            absorption / self.mesophyll_area_fraction
        )
        self.chii.value = default_scalar_type(chii)
        self.compensation.value = default_scalar_type(compensation)
        solution = self.problem.solve()
        gradient = self.compute_gradient(solution)
        return solution, self.analyze(solution, gradient)


# =======================================================================
#                         DIFFUSION SOLVER
# =======================================================================


class DiffusionSolver(BaseSolver):
    def __init__(self, solver_ctx: SolverContext, mesh_ctx: MeshContext) -> None:
        super().__init__(solver_ctx, mesh_ctx)
        #
        chi = ufl.TrialFunction(self.functionspace)
        v = ufl.TestFunction(self.functionspace)

        a = ufl.inner(ufl.grad(chi), ufl.grad(v)) * self.dx(self.tags.AIRSPACE)
        L = (
            fem.Constant(self.mesh_ctx.mesh, default_scalar_type(0.0))  # type: ignore
            * v
            * self.dx(self.tags.AIRSPACE)
        )

        bc_inlet = fem.dirichletbc(self.chii, self.inlet_dofs, self.functionspace)
        bc_top = fem.dirichletbc(self.chit, self.top_dofs, self.functionspace)

        self.problem = LinearProblem(
            a,
            L,
            bcs=[bc_inlet, bc_top],
            petsc_options=self.solver_ctx.petsc_options,
        )
        return

    def solve_for(
        self, chii: float, chit: float
    ) -> tuple[fem.Function, dict[str, Any]]:
        self.chii.value = default_scalar_type(chii)
        self.chit.value = default_scalar_type(chit)
        solution = self.problem.solve()
        gradient = self.compute_gradient(solution)
        return solution, self.analyze(solution, gradient)
