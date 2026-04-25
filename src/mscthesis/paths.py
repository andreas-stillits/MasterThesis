from __future__ import annotations

from pathlib import Path

from stillib_paths import PathLike, PathsBase, child_paths, path_field

# =======================================================================
#                         PIPE PATHS
# =======================================================================


class MeshFilePaths(PathsBase):
    def __init__(
        self, base: PathLike, specifier: int | float, format: str = "03d"
    ) -> None:
        super().__init__(base)
        self.specifier = specifier
        self.format = format

    @path_field(kind="file")
    def file(self) -> Path:
        return self.base / f"mesh_{self.specifier:{self.format}}.msh"


class ValidationPaths(PathsBase):
    def __init__(self, base: PathLike, name: str) -> None:
        super().__init__(base)
        self.name = name

    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / self.name

    @path_field(kind="dir")
    def meshes(self) -> Path:
        return self.root / "meshes"

    @path_field(kind="file")
    def plot(self) -> Path:
        return self.root / "plot.pdf"

    @path_field(kind="file")
    def results(self) -> Path:
        return self.root / "results.csv"

    @path_field(kind="file")
    def config(self) -> Path:
        return self.root / "config.json"

    def mesh(self, scale_factor: float) -> MeshFilePaths:
        return MeshFilePaths(self.meshes.ensure(), scale_factor, format=".2f")


class ExperimentPaths(PathsBase):
    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "experiments"

    @path_field(kind="dir")
    def meshes(self) -> Path:
        return self.root / "meshes"

    @path_field(kind="dir")
    def plots(self) -> Path:
        return self.root / "plots"

    @path_field(kind="file")
    def results(self) -> Path:
        return self.root / "results.csv"

    @path_field(kind="file")
    def config(self) -> Path:
        return self.root / "config.json"

    def mesh(self, index: int) -> MeshFilePaths:
        return MeshFilePaths(self.meshes.ensure(), index, format="03d")


class PipePaths(PathsBase):
    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "pipes"

    @child_paths
    def experiments(self) -> ExperimentPaths:
        return ExperimentPaths(self.root.path)

    @path_field(kind="dir")
    def validations(self) -> Path:
        return self.root / "validations"

    def validation(self, name: str) -> ValidationPaths:
        return ValidationPaths(self.validations.ensure(), name)


# =======================================================================
#                         SAMPLE PATHS
# =======================================================================


class ProcessPaths(PathsBase):
    def __init__(self, base: PathLike, name: str) -> None:
        super().__init__(base)
        self.name = name

    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / self.name

    @path_field(kind="file")
    def config(self) -> Path:
        return self.root / "config.json"

    @path_field(kind="file")
    def manifest(self) -> Path:
        return self.root / "manifest.json"


class SynthesisPaths(ProcessPaths):
    @path_field(kind="file")
    def voxels(self) -> Path:
        return self.root / "voxels.npy"


class TriangulationPaths(ProcessPaths):
    @path_field(kind="file")
    def mesh(self) -> Path:
        return self.root / "mesh.stl"

    @path_field(kind="file")
    def cadmodel(self) -> Path:
        return self.root / "cad_model.brep"


class MeshingPaths(ProcessPaths):
    @path_field(kind="file")
    def mesh(self) -> Path:
        return self.root / "mesh.msh"


class SolutionPaths(ProcessPaths):
    @path_field(kind="dir")
    def solution(self) -> Path:
        return self.root / "solution.bp"


class SolutionsPaths(PathsBase):
    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "solutions"

    @child_paths
    def photoactive(self) -> SolutionPaths:
        return SolutionPaths(self.root.ensure(), "photoactive")

    @child_paths
    def diffusion(self) -> SolutionPaths:
        return SolutionPaths(self.root.ensure(), "diffusion")


class SamplePaths(PathsBase):
    def __init__(self, base: PathLike, sample_id: str) -> None:
        super().__init__(base)
        self.sample_id = sample_id

    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "samples" / self.sample_id

    @child_paths
    def synthesis(self) -> SynthesisPaths:
        return SynthesisPaths(self.root.ensure(), "synthesis")

    @child_paths
    def triangulation(self) -> TriangulationPaths:
        return TriangulationPaths(self.root.ensure(), "triangulation")

    @child_paths
    def meshing(self) -> MeshingPaths:
        return MeshingPaths(self.root.ensure(), "meshing")

    @child_paths
    def solutions(self) -> SolutionsPaths:
        return SolutionsPaths(self.root.ensure())


# =======================================================================
#                               ROOT PATHS
# =======================================================================


class ProjectPaths(PathsBase):
    @child_paths
    def pipes(self) -> PipePaths:
        return PipePaths(self.base)

    def sample(self, sample_id: str) -> SamplePaths:
        return SamplePaths(self.base, sample_id)
