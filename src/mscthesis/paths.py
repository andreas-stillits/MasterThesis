from __future__ import annotations

from pathlib import Path

from stillib_paths import PathLike, PathsBase, child_paths, path_field

# =======================================================================
#                         PIPE PATHS
# =======================================================================


class ValidationPaths(PathsBase):
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
    def plots(self) -> Path:
        return self.root / "plots.pdf"


class PipeMeshPaths(PathsBase):
    def __init__(self, base: PathLike, index: int) -> None:
        super().__init__(base)
        self.index = index

    @path_field(kind="file")
    def file(self) -> Path:
        return (
            self.base / f"mesh_{self.index:03d}.msh"
        )  # zero-padded index, e.g. 7 -> 007


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

    def mesh(self, index: int) -> PipeMeshPaths:
        return PipeMeshPaths(self.meshes.path, index)


class PipePaths(PathsBase):
    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "pipes"

    @child_paths
    def experiments(self) -> ExperimentPaths:
        self.root.ensure()
        return ExperimentPaths(self.root.path)

    def validation(self, name: str) -> ValidationPaths:
        self.root.ensure()
        return ValidationPaths(self.root.path, name)


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
        self.root.ensure()
        return SolutionPaths(self.root.path, "photoactive")

    @child_paths
    def diffusion(self) -> SolutionPaths:
        self.root.ensure()
        return SolutionPaths(self.root.path, "diffusion")


class SamplePaths(PathsBase):
    def __init__(self, base: PathLike, sample_id: str) -> None:
        super().__init__(base)
        self.sample_id = sample_id

    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "samples" / self.sample_id

    @child_paths
    def synthesis(self) -> SynthesisPaths:
        self.root.ensure()
        return SynthesisPaths(self.root.path, "synthesis")

    @child_paths
    def triangulation(self) -> TriangulationPaths:
        self.root.ensure()
        return TriangulationPaths(self.root.path, "triangulation")

    @child_paths
    def meshing(self) -> MeshingPaths:
        self.root.ensure()
        return MeshingPaths(self.root.path, "meshing")

    @child_paths
    def solutions(self) -> SolutionsPaths:
        self.root.ensure()
        return SolutionsPaths(self.root.path)


# =======================================================================
#                               ROOT PATHS
# =======================================================================


class ProjectPaths(PathsBase):
    @child_paths
    def pipes(self) -> PipePaths:
        return PipePaths(self.base)

    def sample(self, sample_id: str) -> SamplePaths:
        return SamplePaths(self.base, sample_id)
