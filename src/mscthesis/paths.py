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

    @child_paths
    def synthesis(self) -> SynthesisPaths:
        return SynthesisPaths(self.root.ensure(), "synthesis")

    @child_paths
    def triangulation(self) -> TriangulationPaths:
        return TriangulationPaths(self.root.ensure(), "triangulation")

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

    @path_field(kind="file")
    def snapshot(self) -> Path:
        return self.root / "rng_snapshot.json"

    @path_field(kind="file")
    def geometry(self) -> Path:
        return self.root / "geometry.json"


class TriangulationPaths(ProcessPaths):
    @path_field(kind="file")
    def mesh(self) -> Path:
        return self.root / "mesh.stl"

    @path_field(kind="file")
    def cadmodel(self) -> Path:
        return self.root / "cad_model.brep"


class MeshPaths(ProcessPaths):
    @path_field(kind="file")
    def mesh(self) -> Path:
        return self.root / "mesh.msh"


class DiffusionPaths(ProcessPaths):
    @path_field(kind="dir")
    def solution(self) -> Path:
        return self.root / "solution.bp"


class ScanningPaths(ProcessPaths):
    @path_field(kind="file")
    def scan(self) -> Path:
        return self.root / "scan.csv"

    @path_field(kind="dir")
    def plots(self) -> Path:
        return self.root / "plots"


class SamplePaths(PathsBase):
    def __init__(self, base: PathLike, sample_id: str) -> None:
        super().__init__(base)
        self.sample_id = sample_id
        self.format = "02d"
        return

    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / self.sample_id

    @child_paths
    def synthesis(self) -> SynthesisPaths:
        return SynthesisPaths(self.root.ensure(), "synthesis")

    @child_paths
    def triangulation(self) -> TriangulationPaths:
        return TriangulationPaths(self.root.ensure(), "triangulation")

    def meshing(self, specifier: int | None = None) -> MeshPaths:
        if specifier is None:
            return MeshPaths(self.root.ensure(), "meshing")
        elif isinstance(specifier, int):
            return MeshPaths(self.root / "meshing", f"mesh_{specifier:{self.format}}")

    def diffusion(self, specifier: int | None = None) -> DiffusionPaths:
        if specifier is None:
            return DiffusionPaths(self.root.ensure(), "diffusion")
        elif isinstance(specifier, int):
            return DiffusionPaths(
                self.root / "diffusion", f"diffusion_{specifier:{self.format}}"
            )

    def scanning(self, specifier: int | None = None) -> ScanningPaths:
        if specifier is None:
            return ScanningPaths(self.root.ensure(), "scanning")
        elif isinstance(specifier, int):
            scanning_root = self.root / "scanning"
            return ScanningPaths(scanning_root, f"scanning_{specifier:{self.format}}")


class CandidatePaths(PathsBase):
    def __init__(self, base: PathLike, sample_id: str) -> None:
        super().__init__(base)
        self.sample_id = sample_id
        return

    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / self.sample_id

    @child_paths
    def synthesis(self) -> SynthesisPaths:
        return SynthesisPaths(self.root.ensure(), "synthesis")


# =======================================================================
#                               ROOT PATHS
# =======================================================================


class ProjectPaths(PathsBase):
    @child_paths
    def pipes(self) -> PipePaths:
        return PipePaths(self.base)

    @path_field(kind="dir")
    def samples(self) -> Path:
        return self.base / "samples"

    @path_field(kind="dir")
    def candidates(self) -> Path:
        return self.base / "candidates"

    @path_field(kind="dir")
    def selected(self) -> Path:
        return self.base / "selected"

    @path_field(kind="dir")
    def validations(self) -> Path:
        return self.base / "validations"

    @path_field(kind="dir")
    def failures(self) -> Path:
        return self.base / "failures"

    @path_field(kind="dir")
    def results(self) -> Path:
        return self.base / "results"

    @path_field(kind="file")
    def index(self) -> Path:
        return self.base / "index.csv"

    @path_field(kind="file")
    def diffusion_index(self) -> Path:
        return self.results.ensure() / "diffusion_solutions.csv"

    @path_field(kind="file")
    def diffusion_summary(self) -> Path:
        return self.results.ensure() / "diffusion_summary.csv"

    @path_field(kind="file")
    def photoactive_index(self) -> Path:
        return self.results.ensure() / "photoactive_scans.csv"

    @path_field(kind="file")
    def photoactive_summary(self) -> Path:
        return self.results.ensure() / "photoactive_summary.csv"

    @path_field(kind="file")
    def collection(self) -> Path:
        return self.results.ensure() / "collection.csv"

    def sample(self, sample_id: str) -> SamplePaths:
        return SamplePaths(self.samples.ensure(), sample_id)

    def candidate_sample(self, sample_id: str) -> CandidatePaths:
        return CandidatePaths(self.candidates.ensure(), sample_id)

    def selected_sample(self, sample_id: str) -> SamplePaths:
        return SamplePaths(self.selected.ensure(), sample_id)

    def validation(self, name: str) -> ValidationPaths:
        return ValidationPaths(self.validations.ensure(), name)
