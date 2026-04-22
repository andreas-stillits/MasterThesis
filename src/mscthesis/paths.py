from __future__ import annotations

from pathlib import Path

from stillib_paths import PathLike, PathsBase, child_paths, path_field


class PipePaths(PathsBase):
    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "pipes"

    @path_field(kind="dir")
    def meshes(self) -> Path:
        self.root.ensure()
        return self.root / "meshes"


class SamplePaths(PathsBase):
    def __init__(self, base: PathLike, sample_id: str) -> None:
        super().__init__(base)
        self.sample_id = sample_id

    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "samples" / self.sample_id

    @path_field(kind="dir")
    def synthesis(self) -> Path:
        self.root.ensure()
        return self.root / "synthesis"


class ProjectPaths(PathsBase):

    @child_paths
    def pipes(self) -> PipePaths:
        return PipePaths(self.base)

    @path_field(kind="dir")
    def samples(self) -> Path:
        return self.base / "samples"

    def sample(self, sample_id: str) -> SamplePaths:
        self.samples.ensure()
        return SamplePaths(self.base, sample_id)
