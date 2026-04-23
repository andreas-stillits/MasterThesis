from __future__ import annotations

from dataclasses import dataclass
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


class SynthesisPaths(PathsBase):
    @path_field(kind="dir")
    def root(self) -> Path:
        return self.base / "synthesis"

    @path_field(kind="file")
    def voxels(self) -> Path:
        return self.base / "voxels.npy"

    @path_field(kind="file")
    def config(self) -> Path:
        return self.base / "config.json"

    @path_field(kind="file")
    def manifest(self) -> Path:
        return self.base / "manifest.json"


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
        return SynthesisPaths(self.root.path)


class ProjectPaths(PathsBase):
    @child_paths
    def pipes(self) -> PipePaths:
        return PipePaths(self.base)

    def sample(self, sample_id: str) -> SamplePaths:
        return SamplePaths(self.base, sample_id)
