"""On-disk artifact cache for yt2llm.

Each stage's output lives at a path that includes a short hash of everything that
affects it. Stale cache hits are impossible by construction: change a parameter ->
new hash -> cache miss -> stage runs.

This module exposes:
  - ArtifactPaths: resolves the canonical on-disk path for each stage's output.
  - cached(): the one-and-only stage wrapper. Loads from disk if present; else
    invokes the producer and writes the result atomically (temp + rename).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class ArtifactPaths:
    """Resolves canonical cache paths for a single video.

    Hashes are caller-supplied (the pipeline computes them from stage inputs).
    """

    cache_dir: Path
    video_id: str

    @property
    def root(self) -> Path:
        return self.cache_dir / self.video_id

    @property
    def metadata(self) -> Path:
        return self.root / "metadata.json"

    @property
    def metadata_raw(self) -> Path:
        return self.root / "metadata.raw.json"

    def audio(self, *, compression_hash: str) -> Path:
        return self.root / f"audio-{compression_hash}.opus"

    def transcript(self, *, input_hash: str) -> Path:
        return self.root / f"transcript-{input_hash}.json"

    def transcript_raw(self, *, input_hash: str) -> Path:
        return self.root / f"transcript-{input_hash}.raw.json"

    def cleaned(self, *, input_hash: str) -> Path:
        return self.root / f"cleaned-{input_hash}.json"

    def structured(self, *, input_hash: str) -> Path:
        return self.root / f"structured-{input_hash}.json"
