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

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

T = TypeVar("T")

FINGERPRINT_LENGTH = 12


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


def cached(
    *,
    path: Path,
    produce: Callable[[], T],
    load: Callable[[Path], T],
    dump: Callable[[T, Path], None],
) -> T:
    """Read the artifact at `path` if present; else produce, write atomically, return.

    Atomicity: dump writes to a sibling `.tmp` file, which is then renamed via
    `Path.replace` (POSIX rename is atomic). On producer failure, no artifact is
    left behind.
    """
    if path.exists():
        return load(path)

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        result = produce()
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise

    try:
        dump(result, tmp)
        tmp.replace(path)
    except BaseException:
        if tmp.exists():
            tmp.unlink()
        raise

    return result


def fingerprint(*parts: Any) -> str:  # noqa: ANN401  -- accepts arbitrary JSON-serializable inputs
    """Deterministic short hash of the given parts, suitable for cache keys in filenames.

    Order-sensitive. Stable across processes and Python versions (uses SHA-256 over a
    canonical JSON encoding of the parts tuple).
    """
    canonical = json.dumps(parts, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha256(canonical).hexdigest()
    return digest[:FINGERPRINT_LENGTH]
