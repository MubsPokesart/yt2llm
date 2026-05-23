"""Transcribe stage: backend dispatcher.

Backends live in stages/transcribe_backends/. This module:
  - resolves which backend to use given Config (auto / explicit)
  - dispatches to the chosen backend
  - is the place chunking integrates (added in a later task)
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Any, Literal

from yt2md.errors import ConfigError
from yt2md.stages.chunk import needs_chunking, split_at_silence, stitch_transcripts
from yt2md.stages.transcribe_backends.local import transcribe_local
from yt2md.stages.transcribe_backends.openai import transcribe_openai

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from yt2md.config import Config
    from yt2md.models import Transcript, VideoMetadata

ResolvedBackend = Literal["openai_transcribe", "local_whisper"]


def resolve_backend(cfg: Config) -> ResolvedBackend:
    """Resolve the transcription backend based on Config and installed packages.

    Raises ConfigError if the configured (or auto-resolved) backend is unavailable.
    """
    choice = cfg.transcription_backend

    if choice == "openai_transcribe":
        if cfg.openai_api_key is None:
            msg = "transcription_backend=openai_transcribe requires OPENAI_API_KEY to be set"
            raise ConfigError(msg)
        return "openai_transcribe"

    if choice == "local_whisper":
        if not _faster_whisper_installed():
            msg = (
                "transcription_backend=local_whisper requires faster-whisper (the [local] extra). "
                "Install with: pip install yt2llm[local]"
            )
            raise ConfigError(msg)
        return "local_whisper"

    # Auto mode: prefer openai, fall back to local, error if neither available.
    if cfg.openai_api_key is not None:
        return "openai_transcribe"
    if _faster_whisper_installed():
        return "local_whisper"
    msg = (
        "No transcription backend available. Set OPENAI_API_KEY or install "
        "the local extra: pip install yt2llm[local]"
    )
    raise ConfigError(msg)


def _faster_whisper_installed() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def transcribe(
    audio: Path,
    metadata: VideoMetadata,
    *,
    cfg: Config,
) -> tuple[Transcript, list[dict[str, Any]]]:
    """Transcribe `audio`. Dispatches to backend, chunks if needed.

    Returns (stitched_transcript, list_of_raw_responses).
    """
    backend = resolve_backend(cfg)
    backend_fn = _backend_function(backend)

    if not needs_chunking(audio, backend=backend, cfg=cfg):
        transcript, raw = backend_fn(audio, metadata, cfg=cfg)
        return transcript, [raw]

    chunks = split_at_silence(audio, backend=backend, cfg=cfg)
    chunk_transcripts: list[Transcript] = []
    chunk_raws: list[dict[str, Any]] = []
    for chunk in chunks:
        t, raw = backend_fn(chunk.path, metadata, cfg=cfg)
        chunk_transcripts.append(t)
        chunk_raws.append(raw)

    stitched = stitch_transcripts(
        chunk_transcripts,
        offsets_s=[c.start_offset_s for c in chunks],
    )
    return stitched, chunk_raws


def _backend_function(
    backend: ResolvedBackend,
) -> Callable[..., tuple[Transcript, dict[str, Any]]]:
    if backend == "openai_transcribe":
        return transcribe_openai
    if backend == "local_whisper":
        return transcribe_local
    msg = f"Unknown resolved backend: {backend}"
    raise ConfigError(msg)
