"""Transcribe stage: backend dispatcher.

Backends live in stages/transcribe_backends/. This module:
  - resolves which backend to use given Config (auto / explicit)
  - dispatches to the chosen backend
  - is the place chunking integrates (added in a later task)
"""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING, Literal

from yt2md.errors import ConfigError

if TYPE_CHECKING:
    from yt2md.config import Config

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
