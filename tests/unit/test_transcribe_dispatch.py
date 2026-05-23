"""Tests for resolve_backend() — auto, explicit, and fallback semantics."""

from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.errors import ConfigError
from yt2md.stages.transcribe import resolve_backend


def _cfg(**kwargs: object) -> Config:
    defaults: dict[str, object] = {"google_api_key": "g"}
    defaults.update(kwargs)
    return Config(**defaults)  # type: ignore[arg-type]


class TestExplicit:
    def test_explicit_openai_requires_api_key(self) -> None:
        with pytest.raises(ConfigError, match="OPENAI_API_KEY"):
            resolve_backend(_cfg(transcription_backend="openai_transcribe", openai_api_key=None))

    def test_explicit_openai_succeeds_with_key(self) -> None:
        cfg = _cfg(transcription_backend="openai_transcribe", openai_api_key="key")
        assert resolve_backend(cfg) == "openai_transcribe"

    def test_explicit_local_requires_faster_whisper(self) -> None:
        cfg = _cfg(transcription_backend="local_whisper")
        with (
            patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=False),
            pytest.raises(ConfigError, match="faster-whisper"),
        ):
            resolve_backend(cfg)

    def test_explicit_local_succeeds_when_installed(self) -> None:
        cfg = _cfg(transcription_backend="local_whisper")
        with patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=True):
            assert resolve_backend(cfg) == "local_whisper"


class TestAuto:
    def test_auto_picks_openai_when_key_present(self) -> None:
        cfg = _cfg(transcription_backend="auto", openai_api_key="key")
        assert resolve_backend(cfg) == "openai_transcribe"

    def test_auto_falls_back_to_local_when_key_missing(self) -> None:
        cfg = _cfg(transcription_backend="auto", openai_api_key=None)
        with patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=True):
            assert resolve_backend(cfg) == "local_whisper"

    def test_auto_hard_error_when_neither_available(self) -> None:
        cfg = _cfg(transcription_backend="auto", openai_api_key=None)
        with (
            patch("yt2md.stages.transcribe._faster_whisper_installed", return_value=False),
            pytest.raises(ConfigError, match="No transcription backend"),
        ):
            resolve_backend(cfg)
