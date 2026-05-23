"""Tests for Config defaults (no env, no TOML)."""

import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from yt2md.config import Config


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Strip any YT2MD_* env vars so defaults are observable."""
    for key in list(os.environ):
        if key.startswith("YT2MD_") or key in {"OPENAI_API_KEY", "GOOGLE_API_KEY"}:
            monkeypatch.delenv(key, raising=False)


class TestDefaults:
    def test_default_output_dir(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.output_dir == Path("./output")

    def test_default_cache_dir(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.cache_dir == Path("./cache")

    def test_default_audio_bitrate(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.audio_bitrate_kbps == 32

    def test_default_transcription_backend(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.transcription_backend == "auto"

    def test_default_force_and_no_cache_false(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.force is False
        assert cfg.no_cache is False

    def test_google_api_key_required(self) -> None:
        with pytest.raises(ValidationError):
            Config()  # type: ignore[call-arg]

    def test_openai_api_key_optional(self) -> None:
        cfg = Config(google_api_key="g")  # type: ignore[arg-type]
        assert cfg.openai_api_key is None
