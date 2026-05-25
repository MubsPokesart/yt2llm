"""Tests for Config precedence: env > TOML > defaults."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from yt2md.config import Config


class TestEnvOverridesDefault:
    def test_output_dir_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_OUTPUT_DIR", "/tmp/yt2md-out")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.output_dir == Path("/tmp/yt2md-out")

    def test_audio_bitrate_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_AUDIO_BITRATE_KBPS", "64")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.audio_bitrate_kbps == 64

    def test_backend_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_TRANSCRIPTION_BACKEND", "local_whisper")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.transcription_backend == "local_whisper"

    def test_invalid_backend_value_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_TRANSCRIPTION_BACKEND", "not_valid")
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        with pytest.raises(ValidationError):
            Config()  # type: ignore[call-arg]


class TestDefaultsAreCompatibleWithProduct:
    def test_default_transcription_model_supports_verbose_json(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The default OpenAI model must return segment+word timestamps.

        yt2llm's Tier 3 markdown requires timestamped segments, which OpenAI only
        delivers via response_format=verbose_json. Only whisper-1 currently supports
        verbose_json; gpt-4o-transcribe(-mini) do not. A default that breaks every
        first-run user is not acceptable.
        """
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.transcription_model == "whisper-1"

    def test_default_structuring_model_is_gemini_2_5_flash(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Pin the structuring model to gemini-2.5-flash.

        Workhorse model on v1beta: stable, well-priced, handles structured output.
        gemini-3-flash is preview-only and 404s on consumer Developer API keys;
        gemini-3.5-flash is materially pricier without commensurate quality gains
        for this workload. A silent swap to either should be a deliberate decision,
        which this pin makes explicit.
        """
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "g")
        cfg = Config()  # type: ignore[call-arg]
        assert cfg.structuring_model == "gemini-2.5-flash"


class TestKwargOverridesEnv:
    def test_kwarg_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_OUTPUT_DIR", "/from/env")
        cfg = Config(google_api_key="g", output_dir=Path("/from/kwarg"))  # type: ignore[arg-type]
        assert cfg.output_dir == Path("/from/kwarg")


class TestSecretsNotLogged:
    def test_secret_str_repr_hidden(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("YT2MD_GOOGLE_API_KEY", "supersecret")
        cfg = Config()  # type: ignore[call-arg]
        assert "supersecret" not in repr(cfg)
        assert cfg.google_api_key.get_secret_value() == "supersecret"
