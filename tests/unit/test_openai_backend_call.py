"""Tests for transcribe_openai() — OpenAI SDK call mocked."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from yt2md.config import Config
from yt2md.errors import ConfigError, TranscriptionError
from yt2md.models import VideoMetadata
from yt2md.stages.transcribe_backends.openai import transcribe_openai


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        openai_api_key="okey",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "a.opus"
    p.write_bytes(b"\x00" * 100)
    return p


@pytest.fixture
def fake_metadata(huberman_metadata: VideoMetadata) -> VideoMetadata:
    return huberman_metadata


@pytest.fixture
def fake_response(fixtures_dir: Path) -> SimpleNamespace:
    raw = json.loads(
        (fixtures_dir / "transcripts" / "openai_raw_sample.json").read_text(encoding="utf-8")
    )
    obj = SimpleNamespace()
    obj.model_dump = lambda: raw
    return obj


class TestTranscribeOpenAI:
    def test_returns_transcript_and_raw(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_response: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcript, raw = transcribe_openai(fake_audio, fake_metadata, cfg=cfg)

        assert transcript.backend == "openai_transcribe"
        assert transcript.duration_s == pytest.approx(8.0)
        assert raw == fake_response.model_dump()

    def test_passes_vocab_hint_as_prompt(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_response: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcribe_openai(fake_audio, fake_metadata, cfg=cfg)

            kwargs = client.audio.transcriptions.create.call_args.kwargs
            assert "prompt" in kwargs
            assert "Huberman" in kwargs["prompt"]

    def test_skipped_hint_when_disabled(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_response: SimpleNamespace,
    ) -> None:
        cfg_no_hint = cfg.model_copy(update={"use_transcription_hint": False})
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcribe_openai(fake_audio, fake_metadata, cfg=cfg_no_hint)

            kwargs = client.audio.transcriptions.create.call_args.kwargs
            assert not kwargs.get("prompt")

    def test_uploads_with_ogg_extension_not_opus(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_response: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcribe_openai(fake_audio, fake_metadata, cfg=cfg)

            file_arg = client.audio.transcriptions.create.call_args.kwargs["file"]
            assert isinstance(file_arg, tuple), (
                "OpenAI rejects .opus extension; the SDK call must send "
                "a (filename, fileobj, content_type) tuple with .ogg filename."
            )
            filename = file_arg[0]
            assert filename.endswith(".ogg")
            assert not filename.endswith(".opus")

    def test_api_error_raises_typed(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.side_effect = RuntimeError("boom")
            openai_cls.return_value = client

            with pytest.raises(TranscriptionError):
                transcribe_openai(fake_audio, fake_metadata, cfg=cfg)

    @pytest.mark.parametrize(
        "incompatible_model",
        ["gpt-4o-transcribe", "gpt-4o-mini-transcribe"],
    )
    def test_rejects_models_without_verbose_json_support(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        incompatible_model: str,
    ) -> None:
        """yt2llm needs segment+word timestamps, which only verbose_json carries.

        OpenAI's gpt-4o-transcribe family does not support response_format=verbose_json
        (only 'json' and 'text'), so the API rejects our request with HTTP 400 at runtime.
        We must catch this at config-validation time with a clear remediation message,
        BEFORE issuing the API call.
        """
        cfg_bad = cfg.model_copy(update={"transcription_model": incompatible_model})
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            openai_cls.return_value = client

            with pytest.raises(ConfigError, match=r"verbose_json|whisper-1") as exc_info:
                transcribe_openai(fake_audio, fake_metadata, cfg=cfg_bad)

            assert incompatible_model in str(exc_info.value), (
                "Error message should name the rejected model so the user knows what to fix."
            )
            client.audio.transcriptions.create.assert_not_called()

    def test_accepts_whisper_1_model(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_response: SimpleNamespace,
    ) -> None:
        """whisper-1 is the only OpenAI model that returns verbose_json with timestamps."""
        cfg_ok = cfg.model_copy(update={"transcription_model": "whisper-1"})
        with patch("yt2md.stages.transcribe_backends.openai.OpenAI") as openai_cls:
            client = MagicMock()
            client.audio.transcriptions.create.return_value = fake_response
            openai_cls.return_value = client

            transcript, _raw = transcribe_openai(fake_audio, fake_metadata, cfg=cfg_ok)

        assert transcript.model_id == "whisper-1"
