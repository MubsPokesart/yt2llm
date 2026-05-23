"""Tests for transcribe_local() — faster-whisper mocked."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from yt2md.config import Config
from yt2md.errors import ConfigError
from yt2md.models import VideoMetadata
from yt2md.stages.transcribe_backends.local import transcribe_local


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
        local_whisper_model="tiny",
    )


@pytest.fixture
def fake_audio(tmp_path: Path) -> Path:
    p = tmp_path / "a.opus"
    p.write_bytes(b"\x00" * 100)
    return p


@pytest.fixture
def fake_metadata(huberman_metadata: VideoMetadata) -> VideoMetadata:
    return huberman_metadata


def _fake_segment(start: float, end: float, text: str) -> SimpleNamespace:
    word = SimpleNamespace(word=text.strip(), start=start, end=end)
    return SimpleNamespace(
        start=start,
        end=end,
        text=text,
        words=[word],
    )


@pytest.fixture
def fake_segments() -> list[SimpleNamespace]:
    return [
        _fake_segment(0.0, 5.0, "Hello world."),
        _fake_segment(5.0, 10.0, "This is local whisper."),
    ]


@pytest.fixture
def fake_info() -> SimpleNamespace:
    return SimpleNamespace(language="en", duration=10.0)


class TestTranscribeLocal:
    def test_returns_transcript(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_segments: list[SimpleNamespace],
        fake_info: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local._import_faster_whisper") as imp:
            model = MagicMock()
            model.transcribe.return_value = (iter(fake_segments), fake_info)
            imp.return_value = MagicMock(return_value=model)

            t, raw = transcribe_local(fake_audio, fake_metadata, cfg=cfg)

        assert t.backend == "local_whisper"
        assert t.model_id.startswith("faster-whisper")
        assert t.duration_s == pytest.approx(10.0)
        assert t.speakers == []
        assert isinstance(raw, dict)

    def test_segments_normalized(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_segments: list[SimpleNamespace],
        fake_info: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local._import_faster_whisper") as imp:
            model = MagicMock()
            model.transcribe.return_value = (iter(fake_segments), fake_info)
            imp.return_value = MagicMock(return_value=model)

            t, _ = transcribe_local(fake_audio, fake_metadata, cfg=cfg)

        assert len(t.segments) == 2
        assert t.segments[0].text == "Hello world."
        assert t.segments[0].start == pytest.approx(0.0)
        assert t.segments[0].end == pytest.approx(5.0)

    def test_passes_initial_prompt(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
        fake_segments: list[SimpleNamespace],
        fake_info: SimpleNamespace,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local._import_faster_whisper") as imp:
            model = MagicMock()
            model.transcribe.return_value = (iter(fake_segments), fake_info)
            imp.return_value = MagicMock(return_value=model)

            transcribe_local(fake_audio, fake_metadata, cfg=cfg)

            kwargs = model.transcribe.call_args.kwargs
            assert "initial_prompt" in kwargs
            assert "Huberman" in kwargs["initial_prompt"]

    def test_raises_config_error_if_faster_whisper_missing(
        self,
        cfg: Config,
        fake_audio: Path,
        fake_metadata: VideoMetadata,
    ) -> None:
        with patch("yt2md.stages.transcribe_backends.local._import_faster_whisper") as imp:
            imp.side_effect = ImportError("no module")
            with pytest.raises(ConfigError, match="faster-whisper"):
                transcribe_local(fake_audio, fake_metadata, cfg=cfg)
