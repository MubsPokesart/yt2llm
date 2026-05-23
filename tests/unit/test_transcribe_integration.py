"""Tests for transcribe() — backend dispatch with chunking integration."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.models import Segment, Transcript, VideoMetadata, Word
from yt2md.stages.chunk import Chunk
from yt2md.stages.transcribe import transcribe


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
    p = tmp_path / "audio.opus"
    p.write_bytes(b"\x00" * 1000)
    return p


def _tx(duration: float, text: str) -> Transcript:
    return Transcript(
        language="en",
        duration_s=duration,
        backend="openai_transcribe",
        model_id="gpt-4o-transcribe",
        chunked=False,
        segments=[
            Segment(
                start=0.0,
                end=duration,
                text=text,
                speaker=None,
                words=[Word(text=text, start=0.0, end=duration, speaker=None)],
            ),
        ],
        speakers=[],
    )


class TestNoChunking:
    def test_single_pass(
        self, cfg: Config, fake_audio: Path, huberman_metadata: VideoMetadata
    ) -> None:
        t_ret = _tx(60.0, "no chunking happened")
        with (
            patch("yt2md.stages.transcribe.needs_chunking", return_value=False),
            patch("yt2md.stages.transcribe.transcribe_openai", return_value=(t_ret, {})),
        ):
            result, _raw = transcribe(fake_audio, huberman_metadata, cfg=cfg)
        assert result.chunked is False
        assert result.segments[0].text == "no chunking happened"


class TestChunking:
    def test_multi_chunk_stitched(
        self, cfg: Config, fake_audio: Path, huberman_metadata: VideoMetadata
    ) -> None:
        chunk1 = Chunk(path=fake_audio, start_offset_s=0.0, duration_s=30.0)
        chunk2 = Chunk(path=fake_audio, start_offset_s=30.0, duration_s=30.0)

        with (
            patch("yt2md.stages.transcribe.needs_chunking", return_value=True),
            patch("yt2md.stages.transcribe.split_at_silence", return_value=[chunk1, chunk2]),
            patch(
                "yt2md.stages.transcribe.transcribe_openai",
                side_effect=[(_tx(30.0, "first"), {}), (_tx(30.0, "second"), {})],
            ),
        ):
            result, _raw = transcribe(fake_audio, huberman_metadata, cfg=cfg)

        assert result.chunked is True
        assert len(result.segments) == 2
        # Second chunk's segment was offset by 30s
        assert result.segments[1].start == pytest.approx(30.0)
        assert result.segments[1].end == pytest.approx(60.0)
