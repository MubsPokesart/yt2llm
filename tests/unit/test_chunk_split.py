"""Tests for needs_chunking() and split_at_silence() — ffmpeg/ffprobe mocked."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.stages.chunk import needs_chunking, split_at_silence


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(google_api_key="g", cache_dir=tmp_path)  # type: ignore[arg-type]


class TestNeedsChunking:
    def test_small_file_no_chunking(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "small.opus"
        audio.write_bytes(b"\x00" * (5 * 1024 * 1024))  # 5MB
        with patch("yt2md.stages.chunk._ffprobe_duration", return_value=600.0):
            assert needs_chunking(audio, backend="openai_transcribe", cfg=cfg) is False

    def test_large_file_chunks(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "large.opus"
        audio.write_bytes(b"\x00" * (25 * 1024 * 1024))  # 25MB
        with patch("yt2md.stages.chunk._ffprobe_duration", return_value=600.0):
            assert needs_chunking(audio, backend="openai_transcribe", cfg=cfg) is True

    def test_long_duration_chunks_even_if_small_file(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "long.opus"
        audio.write_bytes(b"\x00" * 1000)
        with patch("yt2md.stages.chunk._ffprobe_duration", return_value=4 * 3600.0):
            assert needs_chunking(audio, backend="openai_transcribe", cfg=cfg) is True


class TestSplitAtSilence:
    def test_returns_chunks_with_paths_and_offsets(self, cfg: Config, tmp_path: Path) -> None:
        audio = tmp_path / "in.opus"
        audio.write_bytes(b"\x00" * 1000)

        with (
            patch("yt2md.stages.chunk._ffprobe_duration", return_value=3600.0),
            patch("yt2md.stages.chunk._detect_silences", return_value=[1200.0, 2400.0]),
            patch("yt2md.stages.chunk._cut_chunk") as cut,
        ):
            chunks = split_at_silence(audio, backend="openai_transcribe", cfg=cfg)

        assert len(chunks) == 3
        assert [c.start_offset_s for c in chunks] == [
            pytest.approx(0.0),
            pytest.approx(1200.0),
            pytest.approx(2400.0),
        ]
        assert cut.call_count == 3
