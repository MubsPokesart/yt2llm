"""Tests for compress() — ffmpeg subprocess mocked."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.errors import TranscriptionError
from yt2md.stages.compress import compress


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
        audio_bitrate_kbps=32,
    )


class TestCompress:
    def test_invokes_ffmpeg_with_canonical_args(self, cfg: Config, tmp_path: Path) -> None:
        src = tmp_path / "source.m4a"
        src.write_bytes(b"x")
        out = tmp_path / "out.opus"

        with patch("yt2md.stages.compress.subprocess.run") as run:
            run.return_value.returncode = 0
            compress(source=src, destination=out, cfg=cfg)
            args = run.call_args[0][0]

        assert args[0] == "ffmpeg"
        assert "-i" in args
        assert str(src) in args
        assert str(out) in args
        assert "-vn" in args  # no video
        assert "-ac" in args
        assert "1" in args  # mono
        assert "-c:a" in args
        assert "libopus" in args
        assert "-b:a" in args
        assert "32k" in args

    def test_creates_output_directory(self, cfg: Config, tmp_path: Path) -> None:
        src = tmp_path / "source.m4a"
        src.write_bytes(b"x")
        out = tmp_path / "nested" / "out.opus"

        with patch("yt2md.stages.compress.subprocess.run") as run:
            run.return_value.returncode = 0
            compress(source=src, destination=out, cfg=cfg)

        assert out.parent.exists()

    def test_ffmpeg_failure_raises_typed(self, cfg: Config, tmp_path: Path) -> None:
        src = tmp_path / "source.m4a"
        src.write_bytes(b"x")
        out = tmp_path / "out.opus"

        with patch("yt2md.stages.compress.subprocess.run") as run:
            run.side_effect = subprocess.CalledProcessError(1, ["ffmpeg"], stderr="boom")
            with pytest.raises(TranscriptionError, match="ffmpeg"):
                compress(source=src, destination=out, cfg=cfg)
