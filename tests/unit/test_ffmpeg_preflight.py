"""Tests for ffmpeg / ffprobe preflight and its integration into compress + chunk."""

from pathlib import Path
from unittest.mock import patch

import pytest

from yt2md.config import Config
from yt2md.errors import ConfigError
from yt2md.ffmpeg_preflight import require_ffmpeg_tool
from yt2md.stages.compress import compress


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
        audio_bitrate_kbps=32,
    )


class TestRequireFfmpegTool:
    def test_raises_when_missing(self) -> None:
        with (
            patch("yt2md.ffmpeg_preflight.shutil.which", return_value=None),
            pytest.raises(ConfigError, match="ffmpeg not found"),
        ):
            require_ffmpeg_tool("ffmpeg")

    def test_passes_when_present(self) -> None:
        with patch("yt2md.ffmpeg_preflight.shutil.which", return_value="/usr/bin/ffmpeg"):
            require_ffmpeg_tool("ffmpeg")  # no raise

    def test_message_includes_install_hint(self) -> None:
        with (
            patch("yt2md.ffmpeg_preflight.shutil.which", return_value=None),
            pytest.raises(ConfigError, match="Install ffmpeg"),
        ):
            require_ffmpeg_tool("ffprobe")


class TestCompressPreflight:
    """compress() must surface a typed ConfigError, not raw FileNotFoundError, when
    ffmpeg is missing — this was a real failure mode on a fresh Windows install."""

    def test_compress_raises_config_error_when_ffmpeg_missing(
        self, cfg: Config, tmp_path: Path
    ) -> None:
        src = tmp_path / "source.m4a"
        src.write_bytes(b"x")
        out = tmp_path / "out.opus"
        with (
            patch("yt2md.ffmpeg_preflight.shutil.which", return_value=None),
            pytest.raises(ConfigError, match="ffmpeg not found"),
        ):
            compress(source=src, destination=out, cfg=cfg)
