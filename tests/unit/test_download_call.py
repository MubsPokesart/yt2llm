"""Tests for download() — yt-dlp call mocked at the SDK boundary."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from yt_dlp.utils import DownloadError as YtdlError  # type: ignore[import-untyped]

from yt2md.config import Config
from yt2md.errors import LivestreamNotEndedError, VideoUnavailableError
from yt2md.stages.download import download


@pytest.fixture
def cfg(tmp_path: Path) -> Config:
    return Config(
        google_api_key="g",  # type: ignore[arg-type]
        cache_dir=tmp_path,
        output_dir=tmp_path / "out",
    )


@pytest.fixture
def fake_info(fixtures_dir: Path) -> dict[str, Any]:
    return json.loads(
        (fixtures_dir / "metadata" / "ytdlp_raw_sample.json").read_text(encoding="utf-8")
    )


class TestDownloadHappyPath:
    def test_returns_audio_path_metadata_and_raw(
        self, cfg: Config, fake_info: dict[str, Any], tmp_path: Path
    ) -> None:
        # Simulate yt-dlp writing an audio file and producing info_dict.
        fake_audio = tmp_path / "abc123" / "source_audio.m4a"
        fake_audio.parent.mkdir(parents=True)
        fake_audio.write_bytes(b"\x00" * 100)

        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda _self: _self
        fake_ydl.__exit__ = lambda _self, *_args: False
        fake_ydl.extract_info.return_value = fake_info
        fake_ydl.prepare_filename.return_value = str(fake_audio)

        with patch("yt2md.stages.download.YoutubeDL", return_value=fake_ydl):
            audio_path, metadata, raw = download(
                "https://www.youtube.com/watch?v=abc123",
                cfg=cfg,
            )

        assert audio_path == fake_audio
        assert metadata.video_id == "abc123"
        assert raw == fake_info


class TestDownloadErrorMapping:
    def test_yt_dlp_private_raises_video_unavailable(self, cfg: Config) -> None:
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda _self: _self
        fake_ydl.__exit__ = lambda _self, *_args: False
        fake_ydl.extract_info.side_effect = YtdlError("ERROR: [youtube] Private video")

        with (
            patch("yt2md.stages.download.YoutubeDL", return_value=fake_ydl),
            pytest.raises(VideoUnavailableError),
        ):
            download("https://www.youtube.com/watch?v=x", cfg=cfg)


class TestDownloadLivestream:
    def test_live_video_raises_livestream_error(
        self, cfg: Config, fake_info: dict[str, Any]
    ) -> None:
        fake_info["is_live"] = True
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda _self: _self
        fake_ydl.__exit__ = lambda _self, *_args: False
        fake_ydl.extract_info.return_value = fake_info

        with (
            patch("yt2md.stages.download.YoutubeDL", return_value=fake_ydl),
            pytest.raises(LivestreamNotEndedError),
        ):
            download("https://www.youtube.com/watch?v=x", cfg=cfg)


class TestCookiesPassthrough:
    def test_cookies_from_browser_in_yt_dlp_opts(self, cfg: Config, tmp_path: Path) -> None:
        cfg_with_cookies = cfg.model_copy(update={"cookies_from_browser": "firefox"})
        fake_audio = tmp_path / "x" / "source_audio.m4a"
        fake_audio.parent.mkdir(parents=True)
        fake_audio.write_bytes(b"x")
        fake_ydl = MagicMock()
        fake_ydl.__enter__ = lambda _self: _self
        fake_ydl.__exit__ = lambda _self, *_args: False
        fake_ydl.extract_info.return_value = {
            "id": "x",
            "title": "T",
            "channel": "C",
            "channel_id": "UC",
            "upload_date": "20240101",
            "duration": 60,
            "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        }
        fake_ydl.prepare_filename.return_value = str(fake_audio)

        with patch("yt2md.stages.download.YoutubeDL") as ydl_class:
            ydl_class.return_value = fake_ydl
            download("https://www.youtube.com/watch?v=x", cfg=cfg_with_cookies)
            opts = ydl_class.call_args[0][0]
            assert opts.get("cookiesfrombrowser") == ("firefox",)
