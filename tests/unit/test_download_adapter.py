"""Tests for yt-dlp info_dict → VideoMetadata adapter (no network)."""

import json
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from yt2md.errors import LivestreamNotEndedError, VideoUnavailableError
from yt2md.models import VideoMetadata
from yt2md.stages.download import normalize_metadata


@pytest.fixture
def raw_info_dict(fixtures_dir: Path) -> dict[str, Any]:
    return json.loads(
        (fixtures_dir / "metadata" / "ytdlp_raw_sample.json").read_text(encoding="utf-8")
    )


class TestNormalizeMetadata:
    def test_extracts_basic_fields(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert m.video_id == "abc123"
        assert m.title == "Dopamine, Motivation & Drive"
        assert m.channel == "Huberman Lab"
        assert m.channel_id == "UCxxxx"
        assert m.duration_s == pytest.approx(5025.0)

    def test_upload_date_to_published_date(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert m.published_date == date(2024, 3, 15)

    def test_chapters_mapped(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert len(m.chapters) == 2
        assert m.chapters[0].title == "Introduction"
        assert m.chapters[0].start_s == pytest.approx(0.0)
        assert m.chapters[0].end_s == pytest.approx(60.0)

    def test_missing_chapters_yields_empty_list(self) -> None:
        m = normalize_metadata({
            "id": "x",
            "title": "T",
            "channel": "C",
            "channel_id": "UC",
            "upload_date": "20240101",
            "duration": 60,
            "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        })
        assert m.chapters == []

    def test_missing_tags_yields_empty_list(self) -> None:
        m = normalize_metadata({
            "id": "x",
            "title": "T",
            "channel": "C",
            "channel_id": "UC",
            "upload_date": "20240101",
            "duration": 60,
            "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        })
        assert m.tags == []

    def test_missing_language_is_none(self) -> None:
        m = normalize_metadata({
            "id": "x",
            "title": "T",
            "channel": "C",
            "channel_id": "UC",
            "upload_date": "20240101",
            "duration": 60,
            "description": "",
            "webpage_url": "https://www.youtube.com/watch?v=x",
        })
        assert m.language is None

    def test_returns_videometadata_instance(self, raw_info_dict: dict[str, Any]) -> None:
        m = normalize_metadata(raw_info_dict)
        assert isinstance(m, VideoMetadata)


class TestLiveDetection:
    def test_is_live_raises_livestream_error(self) -> None:
        with pytest.raises(LivestreamNotEndedError):
            normalize_metadata({
                "id": "x",
                "title": "T",
                "channel": "C",
                "channel_id": "UC",
                "upload_date": "20240101",
                "duration": 0,
                "description": "",
                "webpage_url": "https://www.youtube.com/watch?v=x",
                "is_live": True,
            })

    def test_live_status_post_live_ok(self, raw_info_dict: dict[str, Any]) -> None:
        # A finished livestream is fine — has full audio.
        raw_info_dict["live_status"] = "post_live"
        m = normalize_metadata(raw_info_dict)
        assert m.video_id == "abc123"


class TestNoAudio:
    def test_zero_duration_raises_unavailable(self) -> None:
        with pytest.raises(VideoUnavailableError):
            normalize_metadata({
                "id": "x",
                "title": "T",
                "channel": "C",
                "channel_id": "UC",
                "upload_date": "20240101",
                "duration": 0,
                "description": "",
                "webpage_url": "https://www.youtube.com/watch?v=x",
            })
