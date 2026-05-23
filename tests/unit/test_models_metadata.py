"""Tests for Chapter and VideoMetadata models."""

from datetime import date

import pytest
from pydantic import ValidationError

from yt2md.models import Chapter, VideoMetadata


class TestChapter:
    def test_chapter(self) -> None:
        c = Chapter(title="Intro", start_s=0.0, end_s=60.0)
        assert c.title == "Intro"

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Chapter(title="bad", start_s=10.0, end_s=5.0)


class TestVideoMetadata:
    def test_minimal(self) -> None:
        m = VideoMetadata(
            video_id="abc123",
            url="https://www.youtube.com/watch?v=abc123",
            title="Test Video",
            channel="Test Channel",
            channel_id="UCxxxx",
            published_date=date(2024, 3, 15),
            duration_s=5025.0,
            description="A test video",
            chapters=[],
            tags=[],
            language=None,
        )
        assert m.video_id == "abc123"
        assert m.published_date == date(2024, 3, 15)
        assert m.chapters == []

    def test_with_chapters(self) -> None:
        m = VideoMetadata(
            video_id="abc",
            url="https://www.youtube.com/watch?v=abc",
            title="T",
            channel="C",
            channel_id="UC1",
            published_date=date(2025, 1, 1),
            duration_s=120.0,
            description="",
            chapters=[
                Chapter(title="Intro", start_s=0.0, end_s=30.0),
                Chapter(title="Main", start_s=30.0, end_s=120.0),
            ],
            tags=["science", "neuroscience"],
            language="en",
        )
        assert len(m.chapters) == 2
        assert m.tags == ["science", "neuroscience"]
