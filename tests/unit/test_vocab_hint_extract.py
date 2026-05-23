"""Tests for vocab_hint.extract_hints — categorized extraction from VideoMetadata."""

from datetime import date

from yt2md.models import VideoMetadata
from yt2md.vocab_hint import extract_hints


def _meta(*, title: str = "T", channel: str = "C", description: str = "") -> VideoMetadata:
    return VideoMetadata(
        video_id="v",
        url="https://www.youtube.com/watch?v=v",
        title=title,
        channel=channel,
        channel_id="UC1",
        published_date=date(2025, 1, 1),
        duration_s=60.0,
        description=description,
        chapters=[],
        tags=[],
        language="en",
    )


class TestExtractPeople:
    def test_title_case_two_words(self) -> None:
        m = _meta(description="Featuring Andrew Huberman from Stanford.")
        h = extract_hints(m)
        assert "Andrew Huberman" in h.people

    def test_title_case_three_words(self) -> None:
        m = _meta(description="An interview with Mary Lou Jepsen.")
        h = extract_hints(m)
        assert "Mary Lou Jepsen" in h.people

    def test_lowercase_not_extracted(self) -> None:
        m = _meta(description="just regular sentence content here")
        h = extract_hints(m)
        assert h.people == []


class TestExtractTitleAndChannel:
    def test_title_present(self) -> None:
        m = _meta(title="Dopamine and Drive")
        h = extract_hints(m)
        assert h.title == "Dopamine and Drive"

    def test_channel_present(self) -> None:
        m = _meta(channel="Huberman Lab")
        h = extract_hints(m)
        assert h.channel == "Huberman Lab"
