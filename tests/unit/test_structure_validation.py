"""Tests for validate_structured_doc — semantic checks beyond Pydantic shape."""

from datetime import date

import pytest

from yt2md.errors import InvalidStructuredOutputError
from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    Quote,
    SpeakerMapping,
    StructuredDoc,
    Takeaway,
    Transcript,
    VideoMetadata,
)
from yt2md.stages.structure import validate_structured_doc


def _doc(
    *,
    title: str = "T",
    video_id: str = "vid",
    takeaways: list[Takeaway] | None = None,
    tldr: str = "Non-empty.",
    quotes: list[Quote] | None = None,
    speaker_mappings: list[SpeakerMapping] | None = None,
) -> StructuredDoc:
    return StructuredDoc(
        frontmatter=Frontmatter(
            title=title,
            channel="C",
            url="u",
            video_id=video_id,
            published=date(2025, 1, 1),
            duration_seconds=10,
            captured_at=date(2026, 5, 23),
            schema_version=CURRENT_SCHEMA_VERSION,
            genre="podcast",
            speakers=["A"],
            topics=[],
            people_mentioned=[],
            works_mentioned=[],
        ),
        tldr=tldr,
        takeaways=takeaways or [Takeaway(text="a", timestamp_s=0.0)] * 3,
        concepts=[],
        references=[],
        quotes=quotes or [],
        sections=[],
        open_questions=[],
        speaker_mappings=speaker_mappings or [],
    )


def _meta(title: str = "T", video_id: str = "vid", duration_s: float = 10.0) -> VideoMetadata:
    return VideoMetadata(
        video_id=video_id,
        url="u",
        title=title,
        channel="C",
        channel_id="UC",
        published_date=date(2025, 1, 1),
        duration_s=duration_s,
        description="",
        chapters=[],
        tags=[],
        language=None,
    )


def _transcript(duration_s: float = 10.0) -> Transcript:
    return Transcript(
        language="en",
        duration_s=duration_s,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=[],
    )


class TestValidationSuccess:
    def test_minimal_valid_doc(self) -> None:
        validate_structured_doc(_doc(), transcript=_transcript(), metadata=_meta())


class TestRequiredFields:
    def test_takeaways_must_be_3_or_more(self) -> None:
        d = _doc(takeaways=[Takeaway(text="x", timestamp_s=0.0)])
        with pytest.raises(InvalidStructuredOutputError, match="takeaways"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta())

    def test_tldr_nonempty(self) -> None:
        d = _doc(tldr="   ")
        with pytest.raises(InvalidStructuredOutputError, match="tldr"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta())


class TestFrontmatterConsistency:
    def test_title_matches_metadata(self) -> None:
        d = _doc(title="Mismatch")
        with pytest.raises(InvalidStructuredOutputError, match="title"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta(title="Real"))

    def test_video_id_matches_metadata(self) -> None:
        d = _doc(video_id="X")
        with pytest.raises(InvalidStructuredOutputError, match="video_id"):
            validate_structured_doc(d, transcript=_transcript(), metadata=_meta(video_id="Y"))


class TestTimestampRange:
    def test_takeaway_timestamp_in_range(self) -> None:
        d = _doc(
            takeaways=[
                Takeaway(text="x", timestamp_s=0.0),
                Takeaway(text="y", timestamp_s=5.0),
                Takeaway(text="z", timestamp_s=100.0),  # > duration_s
            ]
        )
        with pytest.raises(InvalidStructuredOutputError, match="timestamp"):
            validate_structured_doc(d, transcript=_transcript(duration_s=10.0), metadata=_meta())

    def test_quote_timestamp_in_range(self) -> None:
        d = _doc(quotes=[Quote(text="q", speaker="A", timestamp_s=999.0)])
        with pytest.raises(InvalidStructuredOutputError, match="timestamp"):
            validate_structured_doc(d, transcript=_transcript(duration_s=10.0), metadata=_meta())
