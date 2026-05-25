"""Tests for Key Takeaways section rendering with timestamp deep-links."""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    Frontmatter,
    StructuredDoc,
    Takeaway,
    Transcript,
)
from yt2md.stages.render import render


@pytest.fixture
def doc_with_takeaways() -> StructuredDoc:
    fm = Frontmatter(
        title="Test",
        channel="TC",
        url="https://www.youtube.com/watch?v=abc",
        video_id="abc",
        published=date(2025, 1, 1),
        duration_seconds=600,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["Alice"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[
            Takeaway(text="Dopamine signals anticipation.", timestamp_s=252.0),
            Takeaway(text="It peaks before reward.", timestamp_s=510.5),
        ],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_mappings=[],
    )


@pytest.fixture
def empty_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=600.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=["Alice"],
    )


def test_section_header_present(
    doc_with_takeaways: StructuredDoc, empty_transcript: Transcript
) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    assert "## Key Takeaways" in md


def test_takeaway_text_present(
    doc_with_takeaways: StructuredDoc, empty_transcript: Transcript
) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    assert "Dopamine signals anticipation." in md


def test_timestamp_link_format(
    doc_with_takeaways: StructuredDoc, empty_transcript: Transcript
) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    # 252s -> 04:12 display, &t=252s URL
    assert "[04:12]" in md
    assert "https://www.youtube.com/watch?v=abc&t=252s" in md


def test_fractional_timestamp_truncated_to_int_seconds(
    doc_with_takeaways: StructuredDoc, empty_transcript: Transcript
) -> None:
    md = render(doc_with_takeaways, empty_transcript)
    # 510.5s -> int(510) -> 08:30
    assert "[08:30]" in md
    assert "&t=510s" in md
