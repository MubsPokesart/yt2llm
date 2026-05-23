"""Smoke test: render() returns a non-empty string given a minimal StructuredDoc."""

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
def minimal_doc() -> StructuredDoc:
    fm = Frontmatter(
        title="Test",
        channel="TC",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=60,
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
        tldr="A short TLDR sentence.",
        takeaways=[Takeaway(text="One.", timestamp_s=0.0)],
        concepts=[],
        references=[],
        quotes=[],
        sections=[],
        open_questions=[],
        speaker_name_map={"SPEAKER_00": "Alice"},
    )


@pytest.fixture
def empty_transcript() -> Transcript:
    return Transcript(
        language="en",
        duration_s=60.0,
        backend="openai_transcribe",
        model_id="m",
        chunked=False,
        segments=[],
        speakers=["Alice"],
    )


def test_render_returns_string(minimal_doc: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(minimal_doc, empty_transcript)
    assert isinstance(md, str)
    assert md.strip()


def test_render_starts_with_frontmatter(
    minimal_doc: StructuredDoc, empty_transcript: Transcript
) -> None:
    md = render(minimal_doc, empty_transcript)
    assert md.startswith("---\n")


def test_render_contains_title(minimal_doc: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(minimal_doc, empty_transcript)
    assert "Test" in md


def test_render_contains_tldr(minimal_doc: StructuredDoc, empty_transcript: Transcript) -> None:
    md = render(minimal_doc, empty_transcript)
    assert "A short TLDR sentence." in md
