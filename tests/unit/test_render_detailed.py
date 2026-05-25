"""Tests for Detailed Notes sections and Open Questions."""

from datetime import date

import pytest

from yt2md.models import (
    CURRENT_SCHEMA_VERSION,
    DetailedSection,
    Frontmatter,
    StructuredDoc,
    Transcript,
)
from yt2md.stages.render import render


def _doc(
    sections: list[DetailedSection] | None = None,
    open_questions: list[str] | None = None,
) -> StructuredDoc:
    fm = Frontmatter(
        title="T",
        channel="C",
        url="https://www.youtube.com/watch?v=v",
        video_id="v",
        published=date(2025, 1, 1),
        duration_seconds=60,
        captured_at=date(2026, 5, 23),
        schema_version=CURRENT_SCHEMA_VERSION,
        genre="podcast",
        speakers=["A"],
        topics=[],
        people_mentioned=[],
        works_mentioned=[],
    )
    return StructuredDoc(
        frontmatter=fm,
        tldr="t",
        takeaways=[],
        concepts=[],
        references=[],
        quotes=[],
        sections=sections or [],
        open_questions=open_questions or [],
        speaker_mappings=[],
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
        speakers=["A"],
    )


class TestDetailedSections:
    def test_header_present(self, empty_transcript: Transcript) -> None:
        d = _doc(
            sections=[
                DetailedSection(
                    heading="What dopamine does",
                    body="Huberman explains...",
                    timestamp_s=0.0,
                ),
            ]
        )
        md = render(d, empty_transcript)
        assert "## Detailed Notes" in md

    def test_subheading_with_timestamp(self, empty_transcript: Transcript) -> None:
        d = _doc(
            sections=[
                DetailedSection(
                    heading="Tools to raise dopamine",
                    body="Cold exposure...",
                    timestamp_s=2720.0,
                ),
            ]
        )
        md = render(d, empty_transcript)
        assert "### Tools to raise dopamine" in md
        assert "[45:20]" in md
        assert "&t=2720s" in md

    def test_body_present(self, empty_transcript: Transcript) -> None:
        d = _doc(
            sections=[
                DetailedSection(heading="H", body="Multi-paragraph body content.", timestamp_s=0.0),
            ]
        )
        md = render(d, empty_transcript)
        assert "Multi-paragraph body content." in md


class TestOpenQuestions:
    def test_header_present(self, empty_transcript: Transcript) -> None:
        d = _doc(open_questions=["What about D1 vs D2 receptors?"])
        md = render(d, empty_transcript)
        assert "## Open Questions" in md

    def test_questions_bullet(self, empty_transcript: Transcript) -> None:
        d = _doc(open_questions=["Q1?", "Q2?"])
        md = render(d, empty_transcript)
        assert "- Q1?" in md
        assert "- Q2?" in md

    def test_section_omitted_when_empty(self, empty_transcript: Transcript) -> None:
        d = _doc()
        md = render(d, empty_transcript)
        assert "## Open Questions" not in md
